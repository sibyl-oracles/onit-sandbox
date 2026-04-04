"""
Sandbox MCP Server — tool definitions for safe code execution.

Provides an isolated Docker-based sandbox for developing, running, and testing
projects without affecting the host system. Each session gets its own container
with resource limits, network isolation, and a shared home directory.

Tools:
- sandbox_bash: Execute shell commands (supports background mode, network, git, pip)
- sandbox_check_job: Check status/output of background jobs
- sandbox_get_status: Inspect sandbox state, packages, and resource usage
- sandbox_write_file: Write files into the sandbox
- sandbox_download_file: Copy a file from the sandbox to the local filesystem
- sandbox_upload_file: Copy a file/directory from host into the sandbox
- sandbox_enable_network: Enable persistent internet access
- sandbox_disable_network: Disable internet access
- sandbox_stop: Stop and remove the sandbox container
- sandbox_github_create_repo: Create a new private/public GitHub repository
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import functools
import json
import logging
import os
import platform
import queue
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, Literal, TypeVar, overload

from mcp.server.fastmcp import Context
from pydantic import Field

from onit_sandbox.server import (
    CONTAINER_LABEL,
    DEFAULT_CPU_QUOTA,
    DEFAULT_DATA_MOUNTS,
    DEFAULT_GPU_DEVICES,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_PIP_CACHE_PATH,
    DEFAULT_SHM_SIZE,
    DEFAULT_TIMEOUT,
    FALLBACK_IMAGE,
    MAX_OUTPUT_BYTES,
    SANDBOX_IMAGE,
    SandboxMCPServer,
    parse_data_mounts,
)

logger = logging.getLogger(__name__)

IS_MACOS = platform.system() == "Darwin"


def _container_uid_gid() -> tuple[int, int]:
    """Return (uid, gid) to use for the container's --user flag.

    On Linux, use the host UID/GID so files on bind-mounted volumes have
    correct ownership.  On macOS, Docker Desktop runs containers in a Linux VM
    with transparent file-ownership mapping, so we use a fixed 1000:1000 to
    avoid issues with the host UID (which may not exist in the container).
    """
    if IS_MACOS:
        return 1000, 1000
    return os.getuid(), os.getgid()


@dataclass
class ContainerInfo:
    """Information about a managed container."""

    container_id: str
    session_id: str
    data_path: str
    created_at: datetime
    image: str
    status: str = "created"
    network_enabled: bool = False
    installed_packages: list = field(default_factory=list)


class DockerNotAvailableError(Exception):
    """Raised when Docker is not available on the system."""

    pass


class SandboxManager:
    """
    Manages Docker containers as per-session sandboxes.

    Provides lazy container creation, resource limiting, and cleanup.
    Falls back to local execution if Docker is not available.
    """

    def __init__(self) -> None:
        self._containers: dict[str, ContainerInfo] = {}
        self._lock = threading.Lock()
        self._docker_available: bool | None = None
        self._gpu_available: bool | None = None
        self._gpu_devices: str = DEFAULT_GPU_DEVICES
        self._uid, self._gid = _container_uid_gid()

    @staticmethod
    def _load_github_token() -> str | None:
        """Load GitHub token from keychain or file-based fallback."""
        from onit_sandbox.cli import load_github_token

        return load_github_token()

    @staticmethod
    def _load_hf_token() -> str | None:
        """Load HuggingFace token from keychain or file-based fallback."""
        from onit_sandbox.cli import load_hf_token

        return load_hf_token()

    def _check_docker(self) -> bool:
        """Check if Docker is available and running."""
        if self._docker_available is not None:
            return self._docker_available

        try:
            result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            self._docker_available = result.returncode == 0
            if self._docker_available:
                logger.info("Docker is available")
            else:
                logger.warning("Docker is not running")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._docker_available = False
            logger.warning("Docker is not available on this system")

        return self._docker_available

    def _check_gpu(self) -> bool:
        """Check if a usable NVIDIA GPU is available in Docker.

        First checks that the NVIDIA runtime is registered, then verifies
        an actual GPU is accessible by running ``nvidia-smi`` in a
        throwaway container.  The result is cached so the checks only run
        once per process lifetime.
        """
        if self._gpu_available is not None:
            return self._gpu_available

        try:
            # Step 1: Check if the NVIDIA runtime is registered
            result = subprocess.run(
                ["docker", "info", "-f", "{{json .Runtimes}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            has_runtime = result.returncode == 0 and "nvidia" in result.stdout.lower()

            if not has_runtime:
                self._gpu_available = False
                logger.debug("No NVIDIA GPU runtime found — containers will run CPU-only")
                return self._gpu_available

            # Step 2: Verify an actual GPU is accessible
            gpus_flag = self._docker_gpus_flag()
            result = subprocess.run(
                [
                    "docker",
                    "run",
                    "--rm",
                    "--gpus",
                    gpus_flag,
                    "nvidia/cuda:12.0.0-base-ubuntu22.04",
                    "nvidia-smi",
                    "--query-gpu=name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self._gpu_available = result.returncode == 0 and bool(result.stdout.strip())

            if self._gpu_available:
                gpu_names = result.stdout.strip()
                logger.info("NVIDIA GPU verified — %s", gpu_names)
            else:
                logger.warning(
                    "NVIDIA runtime is installed but no GPU is accessible: %s",
                    result.stderr.strip(),
                )

        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._gpu_available = False
            logger.debug("GPU check failed — containers will run CPU-only")

        return self._gpu_available

    def _docker_gpus_flag(self) -> str:
        """Return the value for ``docker run --gpus <value>``.

        When ``_gpu_devices`` is ``"all"`` we pass ``"all"``; otherwise we
        pass ``'"device=0"'`` / ``'"device=0,1"'`` so Docker maps only the
        requested GPU(s).
        """
        devices = self._gpu_devices.strip()
        if devices.lower() == "all":
            return "all"
        return f'"device={devices}"'

    def _check_image_exists(self, image: str) -> bool:
        """Check if a Docker image exists locally."""
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, timeout=10
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _get_container_name(self, session_id: str) -> str:
        """Generate a container name from session ID."""
        safe_id = session_id.replace("-", "")[:12]
        return f"onit-sandbox-{safe_id}"

    def _create_container(
        self,
        session_id: str,
        data_path: str,
        extra_mounts: list[dict[str, str]] | None = None,
    ) -> ContainerInfo:
        """Create a new Docker container for the session.

        Args:
            session_id: Unique session identifier.
            data_path: Host path for the /workspace volume.
            extra_mounts: Additional volume mounts, each a dict with keys
                ``host``, ``container``, and ``mode`` (``ro`` or ``rw``).
        """
        if not self._check_docker():
            raise DockerNotAvailableError("Docker is not available")

        image = SANDBOX_IMAGE if self._check_image_exists(SANDBOX_IMAGE) else FALLBACK_IMAGE
        if image == FALLBACK_IMAGE:
            logger.warning(
                "Using fallback image %s. Run 'docker/build.sh' to build optimized image.",
                FALLBACK_IMAGE,
            )

        container_name = self._get_container_name(session_id)
        os.makedirs(data_path, exist_ok=True)
        pip_cache = os.path.abspath(DEFAULT_PIP_CACHE_PATH)
        os.makedirs(pip_cache, exist_ok=True)

        # Start on bridge so the network stack is fully initialized (DNS,
        # routing tables, etc.), then immediately disconnect.  This lets
        # enable_network/disable_network reconnect reliably later — unlike
        # --network=none which creates a minimal namespace that can't be
        # reconnected to bridge on many Docker Desktop versions.
        gpu_available = self._check_gpu()

        cmd = [
            "docker",
            "run",
            "--detach",
            "--name",
            container_name,
            "--label",
            f"{CONTAINER_LABEL}=true",
            "--label",
            f"{CONTAINER_LABEL}.session_id={session_id}",
            "--label",
            f"{CONTAINER_LABEL}.created_at={datetime.now().isoformat()}",
            "--volume",
            f"{os.path.abspath(data_path)}:/workspace:rw",
            "--volume",
            f"{pip_cache}:/home/sandbox/.cache/pip:rw",
            "--workdir",
            "/workspace",
            "--memory",
            DEFAULT_MEMORY_LIMIT,
            "--pids-limit",
            str(DEFAULT_PIDS_LIMIT),
            "--shm-size",
            DEFAULT_SHM_SIZE,
            "--ulimit",
            "memlock=-1:-1",
            "--ulimit",
            "stack=67108864:67108864",
            "--dns",
            "8.8.8.8",
            "--dns",
            "8.8.4.4",
            "--user",
            f"{self._uid}:{self._gid}",
            "-e",
            "HOME=/home/sandbox",
            "-e",
            "PATH=/home/sandbox/.local/bin:/usr/local/bin:/usr/bin:/bin",
            "-e",
            f"OMP_NUM_THREADS={os.cpu_count() or 4}",
            "-e",
            "HF_HOME=/home/sandbox/.cache/huggingface",
            "-e",
            "TRANSFORMERS_CACHE=/home/sandbox/.cache/huggingface/transformers",
            "-e",
            "CUDA_DEVICE_ORDER=PCI_BUS_ID",
            "-e",
            "NCCL_P2P_DISABLE=0",
            "-e",
            "NCCL_IB_DISABLE=1",
            "-e",
            "TORCH_CUDNN_V8_API_ENABLED=1",
            "-e",
            "TOKENIZERS_PARALLELISM=true",
            "-e",
            f"MKL_NUM_THREADS={os.cpu_count() or 4}",
            "-e",
            f"NUMEXPR_MAX_THREADS={os.cpu_count() or 4}",
        ]

        # Inject GitHub token for git authentication if configured
        github_token = self._load_github_token()
        if github_token:
            cmd.extend(["-e", f"GITHUB_TOKEN={github_token}"])

        # Inject HuggingFace token if configured
        hf_token = self._load_hf_token()
        if hf_token:
            cmd.extend(["-e", f"HF_TOKEN={hf_token}"])

        # Only apply CPU quota if explicitly set (0 = no limit)
        if DEFAULT_CPU_QUOTA > 0:
            cmd.extend(["--cpu-quota", str(DEFAULT_CPU_QUOTA)])

        # Add extra data volume mounts (e.g. /data:/data:rw)
        for mount in extra_mounts or []:
            cmd.extend(
                [
                    "--volume",
                    f"{mount['host']}:{mount['container']}:{mount['mode']}",
                ]
            )

        if gpu_available:
            cmd.extend(["--gpus", self._docker_gpus_flag()])

        cmd.extend(["--rm", image, "sleep", "infinity"])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                raise RuntimeError(f"Failed to create container: {result.stderr}")

            container_id = result.stdout.strip()

            # Ensure home directory exists (critical for fallback image)
            subprocess.run(
                [
                    "docker",
                    "exec",
                    "-u",
                    "0",
                    container_id,
                    "sh",
                    "-c",
                    "mkdir -p /home/sandbox/.cache/pip /home/sandbox/.local"
                    f" && chown -R {self._uid}:{self._gid} /home/sandbox",
                ],
                capture_output=True,
                timeout=10,
            )

            # Add passwd entry for the host UID so pwd.getpwuid() works
            self._ensure_passwd_entry(container_id)

            # Always configure git identity and safe directory
            self._configure_git_defaults(container_id)

            # Configure git credential helper if GitHub token is available
            if github_token:
                self._configure_git_credentials(container_id)

            # Disconnect from bridge immediately — sandbox is network-isolated
            # by default.  enable_network() will reconnect when needed.
            self.disable_network(container_id)

            info = ContainerInfo(
                container_id=container_id,
                session_id=session_id,
                data_path=data_path,
                created_at=datetime.now(),
                image=image,
                status="running",
            )
            logger.info("Created container %s for session %s", container_name, session_id)
            return info

        except subprocess.TimeoutExpired:
            raise RuntimeError("Timeout creating container")

    def _ensure_passwd_entry(self, container_id: str) -> None:
        """Ensure the container has a passwd entry for the current UID.

        Without this, Python's ``getpass.getuser()`` / ``pwd.getpwuid()``
        raises ``KeyError`` because the host UID has no entry in the
        container's ``/etc/passwd``.
        """
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0",
                container_id,
                "sh",
                "-c",
                f"grep -q ':{self._uid}:' /etc/passwd || "
                f"echo 'sandbox:x:{self._uid}:{self._gid}:sandbox:/home/sandbox:/bin/sh' >> /etc/passwd",
            ],
            capture_output=True,
            timeout=10,
        )

    def _configure_git_defaults(self, container_id: str) -> None:
        """Set git identity and safe directory so commits and operations work out-of-the-box."""
        setup_script = (
            "git config --global user.name 'Sandbox User' && "
            "git config --global user.email 'sandbox@onit.local' && "
            "git config --global --add safe.directory '*' && "
            "git config --global init.defaultBranch main"
        )
        result = subprocess.run(
            ["docker", "exec", container_id, "sh", "-c", setup_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to configure git defaults in container %s: %s",
                container_id,
                result.stderr,
            )

    def _configure_git_credentials(self, container_id: str) -> None:
        """Configure git inside the container to use the GitHub token.

        Sets up a credential helper script that returns the token from the
        GITHUB_TOKEN environment variable, and configures git to use it.
        """
        # Create a git credential helper script that reads GITHUB_TOKEN env var
        # and set git config to use it.  Also mark /workspace as safe.
        setup_script = (
            "mkdir -p /home/sandbox/.local/bin && "
            "cat > /home/sandbox/.local/bin/git-credential-github-token << 'SCRIPT'\n"
            "#!/bin/sh\n"
            '# Only respond to "get" requests\n'
            'if [ "$1" != "get" ]; then exit 0; fi\n'
            "# Read input to check for github.com\n"
            "while IFS= read -r line; do\n"
            '  case "$line" in\n'
            "    host=github.com) MATCH=1 ;;\n"
            '    "") break ;;\n'
            "  esac\n"
            "done\n"
            '[ "$MATCH" = "1" ] || exit 0\n'
            'echo "protocol=https"\n'
            'echo "host=github.com"\n'
            'echo "username=x-access-token"\n'
            'echo "password=$GITHUB_TOKEN"\n'
            'echo ""\n'
            "SCRIPT\n"
            "chmod +x /home/sandbox/.local/bin/git-credential-github-token && "
            "git config --global credential.helper github-token && "
            "git config --global --add safe.directory '*' && "
            "git config --global 'url.https://github.com/.insteadOf' 'git@github.com:'"
        )
        result = subprocess.run(
            ["docker", "exec", container_id, "sh", "-c", setup_script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            logger.warning(
                "Failed to configure git credentials in container %s: %s",
                container_id,
                result.stderr,
            )

    def get_or_create_container(
        self,
        session_id: str,
        data_path: str,
        extra_mounts: list[dict[str, str]] | None = None,
    ) -> ContainerInfo:
        """Get existing container or create a new one for the session."""
        with self._lock:
            if session_id in self._containers:
                info = self._containers[session_id]
                if self._is_container_running(info.container_id):
                    self._ensure_passwd_entry(info.container_id)
                    return info
                else:
                    del self._containers[session_id]

            info = self._create_container(session_id, data_path, extra_mounts=extra_mounts)
            self._containers[session_id] = info
            return info

    def _is_container_running(self, container_id: str) -> bool:
        """Check if a container is still running."""
        try:
            result = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() == "true"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    @staticmethod
    def _is_container_dead_error(output: str) -> bool:
        """Return True if the output indicates the container is dead or gone."""
        markers = (
            "no such container",
            "is not running",
            "is restarting",
            "container is paused",
            "has been removed",
            "can not exec in a stopped",
            "can not exec in a dead",
        )
        lower = output.lower()
        return any(m in lower for m in markers)

    @staticmethod
    def _is_transient_docker_error(output: str) -> bool:
        """Return True if the output indicates a transient Docker daemon error."""
        markers = (
            "connection refused",
            "i/o timeout",
            "daemon is not running",
            "tls handshake timeout",
            "temporary failure",
        )
        lower = output.lower()
        return any(m in lower for m in markers)

    @staticmethod
    def _build_exec_cmd(
        container_id: str,
        command: str,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
    ) -> list[str]:
        """Build a ``docker exec`` command list."""
        cmd = ["docker", "exec", "-w", workdir]
        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([container_id, "sh", "-c", command])
        return cmd

    @overload
    def exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int = ...,
        workdir: str = ...,
        env: dict[str, str] | None = ...,
        *,
        split_output: Literal[True],
    ) -> tuple[int, str, str]: ...

    @overload
    def exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int = ...,
        workdir: str = ...,
        env: dict[str, str] | None = ...,
        split_output: Literal[False] = ...,
    ) -> tuple[int, str]: ...

    def exec_in_container(
        self,
        container_id: str,
        command: str,
        timeout: int = DEFAULT_TIMEOUT,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        split_output: bool = False,
    ) -> tuple[int, str] | tuple[int, str, str]:
        """Execute a command inside a container.

        When split_output is True, returns (returncode, stdout, stderr).
        Otherwise returns (returncode, combined_output) for backward compat.
        Retries once on transient Docker daemon errors.
        """
        cmd = self._build_exec_cmd(container_id, command, workdir, env)

        for attempt in range(2):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                if result.returncode != 0 and attempt == 0:
                    combined = result.stdout + result.stderr
                    if self._is_transient_docker_error(combined):
                        logger.warning("Transient Docker error, retrying: %s", combined[:200])
                        time.sleep(1)
                        continue
                if split_output:
                    return result.returncode, result.stdout, result.stderr
                return result.returncode, result.stdout + result.stderr
            except subprocess.TimeoutExpired:
                if split_output:
                    return -1, "", f"Command timed out after {timeout} seconds"
                return -1, f"Command timed out after {timeout} seconds"
        # Should not reach here, but satisfy type checker
        if split_output:
            return -1, "", "Execution failed after retries"
        return -1, "Execution failed after retries"

    def exec_in_container_streaming(
        self,
        container_id: str,
        command: str,
        timeout: int | None = DEFAULT_TIMEOUT,
        workdir: str = "/workspace",
        env: dict[str, str] | None = None,
        on_output: Callable[[str], None] | None = None,
        activity_callback: Callable[[], None] | None = None,
        activity_interval: int = 60,
    ) -> tuple[int, str, str]:
        """Execute a command inside a container, streaming output line-by-line.

        Calls *on_output* with each line of merged stdout/stderr as it arrives.
        Returns (returncode, stdout, stderr) like the split_output variant.

        If *activity_callback* is provided it is called every *activity_interval*
        seconds while the command is running, regardless of output.

        Environment is extended with PYTHONUNBUFFERED=1 and TERM=dumb so that
        Python flushes immediately and tqdm falls back to newline-based output
        instead of ``\\r`` overwrites.
        """
        if env is None:
            env = {}
        # Force unbuffered Python and dumb terminal for readable progress bars
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("TERM", "dumb")

        cmd = self._build_exec_cmd(container_id, command, workdir, env)

        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        output_queue: queue.Queue[tuple[str, str]] = queue.Queue()

        def _reader(stream: Any, label: str) -> None:
            """Read lines from a stream and put them on the queue."""
            try:
                for raw_line in stream:
                    output_queue.put((label, raw_line))
            finally:
                output_queue.put((label, ""))  # sentinel

        def _cleanup_proc(
            proc: subprocess.Popen[str], t_out: threading.Thread, t_err: threading.Thread
        ) -> None:
            """Gracefully terminate process, then force kill if needed."""
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()
            t_out.join(timeout=2)
            t_err.join(timeout=2)

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # Read stdout and stderr in parallel threads
            t_out = threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True)
            t_err = threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True)
            t_out.start()
            t_err.start()

            deadline = (time.monotonic() + timeout) if timeout is not None else None
            last_activity_ping = time.monotonic()
            sentinels_received = 0

            while sentinels_received < 2:
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        _cleanup_proc(proc, t_out, t_err)
                        return (
                            -1,
                            "".join(stdout_lines),
                            f"Command timed out after {timeout} seconds",
                        )
                    wait_time = min(remaining, 1.0)
                else:
                    wait_time = 1.0

                try:
                    label, line = output_queue.get(timeout=wait_time)
                except queue.Empty:
                    if activity_callback is not None:
                        now = time.monotonic()
                        if now - last_activity_ping >= activity_interval:
                            activity_callback()
                            last_activity_ping = now
                    continue

                if line == "":
                    sentinels_received += 1
                    continue

                if label == "stdout":
                    stdout_lines.append(line)
                else:
                    stderr_lines.append(line)

                if on_output is not None:
                    on_output(line.rstrip("\n"))

                if activity_callback is not None:
                    now = time.monotonic()
                    if now - last_activity_ping >= activity_interval:
                        activity_callback()
                        last_activity_ping = now

            t_out.join(timeout=2)
            t_err.join(timeout=2)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _cleanup_proc(proc, t_out, t_err)

            return proc.returncode, "".join(stdout_lines), "".join(stderr_lines)

        except FileNotFoundError:
            return -1, "", "docker command not found"

    def enable_network(self, container_id: str) -> bool:
        """Temporarily enable network for the container.

        Reconnects the container to the bridge network.  DNS is already
        configured via --dns flags at container creation time.
        """
        try:
            result = subprocess.run(
                ["docker", "network", "connect", "bridge", container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                # May already be connected
                if "already exists" not in result.stderr:
                    logger.warning("Failed to connect network: %s", result.stderr)
                    return False
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def disable_network(self, container_id: str) -> bool:
        """Disable network for the container."""
        try:
            result = subprocess.run(
                ["docker", "network", "disconnect", "bridge", container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                if "is not connected" not in result.stderr:
                    logger.warning("Failed to disconnect network: %s", result.stderr)
                    return False
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def stop_container(self, session_id: str) -> bool:
        """Stop and remove the container for a session."""
        with self._lock:
            if session_id not in self._containers:
                return True

            info = self._containers[session_id]
            try:
                subprocess.run(
                    ["docker", "stop", info.container_id], capture_output=True, timeout=30
                )
                del self._containers[session_id]
                logger.info("Stopped container for session %s", session_id)
                return True
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False

    def cleanup_all(self) -> None:
        """Stop all managed containers."""
        with self._lock:
            session_ids = list(self._containers.keys())

        for session_id in session_ids:
            self.stop_container(session_id)

    def recover_orphans(self) -> int:
        """Discover and stop onit-sandbox containers from prior server runs.

        Queries Docker for containers with the ``onit.sandbox`` label that are
        *not* tracked in ``self._containers``.  These are orphans left behind
        by a crashed or killed server process.

        Returns the number of orphaned containers stopped.
        """
        if not self._check_docker():
            return 0

        try:
            result = subprocess.run(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    f"label={CONTAINER_LABEL}=true",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return 0

            container_ids = [
                cid.strip() for cid in result.stdout.strip().split("\n") if cid.strip()
            ]
            if not container_ids:
                return 0

            # Build set of container IDs we already track
            with self._lock:
                tracked_ids = {info.container_id for info in self._containers.values()}

            orphan_ids = [cid for cid in container_ids if cid not in tracked_ids]
            stopped = 0
            for cid in orphan_ids:
                try:
                    logger.info("Stopping orphaned container %s", cid[:12])
                    subprocess.run(
                        ["docker", "stop", cid],
                        capture_output=True,
                        timeout=30,
                    )
                    stopped += 1
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    logger.warning("Failed to stop orphaned container %s", cid[:12])
            return stopped

        except (subprocess.TimeoutExpired, FileNotFoundError):
            return 0

    def get_container_stats(self, container_id: str) -> dict[str, Any] | None:
        """Get resource usage stats for a container."""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_id],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                stats: dict[str, Any] = json.loads(result.stdout)
                return stats
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
        return None

    def list_installed_packages(self, container_id: str) -> list[str]:
        """List installed Python packages in the container."""
        exit_code, output = self.exec_in_container(
            container_id,
            "pip list --format=freeze 2>/dev/null || pip3 list --format=freeze 2>/dev/null",
            timeout=30,
        )
        if exit_code == 0:
            return [line.strip() for line in output.split("\n") if line.strip()]
        return []


# ---------------------------------------------------------------------------
# Module globals — set by run() at startup or by the parent ToolsMCPServer
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(tempfile.gettempdir(), "onit", "data")
SESSION_ID: str | None = None
DATA_MOUNTS: list[dict[str, str]] = parse_data_mounts(DEFAULT_DATA_MOUNTS)

_manager = SandboxManager()
_server = SandboxMCPServer()
mcp = _server.mcp


# ---------------------------------------------------------------------------
# Monkey-patch: suppress ExceptionGroup on client disconnect (MCP SDK bug)
#
# On Python 3.10, anyio's ExceptionGroup inherits from BaseException, not
# Exception.  The MCP SDK's streamable_http_manager catches Exception but not
# BaseExceptionGroup, so when a client disconnects and the session's TaskGroup
# teardown surfaces leftover task errors, the ExceptionGroup escapes and
# crashes the server.  We patch the lowlevel server's run() to catch it.
# ---------------------------------------------------------------------------
import mcp.server.lowlevel.server as _ll_server

_original_server_run = _ll_server.Server.run


async def _patched_server_run(self, *args, **kwargs):
    try:
        return await _original_server_run(self, *args, **kwargs)
    except BaseException as exc:
        if "ExceptionGroup" in type(exc).__name__:
            logger.info("Client disconnected (suppressed %s)", type(exc).__name__)
        else:
            raise


_ll_server.Server.run = _patched_server_run


def _get_session_id(session_id: str | None = None) -> str:
    """Return a session ID, using the caller-supplied value if given.

    When no *session_id* is provided the module-level default is used.  If that
    is also ``None`` a new UUID is generated once and cached so that all
    subsequent calls within the same process share the same default session.
    """
    if session_id:
        return session_id
    global SESSION_ID
    if SESSION_ID is None:
        SESSION_ID = str(uuid.uuid4())
    return SESSION_ID


def _get_data_path(session_id: str | None = None) -> str:
    """Return the workspace data path, namespaced by *session_id* when given."""
    base = DATA_PATH
    sid = _get_session_id(session_id)
    return os.path.join(base, sid)


def _list_workspace_files(container_id: str) -> set[str]:
    """List files in /workspace to detect newly created files."""
    exit_code, output = _manager.exec_in_container(
        container_id,
        "find /workspace -maxdepth 8 -not -path '*/\\.*' -type f 2>/dev/null",
        timeout=10,
    )
    if exit_code == 0:
        return {
            line.replace("/workspace/", "", 1)
            for line in output.strip().split("\n")
            if line.strip()
        }
    return set()


T = TypeVar("T")

_PROGRESS_HEARTBEAT_INTERVAL = 5.0  # seconds between SSE keep-alive heartbeats
_CTX_TIMEOUT = 15.0  # seconds to wait for ctx.info / ctx.report_progress before giving up


async def _safe_ctx_info(ctx: Context, msg: str) -> None:
    """Send ctx.info with a timeout so a disconnected client cannot hang the server."""
    try:
        await asyncio.wait_for(ctx.info(msg), timeout=_CTX_TIMEOUT)
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning("ctx.info timed out or failed (%s), client may have disconnected", exc)


async def _safe_ctx_progress(ctx: Context, progress: float, total: float) -> None:
    """Send ctx.report_progress with a timeout."""
    try:
        await asyncio.wait_for(
            ctx.report_progress(progress=progress, total=total), timeout=_CTX_TIMEOUT
        )
    except (asyncio.TimeoutError, Exception) as exc:
        logger.warning(
            "ctx.report_progress timed out or failed (%s), client may have disconnected", exc
        )


async def _run_with_progress(
    ctx: Context | None,
    fn: Callable[..., T],
    *args: Any,
    interval: float = _PROGRESS_HEARTBEAT_INTERVAL,
    **kwargs: Any,
) -> T:
    """Run a blocking *fn* in a thread, sending MCP progress heartbeats.

    Heartbeats are ``notifications/progress`` messages sent every *interval*
    seconds.  They keep the SSE connection alive so the client does not hit
    an ``httpx.ReadTimeout`` while the server is busy (e.g. installing large
    packages).

    When *ctx* is ``None`` (e.g. direct calls from tests) the function is
    still offloaded to a thread but no heartbeats are emitted.
    """
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

    if ctx is None:
        # No MCP context — just await the result (tests, direct calls).
        return await future

    heartbeat = 0
    while True:
        done, _ = await asyncio.wait({future}, timeout=interval)
        if done:
            # Final "complete" notification.
            await _safe_ctx_progress(ctx, progress=1, total=1)
            return future.result()
        heartbeat += 1
        await _safe_ctx_progress(ctx, progress=heartbeat, total=heartbeat + 1)


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------



@mcp.tool(
    title="Bash",
    description="""Execute a shell command in the isolated sandbox container.
Only sandbox-local paths are accessible — host paths do not exist here.
No internet by default. Output is streamed in real time.

Parameters:
- command: Shell command to run (e.g. "python train.py --data-dir /data").
- network: Temporarily enable internet for this command (default false).
  Required for: pip install, git clone/pull/push/fetch, curl, wget, apt-get.
- timeout: Max seconds before the command is killed (default: no limit).
- background: Run in the background and return a job_id immediately.
  Use sandbox_check_job to poll status. Ideal for long-running training jobs.

Filesystem layout:
  /workspace          Working directory. All relative paths resolve here.
    .cache/pip        Shared pip cache (persists across sessions).
    .cache/huggingface  HuggingFace cache (models, datasets, tokenizers).
    .local/bin        User-installed executables (on PATH).
  /data               Host data directory (if mounted). Read-write.
  /tmp                Temporary files. Cleared on container restart.

Common operations:
  Install packages:  sandbox_bash(command="pip install numpy torch", network=true)
  Git clone:         sandbox_bash(command="git clone https://github.com/user/repo", network=true)
  Git commit & push: sandbox_bash(command="git add -A && git commit -m 'msg' && git push", network=true)
  List files:        sandbox_bash(command="ls -la /workspace")
  Read file:         sandbox_bash(command="cat /workspace/train.py")
  Find files:        sandbox_bash(command="find /workspace -name '*.py'")""",
)
async def sandbox_bash(
    command: Annotated[
        str | None, Field(description="Shell command to execute, e.g. 'python train.py'.")
    ] = None,
    network: Annotated[
        bool, Field(description="Temporarily enable internet for this command.")
    ] = False,
    timeout: Annotated[
        int | None,
        Field(description="Max seconds before the command is killed (default: no limit)."),
    ] = None,
    background: Annotated[
        bool,
        Field(
            description="Run in background and return a job_id immediately. "
            "Use sandbox_check_job to poll status and retrieve output."
        ),
    ] = False,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    if not command:
        return json.dumps({"status": "error", "error": "No command specified"}, indent=2)

    # --- Background mode: launch command detached and return immediately ---
    if background:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)
        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )

            # Enable network if requested (persistent for background jobs)
            if network and not container_info.network_enabled:
                _manager.enable_network(container_info.container_id)

            job_id = uuid.uuid4().hex[:12]
            jobs_dir = "/workspace/.jobs"
            job_dir = f"{jobs_dir}/{job_id}"

            # Create job directory and launch the command via nohup with output capture.
            # A wrapper script writes the PID, captures exit code, and marks completion.
            # The trap EXIT ensures status is updated on SIGTERM or normal exit.
            wrapper = (
                f"mkdir -p {job_dir} && "
                f"echo 'running' > {job_dir}/status && "
                f"echo {json.dumps(command)} > {job_dir}/command && "
                f"nohup sh -c '"
                f"trap \"echo done > {job_dir}/status\" EXIT; "
                f"({command}) > {job_dir}/stdout.log 2> {job_dir}/stderr.log; "
                f"echo $? > {job_dir}/exitcode; "
                f"echo done > {job_dir}/status' "
                f"> /dev/null 2>&1 & "
                f"echo $! > {job_dir}/pid"
            )

            exit_code, output = _manager.exec_in_container(container_info.container_id, wrapper)
            if exit_code != 0:
                return json.dumps(
                    {"status": "error", "error": f"Failed to launch background job: {output}"},
                    indent=2,
                )

            # Read back the PID that was captured by the wrapper
            _, pid_output = _manager.exec_in_container(
                container_info.container_id,
                f"cat {job_dir}/pid 2>/dev/null",
            )
            job_pid = pid_output.strip() or None

            result_data: dict[str, Any] = {
                "status": "ok",
                "job_id": job_id,
                "session_id": sid,
                "container_id": container_info.container_id[:12],
                "message": f"Background job started. Use sandbox_check_job(job_id='{job_id}') to check status and retrieve output.",
            }
            if job_pid:
                result_data["pid"] = int(job_pid)

            return json.dumps(result_data, indent=2)
        except DockerNotAvailableError:
            return json.dumps({"status": "error", "error": "Docker is not available."}, indent=2)
        except Exception as e:
            logger.exception("Error launching background job")
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    # --- Foreground mode: stream output in real time ---

    # Queue for streaming output lines from the worker thread to the async loop.
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _on_output(line: str) -> None:
        """Callback invoked in the worker thread for each output line."""
        line_queue.put(line)

    def _run_in_container(container_info: ContainerInfo, sid: str) -> str:
        """Execute the command in the given container, managing temp network."""
        temp_network = False
        if network and not container_info.network_enabled:
            temp_network = _manager.enable_network(container_info.container_id)

        try:
            files_before = _list_workspace_files(container_info.container_id)

            exit_code, stdout, stderr = _manager.exec_in_container_streaming(
                container_info.container_id,
                command,
                timeout=timeout,
                on_output=_on_output,
            )

            files_after = _list_workspace_files(container_info.container_id)
            files_created = sorted(files_after - files_before)

            if exit_code == 0:
                status = "ok"
            elif "timed out" in stderr:
                status = "timeout"
            else:
                status = "error"

            result_payload: dict[str, Any] = {
                "status": status,
                "session_id": sid,
                "stdout": stdout[-MAX_OUTPUT_BYTES:],
                "stderr": stderr[-MAX_OUTPUT_BYTES:],
                "returncode": exit_code,
                "files_created": files_created,
            }
            if files_created:
                result_payload["hint"] = (
                    "These files exist only inside the sandbox container. "
                    "Use the sandbox_download_file tool to copy them to your local filesystem."
                )
            return json.dumps(result_payload, indent=2)
        finally:
            if temp_network:
                _manager.disable_network(container_info.container_id)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )

            result_str = _run_in_container(container_info, sid)

            # Check if the container died mid-execution; if so, recreate and retry once
            result_data = json.loads(result_str)
            if (
                result_data.get("returncode") == -1
                and _manager._is_container_dead_error(result_data.get("stderr", ""))
            ):
                logger.warning(
                    "Container %s died during execution, recreating...",
                    container_info.container_id[:12],
                )
                with _manager._lock:
                    _manager._containers.pop(sid, None)
                container_info = _manager.get_or_create_container(
                    sid, data_path, extra_mounts=DATA_MOUNTS
                )
                result_str = _run_in_container(container_info, sid)

            return result_str

        except DockerNotAvailableError:
            return json.dumps(
                {
                    "status": "error",
                    "stderr": "Docker is not available. "
                    "Please install Docker and ensure it is running.",
                    "returncode": -1,
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_bash")
            return json.dumps(
                {
                    "status": "error",
                    "stderr": str(e),
                    "returncode": -1,
                },
                indent=2,
            )
        finally:
            # Signal the async drainer that the worker is done.
            line_queue.put(None)

    # Run _impl in a thread while draining the line queue and forwarding
    # each line to the MCP client via ctx.info().
    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _impl)

    if ctx is None:
        # No MCP context (tests / direct calls) — just await the result.
        return await future

    async def _drain_queue() -> None:
        """Forward all queued output lines to the MCP client."""
        while True:
            try:
                line = line_queue.get_nowait()
            except queue.Empty:
                break
            if line is not None:
                await _safe_ctx_info(ctx, line)

    heartbeat = 0
    while True:
        done, _ = await asyncio.wait({future}, timeout=0.25)
        await _drain_queue()

        if done:
            await _drain_queue()
            await _safe_ctx_progress(ctx, progress=1, total=1)
            return future.result()

        heartbeat += 1
        await _safe_ctx_progress(ctx, progress=heartbeat, total=heartbeat + 1)


@mcp.tool(
    title="Check Background Job",
    description="""Check the status of a background job launched with sandbox_bash(background=True).

- job_id: The job ID returned by sandbox_bash.
- tail: Number of lines to return from the end of stdout/stderr (default 100). Use 0 for all output.""",
)
async def sandbox_check_job(
    job_id: Annotated[
        str | None, Field(description="The job_id returned by sandbox_bash.")
    ] = None,
    tail: Annotated[
        int, Field(description="Number of lines from end of stdout/stderr (default 100, 0=all).")
    ] = 100,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    if not job_id:
        return json.dumps({"status": "error", "error": "No job_id specified"}, indent=2)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)
        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )
            job_dir = f"/workspace/.jobs/{job_id}"

            # Read job status
            exit_code, status_out = _manager.exec_in_container(
                container_info.container_id,
                f"cat {job_dir}/status 2>/dev/null || echo 'not_found'",
            )
            job_status = status_out.strip()

            if job_status == "not_found":
                return json.dumps(
                    {"status": "error", "error": f"Job '{job_id}' not found."},
                    indent=2,
                )

            # If status file says "running", verify the process is actually alive.
            # This catches cases where the process died without updating the status file
            # (e.g. OOM kill, SIGKILL, container restart).
            if job_status == "running":
                _, pid_out = _manager.exec_in_container(
                    container_info.container_id,
                    f"cat {job_dir}/pid 2>/dev/null",
                )
                pid_str = pid_out.strip()
                if pid_str:
                    # kill -0 checks if process exists without sending a signal
                    ec_alive, _ = _manager.exec_in_container(
                        container_info.container_id,
                        f"kill -0 {pid_str} 2>/dev/null",
                    )
                    if ec_alive != 0:
                        # Process is dead but status was never updated — mark as done
                        job_status = "done"
                        # Try to read the exit code if the wrapper managed to write it
                        _, ec_out = _manager.exec_in_container(
                            container_info.container_id,
                            f"cat {job_dir}/exitcode 2>/dev/null",
                        )
                        ec_str = ec_out.strip()
                        if ec_str:
                            try:
                                dead_exitcode = int(ec_str)
                            except ValueError:
                                dead_exitcode = -1
                        else:
                            dead_exitcode = -1
                        # Update the status file so future checks don't repeat this
                        _manager.exec_in_container(
                            container_info.container_id,
                            f"echo done > {job_dir}/status && "
                            f"([ -f {job_dir}/exitcode ] || echo {dead_exitcode} > {job_dir}/exitcode)",
                        )

            # Read exit code if job is done
            returncode = None
            if job_status == "done":
                _, ec_out = _manager.exec_in_container(
                    container_info.container_id,
                    f"cat {job_dir}/exitcode 2>/dev/null",
                )
                try:
                    returncode = int(ec_out.strip())
                except ValueError:
                    returncode = -1

            # Read stdout and stderr (tail N lines)
            tail_cmd = f"tail -n {tail}" if tail > 0 else "cat"
            _, stdout = _manager.exec_in_container(
                container_info.container_id,
                f"{tail_cmd} {job_dir}/stdout.log 2>/dev/null",
            )
            _, stderr = _manager.exec_in_container(
                container_info.container_id,
                f"{tail_cmd} {job_dir}/stderr.log 2>/dev/null",
            )

            # Read PID for reporting
            _, pid_out_check = _manager.exec_in_container(
                container_info.container_id,
                f"cat {job_dir}/pid 2>/dev/null",
            )
            job_pid_str = pid_out_check.strip()

            result: dict[str, Any] = {
                "status": "ok",
                "job_id": job_id,
                "job_status": "running" if job_status == "running" else "completed",
                "session_id": sid,
                "container_id": container_info.container_id[:12],
                "stdout": stdout[-MAX_OUTPUT_BYTES:],
                "stderr": stderr[-MAX_OUTPUT_BYTES:],
            }
            if returncode is not None:
                result["returncode"] = returncode
                result["job_status"] = "completed"
            if job_pid_str:
                try:
                    result["pid"] = int(job_pid_str)
                except ValueError:
                    pass

            return json.dumps(result, indent=2)

        except DockerNotAvailableError:
            return json.dumps({"status": "error", "error": "Docker is not available."}, indent=2)
        except Exception as e:
            logger.exception("Error in sandbox_check_job")
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Write File to Sandbox",
    description="""Write content to a file in the sandbox.
Parent directories are created automatically.
Relative paths resolve from /workspace.

- path: Destination path in the sandbox (relative to /workspace or absolute).
- content: File content (text, or base64 when encoding="base64").
- encoding: "utf-8" (default) or "base64" for binary files.""",
)
async def sandbox_write_file(
    path: Annotated[
        str | None,
        Field(description="Destination path in the sandbox (relative to /workspace or absolute)."),
    ] = None,
    content: Annotated[
        str | None, Field(description="File content (text, or base64 when encoding='base64').")
    ] = None,
    encoding: Annotated[
        str, Field(description="'utf-8' (default) or 'base64' for binary files.")
    ] = "utf-8",
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    if not path:
        return json.dumps({"status": "error", "error": "No path specified"}, indent=2)
    if content is None:
        return json.dumps({"status": "error", "error": "No content specified"}, indent=2)
    if encoding not in ("utf-8", "base64"):
        return json.dumps(
            {"status": "error", "error": "encoding must be 'utf-8' or 'base64'"}, indent=2
        )

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )

            # Normalise to absolute path inside container
            if not path.startswith("/"):
                container_path = f"/workspace/{path}"
            else:
                container_path = path

            # Ensure parent directory exists
            parent_dir = os.path.dirname(container_path)
            if parent_dir and parent_dir != "/":
                mkdir_code, mkdir_out = _manager.exec_in_container(
                    container_info.container_id,
                    f"mkdir -p '{parent_dir}'",
                    timeout=10,
                )
                if mkdir_code != 0:
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Failed to create directory "
                            f"'{parent_dir}': {mkdir_out.strip()}",
                        },
                        indent=2,
                    )

            # Write content via stdin using docker exec + sh -c "cat > file"
            # For base64 encoding, decode to binary and use docker cp instead
            if encoding == "base64":
                try:
                    raw = base64.b64decode(content)
                except Exception as e:
                    return json.dumps(
                        {"status": "error", "error": f"Invalid base64 content: {e}"},
                        indent=2,
                    )

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, os.path.basename(container_path))
                    with open(tmp_path, "wb") as f:
                        f.write(raw)

                    result = subprocess.run(
                        [
                            "docker",
                            "cp",
                            tmp_path,
                            f"{container_info.container_id}:{container_path}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    if result.returncode != 0:
                        return json.dumps(
                            {
                                "status": "error",
                                "error": f"Failed to write file: {result.stderr.strip()}",
                            },
                            indent=2,
                        )

                expected_bytes = len(raw)
            else:
                cmd = [
                    "docker",
                    "exec",
                    "-i",
                    "-w",
                    "/workspace",
                    container_info.container_id,
                    "sh",
                    "-c",
                    f"cat > '{container_path}'",
                ]
                result = subprocess.run(
                    cmd,
                    input=content,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if result.returncode != 0:
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Failed to write file: {result.stderr.strip()}",
                        },
                        indent=2,
                    )

                expected_bytes = len(content.encode())

            # Verify the file was actually written with correct size
            verify_code, verify_out = _manager.exec_in_container(
                container_info.container_id,
                f"test -f '{container_path}' && stat -c '%s' '{container_path}'",
                timeout=10,
            )
            if verify_code != 0:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Write appeared to succeed but "
                        f"file not found at {container_path}",
                    },
                    indent=2,
                )

            try:
                actual_bytes = int(verify_out.strip())
            except (ValueError, TypeError):
                actual_bytes = -1

            if actual_bytes != expected_bytes:
                return json.dumps(
                    {
                        "status": "error",
                        "error": (
                            f"Write verification failed: expected {expected_bytes} bytes "
                            f"but file has {actual_bytes} bytes at {container_path}"
                        ),
                    },
                    indent=2,
                )

            return json.dumps(
                {
                    "status": "ok",
                    "session_id": sid,
                    "path": container_path,
                    "size_bytes": expected_bytes,
                    "verified": True,
                },
                indent=2,
            )

        except DockerNotAvailableError:
            return json.dumps(
                {"status": "error", "error": "Docker is not available."},
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_write_file")
            return json.dumps(
                {"status": "error", "error": str(e)},
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)



@mcp.tool(
    title="Sandbox Status",
    description="""Returns sandbox state, installed packages,
GPU availability, and data_mounts (pre-mounted host directories).""",
)
async def sandbox_get_status(
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        try:
            docker_available = _manager._check_docker()
            gpu_available = _manager._check_gpu() if docker_available else False
            sid = _get_session_id(session_id)

            if not docker_available:
                return json.dumps(
                    {
                        "status": "not_created",
                        "python_version": None,
                        "installed_packages": [],
                        "disk_usage_mb": 0,
                        "uptime_seconds": 0,
                        "gpu_available": False,
                    },
                    indent=2,
                )

            with _manager._lock:
                if sid not in _manager._containers:
                    return json.dumps(
                        {
                            "status": "not_created",
                            "python_version": None,
                            "installed_packages": [],
                            "disk_usage_mb": 0,
                            "uptime_seconds": 0,
                            "gpu_available": gpu_available,
                        },
                        indent=2,
                    )
                info = _manager._containers[sid]

            running = _manager._is_container_running(info.container_id)
            if not running:
                return json.dumps(
                    {
                        "status": "stopped",
                        "python_version": None,
                        "installed_packages": [],
                        "disk_usage_mb": 0,
                        "uptime_seconds": 0,
                        "gpu_available": gpu_available,
                    },
                    indent=2,
                )

            # Get Python version
            exit_code, py_version = _manager.exec_in_container(
                info.container_id,
                "python3 --version 2>&1 | awk '{print $2}'",
                timeout=10,
            )
            python_version = py_version.strip() if exit_code == 0 else None

            # Get installed packages
            packages = _manager.list_installed_packages(info.container_id)
            # Parse package names (format: "name==version")
            package_names = [p.split("==")[0] for p in packages]

            # Get disk usage
            exit_code, du_output = _manager.exec_in_container(
                info.container_id,
                "du -sm /workspace 2>/dev/null | awk '{print $1}'",
                timeout=10,
            )
            try:
                disk_usage_mb = float(du_output.strip()) if exit_code == 0 else 0
            except (ValueError, TypeError):
                disk_usage_mb = 0

            # Calculate uptime
            uptime_seconds = (datetime.now() - info.created_at).total_seconds()

            # List workspace files for filesystem awareness
            workspace_files = sorted(_list_workspace_files(info.container_id))

            # Report configured data mounts
            mount_info = [
                {"host": m["host"], "container": m["container"], "mode": m["mode"]}
                for m in DATA_MOUNTS
            ]

            return json.dumps(
                {
                    "status": "running",
                    "session_id": sid,
                    "python_version": python_version,
                    "installed_packages": package_names,
                    "disk_usage_mb": disk_usage_mb,
                    "uptime_seconds": round(uptime_seconds),
                    "gpu_available": gpu_available,
                    "network_enabled": info.network_enabled,
                    "data_mounts": mount_info,
                    "workspace_files": workspace_files[:200],
                },
                indent=2,
            )

        except Exception as e:
            logger.exception("Error in sandbox_get_status")
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Enable Network",
    description="""Enable persistent internet access.
Stays on until sandbox_disable_network is called.""",
)
async def sandbox_enable_network(
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )
            success = _manager.enable_network(container_info.container_id)
            if success:
                container_info.network_enabled = True
                return json.dumps(
                    {"status": "ok", "session_id": sid, "network_enabled": True},
                    indent=2,
                )
            return json.dumps(
                {"status": "error", "error": "Failed to enable network access"},
                indent=2,
            )
        except DockerNotAvailableError:
            return json.dumps(
                {"status": "error", "error": "Docker is not available"},
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_enable_network")
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Disable Network",
    description="Disable internet access, restoring network isolation.",
)
async def sandbox_disable_network(
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )
            success = _manager.disable_network(container_info.container_id)
            if success:
                container_info.network_enabled = False
                return json.dumps(
                    {"status": "ok", "session_id": sid, "network_enabled": False},
                    indent=2,
                )
            return json.dumps(
                {"status": "error", "error": "Failed to disable network access"},
                indent=2,
            )
        except DockerNotAvailableError:
            return json.dumps(
                {"status": "error", "error": "Docker is not available"},
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_disable_network")
            return json.dumps({"status": "error", "error": str(e)}, indent=2)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Download File from Sandbox",
    description="""Copy a file from the sandbox to the server filesystem.

- path: Path in sandbox (relative to /workspace or absolute).
- dest: Absolute server path to save to (default: session downloads directory).""",
)
async def sandbox_download_file(
    path: Annotated[
        str | None, Field(description="Path in sandbox (relative to /workspace or absolute).")
    ] = None,
    dest: Annotated[
        str | None,
        Field(description="Destination filename hint for the agent (not used server-side)."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    data_path: Annotated[str | None, Field(description="Agent data directory.")] = None,
    ctx: Context | None = None,
) -> str:
    if not path:
        return json.dumps({"status": "error", "error": "No path specified"}, indent=2)

    progress_queue: queue.Queue[str | None] = queue.Queue()

    def _impl() -> str:
        sid = _get_session_id(session_id)
        session_data_path = data_path if data_path else _get_data_path(sid)

        try:
            # Normalise path to absolute
            if not path.startswith("/"):
                container_path = f"/workspace/{path}"
            else:
                container_path = path

            dl_filename = os.path.basename(container_path)

            progress_queue.put("Resolving sandbox container…")
            container_info = _manager.get_or_create_container(
                sid, session_data_path, extra_mounts=DATA_MOUNTS
            )

            # docker cp to a server-local temp directory, then return as base64
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_dest = os.path.join(tmpdir, dl_filename)

                progress_queue.put(f"Copying {container_path} from sandbox…")
                result = subprocess.run(
                    [
                        "docker",
                        "cp",
                        f"{container_info.container_id}:{container_path}",
                        tmp_dest,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                if result.returncode != 0:
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"docker cp failed: {result.stderr.strip()}",
                        },
                        indent=2,
                    )

                if not os.path.exists(tmp_dest):
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"File not found after docker cp: {container_path}",
                        },
                        indent=2,
                    )
                if os.path.isdir(tmp_dest):
                    return json.dumps(
                        {
                            "status": "error",
                            "error": f"Path is a directory, not a file: {container_path}",
                        },
                        indent=2,
                    )

                file_size = os.path.getsize(tmp_dest)
                progress_queue.put(f"Reading file ({file_size:,} bytes)…")
                with open(tmp_dest, "rb") as f:
                    file_bytes = f.read()

            progress_queue.put(f"Encoding {dl_filename} ({len(file_bytes):,} bytes)…")
            encoded = base64.b64encode(file_bytes).decode()

            return json.dumps(
                {
                    "status": "ok",
                    "file_name": dl_filename,
                    "file_data_base64": encoded,
                    "mime_type": "application/octet-stream",
                    "size_bytes": len(file_bytes),
                    "session_id": sid,
                },
                indent=2,
            )

        except DockerNotAvailableError:
            return json.dumps(
                {"status": "error", "error": "Docker is not available."},
                indent=2,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {"status": "error", "error": "Timed out copying file from sandbox."},
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_download_file")
            return json.dumps(
                {"status": "error", "error": str(e)},
                indent=2,
            )
        finally:
            progress_queue.put(None)

    loop = asyncio.get_running_loop()
    future = loop.run_in_executor(None, _impl)

    if ctx is None:
        return await future

    async def _drain() -> None:
        while True:
            try:
                msg = progress_queue.get_nowait()
            except queue.Empty:
                break
            if msg is not None:
                await _safe_ctx_info(ctx, msg)

    heartbeat = 0
    while True:
        done, _ = await asyncio.wait({future}, timeout=0.25)
        await _drain()
        if done:
            await _safe_ctx_progress(ctx, progress=1, total=1)
            return future.result()
        heartbeat += 1
        await _safe_ctx_progress(ctx, progress=heartbeat, total=heartbeat + 1)


@mcp.tool(
    title="Upload File to Sandbox",
    description="""Copy a file or directory into the sandbox.
Provide exactly one of src, data, or content.

- src: Absolute server path to copy from.
- data: Base64-encoded file content (requires filename).
- content: Plain UTF-8 text content (requires filename).
- filename: Name for the file when using data or content.
- dest: Destination in sandbox (relative or absolute;
  default: /workspace/<filename>).""",
)
async def sandbox_upload_file(
    src: Annotated[str | None, Field(description="Absolute server path to copy from.")] = None,
    data: Annotated[
        str | None, Field(description="Base64-encoded file content (requires filename).")
    ] = None,
    content: Annotated[
        str | None, Field(description="Plain UTF-8 text content (requires filename).")
    ] = None,
    filename: Annotated[
        str | None, Field(description="Name for the file when using data or content.")
    ] = None,
    dest: Annotated[
        str | None,
        Field(description="Destination in sandbox (relative to /workspace or absolute)."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    data_path: Annotated[str | None, Field(description="Agent data directory.")] = None,
    ctx: Context | None = None,
) -> str:
    modes_set = sum(x is not None for x in (src, data, content))
    if modes_set == 0:
        return json.dumps(
            {"status": "error", "error": "Provide one of: src, data, or content"}, indent=2
        )
    if modes_set > 1:
        return json.dumps(
            {"status": "error", "error": "Provide only one of: src, data, or content"}, indent=2
        )
    if (data is not None or content is not None) and not filename:
        return json.dumps(
            {"status": "error", "error": "filename is required when using data or content"},
            indent=2,
        )

    def _impl() -> str:
        sid = _get_session_id(session_id)
        session_data_path = data_path if data_path else _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, session_data_path, extra_mounts=DATA_MOUNTS
            )

            # --- Inline data mode: decode base64 to temp file, then docker cp ---
            if data is not None:
                assert filename is not None
                try:
                    raw = base64.b64decode(data)
                except Exception as e:
                    return json.dumps(
                        {"status": "error", "error": f"Invalid base64 data: {e}"},
                        indent=2,
                    )

                # Determine destination inside the container
                if dest:
                    container_dest = dest if dest.startswith("/") else f"/workspace/{dest}"
                else:
                    container_dest = f"/workspace/{filename}"

                # Ensure parent directory exists in container
                parent_dir = os.path.dirname(container_dest)
                if parent_dir and parent_dir != "/":
                    _manager.exec_in_container(
                        container_info.container_id,
                        f"mkdir -p '{parent_dir}'",
                        timeout=10,
                    )

                # Write decoded content to a temp file, then docker cp it in
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, filename)
                    with open(tmp_path, "wb") as f:
                        f.write(raw)

                    result = subprocess.run(
                        [
                            "docker",
                            "cp",
                            tmp_path,
                            f"{container_info.container_id}:{container_dest}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode != 0:
                        return json.dumps(
                            {
                                "status": "error",
                                "error": f"docker cp failed: {result.stderr.strip()}",
                            },
                            indent=2,
                        )

                return json.dumps(
                    {
                        "status": "ok",
                        "session_id": sid,
                        "mode": "inline",
                        "dest": container_dest,
                        "size_bytes": len(raw),
                    },
                    indent=2,
                )

            # --- Plain text content mode: write UTF-8 text, then docker cp ---
            if content is not None:
                assert filename is not None
                raw = content.encode("utf-8")

                # Determine destination inside the container
                if dest:
                    container_dest = dest if dest.startswith("/") else f"/workspace/{dest}"
                else:
                    container_dest = f"/workspace/{filename}"

                # Ensure parent directory exists in container
                parent_dir = os.path.dirname(container_dest)
                if parent_dir and parent_dir != "/":
                    _manager.exec_in_container(
                        container_info.container_id,
                        f"mkdir -p '{parent_dir}'",
                        timeout=10,
                    )

                # Write content to a temp file, then docker cp it in
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = os.path.join(tmpdir, filename)
                    with open(tmp_path, "wb") as f:
                        f.write(raw)

                    result = subprocess.run(
                        [
                            "docker",
                            "cp",
                            tmp_path,
                            f"{container_info.container_id}:{container_dest}",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=300,
                    )
                    if result.returncode != 0:
                        return json.dumps(
                            {
                                "status": "error",
                                "error": f"docker cp failed: {result.stderr.strip()}",
                            },
                            indent=2,
                        )

                return json.dumps(
                    {
                        "status": "ok",
                        "session_id": sid,
                        "mode": "content",
                        "dest": container_dest,
                        "size_bytes": len(raw),
                    },
                    indent=2,
                )

            # --- Host path mode: copy from server filesystem ---
            assert src is not None
            abs_src = os.path.abspath(src)

            if not os.path.exists(abs_src):
                hint = ""
                # Detect common remote-path patterns
                if src.startswith("/Users/") or src.startswith("C:\\"):
                    hint = (
                        " This looks like a path from a different machine. "
                        "src must be an absolute path on the server's filesystem. "
                        "Use data or content parameters for remote files."
                    )
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Path does not exist on the server: {abs_src}.{hint}",
                    },
                    indent=2,
                )

            # Determine destination inside the container
            if dest:
                container_dest = dest if dest.startswith("/") else f"/workspace/{dest}"
            else:
                container_dest = f"/workspace/{os.path.basename(abs_src)}"

            # Ensure parent directory exists in container
            parent_dir = os.path.dirname(container_dest)
            if parent_dir and parent_dir != "/":
                _manager.exec_in_container(
                    container_info.container_id,
                    f"mkdir -p '{parent_dir}'",
                    timeout=10,
                )

            # Use docker cp to copy file/directory into the container
            result = subprocess.run(
                ["docker", "cp", abs_src, f"{container_info.container_id}:{container_dest}"],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode != 0:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"docker cp failed: {result.stderr.strip()}",
                    },
                    indent=2,
                )

            # Get size info
            if os.path.isdir(abs_src):
                exit_code, du_out = _manager.exec_in_container(
                    container_info.container_id,
                    f"du -sb '{container_dest}' 2>/dev/null | awk '{{print $1}}'",
                    timeout=30,
                )
                try:
                    size_bytes = int(du_out.strip()) if exit_code == 0 else None
                except (ValueError, TypeError):
                    size_bytes = None
            else:
                try:
                    size_bytes = os.path.getsize(abs_src)
                except OSError:
                    size_bytes = None

            resp = {
                "status": "ok",
                "session_id": sid,
                "mode": "host_path",
                "src": abs_src,
                "dest": container_dest,
                "size_bytes": size_bytes,
            }
            return json.dumps(resp, indent=2)

        except DockerNotAvailableError:
            return json.dumps(
                {"status": "error", "error": "Docker is not available."},
                indent=2,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {"status": "error", "error": "Timed out uploading to sandbox."},
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_upload_file")
            return json.dumps(
                {"status": "error", "error": str(e)},
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)



@mcp.tool(
    title="Create GitHub Repo",
    description="""Create a new GitHub repository.

Requires a GitHub Personal Access Token configured via 'onit-sandbox setup'.
The repository is created under the authenticated user by default, or under
a specified organization.  Returns the repo URL and clone URL on success.""",
)
async def sandbox_github_create_repo(
    name: Annotated[
        str,
        Field(description="Repository name (e.g. 'my-project')."),
    ],
    description: Annotated[
        str | None,
        Field(description="Optional repository description."),
    ] = None,
    private: Annotated[
        bool,
        Field(description="Whether the repository should be private. Defaults to true."),
    ] = True,
    org: Annotated[
        str | None,
        Field(
            description="GitHub organization to create the repo under. "
            "If omitted, creates under the authenticated user."
        ),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    import urllib.request
    import urllib.error

    def _impl() -> str:
        token = _manager._load_github_token()
        if not token:
            return json.dumps(
                {
                    "status": "error",
                    "output": "GitHub token not configured. "
                    "Run 'onit-sandbox setup' on the host to store a GitHub Personal Access Token.",
                },
                indent=2,
            )

        payload: dict[str, Any] = {"name": name, "private": private}
        if description:
            payload["description"] = description

        if org:
            api_url = f"https://api.github.com/orgs/{org}/repos"
        else:
            api_url = "https://api.github.com/user/repos"

        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            api_url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode())
                return json.dumps(
                    {
                        "status": "ok",
                        "name": resp_data.get("full_name"),
                        "url": resp_data.get("html_url"),
                        "clone_url": resp_data.get("clone_url"),
                        "ssh_url": resp_data.get("ssh_url"),
                        "private": resp_data.get("private"),
                    },
                    indent=2,
                )
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if e.fp else ""
            try:
                error_detail = json.loads(error_body).get("message", error_body)
            except (json.JSONDecodeError, AttributeError):
                error_detail = error_body
            return json.dumps(
                {"status": "error", "output": f"GitHub API error {e.code}: {error_detail}"},
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {"status": "error", "output": f"Request failed: {e}"},
                indent=2,
            )

    return await asyncio.get_event_loop().run_in_executor(None, _impl)


@mcp.tool(
    title="Stop Sandbox",
    description="""Stop and remove the sandbox container for the current session.

Call this when you are completely done with the sandbox and want to release its
resources.  The container (and any data not persisted to the shared volume) will
be destroyed.  A new container will be created automatically if any sandbox tool
is called again for the same session.""",
)
async def sandbox_stop(
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    sid = _get_session_id(session_id)

    with _manager._lock:
        info = _manager._containers.get(sid)

    if info is None:
        return json.dumps(
            {"status": "ok", "message": f"No active container for session {sid}."},
            indent=2,
        )

    # Run stop_container in an executor to avoid blocking the event loop
    # (docker stop can take up to 30s).
    stopped = await _run_with_progress(ctx, _manager.stop_container, sid)
    if stopped:
        logger.info("Agent requested stop for session %s", sid)
        return json.dumps(
            {"status": "ok", "message": f"Container for session {sid} stopped and removed."},
            indent=2,
        )
    return json.dumps(
        {"status": "error", "error": f"Failed to stop container for session {sid}."},
        indent=2,
    )



# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run(
    transport: str = "sse",
    host: str = "0.0.0.0",
    port: int = 18202,
    path: str = "/sse",
    options: dict[str, Any] | None = None,
) -> None:
    """Start the sandbox MCP server.

    Args:
        transport: Transport type ("sse" or "stdio").
        host: Host to bind to.
        port: Port to bind to.
        path: SSE endpoint path.
        options: Dict with optional 'data_path', 'session_id', and 'verbose' overrides.
    """
    global DATA_PATH, SESSION_ID, DATA_MOUNTS

    if options is None:
        options = {}

    if options.get("verbose"):
        logging.basicConfig(level=logging.INFO)

    if "data_path" in options:
        DATA_PATH = options["data_path"]
    if "session_id" in options:
        SESSION_ID = options["session_id"]
    if "data_mounts" in options:
        DATA_MOUNTS = parse_data_mounts(options["data_mounts"])
    if "gpu_devices" in options:
        _manager._gpu_devices = options["gpu_devices"]
        logger.info("GPU device selection: %s", options["gpu_devices"])
    if DATA_MOUNTS:
        logger.info(
            "Data mounts configured: %s",
            ", ".join(f"{m['host']}:{m['container']}:{m['mode']}" for m in DATA_MOUNTS),
        )

    _server.host = host
    _server.port = port
    _server.path = path
    _server.transport = transport

    # Recover orphaned containers from prior server runs before accepting traffic.
    orphans = _manager.recover_orphans()
    if orphans:
        logger.info("Recovered and stopped %d orphaned container(s) from a previous run", orphans)

    # Register cleanup handlers so containers are stopped on exit (including Ctrl+C)
    atexit.register(cleanup_all_sandboxes)

    def _shutdown_handler(signum: int, frame: Any) -> None:
        logger.info("Received signal %s, cleaning up containers...", signum)
        cleanup_all_sandboxes()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)

    logger.info(
        "Starting Sandbox MCP Server on http://%s:%s%s (transport: %s)",
        host,
        port,
        path,
        transport,
    )
    _server.run()


def cleanup_sandbox(session_id: str) -> bool:
    """Cleanup function to be called during session teardown."""
    return _manager.stop_container(session_id)


def cleanup_all_sandboxes() -> None:
    """Cleanup all sandbox containers. Call during application shutdown."""
    _manager.cleanup_all()
