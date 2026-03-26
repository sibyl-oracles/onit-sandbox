"""
Sandbox MCP Server — tool definitions for safe code execution.

Provides an isolated Docker-based sandbox for developing, running, and testing
Python projects without affecting the host system. Each session gets its own
container with resource limits, network isolation, and a shared home directory.

Tools:
- sandbox_install_packages: Install Python packages in the sandbox
- sandbox_run_code: Execute commands in the isolated sandbox
- sandbox_get_status: Inspect sandbox state, packages, and resource usage
- sandbox_write_file: Write files into the sandbox
- sandbox_list_files: List files in the sandbox filesystem
- sandbox_download_file: Copy a file from the sandbox to the local filesystem
- sandbox_upload_file: Copy a file/directory from host into the sandbox
- sandbox_read_file: Read file contents from the sandbox
- sandbox_enable_network: Enable persistent internet access
- sandbox_disable_network: Disable internet access
- sandbox_filesystem_info: Explain the sandbox filesystem layout and mounts

Git tools (use 'onit-sandbox setup' for GitHub authentication):
- sandbox_git_clone: Clone a repository into the sandbox
- sandbox_git_status: Show working tree status
- sandbox_git_add: Stage files for commit
- sandbox_git_commit: Create a commit
- sandbox_git_pull: Pull from a remote repository
- sandbox_git_push: Push to a remote repository
- sandbox_git_branch: List, create, or delete branches
- sandbox_git_checkout: Switch branches or restore files
- sandbox_git_log: Show commit history
- sandbox_git_diff: Show changes between commits/working tree
- sandbox_git_init: Initialize a new repository
- sandbox_git_remote: Manage remote repositories
- sandbox_git_stash: Stash or restore uncommitted changes
"""

from __future__ import annotations

import asyncio
import atexit
import base64
import functools
import json
import logging
import os
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
    DEFAULT_CPU_QUOTA,
    DEFAULT_DATA_MOUNTS,
    DEFAULT_GPU_DEVICES,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_PIP_CACHE_PATH,
    DEFAULT_SHM_SIZE,
    DEFAULT_TIMEOUT,
    FALLBACK_IMAGE,
    INSTALL_TIMEOUT,
    MAX_OUTPUT_BYTES,
    SANDBOX_IMAGE,
    SandboxMCPServer,
    parse_data_mounts,
)

logger = logging.getLogger(__name__)


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
            data_path: Host path for the /home/sandbox volume.
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
            "--volume",
            f"{os.path.abspath(data_path)}:/home/sandbox:rw",
            "--volume",
            f"{pip_cache}:/home/sandbox/.cache/pip:rw",
            "--workdir",
            "/home/sandbox",
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
            f"{os.getuid()}:{os.getgid()}",
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

        # Add extra data volume mounts (e.g. /data:/data:ro)
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
                    f" && chown -R {os.getuid()}:{os.getgid()} /home/sandbox",
                ],
                capture_output=True,
                timeout=10,
            )

            # Add passwd entry for the host UID so pwd.getpwuid() works
            self._ensure_passwd_entry(container_id)

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
        uid, gid = os.getuid(), os.getgid()
        subprocess.run(
            [
                "docker",
                "exec",
                "-u",
                "0",
                container_id,
                "sh",
                "-c",
                f"grep -q ':{uid}:' /etc/passwd || "
                f"echo 'sandbox:x:{uid}:{gid}:sandbox:/home/sandbox:/bin/sh' >> /etc/passwd",
            ],
            capture_output=True,
            timeout=10,
        )

    def _configure_git_credentials(self, container_id: str) -> None:
        """Configure git inside the container to use the GitHub token.

        Sets up a credential helper script that returns the token from the
        GITHUB_TOKEN environment variable, and configures git to use it.
        """
        # Create a git credential helper script that reads GITHUB_TOKEN env var
        # and set git config to use it.  Also mark /home/sandbox as safe.
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
            "git config --global --add safe.directory '*'"
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
    def _build_exec_cmd(
        container_id: str,
        command: str,
        workdir: str = "/home/sandbox",
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
        workdir: str = "/home/sandbox",
        env: dict[str, str] | None = None,
        split_output: bool = False,
    ) -> tuple[int, str] | tuple[int, str, str]:
        """Execute a command inside a container.

        When split_output is True, returns (returncode, stdout, stderr).
        Otherwise returns (returncode, combined_output) for backward compat.
        """
        cmd = self._build_exec_cmd(container_id, command, workdir, env)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if split_output:
                return result.returncode, result.stdout, result.stderr
            output = result.stdout + result.stderr
            return result.returncode, output
        except subprocess.TimeoutExpired:
            if split_output:
                return -1, "", f"Command timed out after {timeout} seconds"
            return -1, f"Command timed out after {timeout} seconds"

    def exec_in_container_streaming(
        self,
        container_id: str,
        command: str,
        timeout: int | None = DEFAULT_TIMEOUT,
        workdir: str = "/home/sandbox",
        env: dict[str, str] | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command inside a container, streaming output line-by-line.

        Calls *on_output* with each line of merged stdout/stderr as it arrives.
        Returns (returncode, stdout, stderr) like the split_output variant.

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
    """List files in /home/sandbox to detect newly created files."""
    exit_code, output = _manager.exec_in_container(
        container_id,
        "find /home/sandbox -maxdepth 8 -not -path '*/\\.*' -type f 2>/dev/null",
        timeout=10,
    )
    if exit_code == 0:
        return {
            line.replace("/home/sandbox/", "", 1)
            for line in output.strip().split("\n")
            if line.strip()
        }
    return set()


T = TypeVar("T")

_PROGRESS_HEARTBEAT_INTERVAL = 5.0  # seconds between SSE keep-alive heartbeats


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
            await ctx.report_progress(progress=1, total=1)
            return future.result()
        heartbeat += 1
        await ctx.report_progress(progress=heartbeat, total=heartbeat + 1)


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------


@mcp.tool(
    title="Install Python Packages",
    description="""Install Python packages via pip. Network is enabled automatically during install.

- packages: Space-separated package names, e.g. "numpy matplotlib scipy==1.12.0".""",
)
async def sandbox_install_packages(
    packages: Annotated[
        str | None,
        Field(description="Space-separated package names, e.g. 'numpy matplotlib scipy==1.12.0'."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    if not packages:
        return json.dumps({"status": "error", "error": "No packages specified"}, indent=2)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )
            if not _manager.enable_network(container_info.container_id):
                return json.dumps(
                    {
                        "status": "error",
                        "installed": [],
                        "output": "Failed to enable network access for pip install.",
                    },
                    indent=2,
                )

            try:
                exit_code, output = _manager.exec_in_container(
                    container_info.container_id,
                    f"pip install --user {packages}",
                    timeout=INSTALL_TIMEOUT,
                )

                installed = []
                if exit_code == 0:
                    # Parse successfully installed packages from pip output
                    for line in output.split("\n"):
                        if line.strip().startswith("Successfully installed"):
                            installed = [pkg.rsplit("-", 1)[0] for pkg in line.strip().split()[2:]]
                            break
                    if not installed:
                        # Fallback: assume requested packages were installed
                        installed = [p for p in packages.split() if not p.startswith("-")]
                    container_info.installed_packages.extend(installed)

                return json.dumps(
                    {
                        "status": "ok" if exit_code == 0 else "error",
                        "session_id": sid,
                        "installed": installed,
                        "output": output[-MAX_OUTPUT_BYTES:],
                    },
                    indent=2,
                )
            finally:
                if not container_info.network_enabled:
                    _manager.disable_network(container_info.container_id)

        except DockerNotAvailableError:
            return json.dumps(
                {
                    "status": "error",
                    "installed": [],
                    "output": "Docker is not available. "
                    "Please install Docker and ensure it is running.",
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_install_packages")
            return json.dumps(
                {"status": "error", "installed": [], "output": str(e)},
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Run Code",
    description="""Execute a shell command in the isolated sandbox container. Only sandbox-local
paths (e.g. /home/sandbox) are accessible — host paths do not exist here.
No internet by default. Output is streamed in real time.

- command: Shell command to run (e.g. "python train.py --data-dir /data").
- network: Temporarily enable internet for this command (default false).
- timeout: Max seconds before the command is killed (default: no limit).""",
)
async def sandbox_run_code(
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
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    if not command:
        return json.dumps({"status": "error", "error": "No command specified"}, indent=2)

    # Queue for streaming output lines from the worker thread to the async loop.
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _on_output(line: str) -> None:
        """Callback invoked in the worker thread for each output line."""
        line_queue.put(line)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )

            # Enable network temporarily if requested (and not already persistent)
            temp_network = False
            if network and not container_info.network_enabled:
                temp_network = _manager.enable_network(container_info.container_id)

            try:
                # Snapshot files before execution
                files_before = _list_workspace_files(container_info.container_id)

                exit_code, stdout, stderr = _manager.exec_in_container_streaming(
                    container_info.container_id,
                    command,
                    timeout=timeout,
                    on_output=_on_output,
                )

                # Detect new files
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
            logger.exception("Error in sandbox_run_code")
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
                await ctx.info(line)

    heartbeat = 0
    while True:
        done, _ = await asyncio.wait({future}, timeout=0.25)
        await _drain_queue()

        if done:
            await _drain_queue()
            await ctx.report_progress(progress=1, total=1)
            return future.result()

        heartbeat += 1
        await ctx.report_progress(progress=heartbeat, total=heartbeat + 1)


@mcp.tool(
    title="Write File to Sandbox",
    description="""Write content to a file in the sandbox.
Parent directories are created automatically.
Relative paths resolve from /home/sandbox.

- path: Destination path in the sandbox (relative to /home/sandbox or absolute).
- content: File content (text, or base64 when encoding="base64").
- encoding: "utf-8" (default) or "base64" for binary files.""",
)
async def sandbox_write_file(
    path: Annotated[
        str | None,
        Field(description="Destination path in the sandbox (relative to /home/sandbox or absolute)."),
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
                container_path = f"/home/sandbox/{path}"
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
                    "/home/sandbox",
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
    title="List Files in Sandbox",
    description="""List files in the sandbox.
Use absolute paths to explore data mounts (e.g. "/data").

- path: Directory to list, relative to /home/sandbox or absolute (default ".")
- max_depth: Recursion depth 1-10 (default 3)""",
)
async def sandbox_list_files(
    path: Annotated[
        str, Field(description="Directory to list (relative to /home/sandbox or absolute).")
    ] = ".",
    max_depth: Annotated[int, Field(description="Recursion depth, 1-10.")] = 3,
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

            # Normalise to absolute path inside container
            if not path.startswith("/"):
                container_path = f"/home/sandbox/{path}"
            else:
                container_path = path

            clamped_depth = min(max(max_depth, 1), 10)

            exit_code, output = _manager.exec_in_container(
                container_info.container_id,
                f"find '{container_path}' -maxdepth {clamped_depth} -not -path '*/\\.*' "
                f"\\( -type f -o -type d \\) 2>/dev/null | sort | head -500",
                timeout=15,
            )

            files: list[str] = []
            if exit_code == 0 and output.strip():
                files = [
                    line.replace("/home/sandbox/", "", 1)
                    for line in output.strip().split("\n")
                    if line.strip()
                ]

            return json.dumps(
                {
                    "status": "ok",
                    "session_id": sid,
                    "path": path,
                    "files": files,
                    "count": len(files),
                },
                indent=2,
            )

        except DockerNotAvailableError:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Docker is not available.",
                    "path": path,
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_list_files")
            return json.dumps(
                {"status": "error", "error": str(e), "path": path},
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
                "du -sm /home/sandbox 2>/dev/null | awk '{print $1}'",
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

- path: Path in sandbox (relative to /home/sandbox or absolute).
- dest: Absolute server path to save to (default: session downloads directory).""",
)
async def sandbox_download_file(
    path: Annotated[
        str | None, Field(description="Path in sandbox (relative to /home/sandbox or absolute).")
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
                container_path = f"/home/sandbox/{path}"
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
                await ctx.info(msg)

    heartbeat = 0
    while True:
        done, _ = await asyncio.wait({future}, timeout=0.25)
        await _drain()
        if done:
            await ctx.report_progress(progress=1, total=1)
            return future.result()
        heartbeat += 1
        await ctx.report_progress(progress=heartbeat, total=heartbeat + 1)


@mcp.tool(
    title="Upload File to Sandbox",
    description="""Copy a file or directory into the sandbox.
Provide exactly one of src, data, or content.

- src: Absolute server path to copy from.
- data: Base64-encoded file content (requires filename).
- content: Plain UTF-8 text content (requires filename).
- filename: Name for the file when using data or content.
- dest: Destination in sandbox (relative or absolute;
  default: /home/sandbox/<filename>).""",
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
        Field(description="Destination in sandbox (relative to /home/sandbox or absolute)."),
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
                    container_dest = dest if dest.startswith("/") else f"/home/sandbox/{dest}"
                else:
                    container_dest = f"/home/sandbox/{filename}"

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
                    container_dest = dest if dest.startswith("/") else f"/home/sandbox/{dest}"
                else:
                    container_dest = f"/home/sandbox/{filename}"

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
                container_dest = dest if dest.startswith("/") else f"/home/sandbox/{dest}"
            else:
                container_dest = f"/home/sandbox/{os.path.basename(abs_src)}"

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
    title="Read File from Sandbox",
    description="""Read file contents from the sandbox.
For large binary files, use sandbox_download_file.

- path: Path in sandbox (relative to /home/sandbox or absolute)
- max_bytes: Max bytes to read (default 100000, cap 1MB)
- offset: Byte offset to start from (default 0, useful for tailing logs)""",
)
async def sandbox_read_file(
    path: Annotated[
        str | None, Field(description="Path in sandbox (relative to /home/sandbox or absolute).")
    ] = None,
    max_bytes: Annotated[int, Field(description="Max bytes to read (cap 1MB).")] = 100000,
    offset: Annotated[
        int, Field(description="Byte offset to start from (useful for tailing logs).")
    ] = 0,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    if not path:
        return json.dumps({"status": "error", "error": "No path specified"}, indent=2)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(
                sid, data_path, extra_mounts=DATA_MOUNTS
            )

            # Normalise to absolute path inside container
            if not path.startswith("/"):
                container_path = f"/home/sandbox/{path}"
            else:
                container_path = path

            # Get file size first
            exit_code, size_out = _manager.exec_in_container(
                container_info.container_id,
                f"stat -c '%s' '{container_path}' 2>/dev/null",
                timeout=10,
            )
            if exit_code != 0:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"File not found or not readable: {container_path}",
                    },
                    indent=2,
                )

            try:
                total_size = int(size_out.strip())
            except (ValueError, TypeError):
                total_size = 0

            # Read content with dd for precise offset/length control
            clamped_max = min(max_bytes, 1_000_000)  # hard cap at 1MB
            dd_cmd = f"dd if='{container_path}' bs=1 skip={offset} count={clamped_max} 2>/dev/null"
            exit_code, content = _manager.exec_in_container(
                container_info.container_id,
                dd_cmd,
                timeout=30,
            )

            if exit_code != 0:
                return json.dumps(
                    {
                        "status": "error",
                        "error": f"Failed to read file: {container_path}",
                    },
                    indent=2,
                )

            truncated = (total_size - offset) > clamped_max

            return json.dumps(
                {
                    "status": "ok",
                    "session_id": sid,
                    "content": content,
                    "size_bytes": total_size,
                    "offset": offset,
                    "bytes_read": len(content),
                    "truncated": truncated,
                },
                indent=2,
            )

        except DockerNotAvailableError:
            return json.dumps(
                {"status": "error", "error": "Docker is not available."},
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_read_file")
            return json.dumps(
                {"status": "error", "error": str(e)},
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)


# ---------------------------------------------------------------------------
# Git MCP Tool definitions
# ---------------------------------------------------------------------------


def _run_git_command(
    command: str,
    session_id: str | None = None,
    network: bool = False,
    timeout: int = 60,
) -> str:
    """Run a git command inside the sandbox container.

    Handles container creation, optional network access, and returns
    a JSON-encoded result dict.
    """
    sid = _get_session_id(session_id)
    data_path = _get_data_path(sid)

    try:
        container_info = _manager.get_or_create_container(sid, data_path, extra_mounts=DATA_MOUNTS)

        temp_network = False
        if network and not container_info.network_enabled:
            temp_network = _manager.enable_network(container_info.container_id)
            if not temp_network:
                return json.dumps(
                    {
                        "status": "error",
                        "session_id": sid,
                        "output": "Failed to enable network for git operation. "
                        "The container could not connect to the network.",
                        "returncode": -1,
                    },
                    indent=2,
                )

        try:
            exit_code, output = _manager.exec_in_container(
                container_info.container_id,
                command,
                timeout=timeout,
            )

            # Detect authentication failures that git may not surface clearly
            status = "ok" if exit_code == 0 else "error"
            lower_output = output.lower()
            if exit_code != 0 and (
                "authentication" in lower_output
                or "could not read username" in lower_output
                or "permission denied" in lower_output
                or "403" in output
                or "401" in output
            ):
                output += (
                    "\n\nHint: GitHub authentication may not be configured. "
                    "Run 'onit-sandbox setup' on the host to store a GitHub token."
                )

            return json.dumps(
                {
                    "status": status,
                    "session_id": sid,
                    "output": output[-MAX_OUTPUT_BYTES:],
                    "returncode": exit_code,
                },
                indent=2,
            )
        finally:
            if temp_network:
                _manager.disable_network(container_info.container_id)

    except DockerNotAvailableError:
        return json.dumps(
            {
                "status": "error",
                "output": "Docker is not available. "
                "Please install Docker and ensure it is running.",
                "returncode": -1,
            },
            indent=2,
        )
    except Exception as e:
        logger.exception("Error in git command: %s", command)
        return json.dumps(
            {"status": "error", "output": str(e), "returncode": -1},
            indent=2,
        )


@mcp.tool(
    title="Git Clone",
    description="""Clone a git repository into the sandbox workspace.
Network is enabled automatically during clone.

- url: Repository URL (HTTPS). For private repos, run 'onit-sandbox setup' first.
- path: Optional target directory name (relative to /home/sandbox).""",
)
async def sandbox_git_clone(
    url: Annotated[
        str,
        Field(
            description="Repository URL to clone (HTTPS), e.g. 'https://github.com/user/repo.git'."
        ),
    ],
    path: Annotated[
        str | None,
        Field(description="Target directory name relative to /home/sandbox. Defaults to repo name."),
    ] = None,
    branch: Annotated[
        str | None,
        Field(description="Branch to clone. Defaults to the remote default branch."),
    ] = None,
    depth: Annotated[
        int | None,
        Field(description="Create a shallow clone with this many commits. Omit for full history."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cmd = "git clone"
        if branch:
            cmd += f" --branch {branch}"
        if depth and depth > 0:
            cmd += f" --depth {depth}"
        cmd += f" {url}"
        if path:
            cmd += f" {path}"
        return _run_git_command(cmd, session_id=session_id, network=True, timeout=300)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Status",
    description="""Show the working tree status of a git repository in the sandbox.

- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_status(
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        return _run_git_command(f"{cd}git status", session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Add",
    description="""Stage files for the next commit.

- files: Space-separated file paths or patterns to stage. Use '.' to stage all changes.
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_add(
    files: Annotated[
        str,
        Field(description="Files to stage, e.g. '.' or 'src/main.py tests/' or '-A'."),
    ],
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        return _run_git_command(f"{cd}git add {files}", session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Commit",
    description="""Create a git commit with staged changes.

- message: Commit message.
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.
- all: If true, automatically stage all modified/deleted files before committing (-a).""",
)
async def sandbox_git_commit(
    message: Annotated[
        str,
        Field(description="Commit message."),
    ],
    all: Annotated[
        bool,
        Field(description="Stage all modified/deleted files before committing (-a flag)."),
    ] = False,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        # Use heredoc to safely pass the commit message
        all_flag = " -a" if all else ""
        cmd = f"{cd}git commit{all_flag} -m \"$(cat <<'COMMITMSG'\n{message}\nCOMMITMSG\n)\""
        return _run_git_command(cmd, session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Pull",
    description="""Pull changes from a remote repository. Network is enabled automatically.

- remote: Remote name (default: origin).
- branch: Branch to pull. Defaults to current branch.
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_pull(
    remote: Annotated[
        str,
        Field(description="Remote name (default: origin)."),
    ] = "origin",
    branch: Annotated[
        str | None,
        Field(description="Branch to pull. Defaults to the current tracking branch."),
    ] = None,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        branch_arg = f" {branch}" if branch else ""
        return _run_git_command(
            f"{cd}git pull {remote}{branch_arg}",
            session_id=session_id,
            network=True,
        )

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Push",
    description="""Push commits to a remote repository. Network is enabled automatically.
For private repos, run 'onit-sandbox setup' to configure GitHub authentication first.

- remote: Remote name (default: origin).
- branch: Branch to push. Defaults to current branch.
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_push(
    remote: Annotated[
        str,
        Field(description="Remote name (default: origin)."),
    ] = "origin",
    branch: Annotated[
        str | None,
        Field(description="Branch to push. Defaults to the current branch."),
    ] = None,
    set_upstream: Annotated[
        bool,
        Field(description="Set upstream tracking reference (-u flag)."),
    ] = False,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        upstream_flag = " -u" if set_upstream else ""
        branch_arg = f" {branch}" if branch else ""
        # Use --porcelain for machine-readable output and 2>&1 to capture
        # progress/errors from stderr
        result = _run_git_command(
            f"{cd}git push --porcelain{upstream_flag} {remote}{branch_arg} 2>&1",
            session_id=session_id,
            network=True,
            timeout=120,
        )
        # Parse the result to add clarity about what happened
        parsed = json.loads(result)
        if parsed.get("status") == "ok":
            output = parsed.get("output", "")
            if "rejected" in output:
                parsed["status"] = "error"
                parsed["output"] = output + (
                    "\n\nPush was rejected by the remote. "
                    "You may need to pull first or force push."
                )
            elif "up to date" in output.lower() or "[up to date]" in output:
                parsed["output"] = output + (
                    "\n\nNothing was pushed — the remote already has these commits."
                )
        return json.dumps(parsed, indent=2)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Branch",
    description="""List, create, or delete branches.

- name: Branch name to create. Omit to list branches.
- delete: If true, delete the named branch.
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_branch(
    name: Annotated[
        str | None,
        Field(description="Branch name to create. Omit to list all branches."),
    ] = None,
    delete: Annotated[
        bool,
        Field(description="Delete the named branch instead of creating it."),
    ] = False,
    all: Annotated[
        bool,
        Field(description="List both local and remote-tracking branches (-a)."),
    ] = False,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        if name and delete:
            cmd = f"{cd}git branch -d {name}"
        elif name:
            cmd = f"{cd}git branch {name}"
        else:
            all_flag = " -a" if all else ""
            cmd = f"{cd}git branch{all_flag}"
        return _run_git_command(cmd, session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Checkout",
    description="""Switch branches or restore working tree files.

- target: Branch name, tag, or commit hash to check out.
- create: If true, create a new branch and switch to it (-b flag).
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_checkout(
    target: Annotated[
        str,
        Field(description="Branch name, tag, or commit hash to check out."),
    ],
    create: Annotated[
        bool,
        Field(description="Create a new branch and switch to it (-b flag)."),
    ] = False,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        create_flag = " -b" if create else ""
        return _run_git_command(f"{cd}git checkout{create_flag} {target}", session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Log",
    description="""Show commit history.

- max_count: Maximum number of commits to show (default: 20).
- oneline: If true, use condensed one-line format.
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_log(
    max_count: Annotated[
        int,
        Field(description="Maximum number of commits to show."),
    ] = 20,
    oneline: Annotated[
        bool,
        Field(description="Use condensed one-line format (--oneline)."),
    ] = True,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        oneline_flag = " --oneline" if oneline else ""
        return _run_git_command(
            f"{cd}git log -n {max_count}{oneline_flag}",
            session_id=session_id,
        )

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Diff",
    description="""Show changes between commits, working tree, and staging area.

- staged: If true, show staged changes (--cached/--staged).
- target: Compare against a specific commit, branch, or ref.
- path: Directory of the git repo (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_diff(
    staged: Annotated[
        bool,
        Field(description="Show only staged changes (--staged)."),
    ] = False,
    target: Annotated[
        str | None,
        Field(
            description="Compare against a specific commit, branch, or ref (e.g. 'HEAD~1', 'main')."
        ),
    ] = None,
    files: Annotated[
        str | None,
        Field(description="Limit diff to specific files (space-separated paths)."),
    ] = None,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        staged_flag = " --staged" if staged else ""
        target_arg = f" {target}" if target else ""
        files_arg = f" -- {files}" if files else ""
        return _run_git_command(
            f"{cd}git diff{staged_flag}{target_arg}{files_arg}",
            session_id=session_id,
        )

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Init",
    description="""Initialize a new git repository in the sandbox workspace.

- path: Directory to initialize (relative to /home/sandbox). Defaults to /home/sandbox.""",
)
async def sandbox_git_init(
    path: Annotated[
        str | None,
        Field(
            description="Directory to initialize relative to /home/sandbox. Defaults to /home/sandbox."
        ),
    ] = None,
    default_branch: Annotated[
        str | None,
        Field(description="Name for the initial branch (e.g. 'main'). Defaults to git's default."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        target = f"/home/sandbox/{path}" if path else "/home/sandbox"
        branch_flag = f" --initial-branch={default_branch}" if default_branch else ""
        return _run_git_command(f"git init{branch_flag} {target}", session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Remote",
    description="""Manage remote repositories.

- action: 'list' (default), 'add', or 'remove'.
- name: Remote name (required for add/remove).
- url: Remote URL (required for add).""",
)
async def sandbox_git_remote(
    action: Annotated[
        str,
        Field(description="Action: 'list', 'add', or 'remove'."),
    ] = "list",
    name: Annotated[
        str | None,
        Field(description="Remote name (required for add/remove)."),
    ] = None,
    url: Annotated[
        str | None,
        Field(description="Remote URL (required for add)."),
    ] = None,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        if action == "add":
            if not name or not url:
                return json.dumps(
                    {"status": "error", "output": "Both 'name' and 'url' are required for 'add'."},
                    indent=2,
                )
            cmd = f"{cd}git remote add {name} {url}"
        elif action == "remove":
            if not name:
                return json.dumps(
                    {"status": "error", "output": "'name' is required for 'remove'."},
                    indent=2,
                )
            cmd = f"{cd}git remote remove {name}"
        else:
            cmd = f"{cd}git remote -v"
        return _run_git_command(cmd, session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Stash",
    description="""Stash or restore uncommitted changes.

- action: 'push' (default), 'pop', 'list', 'apply', or 'drop'.
- message: Optional message for stash push.""",
)
async def sandbox_git_stash(
    action: Annotated[
        str,
        Field(description="Action: 'push' (default), 'pop', 'list', 'apply', or 'drop'."),
    ] = "push",
    message: Annotated[
        str | None,
        Field(description="Optional message for stash push."),
    ] = None,
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        cd = f"cd /home/sandbox/{path} && " if path else ""
        if action == "push" and message:
            cmd = f'{cd}git stash push -m "{message}"'
        else:
            cmd = f"{cd}git stash {action}"
        return _run_git_command(cmd, session_id=session_id)

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Git Auth Status",
    description="""Check whether GitHub authentication is configured in the sandbox.
Verifies the credential helper and token availability. Call this before push/pull
to private repos if you encounter authentication errors.""",
)
async def sandbox_git_auth_status(
    path: Annotated[
        str | None,
        Field(description="Git repo directory relative to /home/sandbox. Defaults to /home/sandbox."),
    ] = None,
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

            cd = f"cd /home/sandbox/{path} && " if path else ""
            # Check: 1) credential helper configured, 2) GITHUB_TOKEN env var set,
            # 3) helper script exists
            check_script = (
                f"{cd}"
                "echo '=== Credential Helper ===' && "
                "git config --global credential.helper && "
                "echo '=== Token Available ===' && "
                '([ -n "$GITHUB_TOKEN" ] && echo "yes (${#GITHUB_TOKEN} chars)" || echo "no") && '
                "echo '=== Helper Script ===' && "
                "([ -x /home/sandbox/.local/bin/git-credential-github-token ] && echo 'exists' || echo 'missing') && "
                "echo '=== Git User Config ===' && "
                "git config --global user.name 2>/dev/null || echo '(not set)' && "
                "git config --global user.email 2>/dev/null || echo '(not set)'"
            )
            exit_code, output = _manager.exec_in_container(
                container_info.container_id,
                check_script,
                timeout=10,
            )

            # Determine overall auth readiness
            has_token = "yes (" in output
            has_helper = "exists" in output
            auth_ready = has_token and has_helper

            return json.dumps(
                {
                    "status": "ok",
                    "session_id": sid,
                    "auth_configured": auth_ready,
                    "details": output.strip(),
                    "hint": None
                    if auth_ready
                    else "GitHub token not configured. Run 'onit-sandbox setup' on the host to store a GitHub Personal Access Token.",
                },
                indent=2,
            )
        except Exception as e:
            return json.dumps(
                {"status": "error", "output": str(e)},
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Sandbox Filesystem Info",
    description="""Explains the filesystem layout inside the sandbox container.
Returns a description of key directories, their purpose, and current mount configuration.
Call this tool first to orient yourself before working with files in the sandbox.""",
)
async def sandbox_filesystem_info(
    session_id: Annotated[str | None, Field(description="Session identifier.")] = None,
    ctx: Context | None = None,
) -> str:
    mounts_info = []
    for m in DATA_MOUNTS:
        mounts_info.append(
            f"  {m['container']}  (mounted from host, mode: {m['mode']})"
        )

    mounts_section = "\n".join(mounts_info) if mounts_info else "  (none configured)"

    info = f"""\
=== Sandbox Container Filesystem Layout ===

/home/sandbox       Default home directory and working directory. This is
                    where you should create, edit, and run code. All relative
                    paths resolve here.
  .cache/pip        Shared pip cache — persists across sessions for faster
                    package installs.
  .cache/huggingface  HuggingFace cache (models, datasets, tokenizers).
  .local/bin        User-installed executables (on PATH).

/data               Optional data directory. Typically mounted read-only for
                    datasets, CSVs, or other input files the agent should
                    read but not modify.

/tmp                Temporary files. Cleared on container restart.

--- Environment ---
USER:               sandbox (uid 1000)
HOME:               /home/sandbox
HF_HOME:            /home/sandbox/.cache/huggingface
PATH includes:      /home/sandbox/.local/bin

--- Active Data Mounts ---
{mounts_section}

--- Tips ---
- Use /home/sandbox for all code and output files.
- Install packages with sandbox_install_packages (pip).
- Network is disabled by default; use sandbox_enable_network or the
  network=true flag on sandbox_run_code to access the internet.
- Use sandbox_list_files to explore directory contents.
"""
    return json.dumps({"status": "ok", "filesystem_info": info}, indent=2)


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
