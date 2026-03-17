"""
Sandbox MCP Server — tool definitions for safe code execution.

Provides an isolated Docker-based sandbox for developing, running, and testing
Python projects without affecting the host system. Each session gets its own
container with resource limits, network isolation, and a shared workspace.

Tools:
- sandbox_install_packages: Install Python packages in the sandbox
- sandbox_run_code: Execute commands in the isolated sandbox
- sandbox_get_status: Inspect sandbox state, packages, and resource usage
- sandbox_write_file: Write files into the sandbox
- sandbox_list_files: List files in the sandbox filesystem
- sandbox_download_file: Copy a file from the sandbox to the local filesystem
- sandbox_enable_network: Enable persistent internet access
- sandbox_disable_network: Disable internet access
"""

from __future__ import annotations

import asyncio
import atexit
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
from typing import Any, Literal, TypeVar, overload

from mcp.server.fastmcp import Context

from onit_sandbox.server import (
    DEFAULT_CPU_QUOTA,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_PIP_CACHE_PATH,
    DEFAULT_TIMEOUT,
    FALLBACK_IMAGE,
    INSTALL_TIMEOUT,
    MAX_TIMEOUT,
    SANDBOX_IMAGE,
    SandboxMCPServer,
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
        """Check if the NVIDIA Container Toolkit is available.

        Runs ``docker info`` and looks for the ``nvidia`` runtime, which is
        installed by the NVIDIA Container Toolkit.  The result is cached so
        the subprocess only runs once per process lifetime.
        """
        if self._gpu_available is not None:
            return self._gpu_available

        try:
            result = subprocess.run(
                ["docker", "info", "-f", "{{json .Runtimes}}"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._gpu_available = result.returncode == 0 and "nvidia" in result.stdout.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self._gpu_available = False

        if self._gpu_available:
            logger.info("NVIDIA GPU runtime detected — containers will use --gpus all")
        else:
            logger.debug("No NVIDIA GPU runtime found — containers will run CPU-only")

        return self._gpu_available

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

    def _create_container(self, session_id: str, data_path: str) -> ContainerInfo:
        """Create a new Docker container for the session."""
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
            f"{os.path.abspath(data_path)}:/workspace:rw",
            "--volume",
            f"{pip_cache}:/home/sandbox/.cache/pip:rw",
            "--workdir",
            "/workspace",
            "--memory",
            DEFAULT_MEMORY_LIMIT,
            "--cpu-quota",
            str(DEFAULT_CPU_QUOTA),
            "--pids-limit",
            str(DEFAULT_PIDS_LIMIT),
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
            "OMP_NUM_THREADS=4",
        ]

        if gpu_available:
            cmd.extend(["--gpus", "all"])

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

    def get_or_create_container(self, session_id: str, data_path: str) -> ContainerInfo:
        """Get existing container or create a new one for the session."""
        with self._lock:
            if session_id in self._containers:
                info = self._containers[session_id]
                if self._is_container_running(info.container_id):
                    return info
                else:
                    del self._containers[session_id]

            info = self._create_container(session_id, data_path)
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
        timeout: int = DEFAULT_TIMEOUT,
        workdir: str = "/workspace",
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
            """Kill process, close pipes, and join reader threads."""
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

            deadline = time.monotonic() + timeout
            sentinels_received = 0

            while sentinels_received < 2:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _cleanup_proc(proc, t_out, t_err)
                    return -1, "".join(stdout_lines), f"Command timed out after {timeout} seconds"

                try:
                    label, line = output_queue.get(timeout=min(remaining, 1.0))
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
    """List files in /workspace to detect newly created files."""
    exit_code, output = _manager.exec_in_container(
        container_id,
        "find /workspace -maxdepth 3 -type f 2>/dev/null",
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
    description="""Install Python packages inside the isolated Docker sandbox using pip.
Packages are installed in the sandbox container, NOT on your local system.
Call this BEFORE running code that requires external libraries.
Multiple packages can be specified in a single call, separated by spaces.
Network access is enabled automatically for the duration of the install.

Args:
- packages: Space-separated package names with optional versions
  (e.g., "numpy matplotlib", "scipy==1.12.0 pandas>=2.0")
- session_id: Optional identifier to isolate this sandbox from other agents.
  Calls with the same session_id share a container; different IDs get separate containers.

Returns JSON: {packages, installed, status, output}

Examples:
  sandbox_install_packages(packages="numpy matplotlib")
  sandbox_install_packages(packages="torch torchvision --index-url https://download.pytorch.org/whl/cpu")""",
)
async def sandbox_install_packages(
    packages: str | None = None,
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    if not packages:
        return json.dumps({"status": "error", "error": "No packages specified"}, indent=2)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(sid, data_path)
            if not _manager.enable_network(container_info.container_id):
                return json.dumps(
                    {
                        "packages": packages,
                        "installed": [],
                        "status": "error",
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
                        "packages": packages,
                        "installed": installed,
                        "status": "ok" if exit_code == 0 else "error",
                        "output": output[-5000:],
                        "workspace_root": "/workspace",
                        "host_data_path": data_path,
                    },
                    indent=2,
                )
            finally:
                if not container_info.network_enabled:
                    _manager.disable_network(container_info.container_id)

        except DockerNotAvailableError:
            return json.dumps(
                {
                    "packages": packages,
                    "installed": [],
                    "status": "error",
                    "output": "Docker is not available. "
                    "Please install Docker and ensure it is running.",
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_install_packages")
            return json.dumps(
                {"packages": packages, "installed": [], "status": "error", "output": str(e)},
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="Run Code",
    description="""Execute a command inside an isolated Docker sandbox container.

CRITICAL — Sandbox isolation:
The sandbox is a SEPARATE Docker container with its OWN filesystem.
It is NOT your local filesystem. You CANNOT directly access files
from your local machine or working directory inside the sandbox.
To make code available in the sandbox, use the sandbox_write_file tool to
create files, or pass code inline via "python -c '...'".
To discover what files already exist (e.g., a mounted codebase),
use the sandbox_list_files tool or sandbox_get_status.
Files created inside the sandbox (plots, CSVs, etc.) exist only in
the sandbox's /workspace directory — they are NOT automatically
available on your local filesystem.
To retrieve sandbox files locally, use the sandbox_download_file tool.

IMPORTANT — Network access:
By default the sandbox has NO internet access. If your code needs to
download datasets, call APIs, or access any remote resource, you MUST
either set network=True on this call, or call sandbox_enable_network
first for persistent access across multiple commands.

Args:
- command: The command to execute inside the sandbox
  (e.g., "python main.py", "python -c 'print(1+1)'")
- timeout: Max seconds to wait (default: 120, max: 3600).
  Configurable via SANDBOX_MAX_TIMEOUT env var.
- network: Set to True to temporarily enable internet access for
  this command. The network is disabled again after the command
  finishes. For persistent access, use sandbox_enable_network.
- session_id: Optional identifier to isolate this sandbox from
  other agents. Same session_id = shared container.

Returns JSON: {command, stdout, stderr, returncode, status,
files_created}

Examples:
  sandbox_run_code(command="python main.py")
  sandbox_run_code(command="python simulate_ekf.py", timeout=300)
  sandbox_run_code(command="python download_data.py", network=True)""",
)
async def sandbox_run_code(
    command: str | None = None,
    timeout: int = 120,
    network: bool = False,
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    if not command:
        return json.dumps({"status": "error", "error": "No command specified"}, indent=2)

    clamped_timeout = min(timeout, MAX_TIMEOUT)

    # Queue for streaming output lines from the worker thread to the async loop.
    line_queue: queue.Queue[str | None] = queue.Queue()

    def _on_output(line: str) -> None:
        """Callback invoked in the worker thread for each output line."""
        line_queue.put(line)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(sid, data_path)

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
                    timeout=clamped_timeout,
                    on_output=_on_output,
                )

                # Detect new files
                files_after = _list_workspace_files(container_info.container_id)
                files_created = sorted(files_after - files_before)

                if exit_code == -1 and "timed out" in stderr:
                    status = "timeout"
                elif exit_code == 0:
                    status = "ok"
                else:
                    status = "error"

                result_payload: dict[str, Any] = {
                    "command": command,
                    "stdout": stdout[-10000:],
                    "stderr": stderr[-10000:],
                    "returncode": exit_code,
                    "status": status,
                    "files_created": files_created,
                    "workspace_root": "/workspace",
                    "host_data_path": data_path,
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
                    "command": command,
                    "stdout": "",
                    "stderr": "Docker is not available. "
                    "Please install Docker and ensure it is running.",
                    "returncode": -1,
                    "status": "error",
                    "files_created": [],
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_run_code")
            return json.dumps(
                {
                    "command": command,
                    "stdout": "",
                    "stderr": str(e),
                    "returncode": -1,
                    "status": "error",
                    "files_created": [],
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
    description="""Write content to a file inside the sandbox container.

This is the recommended way to create or update files in the sandbox.
The sandbox has its OWN filesystem — this tool writes directly into it.

The file path is relative to /workspace (the sandbox root) unless an
absolute path starting with / is given.

Parent directories are created automatically.

Args:
- file_path: Path of the file to create/overwrite inside the sandbox.
  Relative paths are resolved from /workspace
  (e.g., "src/main.py" → /workspace/src/main.py).
- content: The full text content to write to the file.
- session_id: Optional identifier for the target sandbox.

Returns JSON: {status, file_path, size_bytes, workspace_root}

Examples:
  sandbox_write_file(file_path="main.py", content="print('hello')")
  sandbox_write_file(file_path="src/utils.py", content="def add(a, b): return a + b")""",
)
async def sandbox_write_file(
    file_path: str | None = None,
    content: str | None = None,
    session_id: str | None = None,
    path: str | None = None,
    ctx: Context | None = None,
) -> str:
    # Accept "path" as an alias for "file_path" (common LLM mis-naming)
    if not file_path and path:
        file_path = path
    if not file_path:
        return json.dumps({"status": "error", "error": "No file_path specified"}, indent=2)
    if content is None:
        return json.dumps({"status": "error", "error": "No content specified"}, indent=2)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(sid, data_path)

            # Normalise to absolute path inside container
            if not file_path.startswith("/"):
                container_path = f"/workspace/{file_path}"
            else:
                container_path = file_path

            # Ensure parent directory exists
            parent_dir = os.path.dirname(container_path)
            if parent_dir and parent_dir != "/":
                _manager.exec_in_container(
                    container_info.container_id,
                    f"mkdir -p '{parent_dir}'",
                    timeout=10,
                )

            # Write content via stdin using docker exec + sh -c "cat > file"
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
                        "file_path": file_path,
                    },
                    indent=2,
                )

            size_bytes = len(content.encode())

            return json.dumps(
                {
                    "status": "ok",
                    "file_path": file_path,
                    "container_path": container_path,
                    "size_bytes": size_bytes,
                    "workspace_root": "/workspace",
                    "host_data_path": data_path,
                },
                indent=2,
            )

        except DockerNotAvailableError:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Docker is not available.",
                    "file_path": file_path,
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_write_file")
            return json.dumps(
                {"status": "error", "error": str(e), "file_path": file_path},
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)


@mcp.tool(
    title="List Files in Sandbox",
    description="""List files and directories inside the sandbox container.

Use this to explore the sandbox filesystem and discover what files are
available — especially useful when a codebase has been mounted into the
sandbox and you need to understand its structure before working with it.

Args:
- path: Directory path to list, relative to /workspace or absolute.
  Defaults to "." (the /workspace root).
- max_depth: How many levels deep to recurse (default: 3).
  Use 1 for a shallow listing, higher values for deeper exploration.
- session_id: Optional identifier for the target sandbox.

Returns JSON: {status, path, workspace_root, host_data_path, files}

Examples:
  sandbox_list_files()  # list /workspace root
  sandbox_list_files(path="src", max_depth=2)
  sandbox_list_files(path=".", max_depth=1)  # shallow listing""",
)
async def sandbox_list_files(
    path: str = ".",
    max_depth: int = 3,
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(sid, data_path)

            # Normalise to absolute path inside container
            if not path.startswith("/"):
                container_path = f"/workspace/{path}"
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
                    line.replace("/workspace/", "", 1)
                    for line in output.strip().split("\n")
                    if line.strip()
                ]

            return json.dumps(
                {
                    "status": "ok",
                    "path": path,
                    "workspace_root": "/workspace",
                    "host_data_path": data_path,
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
    description="""Check the status of the isolated Docker sandbox container.
Shows whether the sandbox is running, what packages are installed, resource usage,
and a listing of files currently in the workspace.
This reports on the sandbox environment, which is separate from your local system.

The response includes:
- workspace_root: The root directory inside the container (/workspace)
- host_data_path: The corresponding directory on the host filesystem
- workspace_files: Files currently present in the workspace (up to 200)

Use this tool first to orient yourself and discover what files exist in
the sandbox before running code.

Args:
- session_id: Optional identifier to check a specific agent's sandbox.

Returns JSON: {status, python_version, installed_packages,
disk_usage_mb, uptime_seconds, gpu_available, network_enabled,
workspace_root, host_data_path, workspace_files}""",
)
async def sandbox_get_status(
    session_id: str | None = None,
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

            return json.dumps(
                {
                    "status": "running",
                    "python_version": python_version,
                    "installed_packages": package_names,
                    "disk_usage_mb": disk_usage_mb,
                    "uptime_seconds": round(uptime_seconds),
                    "gpu_available": gpu_available,
                    "network_enabled": info.network_enabled,
                    "workspace_root": "/workspace",
                    "host_data_path": info.data_path,
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
    description="""Enable persistent internet access for the sandbox.

Call this BEFORE starting work that requires sustained network access,
such as downloading datasets, training with remote logging, calling
external APIs, or any multi-step workflow that needs the internet.

Once enabled, ALL subsequent sandbox_run_code and sandbox_install_packages calls in
this session will have internet access until you call
sandbox_disable_network.

Prefer this over setting network=True on every sandbox_run_code call when
you have multiple commands that need internet access.

Args:
- session_id: Optional identifier for the target sandbox.

Returns JSON: {status, network_enabled}""",
)
async def sandbox_enable_network(
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(sid, data_path)
            success = _manager.enable_network(container_info.container_id)
            if success:
                container_info.network_enabled = True
                return json.dumps(
                    {"status": "ok", "network_enabled": True},
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
    description="""Disable internet access for the sandbox, restoring
network isolation.

Call this when you are done downloading data or accessing remote
services and want to restore the sandbox's default isolated state.

Args:
- session_id: Optional identifier for the target sandbox.

Returns JSON: {status, network_enabled}""",
)
async def sandbox_disable_network(
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(sid, data_path)
            success = _manager.disable_network(container_info.container_id)
            if success:
                container_info.network_enabled = False
                return json.dumps(
                    {"status": "ok", "network_enabled": False},
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
    description="""Copy a file from the sandbox container to the local (agent) filesystem.

IMPORTANT: This is the ONLY correct way to retrieve files from the sandbox.
Do NOT use shell commands like 'docker cp' or other workarounds — always
use sandbox_download_file to copy files out of the sandbox.

Use this whenever sandbox code creates output files — plots, CSVs, model
checkpoints, logs, etc. — so they are available on your local filesystem
for inspection, further processing, or delivery to the user.

The sandbox's /workspace is an isolated Docker filesystem. Files there
are NOT directly accessible from your local tools. This tool bridges
that gap by copying files out of the container.

Args:
- sandbox_path: Path of the file inside the sandbox to copy.
  Can be relative to /workspace (e.g., "output/plot.png") or
  absolute (e.g., "/workspace/output/plot.png").
- dest_path: Absolute path on the local filesystem where the file
  should be written (e.g., "/home/user/results/plot.png").
  Parent directories are created automatically.
- session_id: Optional identifier for the target sandbox.

Returns JSON: {status, sandbox_path, dest_path, size_bytes}

Examples:
  sandbox_download_file(sandbox_path="results/plot.png", dest_path="/home/user/plot.png")
  sandbox_download_file(sandbox_path="model.pt", dest_path="/tmp/model.pt")""",
)
async def sandbox_download_file(
    sandbox_path: str | None = None,
    dest_path: str | None = None,
    session_id: str | None = None,
    ctx: Context | None = None,
) -> str:
    if not sandbox_path:
        return json.dumps({"status": "error", "error": "No sandbox_path specified"}, indent=2)
    if not dest_path:
        return json.dumps({"status": "error", "error": "No dest_path specified"}, indent=2)

    def _impl() -> str:
        sid = _get_session_id(session_id)
        data_path = _get_data_path(sid)

        try:
            container_info = _manager.get_or_create_container(sid, data_path)

            # Normalise sandbox_path to absolute
            if not sandbox_path.startswith("/"):
                container_path = f"/workspace/{sandbox_path}"
            else:
                container_path = sandbox_path

            # Ensure destination directory exists on the host
            os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)

            # Use docker cp to copy the file out (fails with clear error if missing)
            result = subprocess.run(
                [
                    "docker",
                    "cp",
                    f"{container_info.container_id}:{container_path}",
                    dest_path,
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
                        "sandbox_path": sandbox_path,
                        "dest_path": dest_path,
                    },
                    indent=2,
                )

            try:
                size_bytes = os.path.getsize(dest_path)
            except OSError:
                size_bytes = None

            return json.dumps(
                {
                    "status": "ok",
                    "sandbox_path": sandbox_path,
                    "dest_path": dest_path,
                    "size_bytes": size_bytes,
                },
                indent=2,
            )

        except DockerNotAvailableError:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Docker is not available.",
                    "sandbox_path": sandbox_path,
                    "dest_path": dest_path,
                },
                indent=2,
            )
        except subprocess.TimeoutExpired:
            return json.dumps(
                {
                    "status": "error",
                    "error": "Timed out copying file from sandbox.",
                    "sandbox_path": sandbox_path,
                    "dest_path": dest_path,
                },
                indent=2,
            )
        except Exception as e:
            logger.exception("Error in sandbox_download_file")
            return json.dumps(
                {
                    "status": "error",
                    "error": str(e),
                    "sandbox_path": sandbox_path,
                    "dest_path": dest_path,
                },
                indent=2,
            )

    return await _run_with_progress(ctx, _impl)


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
    global DATA_PATH, SESSION_ID

    if options is None:
        options = {}

    if options.get("verbose"):
        logging.basicConfig(level=logging.INFO)

    if "data_path" in options:
        DATA_PATH = options["data_path"]
    if "session_id" in options:
        SESSION_ID = options["session_id"]

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
