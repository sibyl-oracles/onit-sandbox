"""
Sandbox MCP Server — tool definitions for safe code execution.

Provides an isolated Docker-based sandbox for developing, running, and testing
Python projects without affecting the host system. Each session gets its own
container with resource limits, network isolation, and a shared workspace.

Tools:
- install_packages: Install Python packages in the sandbox
- run_code: Execute commands in the isolated sandbox
- sandbox_status: Inspect sandbox state, packages, and resource usage
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, overload

from onit_sandbox.server import (
    DEFAULT_CPU_QUOTA,
    DEFAULT_MEMORY_LIMIT,
    DEFAULT_PIDS_LIMIT,
    DEFAULT_TIMEOUT,
    FALLBACK_IMAGE,
    INSTALL_TIMEOUT,
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
            "1000:1000",
            "-e",
            "HOME=/home/sandbox",
            "-e",
            "PATH=/home/sandbox/.local/bin:/usr/local/bin:/usr/bin:/bin",
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
                    " && chown -R 1000:1000 /home/sandbox",
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
        cmd = ["docker", "exec", "-w", workdir]

        if env:
            for key, value in env.items():
                cmd.extend(["-e", f"{key}={value}"])

        cmd.extend([container_id, "sh", "-c", command])

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
            for session_id in list(self._containers.keys()):
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


def _get_session_id() -> str:
    return SESSION_ID or str(uuid.uuid4())


def _get_data_path() -> str:
    return DATA_PATH


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


# ---------------------------------------------------------------------------
# MCP Tool definitions
# ---------------------------------------------------------------------------


@mcp.tool(
    title="Install Python Packages",
    description="""Install Python packages in the code execution environment using pip.
Call this BEFORE running code that requires external libraries.
Multiple packages can be specified in a single call, separated by spaces.

Args:
- packages: Space-separated package names with optional versions
  (e.g., "numpy matplotlib", "scipy==1.12.0 pandas>=2.0")

Returns JSON: {packages, installed, status, output}

Examples:
  install_packages(packages="numpy matplotlib")
  install_packages(packages="torch torchvision --index-url https://download.pytorch.org/whl/cpu")""",
)
def install_packages(packages: str | None = None) -> str:
    if not packages:
        return json.dumps({"status": "error", "error": "No packages specified"}, indent=2)

    session_id = _get_session_id()
    data_path = _get_data_path()

    try:
        container_info = _manager.get_or_create_container(session_id, data_path)
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
                },
                indent=2,
            )
        finally:
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
        logger.exception("Error in install_packages")
        return json.dumps(
            {"packages": packages, "installed": [], "status": "error", "output": str(e)},
            indent=2,
        )


@mcp.tool(
    title="Run Code",
    description="""Execute a command in the code execution environment.
Use this to run Python scripts, shell commands, or any program inside the sandbox.
Files written by write_file are available in the working directory.
Generated output files (plots, CSVs, etc.) will also appear in the working directory.

Args:
- command: The command to execute (e.g., "python main.py", "python -c 'print(1+1)'")
- timeout: Max seconds to wait (default: 120)

Returns JSON: {command, stdout, stderr, returncode, status, files_created}

Examples:
  run_code(command="python main.py")
  run_code(command="python simulate_ekf.py", timeout=300)
  run_code(command="python -c 'import numpy; print(numpy.__version__)'")""",
)
def run_code(
    command: str | None = None,
    timeout: int = 120,
) -> str:
    if not command:
        return json.dumps({"status": "error", "error": "No command specified"}, indent=2)

    timeout = min(timeout, 600)
    session_id = _get_session_id()
    data_path = _get_data_path()

    try:
        container_info = _manager.get_or_create_container(session_id, data_path)

        # Snapshot files before execution
        files_before = _list_workspace_files(container_info.container_id)

        exit_code, stdout, stderr = _manager.exec_in_container(
            container_info.container_id,
            command,
            timeout=timeout,
            split_output=True,
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

        return json.dumps(
            {
                "command": command,
                "stdout": stdout[-10000:],
                "stderr": stderr[-10000:],
                "returncode": exit_code,
                "status": status,
                "files_created": files_created,
            },
            indent=2,
        )

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
        logger.exception("Error in run_code")
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


@mcp.tool(
    title="Sandbox Status",
    description="""Check the status of the code execution environment.
Shows whether the sandbox is running, what packages are installed, and resource usage.

Returns JSON: {status, python_version, installed_packages,
disk_usage_mb, uptime_seconds, gpu_available}""",
)
def sandbox_status() -> str:
    try:
        docker_available = _manager._check_docker()
        gpu_available = _manager._check_gpu() if docker_available else False
        session_id = _get_session_id()

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
            if session_id not in _manager._containers:
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
            info = _manager._containers[session_id]

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

        return json.dumps(
            {
                "status": "running",
                "python_version": python_version,
                "installed_packages": package_names,
                "disk_usage_mb": disk_usage_mb,
                "uptime_seconds": round(uptime_seconds),
                "gpu_available": gpu_available,
            },
            indent=2,
        )

    except Exception as e:
        logger.exception("Error in sandbox_status")
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


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
