"""
Tests for the Sandbox MCP Server

These tests verify the sandbox functionality with and without Docker.
"""

import asyncio
import base64
import json
import os

import pytest

import onit_sandbox.mcp_server as mcp_module
from onit_sandbox.mcp_server import (
    SandboxManager,
    cleanup_sandbox,
    sandbox_disable_network,
    sandbox_download_file,
    sandbox_enable_network,
    sandbox_get_status,
    sandbox_bash,
)


def _run(coro):
    """Helper to run an async tool function from sync test code."""
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def sandbox_session(tmp_path):
    """Set module globals for a test session, then clean up."""
    session_id = "test-session"
    mcp_module.DATA_PATH = str(tmp_path)
    mcp_module.SESSION_ID = session_id
    yield session_id, tmp_path
    cleanup_sandbox(session_id)
    mcp_module.SESSION_ID = None


class TestSandboxManager:
    """Tests for the SandboxManager class."""

    def test_manager_creation(self):
        """Test that manager can be created."""
        manager = SandboxManager()
        assert manager is not None
        assert manager._containers == {}

    def test_docker_check(self):
        """Test Docker availability check."""
        manager = SandboxManager()
        # This will return True or False depending on the system
        result = manager._check_docker()
        assert isinstance(result, bool)

    def test_container_name_generation(self):
        """Test container name generation from session ID."""
        manager = SandboxManager()
        session_id = "test-session-12345678"
        name = manager._get_container_name(session_id)
        assert name.startswith("onit-sandbox-")
        assert len(name) <= 30  # Docker has name length limits


class TestSandboxTools:
    """Tests for the MCP tool functions."""

    def test_sandbox_get_status_no_container(self):
        """Test sandbox_get_status when no container exists."""
        mcp_module.SESSION_ID = "nonexistent-session-id"
        try:
            result = _run(sandbox_get_status())
            data = json.loads(result)
            assert data["status"] in ("not_created", "error")
        finally:
            mcp_module.SESSION_ID = None

    def test_sandbox_bash_no_command(self):
        """Test sandbox_bash with no command specified."""
        result = _run(sandbox_bash(command=None))
        data = json.loads(result)
        assert data["status"] == "error"

    def test_sandbox_download_file_no_path(self):
        """Test sandbox_download_file with no path specified."""
        result = _run(sandbox_download_file(path=None, dest="/tmp/out.txt"))
        data = json.loads(result)
        assert data["status"] == "error"

    def test_sandbox_download_file_no_dest(self):
        """Test sandbox_download_file with no dest specified — still works (dest is optional)."""
        result = _run(sandbox_download_file(path="file.txt", dest=None))
        data = json.loads(result)
        # No container exists, so it errors on container creation, not on missing dest
        assert data["status"] == "error"


@pytest.mark.skipif(
    os.system("docker info > /dev/null 2>&1") != 0,
    reason="Docker not available",
)
class TestSandboxWithDocker:
    """Tests that require Docker to be available."""

    def test_sandbox_bash_simple_command(self, sandbox_session):
        """Test running a simple command in sandbox."""
        result = _run(sandbox_bash(command="echo 'Hello, World!'"))
        data = json.loads(result)

        assert data["status"] == "ok"
        assert "Hello, World!" in data["stdout"]
        assert data["returncode"] == 0
        assert isinstance(data["files_created"], list)

    def test_sandbox_bash_python(self, sandbox_session):
        """Test running Python code in sandbox."""
        result = _run(sandbox_bash(command="python -c \"print('Python works!')\""))
        data = json.loads(result)

        assert data["status"] == "ok", f"sandbox_bash failed: {data}"
        assert "Python works!" in data["stdout"]

    def test_sandbox_pip_install_and_run(self, sandbox_session):
        """Test installing a package via sandbox_bash and using it."""
        install_result = _run(
            sandbox_bash(command="pip install --user cowsay", network=True, timeout=60),
        )
        install_data = json.loads(install_result)
        assert install_data["status"] == "ok"

        run_result = _run(
            sandbox_bash(command="python -c 'import cowsay; cowsay.cow(\"moo\")'"),
        )
        run_data = json.loads(run_result)
        assert run_data["status"] == "ok"

    def test_files_created_detection(self, sandbox_session):
        """Test that files_created detects new files after execution."""
        result = _run(
            sandbox_bash(command="echo 'test content' > /workspace/testfile.txt"),
        )
        data = json.loads(result)

        if data["status"] == "ok":
            assert "testfile.txt" in data["files_created"]

    def test_sandbox_get_status_running(self, sandbox_session):
        """Test sandbox_get_status when container is running."""
        # First, create a container by running a command
        _run(sandbox_bash(command="echo hello"))

        result = _run(sandbox_get_status())
        data = json.loads(result)

        assert data["status"] == "running"
        assert data["python_version"] is not None
        assert isinstance(data["installed_packages"], list)
        assert isinstance(data["disk_usage_mb"], int | float)
        assert isinstance(data["uptime_seconds"], int)

    def test_sandbox_bash_timeout(self, sandbox_session):
        """Test that sandbox_bash respects timeout."""
        result = _run(sandbox_bash(command="sleep 10", timeout=2))
        data = json.loads(result)

        assert data["status"] == "timeout"
        assert data["returncode"] == -1

    def test_sandbox_bash_with_network(self, sandbox_session):
        """Test sandbox_bash with temporary network access."""
        result = _run(
            sandbox_bash(
                command="python -c \"import urllib.request; print(urllib.request.urlopen('http://example.com').status)\"",
                network=True,
                timeout=30,
            ),
        )
        data = json.loads(result)
        assert data["status"] == "ok"
        assert "200" in data["stdout"]

    def test_sandbox_download_file(self, sandbox_session):
        """Test downloading a file from the sandbox as base64."""
        session_id, tmp_path = sandbox_session

        # Create a file inside the sandbox
        _run(sandbox_bash(command="echo 'download test' > /workspace/dl_test.txt"))

        # Download it — returns base64 content
        result = _run(sandbox_download_file(path="dl_test.txt"))
        data = json.loads(result)

        assert data["status"] == "ok"
        assert data["file_name"] == "dl_test.txt"
        assert data["size_bytes"] > 0
        assert "file_data_base64" in data
        # Verify base64 content decodes correctly
        content = base64.b64decode(data["file_data_base64"]).decode()
        assert "download test" in content

    def test_sandbox_download_file_not_found(self, sandbox_session):
        """Test downloading a nonexistent file from the sandbox."""
        session_id, tmp_path = sandbox_session

        # Ensure sandbox is running
        _run(sandbox_bash(command="echo hello"))

        result = _run(sandbox_download_file(path="no_such_file.txt"))
        data = json.loads(result)

        assert data["status"] == "error"
        assert "not find" in data["error"].lower() or "not found" in data["error"].lower()

    def test_enable_disable_network(self, sandbox_session):
        """Test persistent network enable/disable tools."""
        # Create container first
        _run(sandbox_bash(command="echo hello"))

        # Enable persistent network
        result = _run(sandbox_enable_network())
        data = json.loads(result)
        assert data["status"] == "ok"
        assert data["network_enabled"] is True

        # Verify network works in sandbox_bash without network=True flag
        # (bridge stays connected since sandbox_enable_network connected it)
        result = _run(
            sandbox_bash(
                command="python -c \"import urllib.request; print(urllib.request.urlopen('http://example.com').status)\"",
                timeout=30,
            ),
        )
        run_data = json.loads(result)
        assert run_data["status"] == "ok"
        assert "200" in run_data["stdout"]

        # Check status reports network_enabled
        status_result = _run(sandbox_get_status())
        status_data = json.loads(status_result)
        assert status_data["network_enabled"] is True

        # Disable network
        result = _run(sandbox_disable_network())
        data = json.loads(result)
        assert data["status"] == "ok"
        assert data["network_enabled"] is False

        # Verify status updated
        status_result = _run(sandbox_get_status())
        status_data = json.loads(status_result)
        assert status_data["network_enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
