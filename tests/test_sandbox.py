"""
Tests for the Sandbox MCP Server

These tests verify the sandbox functionality with and without Docker.
"""

import json
import os

import pytest

import onit_sandbox.mcp_server as mcp_module
from onit_sandbox.mcp_server import (
    SandboxManager,
    cleanup_sandbox,
    disable_sandbox_network,
    enable_sandbox_network,
    install_packages,
    run_code,
    sandbox_status,
)


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

    def test_sandbox_status_no_container(self):
        """Test sandbox_status when no container exists."""
        mcp_module.SESSION_ID = "nonexistent-session-id"
        try:
            result = sandbox_status()
            data = json.loads(result)
            assert data["status"] in ("not_created", "error")
        finally:
            mcp_module.SESSION_ID = None

    def test_install_packages_no_packages(self):
        """Test install_packages with no packages specified."""
        result = install_packages(packages=None)
        data = json.loads(result)
        assert data["status"] == "error"

    def test_run_code_no_command(self):
        """Test run_code with no command specified."""
        result = run_code(command=None)
        data = json.loads(result)
        assert data["status"] == "error"


@pytest.mark.skipif(
    os.system("docker info > /dev/null 2>&1") != 0,
    reason="Docker not available",
)
class TestSandboxWithDocker:
    """Tests that require Docker to be available."""

    def test_run_code_simple_command(self, sandbox_session):
        """Test running a simple command in sandbox."""
        result = run_code(command="echo 'Hello, World!'")
        data = json.loads(result)

        assert data["status"] == "ok"
        assert "Hello, World!" in data["stdout"]
        assert data["returncode"] == 0
        assert isinstance(data["files_created"], list)

    def test_run_code_python(self, sandbox_session):
        """Test running Python code in sandbox."""
        result = run_code(command="python -c \"print('Python works!')\"")
        data = json.loads(result)

        assert data["status"] == "ok", f"run_code failed: {data}"
        assert "Python works!" in data["stdout"]

    def test_install_packages_and_run(self, sandbox_session):
        """Test installing a package and using it."""
        # Install a small package
        install_result = install_packages(packages="cowsay")
        install_data = json.loads(install_result)

        assert install_data["status"] == "ok"
        assert isinstance(install_data["installed"], list)
        assert len(install_data["installed"]) > 0

        # Run Python with the package
        run_result = run_code(
            command="python -c 'import cowsay; cowsay.cow(\"moo\")'",
        )
        run_data = json.loads(run_result)
        assert run_data["status"] == "ok"

    def test_files_created_detection(self, sandbox_session):
        """Test that files_created detects new files after execution."""
        result = run_code(
            command="echo 'test content' > /workspace/testfile.txt",
        )
        data = json.loads(result)

        if data["status"] == "ok":
            assert "testfile.txt" in data["files_created"]

    def test_sandbox_status_running(self, sandbox_session):
        """Test sandbox_status when container is running."""
        # First, create a container by running a command
        run_code(command="echo hello")

        result = sandbox_status()
        data = json.loads(result)

        assert data["status"] == "running"
        assert data["python_version"] is not None
        assert isinstance(data["installed_packages"], list)
        assert isinstance(data["disk_usage_mb"], (int, float))
        assert isinstance(data["uptime_seconds"], int)

    def test_run_code_timeout(self, sandbox_session):
        """Test that run_code respects timeout."""
        result = run_code(command="sleep 10", timeout=2)
        data = json.loads(result)

        assert data["status"] == "timeout"
        assert data["returncode"] == -1

    def test_run_code_with_network(self, sandbox_session):
        """Test run_code with temporary network access."""
        result = run_code(
            command="python -c \"import urllib.request; print(urllib.request.urlopen('http://example.com').status)\"",
            network=True,
            timeout=30,
        )
        data = json.loads(result)
        assert data["status"] == "ok"
        assert "200" in data["stdout"]

    def test_enable_disable_network(self, sandbox_session):
        """Test persistent network enable/disable tools."""
        # Create container first
        run_code(command="echo hello")

        # Enable persistent network
        result = enable_sandbox_network()
        data = json.loads(result)
        assert data["status"] == "ok"
        assert data["network_enabled"] is True

        # Verify network works in run_code without network=True flag
        # (bridge stays connected since enable_sandbox_network connected it)
        result = run_code(
            command="python -c \"import urllib.request; print(urllib.request.urlopen('http://example.com').status)\"",
            timeout=30,
        )
        run_data = json.loads(result)
        assert run_data["status"] == "ok"
        assert "200" in run_data["stdout"]

        # Check status reports network_enabled
        status_result = sandbox_status()
        status_data = json.loads(status_result)
        assert status_data["network_enabled"] is True

        # Disable network
        result = disable_sandbox_network()
        data = json.loads(result)
        assert data["status"] == "ok"
        assert data["network_enabled"] is False

        # Verify status updated
        status_result = sandbox_status()
        status_data = json.loads(status_result)
        assert status_data["network_enabled"] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
