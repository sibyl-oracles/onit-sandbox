"""
Sandbox MCP Server configuration and initialization.

Provides the SandboxMCPServer class that wraps FastMCP with
host/port/path configuration and lifecycle management.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# Default server settings
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 18205
DEFAULT_PATH = "/sse"
DEFAULT_DATA_PATH = "/tmp/onit/data"
DEFAULT_PIP_CACHE_PATH = os.getenv("SANDBOX_PIP_CACHE_PATH", "/tmp/onit/pip-cache")

# Docker configuration constants
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "onit-sandbox:latest")
FALLBACK_IMAGE = os.getenv("FALLBACK_IMAGE", "python:3.12-slim")
DEFAULT_MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "2g")
DEFAULT_CPU_QUOTA = int(os.getenv("SANDBOX_CPU_QUOTA", "100000"))  # 1 CPU
DEFAULT_PIDS_LIMIT = int(os.getenv("SANDBOX_PIDS_LIMIT", "100"))
DEFAULT_TIMEOUT = int(os.getenv("SANDBOX_DEFAULT_TIMEOUT", "60"))
MAX_TIMEOUT = int(os.getenv("SANDBOX_MAX_TIMEOUT", "3600"))  # 1 hour
INSTALL_TIMEOUT = int(os.getenv("SANDBOX_INSTALL_TIMEOUT", "300"))


def build_server_url(host: str, port: int, transport: str, path: str = DEFAULT_PATH) -> str:
    """Build the full server endpoint URL for a given transport."""
    if transport == "streamable-http":
        return f"http://{host}:{port}/mcp"
    return f"http://{host}:{port}{path}"


@dataclass
class SandboxMCPServer:
    """
    Configuration and lifecycle wrapper for the Sandbox MCP Server.

    Mirrors the WorkspaceMCPServer pattern from onit-workspace.
    """

    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    path: str = DEFAULT_PATH
    data_path: str = DEFAULT_DATA_PATH
    transport: str = "streamable-http"
    verbose: bool = False

    _mcp: FastMCP | None = field(default=None, init=False, repr=False)

    @property
    def mcp(self) -> FastMCP:
        """Get or create the FastMCP server instance."""
        if self._mcp is None:
            self._mcp = FastMCP("Sandbox MCP Server", host=self.host, port=self.port)
        return self._mcp

    @mcp.setter
    def mcp(self, value: FastMCP) -> None:
        self._mcp = value

    @property
    def url(self) -> str:
        """Full endpoint URL."""
        return build_server_url(self.host, self.port, self.transport, self.path)

    def run(self) -> None:
        """Start the MCP server (blocking)."""
        if self.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        logger.info("Starting Sandbox MCP Server on %s (transport: %s)", self.url, self.transport)

        # Ensure settings reflect current config (in case host/port changed after init)
        self.mcp.settings.host = self.host
        self.mcp.settings.port = self.port

        if self.transport == "stdio":
            self.mcp.run(transport="stdio")
        elif self.transport == "streamable-http":
            self.mcp.run(transport="streamable-http")
        else:
            self.mcp.run(transport="sse")
