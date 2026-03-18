"""
CLI for the Sandbox MCP Server.

Provides start / stop / status subcommands with PID-file management,
mirroring the pattern used in onit-workspace.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from onit_sandbox.server import DEFAULT_HOST, DEFAULT_PORT, build_server_url

# State directory — mirrors ~/.onit-workspace/ convention
STATE_DIR = Path.home() / ".onit-sandbox"
PID_FILE = STATE_DIR / "server.pid"
LOG_FILE = STATE_DIR / "server.log"


def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)


def _get_pid() -> int | None:
    """Read stored PID, or None."""
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_pid(pid: int) -> None:
    _ensure_state_dir()
    PID_FILE.write_text(str(pid))


def _remove_pid() -> None:
    if PID_FILE.exists():
        PID_FILE.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> None:
    """Start the sandbox MCP server."""
    pid = _get_pid()
    if pid and _is_process_running(pid):
        print(f"Server is already running (PID: {pid})")
        return

    if args.foreground:
        _run_foreground(args)
    else:
        _run_background(args)


def _run_foreground(args: argparse.Namespace) -> None:
    """Run the server in the foreground (blocking)."""
    from onit_sandbox.mcp_server import run as mcp_run

    mounts = getattr(args, "mount", None) or []

    print(
        f"Starting Sandbox MCP Server on "
        f"{build_server_url(args.host, args.port, args.transport)} "
        f"(transport: {args.transport})"
    )
    if mounts:
        print(f"Data mounts: {', '.join(mounts)}")

    _write_pid(os.getpid())
    try:
        mcp_run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            options={
                "data_path": getattr(args, "data_path", "/tmp/onit/data"),
                "verbose": args.verbose,
                "data_mounts": mounts,
            },
        )
    finally:
        _remove_pid()


def _run_background(args: argparse.Namespace) -> None:
    """Spawn the server as a detached background process."""
    _ensure_state_dir()

    cmd = [
        sys.executable,
        "-m",
        "onit_sandbox.cli",
        "start",
        "--foreground",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--transport",
        args.transport,
    ]
    if args.verbose:
        cmd.append("--verbose")
    for m in getattr(args, "mount", None) or []:
        cmd.extend(["--mount", m])

    log_fh = open(LOG_FILE, "a")
    process = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )

    time.sleep(1)

    if process.poll() is None:
        _write_pid(process.pid)
        url = build_server_url(args.host, args.port, args.transport)
        print(f"Sandbox MCP Server started on {url} (PID: {process.pid})")
        print(f"Logs: {LOG_FILE}")
    else:
        print("Failed to start server. Check logs at:", LOG_FILE)


def cmd_stop(_args: argparse.Namespace) -> None:
    """Stop the sandbox MCP server."""
    pid = _get_pid()
    if not pid:
        print("Server is not running (no PID file)")
        return

    if not _is_process_running(pid):
        print("Server is not running (stale PID file)")
        _remove_pid()
        return

    print(f"Stopping server (PID: {pid})...")
    try:
        os.kill(pid, signal.SIGTERM)
        for _ in range(10):
            if not _is_process_running(pid):
                break
            time.sleep(0.5)
        else:
            os.kill(pid, signal.SIGKILL)

        print("Server stopped")
    except OSError as e:
        print(f"Error stopping server: {e}")
    finally:
        _remove_pid()
        # Cleanup all containers
        from onit_sandbox.mcp_server import cleanup_all_sandboxes

        cleanup_all_sandboxes()


def cmd_status(_args: argparse.Namespace) -> None:
    """Check server status."""
    pid = _get_pid()
    if pid and _is_process_running(pid):
        print(f"Server is running (PID: {pid})")
    else:
        print("Server is not running")
        if pid:
            _remove_pid()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="onit-sandbox",
        description="Docker-based code execution sandbox MCP server",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # start
    start_p = subparsers.add_parser("start", help="Start the server")
    start_p.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    start_p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind to")
    start_p.add_argument(
        "--transport",
        default="streamable-http",
        choices=["streamable-http", "sse", "stdio"],
        help="Transport type",
    )
    start_p.add_argument(
        "--foreground", "-f", action="store_true", help="Run in foreground (don't daemonize)"
    )
    start_p.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    start_p.add_argument(
        "--data-path", default="/tmp/onit/data", help="Default data directory for file storage"
    )
    start_p.add_argument(
        "--mount",
        action="append",
        default=[],
        metavar="HOST:CONTAINER[:MODE]",
        help=(
            "Mount a host directory into the sandbox container. "
            "MODE is 'ro' (read-only, default) or 'rw'. "
            "Can be specified multiple times. "
            "Example: --mount /data:/data:ro --mount /models:/models:rw"
        ),
    )

    # stop
    subparsers.add_parser("stop", help="Stop the server")

    # status
    subparsers.add_parser("status", help="Check server status")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "stop":
        cmd_stop(args)
    elif args.command == "status":
        cmd_status(args)
    else:
        # Default to "start" when no subcommand is given
        if args.command is None:
            args = parser.parse_args(["start"])
        cmd_start(args)


if __name__ == "__main__":
    main()
