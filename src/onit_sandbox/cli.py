"""
CLI for the Sandbox MCP Server.

Provides start / stop / status / setup subcommands with PID-file management,
mirroring the pattern used in onit-workspace.
"""

from __future__ import annotations

import argparse
import getpass
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from onit_sandbox.server import DEFAULT_HOST, DEFAULT_PORT, build_server_url

# State directory — mirrors ~/.onit-workspace/ convention
STATE_DIR = Path.home() / ".onit-sandbox"
PID_FILE = STATE_DIR / "server.pid"
LOG_FILE = STATE_DIR / "server.log"
GITHUB_TOKEN_FILE = STATE_DIR / "github_token"


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
        opts: dict[str, Any] = {
            "data_path": getattr(args, "data_path", "/tmp/onit/data"),
            "verbose": args.verbose,
            "data_mounts": mounts,
        }
        if getattr(args, "gpu", None) is not None:
            opts["gpu_devices"] = args.gpu
        mcp_run(
            transport=args.transport,
            host=args.host,
            port=args.port,
            options=opts,
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
    if getattr(args, "gpu", None) is not None:
        cmd.extend(["--gpu", args.gpu])

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
# GitHub token management
# ---------------------------------------------------------------------------

_KEYRING_SERVICE = "onit-sandbox"
_KEYRING_USERNAME = "github-token"


def _keyring_available() -> bool:
    """Check if the keyring library is installed and has a usable backend."""
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring

        backend = keyring.get_keyring()
        # fail backend means no real keychain is available
        return not isinstance(backend, FailKeyring)
    except Exception:
        return False


def _save_token_to_keyring(token: str) -> bool:
    """Store token in the OS keychain. Returns True on success."""
    try:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, _KEYRING_USERNAME, token)
        return True
    except Exception:
        return False


def _load_token_from_keyring() -> str | None:
    """Load token from the OS keychain, or None."""
    try:
        import keyring

        token = keyring.get_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        return token if token else None
    except Exception:
        return None


def _delete_token_from_keyring() -> bool:
    """Delete token from the OS keychain. Returns True on success."""
    try:
        import keyring

        keyring.delete_password(_KEYRING_SERVICE, _KEYRING_USERNAME)
        return True
    except Exception:
        return False


def _save_token_to_file(token: str) -> None:
    """Store token in a file with restricted permissions (fallback)."""
    _ensure_state_dir()
    GITHUB_TOKEN_FILE.write_text(token)
    GITHUB_TOKEN_FILE.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0o600


def _load_token_from_file() -> str | None:
    """Load token from the file-based store, or None."""
    if GITHUB_TOKEN_FILE.exists():
        try:
            token = GITHUB_TOKEN_FILE.read_text().strip()
            return token if token else None
        except OSError:
            return None
    return None


def _delete_token_from_file() -> bool:
    """Delete token file. Returns True if it existed."""
    if GITHUB_TOKEN_FILE.exists():
        GITHUB_TOKEN_FILE.unlink()
        return True
    return False


def load_github_token() -> str | None:
    """Load the stored GitHub token.

    Checks the OS keychain first, then falls back to the file-based store.
    """
    if _keyring_available():
        token = _load_token_from_keyring()
        if token:
            return token
    return _load_token_from_file()


def _save_github_token(token: str) -> str:
    """Save token to the best available backend.

    Returns a human-readable description of where it was stored.
    """
    if _keyring_available():
        if _save_token_to_keyring(token):
            # Also remove any leftover plaintext file
            _delete_token_from_file()
            return "OS keychain"
    # Fallback to file
    _save_token_to_file(token)
    return f"{GITHUB_TOKEN_FILE} (mode 600)"


def _delete_github_token() -> bool:
    """Delete token from all backends. Returns True if anything was deleted."""
    deleted = False
    if _keyring_available():
        deleted = _delete_token_from_keyring() or deleted
    deleted = _delete_token_from_file() or deleted
    return deleted


def _token_storage_location() -> str | None:
    """Return a description of where the token is currently stored, or None."""
    if _keyring_available():
        if _load_token_from_keyring():
            return "OS keychain"
    if _load_token_from_file():
        return f"file ({GITHUB_TOKEN_FILE})"
    return None


def cmd_setup(args: argparse.Namespace) -> None:
    """Interactive setup for GitHub authentication."""
    _ensure_state_dir()

    action = getattr(args, "setup_action", None)

    if action == "remove":
        if _delete_github_token():
            print("GitHub token removed.")
        else:
            print("No GitHub token configured.")
        return

    if action == "status":
        token = load_github_token()
        if token:
            masked = token[:4] + "..." + token[-4:] if len(token) > 8 else "****"
            location = _token_storage_location()
            print(f"GitHub token is configured: {masked}")
            print(f"Storage: {location}")
            if not _keyring_available():
                print(
                    "Tip: Install 'keyring' package for OS keychain storage "
                    "(pip install keyring)"
                )
        else:
            print("No GitHub token configured. Run: onit-sandbox setup")
        return

    # Default: configure token
    print("GitHub Authentication Setup")
    print("=" * 40)
    print()
    print("This stores a GitHub Personal Access Token (PAT) for git")
    print("operations inside the sandbox (clone, push, pull, etc.).")
    print()
    if _keyring_available():
        print("Storage: OS keychain (secure)")
    else:
        print(
            "Storage: encrypted file (~/.onit-sandbox/github_token)\n"
            "Tip: Install 'keyring' for OS keychain support (pip install keyring)"
        )
    print()
    print("Create a token at: https://github.com/settings/tokens")
    print("Required scopes: repo (for private repos), or public_repo")
    print()

    # Check for existing token
    existing = load_github_token()
    if existing:
        masked = existing[:4] + "..." + existing[-4:] if len(existing) > 8 else "****"
        print(f"Current token: {masked}")
        confirm = input("Replace existing token? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Keeping existing token.")
            return

    # Read token (hidden input)
    token = getpass.getpass("GitHub token (input hidden): ").strip()
    if not token:
        print("No token provided. Aborting.")
        return

    # Validate token format (basic check)
    if not (
        token.startswith("ghp_")
        or token.startswith("gho_")
        or token.startswith("github_pat_")
        or token.startswith("ghs_")
    ):
        print(
            "Warning: Token doesn't look like a GitHub token "
            "(expected ghp_*, gho_*, github_pat_*, or ghs_* prefix)."
        )
        confirm = input("Save anyway? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborting.")
            return

    location = _save_github_token(token)

    print()
    print(f"Token saved to {location}")
    print("Git operations in the sandbox will now use this token for authentication.")


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
    start_p.add_argument(
        "--gpu",
        default=None,
        metavar="DEVICE",
        help=(
            "GPU device(s) to expose to containers. "
            "Examples: '0', '1', '2', '0,1', 'all' (default). "
            "Overrides CUDA_VISIBLE_DEVICES and SANDBOX_GPU_DEVICES env vars."
        ),
    )

    # stop
    subparsers.add_parser("stop", help="Stop the server")

    # status
    subparsers.add_parser("status", help="Check server status")

    # setup
    setup_p = subparsers.add_parser(
        "setup", help="Configure GitHub authentication for git operations"
    )
    setup_p.add_argument(
        "setup_action",
        nargs="?",
        default=None,
        choices=["remove", "status"],
        help="Optional action: 'remove' to delete token, 'status' to check config",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "stop":
        cmd_stop(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "setup":
        cmd_setup(args)
    else:
        # Default to "start" when no subcommand is given
        if args.command is None:
            args = parser.parse_args(["start"])
        cmd_start(args)


if __name__ == "__main__":
    main()
