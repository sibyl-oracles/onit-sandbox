"""
CLI for the Sandbox MCP Server.

Provides start / stop / status / setup subcommands with PID-file management,
mirroring the pattern used in onit-workspace.
"""

# WARNING: This file has high cyclomatic complexity (CC > 20)
# This indicates complex conditional logic that may benefit from refactoring.
# See analyze_complexity.py for details.

from __future__ import annotations

import argparse
import getpass
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from onit_sandbox.server import DEFAULT_HOST, DEFAULT_PORT, build_server_url

# State directory — mirrors ~/.onit-workspace/ convention
STATE_DIR = Path.home() / ".onit-sandbox"
GITHUB_TOKEN_FILE = STATE_DIR / "github_token"
HF_TOKEN_FILE = STATE_DIR / "hf_token"


def _pid_file(port: int) -> Path:
    return STATE_DIR / f"server-{port}.pid"


def _log_file(port: int) -> Path:
    return STATE_DIR / f"server-{port}.log"


def _ensure_state_dir() -> None:
    STATE_DIR.mkdir(mode=0o700, exist_ok=True)


def _get_pid(port: int) -> int | None:
    """Read stored PID for the given port, or None."""
    pid_file = _pid_file(port)
    if pid_file.exists():
        try:
            return int(pid_file.read_text().strip())
        except (ValueError, OSError):
            pass
    return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_pid(pid: int, port: int) -> None:
    _ensure_state_dir()
    _pid_file(port).write_text(str(pid))


def _remove_pid(port: int) -> None:
    _pid_file(port).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------


def cmd_start(args: argparse.Namespace) -> None:
    """Start the sandbox MCP server."""
    pid = _get_pid(args.port)
    if pid and _is_process_running(pid):
        print(f"Server is already running on port {args.port} (PID: {pid})")
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

    _write_pid(os.getpid(), args.port)
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
        _remove_pid(args.port)


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

    log_fh = open(_log_file(args.port), "a")
    process = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )

    time.sleep(1)

    if process.poll() is None:
        _write_pid(process.pid, args.port)
        url = build_server_url(args.host, args.port, args.transport)
        print(f"Sandbox MCP Server started on {url} (PID: {process.pid})")
        print(f"Logs: {_log_file(args.port)}")
    else:
        print("Failed to start server. Check logs at:", _log_file(args.port))


def cmd_stop(args: argparse.Namespace) -> None:
    """Stop the sandbox MCP server."""
    pid = _get_pid(args.port)
    if not pid:
        print(f"Server is not running on port {args.port} (no PID file)")
        return

    if not _is_process_running(pid):
        print(f"Server is not running on port {args.port} (stale PID file)")
        _remove_pid(args.port)
        return

    print(f"Stopping server on port {args.port} (PID: {pid})...")
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
        _remove_pid(args.port)
        # Cleanup all containers
        from onit_sandbox.mcp_server import cleanup_all_sandboxes

        cleanup_all_sandboxes()


def cmd_status(args: argparse.Namespace) -> None:
    """Check server status."""
    pid = _get_pid(args.port)
    if pid and _is_process_running(pid):
        print(f"Server is running on port {args.port} (PID: {pid})")
    else:
        print(f"Server is not running on port {args.port}")
        if pid:
            _remove_pid(args.port)


# ---------------------------------------------------------------------------
# Keychain-based token management
# ---------------------------------------------------------------------------

_KEYRING_SERVICE = "onit-sandbox"
_KEYRING_GITHUB = "github-token"
_KEYRING_HF = "hf-token"


def _keyring_available() -> bool:
    """Check if the keyring library is installed and has a usable backend."""
    try:
        import keyring
        from keyring.backends.fail import Keyring as FailKeyring

        backend = keyring.get_keyring()
        return not isinstance(backend, FailKeyring)
    except Exception:
        return False


def _require_keyring() -> None:
    """Raise if keyring is not available."""
    if not _keyring_available():
        print(
            "Error: OS keychain is required but not available.\n"
            "Install the 'keyring' package:  pip install keyring\n"
            "On headless Linux you may also need a backend like "
            "'keyrings.alt' or 'SecretStorage'."
        )
        sys.exit(1)


def _save_to_keyring(username: str, token: str) -> bool:
    """Store a token in the OS keychain. Returns True on success."""
    try:
        import keyring

        keyring.set_password(_KEYRING_SERVICE, username, token)
        return True
    except Exception:
        return False


def _load_from_keyring(username: str) -> str | None:
    """Load a token from the OS keychain, or None."""
    try:
        import keyring

        token = keyring.get_password(_KEYRING_SERVICE, username)
        return token if token else None
    except Exception:
        return None


def _delete_from_keyring(username: str) -> bool:
    """Delete a token from the OS keychain. Returns True on success."""
    try:
        import keyring

        keyring.delete_password(_KEYRING_SERVICE, username)
        return True
    except Exception:
        return False


def _load_from_file(path: Path) -> str | None:
    """Load a token from a legacy file-based store, or None."""
    if path.exists():
        try:
            token = path.read_text().strip()
            return token if token else None
        except OSError:
            return None
    return None


def _delete_file(path: Path) -> bool:
    """Delete a legacy token file. Returns True if it existed."""
    if path.exists():
        path.unlink()
        return True
    return False


def _migrate_file_to_keyring(path: Path, username: str) -> None:
    """If a token exists in a legacy file, migrate it to keyring and delete the file."""
    token = _load_from_file(path)
    if token and _keyring_available():
        if _save_to_keyring(username, token):
            _delete_file(path)


# --- GitHub token helpers ---------------------------------------------------


def load_github_token() -> str | None:
    """Load the stored GitHub token from the OS keychain.

    Auto-migrates any legacy file-based token into keychain.
    """
    _migrate_file_to_keyring(GITHUB_TOKEN_FILE, _KEYRING_GITHUB)
    return _load_from_keyring(_KEYRING_GITHUB)


def _save_github_token(token: str) -> None:
    """Save GitHub token to the OS keychain."""
    _require_keyring()
    if not _save_to_keyring(_KEYRING_GITHUB, token):
        print("Error: Failed to save GitHub token to OS keychain.")
        sys.exit(1)
    _delete_file(GITHUB_TOKEN_FILE)  # clean up any legacy file


def _delete_github_token() -> bool:
    """Delete GitHub token from keychain (and any legacy file)."""
    deleted = _delete_from_keyring(_KEYRING_GITHUB)
    deleted = _delete_file(GITHUB_TOKEN_FILE) or deleted
    return deleted


# --- HuggingFace token helpers -----------------------------------------------


def load_hf_token() -> str | None:
    """Load the stored HuggingFace token from the OS keychain.

    Auto-migrates any legacy file-based token into keychain.
    """
    _migrate_file_to_keyring(HF_TOKEN_FILE, _KEYRING_HF)
    return _load_from_keyring(_KEYRING_HF)


def _save_hf_token(token: str) -> None:
    """Save HuggingFace token to the OS keychain."""
    _require_keyring()
    if not _save_to_keyring(_KEYRING_HF, token):
        print("Error: Failed to save HuggingFace token to OS keychain.")
        sys.exit(1)
    _delete_file(HF_TOKEN_FILE)  # clean up any legacy file


def _delete_hf_token() -> bool:
    """Delete HuggingFace token from keychain (and any legacy file)."""
    deleted = _delete_from_keyring(_KEYRING_HF)
    deleted = _delete_file(HF_TOKEN_FILE) or deleted
    return deleted


def _cmd_setup_show() -> None:
    """Display all current setup configuration."""
    from onit_sandbox.server import (
        DEFAULT_CPU_QUOTA,
        DEFAULT_DATA_MOUNTS,
        DEFAULT_GPU_DEVICES,
        DEFAULT_MEMORY_LIMIT,
        DEFAULT_PIP_CACHE_PATH,
        DEFAULT_PIDS_LIMIT,
        DEFAULT_SHM_SIZE,
        DEFAULT_TIMEOUT,
        FALLBACK_IMAGE,
        INSTALL_TIMEOUT,
        MAX_OUTPUT_BYTES,
        MAX_TIMEOUT,
        SANDBOX_IMAGE,
    )

    # --- Tokens --------------------------------------------------------------
    print("Tokens")
    print("=" * 40)

    keyring_ok = _keyring_available()
    if not keyring_ok:
        print("  (!) OS keychain is not available")

    gh_token = load_github_token()
    if gh_token:
        masked = gh_token[:4] + "..." + gh_token[-4:] if len(gh_token) > 8 else "****"
        print(f"  GitHub:       {masked}  (keychain)")
    else:
        print("  GitHub:       not configured")

    hf_token = load_hf_token()
    if hf_token:
        masked = hf_token[:4] + "..." + hf_token[-4:] if len(hf_token) > 8 else "****"
        print(f"  HuggingFace:  {masked}  (keychain)")
    else:
        print("  HuggingFace:  not configured")

    # --- Docker / sandbox ----------------------------------------------------
    print()
    print("Docker / Sandbox")
    print("=" * 40)
    print(f"  Image:            {SANDBOX_IMAGE}")
    print(f"  Fallback image:   {FALLBACK_IMAGE}")
    print(f"  Memory limit:     {DEFAULT_MEMORY_LIMIT}")
    cpu = "no limit" if DEFAULT_CPU_QUOTA == 0 else str(DEFAULT_CPU_QUOTA)
    print(f"  CPU quota:        {cpu}")
    print(f"  PIDs limit:       {DEFAULT_PIDS_LIMIT}")
    print(f"  Shared memory:    {DEFAULT_SHM_SIZE}")
    print(f"  GPU devices:      {DEFAULT_GPU_DEVICES}")

    # --- Timeouts ------------------------------------------------------------
    print()
    print("Timeouts")
    print("=" * 40)
    print(f"  Default:          {DEFAULT_TIMEOUT}s")
    print(f"  Max:              {MAX_TIMEOUT}s")
    print(f"  Install:          {INSTALL_TIMEOUT}s")

    # --- Paths / mounts ------------------------------------------------------
    print()
    print("Paths / Mounts")
    print("=" * 40)
    print(f"  Pip cache:        {DEFAULT_PIP_CACHE_PATH}")
    print(f"  Max output:       {MAX_OUTPUT_BYTES} bytes")
    if DEFAULT_DATA_MOUNTS:
        print(f"  Data mounts:      {DEFAULT_DATA_MOUNTS}")
    else:
        print("  Data mounts:      none")


def cmd_setup(args: argparse.Namespace) -> None:
    """Interactive setup for GitHub and HuggingFace authentication."""
    _ensure_state_dir()

    # --show flag takes priority over positional action
    if getattr(args, "show", False):
        _cmd_setup_show()
        return

    action = getattr(args, "setup_action", None)
    target = getattr(args, "setup_target", None) or "all"

    if action == "remove":
        if target in ("all", "github"):
            if _delete_github_token():
                print("GitHub token removed.")
            else:
                print("No GitHub token configured.")
        if target in ("all", "huggingface"):
            if _delete_hf_token():
                print("HuggingFace token removed.")
            else:
                print("No HuggingFace token configured.")
        return

    if action == "status":
        if not _keyring_available():
            print(
                "Warning: OS keychain is not available.\n"
                "Install 'keyring' package:  pip install keyring\n"
            )

        # GitHub status
        gh_token = load_github_token()
        if gh_token:
            masked = gh_token[:4] + "..." + gh_token[-4:] if len(gh_token) > 8 else "****"
            print(f"GitHub token is configured: {masked}")
            print("  Storage: OS keychain")
        else:
            print("No GitHub token configured.")

        # HuggingFace status
        hf_token = load_hf_token()
        if hf_token:
            masked = hf_token[:4] + "..." + hf_token[-4:] if len(hf_token) > 8 else "****"
            print(f"HuggingFace token is configured: {masked}")
            print("  Storage: OS keychain")
        else:
            print("No HuggingFace token configured.")
        return

    # Default: interactive configuration
    if target in ("all", "github"):
        _setup_github_token()

    if target in ("all", "huggingface"):
        if target == "all":
            print()  # separator between sections
        _setup_hf_token()


def _setup_github_token() -> None:
    """Interactive setup for GitHub token."""
    _require_keyring()

    print("GitHub Authentication Setup")
    print("=" * 40)
    print()
    print("This stores a GitHub Personal Access Token (PAT) in the OS")
    print("keychain for git operations inside the sandbox.")
    print()
    print("Create a token at: https://github.com/settings/tokens")
    print("Required scopes: repo (for private repos), or public_repo")
    print()

    existing = load_github_token()
    if existing:
        masked = existing[:4] + "..." + existing[-4:] if len(existing) > 8 else "****"
        print(f"Current token: {masked}")
        confirm = input("Replace existing token? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Keeping existing token.")
            return

    token = getpass.getpass("GitHub token (input hidden): ").strip()
    if not token:
        print("No token provided. Skipping GitHub setup.")
        return

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
            print("Aborting GitHub token setup.")
            return

    _save_github_token(token)
    print("GitHub token saved to OS keychain.")
    print("Git operations in the sandbox will now use this token for authentication.")


def _setup_hf_token() -> None:
    """Interactive setup for HuggingFace token."""
    _require_keyring()

    print("HuggingFace Authentication Setup")
    print("=" * 40)
    print()
    print("This stores a HuggingFace API token in the OS keychain for")
    print("accessing models, datasets, and other resources inside the sandbox.")
    print()
    print("Create a token at: https://huggingface.co/settings/tokens")
    print()

    existing = load_hf_token()
    if existing:
        masked = existing[:4] + "..." + existing[-4:] if len(existing) > 8 else "****"
        print(f"Current token: {masked}")
        confirm = input("Replace existing token? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Keeping existing token.")
            return

    token = getpass.getpass("HuggingFace token (input hidden): ").strip()
    if not token:
        print("No token provided. Skipping HuggingFace setup.")
        return

    if not token.startswith("hf_"):
        print(
            "Warning: Token doesn't look like a HuggingFace token "
            "(expected hf_* prefix)."
        )
        confirm = input("Save anyway? [y/N]: ").strip().lower()
        if confirm not in ("y", "yes"):
            print("Aborting HuggingFace token setup.")
            return

    _save_hf_token(token)
    print("HuggingFace token saved to OS keychain.")
    print("HuggingFace operations in the sandbox will now use this token.")


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
            "MODE is 'rw' (read-write, default) or 'ro' (read-only). "
            "Can be specified multiple times. "
            "Example: --mount /data:/data --mount /reference:/reference:ro"
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
    stop_p = subparsers.add_parser("stop", help="Stop the server")
    stop_p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port of the server to stop")

    # status
    status_p = subparsers.add_parser("status", help="Check server status")
    status_p.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port of the server to check")

    # setup
    setup_p = subparsers.add_parser(
        "setup", help="Configure authentication tokens (GitHub, HuggingFace)"
    )
    setup_p.add_argument(
        "setup_action",
        nargs="?",
        default=None,
        choices=["remove", "status"],
        help="Optional action: 'remove' to delete token(s), 'status' to check config",
    )
    setup_p.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display current setup configuration",
    )
    setup_p.add_argument(
        "--target",
        dest="setup_target",
        default="all",
        choices=["github", "huggingface", "all"],
        help="Which token to configure (default: all)",
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
