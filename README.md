# onit-sandbox

[![PyPI version](https://img.shields.io/pypi/v/onit-sandbox.svg)](https://pypi.org/project/onit-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/onit-sandbox.svg)](https://pypi.org/project/onit-sandbox/)
[![CI](https://github.com/sibyl-oracles/onit-sandbox/actions/workflows/ci.yml/badge.svg)](https://github.com/sibyl-oracles/onit-sandbox/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Docker-based code execution sandbox [MCP server](https://modelcontextprotocol.io/) for safe Python execution. Gives LLM agents an isolated environment where they can install packages, run code, and produce artifacts — without touching the host system.

## Why onit-sandbox?

Traditional bash-based MCP tools block package managers and restrict operations. **onit-sandbox** lifts those restrictions safely by running everything inside per-session Docker containers with resource limits and network isolation.

- **Install any package** — `pip install numpy matplotlib torch` just works
- **Run full projects** — execute scripts, run tests, build simulations
- **Stay secure** — containers are resource-limited, network-isolated, and non-root

## Architecture

```
┌──────────────────────────────────────┐
│  LLM Agent (chat loop)              │
│  ┌────────────────────────────────┐  │
│  │ SandboxMCPServer               │  │
│  │  install_packages              │  │
│  │  run_code                      │  │
│  │  sandbox_status                │  │
│  └──────────┬─────────────────────┘  │
│             │ Docker CLI              │
│  ┌──────────▼─────────────────────┐  │
│  │ Docker Container (per-session) │  │
│  │  - PyTorch + scientific stack  │  │
│  │  - Full pip access             │  │
│  │  - Workspace bind-mounted      │  │
│  │  - Shared pip cache volume     │  │
│  │  - CPU/mem/network limits      │  │
│  │  - Auto-cleanup on stop        │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Docker** — install from [docker.com](https://docs.docker.com/get-docker/) and make sure the daemon is running (`docker info`)

### Step 1 — Install the Python package

```bash
# From PyPI
pip install onit-sandbox

# Or from source
git clone https://github.com/sibyl-oracles/onit-sandbox.git
cd onit-sandbox
pip install -e .
```

### Step 2 — Build (or pull) the Docker image

The sandbox image comes pre-loaded with PyTorch (CUDA), torchvision, torchaudio, and the common scientific Python stack so agents can `import torch` without waiting for a pip install.

**Option A — Pull from GHCR:**

```bash
docker pull ghcr.io/sibyl-oracles/onit-sandbox:latest
export SANDBOX_IMAGE=ghcr.io/sibyl-oracles/onit-sandbox:latest
```

**Option B — Build locally:**

```bash
cd docker
chmod +x build.sh
./build.sh        # builds onit-sandbox:latest
```

Pre-installed packages:

| Package | Description |
|---------|-------------|
| torch | Deep learning (CUDA) |
| torchvision | Vision models and transforms |
| torchaudio | Audio processing |
| numpy | Numerical computing |
| matplotlib | Plotting and visualization |
| scipy | Scientific computing |
| pandas | Data manipulation |
| scikit-learn | Machine learning |
| sympy | Symbolic mathematics |
| pytest | Testing framework |

> **Fallback:** If no custom image is found, the server falls back to `python:3.12-slim` automatically. Packages can still be installed at runtime via `install_packages`.

### Step 3 — Start the server

```bash
# Start the MCP server (background)
onit-sandbox start

# Start in foreground (useful for debugging)
onit-sandbox start --foreground

# Check status
onit-sandbox status

# Stop the server
onit-sandbox stop
```

The server runs on `http://0.0.0.0:18205/sse` by default.

## MCP Tools

### `install_packages`

Install Python packages inside the sandbox via pip. Network is temporarily enabled for PyPI access, then disabled again.

```json
{
  "packages": "numpy matplotlib scipy",
  "session_id": "my-session"
}
```

### `run_code`

Execute shell commands inside the sandbox. Files written by `write_file` are available in the working directory. Output is captured and returned (truncated to 10 KB).

```json
{
  "command": "python simulation.py",
  "timeout": 120,
  "session_id": "my-session"
}
```

### `sandbox_status`

Check sandbox state — container health, Python version, installed packages, disk usage, uptime, and GPU availability.

```json
{
  "session_id": "my-session"
}
```

## Configuration

### CLI Options

```
onit-sandbox start [OPTIONS]

  --host         Host to bind to (default: 0.0.0.0)
  --port         Port to bind to (default: 18205)
  --transport    sse or stdio (default: sse)
  --foreground   Run in foreground
  --verbose      Enable verbose logging
  --data-path    Workspace data directory (default: /tmp/onit/data)
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_IMAGE` | `onit-sandbox:latest` | Docker image to use |
| `FALLBACK_IMAGE` | `python:3.12-slim` | Fallback if sandbox image unavailable |
| `SANDBOX_MEMORY_LIMIT` | `2g` | Container memory limit |
| `SANDBOX_CPU_QUOTA` | `100000` | CPU quota (100000 = 1 CPU) |
| `SANDBOX_PIDS_LIMIT` | `100` | Max processes in container |
| `SANDBOX_DEFAULT_TIMEOUT` | `60` | Default command timeout (seconds) |
| `SANDBOX_INSTALL_TIMEOUT` | `300` | Package install timeout (seconds) |
| `SANDBOX_PIP_CACHE_PATH` | `/tmp/onit/pip-cache` | Shared pip cache across sessions |

### YAML Configuration

```yaml
# configs/default.yaml
server:
  name: SandboxMCPServer
  host: 0.0.0.0
  port: 18205
  transport: sse

docker:
  sandbox_image: onit-sandbox:latest
  memory_limit: 2g
  cpu_quota: 100000
  pids_limit: 100
  network_disabled_default: true
```

## Integration with OnIt

Add to your OnIt `configs/default.yaml`:

```yaml
mcp_servers:
  - name: SandboxMCPServer
    description: Python code execution sandbox with package installation
    url: http://127.0.0.1:18205/sse
    enabled: true
```

### Programmatic Usage

```python
from onit_sandbox.mcp_server import run, cleanup_sandbox

# Start the server
run(transport="sse", host="0.0.0.0", port=18205)

# Cleanup a session's container
cleanup_sandbox(session_id="my-session")
```

## Security

### Isolation Features

| Layer | Protection |
|-------|-----------|
| Filesystem | Container isolation — only mounted workspace is accessible |
| User | Non-root execution as `sandbox` (UID 1000) |
| Resources | Memory (2 GB), CPU (1 core), PIDs (100) |
| Network | Disabled by default — enabled only during `install_packages` |
| Lifecycle | Containers auto-removed on stop (`--rm` flag) |
| GPU | Optional NVIDIA GPU passthrough when available |

### What's Blocked

- Direct host filesystem access (only mounted workspace)
- Root access inside container
- Network access during code execution
- Unlimited resource consumption

### What's Allowed

- Full PyPI access during `install_packages`
- Read/write to mounted workspace
- Python execution with any installed packages
- Shell commands within resource limits

## Development

### Setup

```bash
# Clone and install with dev dependencies
git clone https://github.com/sibyl-oracles/onit-sandbox.git
cd onit-sandbox
pip install -e ".[dev]"

# Or use the setup script (creates venv, builds Docker image)
chmod +x setup.sh
./setup.sh
```

### Running Tests

```bash
# All tests
pytest tests/ -v

# Skip Docker-dependent tests
pytest tests/ -v -k "not Docker"
```

### Code Quality

```bash
black src/ tests/        # Format
ruff check src/ tests/   # Lint
mypy src/                # Type check
```

### Publishing a Release

1. Update the version in [pyproject.toml](pyproject.toml)
2. Create a GitHub release with a tag matching the version (e.g., `v0.1.0`)
3. The [publish workflow](.github/workflows/publish.yml) automatically builds and uploads to PyPI
4. The [docker workflow](.github/workflows/docker.yml) automatically builds and pushes the container image to GHCR

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Docker not available | Install Docker and start the daemon (`sudo systemctl start docker` on Linux, or open Docker Desktop on macOS) |
| Sandbox image not found | Build locally with `cd docker && ./build.sh` or pull from GHCR |
| Permission denied on workspace | `chmod -R 777 /path/to/workspace` |
| Command timeout | Increase timeout: `run_code(command="...", timeout=300)` |
| Port already in use | Use `--port` flag: `onit-sandbox start --port 18206` |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
