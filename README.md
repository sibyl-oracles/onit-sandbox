# onit-sandbox

A Docker-based code execution sandbox MCP server for safe Python project execution.

## Overview

The Sandbox MCP Server provides isolated Docker containers for executing Python code with full package installation capabilities. Unlike traditional bash MCP servers that block package managers and restrict operations, this sandbox allows agents to:

- **Install packages**: `pip install numpy matplotlib scipy` works seamlessly
- **Run full projects**: Execute Python scripts, run tests, build simulations
- **Stay secure**: All execution is isolated in Docker containers with resource limits

## Architecture

```
┌──────────────────────────────────────┐
│  OnIt Agent (chat loop)              │
│  ┌────────────────────────────────┐  │
│  │ ToolsMCPServer (existing)      │  │
│  │  bash, read_file, write_file…  │  │
│  └────────────────────────────────┘  │
│  ┌────────────────────────────────┐  │
│  │ SandboxMCPServer (NEW)         │  │
│  │  sandbox_install               │  │
│  │  sandbox_run                   │  │
│  │  sandbox_status                │  │
│  │  sandbox_cleanup               │  │
│  └──────────┬─────────────────────┘  │
│             │ Docker SDK              │
│  ┌──────────▼─────────────────────┐  │
│  │ Docker Container (per-session) │  │
│  │  - python:3.12-slim image      │  │
│  │  - Full pip access             │  │
│  │  - Workspace bind-mounted      │  │
│  │  - CPU/mem/network limits      │  │
│  │  - auto-cleanup on session end │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## Installation

### Prerequisites

- Python 3.10+
- Docker (install from [docker.com](https://docs.docker.com/get-docker/))

### Quick Start

```bash
# Clone the repository
git clone https://github.com/sibyl-oracles/onit-sandbox.git
cd onit-sandbox

# Run setup (creates venv, installs deps, builds Docker image)
chmod +x setup.sh
./setup.sh

# Activate virtual environment
source .venv/bin/activate

# Start the server (runs in background)
onit-sandbox start

# Or run in foreground
onit-sandbox start --foreground

# Stop the server
onit-sandbox stop

# Check status
onit-sandbox status
```

### Manual Installation

```bash
# Install Python dependencies
pip install -e .

# Build the Docker image
cd docker
chmod +x build.sh
./build.sh
```

## Tools

### `sandbox_install`

Install Python packages inside the sandbox container.

```json
{
  "packages": "numpy matplotlib scipy",
  "session_id": "my-session",
  "data_path": "/path/to/workspace"
}
```

**Features:**
- Temporarily enables network for PyPI access
- Packages persist for the session lifetime
- Network disabled after installation for security

### `sandbox_run`

Execute commands inside the sandbox container.

```json
{
  "command": "python simulation.py",
  "session_id": "my-session",
  "data_path": "/path/to/workspace",
  "timeout": 60
}
```

**Features:**
- Workspace directory mounted at `/workspace`
- Files written by agents are immediately accessible
- Output captured and returned (truncated to 10KB)

### `sandbox_status`

Check sandbox state and resource usage.

```json
{
  "session_id": "my-session"
}
```

**Returns:**
- Container state (running, stopped)
- Installed packages list
- Resource usage (CPU, memory)
- Creation timestamp

### `sandbox_cleanup`

Stop and remove the sandbox container.

```json
{
  "session_id": "my-session"
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_IMAGE` | `onit-sandbox:latest` | Docker image to use |
| `FALLBACK_IMAGE` | `python:3.12-slim` | Fallback if sandbox image unavailable |
| `SANDBOX_MEMORY_LIMIT` | `512m` | Container memory limit |
| `SANDBOX_CPU_QUOTA` | `100000` | CPU quota (100000 = 1 CPU) |
| `SANDBOX_PIDS_LIMIT` | `100` | Max processes in container |
| `SANDBOX_DEFAULT_TIMEOUT` | `60` | Default command timeout (seconds) |
| `SANDBOX_INSTALL_TIMEOUT` | `300` | Package install timeout (seconds) |

### Server Configuration

```yaml
# configs/default.yaml
server:
  name: SandboxMCPServer
  host: 0.0.0.0
  port: 18205
  transport: sse

docker:
  sandbox_image: onit-sandbox:latest
  memory_limit: 512m
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

### Session Cleanup

Call `cleanup_sandbox(session_id)` during session teardown:

```python
from onit_sandbox.mcp_server import cleanup_sandbox

# In session cleanup code
cleanup_sandbox(session_id)
```

## Security

### Isolation Features

- **Container isolation**: Full filesystem, process, and network isolation
- **Non-root execution**: All code runs as user `sandbox` (UID 1000)
- **Resource limits**: Memory (512MB), CPU (1 core), PIDs (100)
- **Network disabled by default**: Only enabled during `sandbox_install`
- **Auto-cleanup**: Containers removed when stopped (`--rm` flag)

### What's Blocked

- Direct host filesystem access (only mounted workspace)
- Root access inside container
- Network access during code execution
- Unlimited resource consumption

### What's Allowed

- Full PyPI access during `sandbox_install`
- Read/write to mounted workspace
- Python execution with any installed packages
- Shell commands within resource limits

## Pre-installed Packages

The `onit-sandbox:latest` image includes:

- numpy
- matplotlib
- scipy
- pandas
- scikit-learn
- sympy
- pytest

## Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests (skip Docker-dependent tests if Docker unavailable)
pytest tests/ -v -k "not Docker"
```

### Building the Docker Image

```bash
cd docker
./build.sh
```

### Code Style

```bash
# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Type check
mypy src/
```

## Troubleshooting

### Docker not available

```
Error: Docker is not available. Please install Docker and ensure it is running.
```

**Solution:** Install Docker and start the Docker daemon:
- macOS: Install Docker Desktop
- Linux: `sudo systemctl start docker`

### Sandbox image not found

```
Warning: Using fallback image python:3.12-slim
```

**Solution:** Build the optimized image:
```bash
cd docker && ./build.sh
```

### Permission denied on workspace

**Solution:** Ensure the workspace directory is writable:
```bash
chmod -R 777 /path/to/workspace
```

### Container timeout

**Solution:** Increase timeout or optimize your code:
```python
sandbox_run("python slow_script.py", timeout=300)
```

## License

Apache License 2.0 - See [LICENSE](LICENSE) for details.
