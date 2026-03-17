# onit-sandbox

[![PyPI version](https://img.shields.io/pypi/v/onit-sandbox.svg)](https://pypi.org/project/onit-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/onit-sandbox.svg)](https://pypi.org/project/onit-sandbox/)
[![CI](https://github.com/sibyl-oracles/onit-sandbox/actions/workflows/ci.yml/badge.svg)](https://github.com/sibyl-oracles/onit-sandbox/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Docker-based code execution sandbox [MCP server](https://modelcontextprotocol.io/) for safe Python execution. Gives LLM agents an isolated environment where they can install packages, run code, and produce artifacts — without touching the host system. Designed for AI model training, evaluation, and data pipelines.

## Why onit-sandbox?

Traditional bash-based MCP tools block package managers and restrict operations. **onit-sandbox** lifts those restrictions safely by running everything inside per-session Docker containers with resource limits and network isolation.

- **Install any package** — `pip install numpy matplotlib torch` just works
- **Run full projects** — execute scripts, run tests, build simulations
- **Mount data volumes** — bind host directories (datasets, models) into the sandbox
- **Data pipelines** — upload files, run training, read metrics, download artifacts
- **Stay secure** — containers are resource-limited, network-isolated, and non-root

## Architecture

```
┌──────────────────────────────────────┐
│  LLM Agent (chat loop)              │
│  ┌────────────────────────────────┐  │
│  │ SandboxMCPServer               │  │
│  │  install_packages              │  │
│  │  run_code                      │  │
│  │  write_file / read_file        │  │
│  │  upload_file / download_file   │  │
│  │  list_files / sandbox_status   │  │
│  │  enable_network / disable_net  │  │
│  └──────────┬─────────────────────┘  │
│             │ Docker CLI              │
│  ┌──────────▼─────────────────────┐  │
│  │ Docker Container (per-session) │  │
│  │  - PyTorch + scientific stack  │  │
│  │  - ML/data pipeline packages   │  │
│  │  - Full pip access             │  │
│  │  - /workspace bind-mounted     │  │
│  │  - /data mount (optional)      │  │
│  │  - Shared pip cache volume     │  │
│  │  - CPU/mem/network limits      │  │
│  │  - GPU passthrough (optional)  │  │
│  │  - Auto-cleanup on stop        │  │
│  └────────────────────────────────┘  │
└──────────────────────────────────────┘
```

## Getting Started

### Prerequisites

- **Python 3.10+**
- **Docker** — install from [docker.com](https://docs.docker.com/get-docker/) and make sure the daemon is running (`docker info`)
- **NVIDIA Container Toolkit** *(optional, for GPU support)* — required to pass GPUs into containers. Without it, `gpu_available` will be `false` even if the host has a GPU. Install with:

  ```bash
  # Add the NVIDIA container toolkit repo
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

  # Install and configure
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```

  Verify with: `docker info -f '{{json .Runtimes}}'` — the output should contain `"nvidia"`.

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

The sandbox image comes pre-loaded with PyTorch (CUDA), the common scientific Python stack, and ML/data pipeline packages so agents can start training immediately.

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
| datasets | Hugging Face datasets |
| transformers | Hugging Face transformers |
| safetensors | Safe model serialization |
| tensorboard | Training visualization |
| tqdm | Progress bars |
| pyyaml | YAML parsing |
| jsonlines | JSONL file handling |
| pillow | Image processing |

> **Fallback:** If no custom image is found, the server falls back to `python:3.12-slim` automatically. Packages can still be installed at runtime via `install_packages`.

### Step 3 — Start the server

```bash
# Start the MCP server (background)
onit-sandbox start

# Start in foreground (useful for debugging)
onit-sandbox start --foreground

# Start with data volume mounts
onit-sandbox start --mount /data:/data:ro --mount /models:/models:rw

# Check status
onit-sandbox status

# Stop the server
onit-sandbox stop
```

The server runs on `http://0.0.0.0:18205/mcp` by default (streamable-http transport).

## MCP Tools

### `sandbox_install_packages`

Install Python packages inside the sandbox via pip. Network is temporarily enabled for PyPI access, then disabled again.

```json
{
  "packages": "numpy matplotlib scipy",
  "session_id": "my-session"
}
```

### `sandbox_run_code`

Execute shell commands inside the sandbox. Output is captured and returned (configurable limit, default 50 KB). Automatically detects newly created files.

```json
{
  "command": "python train.py --epochs 10",
  "timeout": 3600,
  "network": false,
  "session_id": "my-session"
}
```

### `sandbox_write_file`

Write inline content to a file inside the sandbox. Parent directories are created automatically.

```json
{
  "file_path": "train.py",
  "content": "import torch\n...",
  "session_id": "my-session"
}
```

### `sandbox_read_file`

Read file contents from inside the sandbox without copying to the host. Supports offset and max_bytes for tailing large log files.

```json
{
  "file_path": "output/metrics.json",
  "max_bytes": 100000,
  "offset": 0,
  "session_id": "my-session"
}
```

### `sandbox_upload_file`

Copy a file or directory from the host filesystem into the sandbox. Ideal for large datasets, model weights, or configuration archives.

```json
{
  "host_path": "/data/train.csv",
  "sandbox_path": "data/train.csv",
  "session_id": "my-session"
}
```

### `sandbox_download_file`

Copy a file from the sandbox to the host filesystem. Use for retrieving plots, CSVs, model checkpoints, or any output artifacts.

```json
{
  "sandbox_path": "checkpoints/model_best.pt",
  "dest_path": "/home/user/results/model_best.pt",
  "session_id": "my-session"
}
```

### `sandbox_list_files`

List files and directories inside the sandbox. Supports recursive depth control.

```json
{
  "path": ".",
  "max_depth": 3,
  "session_id": "my-session"
}
```

### `sandbox_get_status`

Check sandbox state — container health, Python version, installed packages, disk usage, uptime, GPU availability, network state, and configured data mounts.

```json
{
  "session_id": "my-session"
}
```

### `sandbox_enable_network` / `sandbox_disable_network`

Toggle persistent internet access. Use `enable_network` before downloading datasets or calling APIs, then `disable_network` to restore isolation.

## Data Mounts

Mount host directories into the sandbox for direct access to datasets, model weights, or shared storage. This avoids copying large files through `upload_file`.

### Via CLI

```bash
# Read-only data, read-write checkpoints
onit-sandbox start \
  --mount /data/datasets:/data:ro \
  --mount /data/checkpoints:/checkpoints:rw

# Multiple mounts
onit-sandbox start \
  --mount /nas/imagenet:/data/imagenet:ro \
  --mount /scratch/outputs:/outputs:rw
```

### Via Environment Variable

```bash
# Comma-separated "host:container:mode" entries
export SANDBOX_DATA_MOUNTS="/data:/data:ro,/models:/models:rw"
onit-sandbox start
```

### Mount Modes

| Mode | Description |
|------|-------------|
| `ro` | Read-only (default) — sandbox can read but not modify host data |
| `rw` | Read-write — sandbox can read and write to the host directory |

### Example: Training Pipeline

```bash
# Start sandbox with dataset and output mounts
onit-sandbox start \
  --mount /data/imagenet:/data:ro \
  --mount /results:/results:rw
```

The AI agent can then:

1. List available data: `sandbox_list_files(path="/data")`
2. Write training code: `sandbox_write_file(file_path="train.py", content="...")`
3. Run training: `sandbox_run_code(command="python train.py", timeout=3600)`
4. Monitor progress: `sandbox_read_file(file_path="train.log")`
5. Results are written directly to `/results` on the host via the `rw` mount

## Configuration

### CLI Options

```
onit-sandbox start [OPTIONS]

  --host         Host to bind to (default: 0.0.0.0)
  --port         Port to bind to (default: 18205)
  --transport    streamable-http, sse, or stdio (default: streamable-http)
  --foreground   Run in foreground
  --verbose      Enable verbose logging
  --data-path    Workspace data directory (default: /tmp/onit/data)
  --mount        Mount host directory into sandbox (repeatable)
                 Format: HOST_PATH:CONTAINER_PATH[:MODE]
                 MODE is "ro" (default) or "rw"
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_IMAGE` | `onit-sandbox:latest` | Docker image to use |
| `FALLBACK_IMAGE` | `python:3.12-slim` | Fallback if sandbox image unavailable |
| `SANDBOX_MEMORY_LIMIT` | `2g` | Container memory limit |
| `SANDBOX_CPU_QUOTA` | `100000` | CPU quota (100000 = 1 CPU) |
| `SANDBOX_PIDS_LIMIT` | `256` | Max processes in container |
| `SANDBOX_DEFAULT_TIMEOUT` | `60` | Default command timeout (seconds) |
| `SANDBOX_MAX_TIMEOUT` | `3600` | Maximum command timeout (seconds) |
| `SANDBOX_INSTALL_TIMEOUT` | `300` | Package install timeout (seconds) |
| `SANDBOX_PIP_CACHE_PATH` | `/tmp/onit/pip-cache` | Shared pip cache across sessions |
| `SANDBOX_DATA_MOUNTS` | *(empty)* | Comma-separated mount specs (e.g. `/data:/data:ro`) |
| `SANDBOX_MAX_OUTPUT_BYTES` | `50000` | Max stdout/stderr bytes per tool call |

### YAML Configuration

```yaml
# configs/default.yaml
server:
  name: SandboxMCPServer
  host: 0.0.0.0
  port: 18205
  transport: streamable-http

docker:
  sandbox_image: onit-sandbox:latest
  memory_limit: 2g
  cpu_quota: 100000
  pids_limit: 256
  network_disabled_default: true
  default_timeout: 60
  max_timeout: 3600
  max_output_bytes: 50000
  data_mounts: []
  # data_mounts: ["/data:/data:ro", "/models:/models:rw"]
```

## Integration with OnIt

Add to your OnIt `configs/default.yaml`:

```yaml
mcp_servers:
  - name: SandboxMCPServer
    description: Python code execution sandbox with package installation
    url: http://127.0.0.1:18205/mcp
    enabled: true
```

### Programmatic Usage

```python
from onit_sandbox.mcp_server import run, cleanup_sandbox

# Start the server with data mounts
run(
    transport="streamable-http",
    host="0.0.0.0",
    port=18205,
    options={
        "data_mounts": ["/data:/data:ro", "/models:/models:rw"],
    },
)

# Cleanup a session's container
cleanup_sandbox(session_id="my-session")
```

## Security

### Isolation Features

| Layer | Protection |
|-------|-----------|
| Filesystem | Container isolation — only workspace and explicit mounts are accessible |
| User | Non-root execution as `sandbox` (UID 1000) |
| Resources | Memory (2 GB), CPU (1 core), PIDs (256) |
| Network | Disabled by default — enabled only during `install_packages` or on request |
| Lifecycle | Containers auto-removed on stop (`--rm` flag) |
| GPU | Optional NVIDIA GPU passthrough when available |
| Data mounts | Default to read-only; read-write must be explicitly requested |

### What's Blocked

- Direct host filesystem access (only workspace and configured mounts)
- Root access inside container
- Network access during code execution (by default)
- Unlimited resource consumption

### What's Allowed

- Full PyPI access during `install_packages` or with network enabled
- Read/write to mounted workspace
- Read access to data mounts (write if `rw` mode)
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
| Command timeout | Increase timeout: `run_code(command="...", timeout=3600)` |
| Port already in use | Use `--port` flag: `onit-sandbox start --port 18206` |
| `gpu_available` is false despite having a GPU | Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — Docker needs it to expose GPUs to containers. See Prerequisites above. |
| Mount path does not exist | The server will attempt to create it; ensure the parent directory is writable |
| Data mount permission denied | Ensure the host directory is readable by UID 1000 (the sandbox user) |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
