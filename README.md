# onit-sandbox

[![PyPI version](https://img.shields.io/pypi/v/onit-sandbox.svg)](https://pypi.org/project/onit-sandbox/)
[![Python](https://img.shields.io/pypi/pyversions/onit-sandbox.svg)](https://pypi.org/project/onit-sandbox/)
[![CI](https://github.com/sibyl-oracles/onit-sandbox/actions/workflows/ci.yml/badge.svg)](https://github.com/sibyl-oracles/onit-sandbox/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

A Docker-based code execution sandbox [MCP server](https://modelcontextprotocol.io/) for safe Python execution. Gives LLM agents an isolated environment where they can install packages, run code, manage git repositories, and produce artifacts — without touching the host system. Designed for AI model training, evaluation, and data pipelines.

## Why onit-sandbox?

Traditional bash-based MCP tools block package managers and restrict operations. **onit-sandbox** lifts those restrictions safely by running everything inside per-session Docker containers with resource limits and network isolation.

- **Install any package** — `pip install numpy matplotlib torch` just works
- **Run full projects** — execute scripts, run tests, build simulations
- **Git operations** — clone, commit, push, pull, branch — all inside the sandbox
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
│  │  git_clone / git_commit / …    │  │
│  └──────────┬─────────────────────┘  │
│             │ Docker CLI              │
│  ┌──────────▼─────────────────────┐  │
│  │ Docker Container (per-session) │  │
│  │  - PyTorch + scientific stack  │  │
│  │  - ML/data pipeline packages   │  │
│  │  - Full pip access             │  │
│  │  - /home/sandbox (home dir)     │  │
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

# With OS keychain support for secure GitHub token storage
pip install onit-sandbox[keyring]

# Or from source
git clone https://github.com/sibyl-oracles/onit-sandbox.git
cd onit-sandbox
pip install -e .
```

### Step 2 — Build (or pull) the Docker image

The sandbox image comes pre-loaded with PyTorch (CUDA), the common scientific Python stack, audio/speech processing tools, large-scale training libraries, and ML/data pipeline packages so agents can start training immediately.

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

> **Rebuilding after updates:** If you have previously built the image, you must rebuild it to pick up new packages or configuration changes. The build script will replace your existing `onit-sandbox:latest` image. Running containers based on the old image are unaffected — stop and restart sessions to use the new image.
>
> ```bash
> # Force a clean rebuild (no layer cache)
> docker buildx build --no-cache --load -t onit-sandbox:latest docker/
>
> # Or use the build script (uses layer cache by default)
> cd docker && ./build.sh
> ```

Pre-installed packages:

| Package | Description |
|---------|-------------|
| **PyTorch ecosystem** | |
| torch | Deep learning (CUDA) |
| torchvision | Vision models and transforms |
| torchaudio | Audio processing |
| **Scientific stack** | |
| numpy (1.26.4) | Numerical computing |
| matplotlib | Plotting and visualization |
| scipy (1.13.0) | Scientific computing |
| Cython (3.0.10) | C-extensions for Python |
| pandas | Data manipulation |
| scikit-learn | Machine learning |
| sympy | Symbolic mathematics |
| pytest | Testing framework |
| **Data pipeline** | |
| datasets | Hugging Face datasets |
| transformers | Hugging Face transformers |
| safetensors | Safe model serialization |
| tensorboard (2.16.2) | Training visualization |
| tqdm | Progress bars |
| pyyaml | YAML parsing |
| jsonlines | JSONL file handling |
| pillow | Image processing |
| **Audio & speech** | |
| phonemizer (3.2.1) | Grapheme-to-phoneme conversion |
| librosa (0.10.1) | Audio analysis |
| soundfile (0.12.1) | Audio file I/O |
| pyloudnorm | Loudness normalization (ITU-R BS.1770) |
| utmos | MOS prediction for speech |
| openai-whisper | Speech recognition |
| jiwer | Word/character error rate metrics |
| **Large-scale training** | |
| accelerate | Distributed training (Hugging Face) |
| deepspeed | Memory-efficient distributed training |
| bitsandbytes | 8-bit/4-bit quantization |
| peft | Parameter-efficient fine-tuning (LoRA, QLoRA) |
| flash-attn | Flash Attention (requires CUDA at build time) |

> **Fallback:** If no custom image is found, the server falls back to `python:3.12-slim` automatically. Packages can still be installed at runtime via `install_packages`.

### Step 3 — Start the server

```bash
# Start the MCP server (background)
onit-sandbox start

# Start in foreground (useful for debugging)
onit-sandbox start --foreground

# Start with data volume mounts (read-write by default)
onit-sandbox start --mount /data:/data --mount /checkpoints:/checkpoints

# Start with a specific GPU (e.g. GPU 2)
onit-sandbox start --gpu 2

# Or use CUDA_VISIBLE_DEVICES (automatically picked up)
CUDA_VISIBLE_DEVICES=2 onit-sandbox start

# Multiple GPUs
onit-sandbox start --gpu 0,1

# Check status
onit-sandbox status

# Stop the server
onit-sandbox stop
```

The server runs on `http://0.0.0.0:18205/mcp` by default (streamable-http transport).

### Step 4 — Configure GitHub authentication (optional)

If you need git operations against private repositories (clone, push, pull), store a GitHub Personal Access Token:

```bash
# Interactive setup — prompts for your token (input is hidden)
onit-sandbox setup

# Check current token status
onit-sandbox setup status

# Remove stored token
onit-sandbox setup remove
```

**Token storage:** The token is stored in the OS keychain (macOS Keychain, GNOME Keyring, KWallet, or Windows Credential Manager) when the `keyring` package is installed. Otherwise it falls back to `~/.onit-sandbox/github_token` with `0o600` permissions.

```bash
# Install keyring for OS keychain support
pip install keyring
```

Create a token at [github.com/settings/tokens](https://github.com/settings/tokens) with the `repo` scope (for private repos) or `public_repo` (for public repos only).

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

### Git Tools

All git tools operate inside the sandbox container. For private repos, run `onit-sandbox setup` first to configure authentication. Network is automatically enabled for operations that need it (clone, pull, push).

**Auto-recovery:** Git tools automatically handle common first-use issues so agents don't waste tool calls on predictable failures:
- **Identity** — `user.name`, `user.email`, and `init.defaultBranch main` are pre-configured in every container.
- **Push upstream** — if `git push` fails because the branch has no upstream, it automatically retries with `-u`.
- **Unstaged changes** — if `git commit` fails because changes exist but aren't staged, it automatically retries with `-a`.

#### `sandbox_git_clone`

Clone a repository into the sandbox workspace.

```json
{
  "url": "https://github.com/user/repo.git",
  "branch": "main",
  "depth": 1,
  "path": "my-repo"
}
```

#### `sandbox_git_status`

Show working tree status.

```json
{
  "path": "my-repo"
}
```

#### `sandbox_git_add`

Stage files for the next commit.

```json
{
  "files": ".",
  "path": "my-repo"
}
```

#### `sandbox_git_commit`

Create a commit with staged changes.

```json
{
  "message": "Add training script",
  "all": true,
  "path": "my-repo"
}
```

#### `sandbox_git_pull`

Pull changes from a remote.

```json
{
  "remote": "origin",
  "branch": "main",
  "path": "my-repo"
}
```

#### `sandbox_git_push`

Push commits to a remote.

```json
{
  "remote": "origin",
  "branch": "main",
  "set_upstream": true,
  "path": "my-repo"
}
```

#### `sandbox_git_branch`

List, create, or delete branches.

```json
{
  "name": "feature/training",
  "delete": false,
  "all": false,
  "path": "my-repo"
}
```

#### `sandbox_git_checkout`

Switch branches or restore files.

```json
{
  "target": "feature/training",
  "create": true,
  "path": "my-repo"
}
```

#### `sandbox_git_log`

Show commit history.

```json
{
  "max_count": 10,
  "oneline": true,
  "path": "my-repo"
}
```

#### `sandbox_git_diff`

Show changes between commits or working tree.

```json
{
  "staged": false,
  "target": "HEAD~1",
  "files": "src/",
  "path": "my-repo"
}
```

#### `sandbox_git_init`

Initialize a new git repository.

```json
{
  "path": "new-project",
  "default_branch": "main"
}
```

#### `sandbox_git_remote`

Manage remote repositories.

```json
{
  "action": "add",
  "name": "origin",
  "url": "https://github.com/user/repo.git",
  "path": "my-repo"
}
```

#### `sandbox_git_stash`

Stash or restore uncommitted changes.

```json
{
  "action": "push",
  "message": "WIP: training changes",
  "path": "my-repo"
}
```

### Example: Git Workflow in the Sandbox

```
1. sandbox_git_clone(url="https://github.com/user/ml-project.git")
2. sandbox_git_checkout(target="experiment/new-loss", create=true, path="ml-project")
3. sandbox_write_file(file_path="ml-project/train.py", content="...")
4. sandbox_run_code(command="cd ml-project && python train.py")
5. sandbox_git_add(files=".", path="ml-project")
6. sandbox_git_commit(message="Add new loss function", path="ml-project")
7. sandbox_git_push(set_upstream=true, path="ml-project")
```

## Data Mounts

Mount host directories into the sandbox for direct access to datasets, model weights, or shared storage. This avoids copying large files through `upload_file`.

### Via CLI

```bash
# Read-write data (default), read-only reference datasets
onit-sandbox start \
  --mount /data/datasets:/data \
  --mount /data/reference:/reference:ro

# Multiple mounts
onit-sandbox start \
  --mount /nas/imagenet:/data/imagenet:ro \
  --mount /scratch/outputs:/outputs
```

### Via Environment Variable

```bash
# Comma-separated "host:container[:mode]" entries (mode defaults to rw)
export SANDBOX_DATA_MOUNTS="/data:/data,/checkpoints:/checkpoints,/reference:/reference:ro"
onit-sandbox start
```

### Mount Modes

| Mode | Description |
|------|-------------|
| `rw` | Read-write (default) — sandbox can read and write to the host directory |
| `ro` | Read-only — sandbox can read but not modify host data |

### Example: Training Pipeline

```bash
# Start sandbox with dataset and output mounts (both read-write by default)
onit-sandbox start \
  --mount /data/imagenet:/data \
  --mount /results:/results
```

The AI agent can then:

1. List available data: `sandbox_list_files(path="/data")`
2. Write training code: `sandbox_write_file(file_path="train.py", content="...")`
3. Run training: `sandbox_run_code(command="python train.py", timeout=3600)`
4. Cache pre-processed datasets: results saved directly to `/data` for reuse across sessions
5. Monitor progress: `sandbox_read_file(file_path="train.log")`
6. Results are written directly to `/results` on the host

## Configuration

### CLI Options

```
onit-sandbox start [OPTIONS]

  --host         Host to bind to (default: 0.0.0.0)
  --port         Port to bind to (default: 18205)
  --transport    streamable-http, sse, or stdio (default: streamable-http)
  --foreground   Run in foreground
  --verbose      Enable verbose logging
  --data-path    Host data directory, mounted as /home/sandbox inside the sandbox (default: /tmp/onit/data)
  --mount        Mount host directory into sandbox (repeatable)
                 Format: HOST_PATH:CONTAINER_PATH[:MODE]
                 MODE is "rw" (default) or "ro"
  --gpu          GPU device(s) to expose to containers
                 Examples: "0", "1", "2", "0,1", "all" (default)
                 Overrides CUDA_VISIBLE_DEVICES and SANDBOX_GPU_DEVICES env vars

onit-sandbox setup [ACTION]

  (no action)    Interactive prompt to store a GitHub token
  status         Show whether a token is configured and where it's stored
  remove         Delete the stored token from keychain and/or file

onit-sandbox stop       Stop the running server
onit-sandbox status     Check server status
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_IMAGE` | `onit-sandbox:latest` | Docker image to use |
| `FALLBACK_IMAGE` | `python:3.12-slim` | Fallback if sandbox image unavailable |
| `SANDBOX_MEMORY_LIMIT` | `64g` | Container memory limit |
| `SANDBOX_CPU_QUOTA` | `0` | CPU quota (0 = no limit, use all available CPUs) |
| `SANDBOX_PIDS_LIMIT` | `4096` | Max processes in container |
| `SANDBOX_SHM_SIZE` | `16g` | Shared memory size (`/dev/shm`) for PyTorch DataLoader workers |
| `SANDBOX_DEFAULT_TIMEOUT` | `600` | Default command timeout (seconds) |
| `SANDBOX_MAX_TIMEOUT` | `86400` | Maximum command timeout — 24 hours (seconds) |
| `SANDBOX_INSTALL_TIMEOUT` | `600` | Package install timeout (seconds) |
| `SANDBOX_PIP_CACHE_PATH` | `/tmp/onit/pip-cache` | Shared pip cache across sessions |
| `SANDBOX_DATA_MOUNTS` | *(empty)* | Comma-separated mount specs (e.g. `/data:/data`). Mode defaults to `rw`; use `:ro` for read-only. |
| `SANDBOX_GPU_DEVICES` | `all` | GPU device(s) to expose (e.g. `0`, `0,1`, `all`). Falls back to `CUDA_VISIBLE_DEVICES` if unset. |
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
  memory_limit: 64g
  cpu_quota: 0           # 0 = no limit
  pids_limit: 4096
  shm_size: 16g          # shared memory for DataLoader workers
  network_disabled_default: true
  default_timeout: 600
  max_timeout: 86400     # 24 hours
  max_output_bytes: 50000
  data_mounts: []
  # data_mounts: ["/data:/data", "/checkpoints:/checkpoints", "/reference:/reference:ro"]
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
        "data_mounts": ["/data:/data", "/checkpoints:/checkpoints"],
    },
)

# Cleanup a session's container
cleanup_sandbox(session_id="my-session")
```

## Security

### Isolation Features

| Layer | Protection |
|-------|-----------|
| Filesystem | Container isolation — only home directory and explicit mounts are accessible |
| User | Non-root execution as `sandbox` (UID 1000) |
| Resources | Memory (64 GB), CPU (no limit), PIDs (4096), shared memory (16 GB) |
| Network | Disabled by default — enabled only during `install_packages`, git clone/push/pull, or on request |
| Lifecycle | Containers auto-removed on stop (`--rm` flag) |
| GPU | Optional NVIDIA GPU passthrough when available |
| Data mounts | Default to read-write for caching pre-processed data; use `:ro` to restrict |
| Credentials | GitHub tokens stored in OS keychain (with `keyring`) or file with `0o600` permissions; injected via env var, never written to disk inside containers |

### What's Blocked

- Direct host filesystem access (only home directory and configured mounts)
- Root access inside container
- Network access during code execution (by default)
- Unlimited resource consumption

### What's Allowed

- Full PyPI access during `install_packages` or with network enabled
- Read/write to home directory
- Read/write access to data mounts (default); read-only if mounted with `:ro`
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
| Permission denied on home directory | `chmod -R 777 /path/to/data` |
| Command timeout | Increase timeout: `run_code(command="...", timeout=86400)` |
| Port already in use | Use `--port` flag: `onit-sandbox start --port 18206` |
| `gpu_available` is false despite having a GPU | Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) — Docker needs it to expose GPUs to containers. See Prerequisites above. |
| Wrong GPU used inside container | Use `--gpu` to select a specific device: `onit-sandbox start --gpu 2`. `CUDA_VISIBLE_DEVICES` on the host is also respected. |
| Mount path does not exist | The server will attempt to create it; ensure the parent directory is writable |
| Data mount permission denied | Ensure the host directory is readable by UID 1000 (the sandbox user) |
| DataLoader crashes with shared memory error | Increase shared memory: `SANDBOX_SHM_SIZE=32g` or reduce `num_workers` |
| OOM during training | Increase memory limit: `SANDBOX_MEMORY_LIMIT=128g`, or use quantization (`bitsandbytes`) / PEFT (`peft`) |
| Old packages after rebuild | Stop running sessions and restart — containers use the image from when they were created |
| Git commit fails with "tell me who you are" | This is auto-configured in new containers. If you see it, restart the session to pick up the fix. |
| Git push fails with "no upstream branch" | Auto-recovered — the tool retries with `-u`. No action needed. |
| Git clone/push fails with 401 | Run `onit-sandbox setup` to configure a GitHub token |
| Git push fails with 403 | Token may lack `repo` scope — create a new token with the correct permissions |
| `onit-sandbox setup status` shows "file" storage | Install `keyring` for OS keychain storage: `pip install keyring` |
| Git operations hang | Network may not be reaching GitHub — check `sandbox_enable_network` and DNS settings |

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
