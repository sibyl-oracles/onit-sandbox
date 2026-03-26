FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

LABEL org.opencontainers.image.title="onit-sandbox"
LABEL org.opencontainers.image.description="Isolated Python execution sandbox for MCP agents"
LABEL org.opencontainers.image.source="https://github.com/sibyl-oracles/onit-sandbox"
LABEL org.opencontainers.image.licenses="Apache-2.0"

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.12 (ships with 24.04), pip, and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    espeak-ng \
    build-essential \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libreadline-dev \
    libsqlite3-dev \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libsndfile1 \
    ffmpeg \
    gfortran \
    git \
    curl \
    wget \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security (remove existing UID 1000 user if present)
RUN if getent passwd 1000 >/dev/null 2>&1; then userdel -r "$(getent passwd 1000 | cut -d: -f1)"; fi \
    && useradd -m -u 1000 sandbox

# Allow pip install system-wide (PEP 668 — safe in a container)
RUN rm -f /usr/lib/python3.12/EXTERNALLY-MANAGED

# Install PyTorch ecosystem (CUDA 12.8) — pinned to match the host driver.
# The container base image is CUDA 12.8.1; the host driver (570.x) supports
# up to CUDA 12.8.  Using the cu128 index ensures PyTorch's bundled CUDA
# runtime stays within what the driver can handle.
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install common scientific packages in system Python
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    matplotlib \
    scipy==1.13.0 \
    Cython==3.0.10 \
    pandas \
    scikit-learn \
    sympy \
    pytest

# Install data pipeline and evaluation packages
RUN pip install --no-cache-dir \
    datasets \
    transformers \
    safetensors \
    tensorboard==2.16.2 \
    tqdm \
    pyyaml \
    jsonlines \
    pillow \
    tokenizers \
    sentencepiece

# Install common ML packages (CV, NLP, RL, classical ML)
RUN pip install --no-cache-dir \
    xgboost \
    lightgbm \
    catboost \
    optuna \
    seaborn \
    plotly \
    networkx \
    opencv-python-headless \
    scikit-image \
    nltk \
    spacy \
    gensim \
    gymnasium \
    stable-baselines3 \
    einops \
    timm \
    onnx \
    onnxruntime \
    mlflow \
    wandb \
    huggingface-hub \
    evaluate

# Install audio and speech packages
RUN pip install --no-cache-dir \
    phonemizer \
    librosa \
    soundfile \
    pyloudnorm \
    openai-whisper \
    jiwer

# Install packages for large-scale training
RUN pip install --no-cache-dir \
    accelerate \
    deepspeed \
    bitsandbytes \
    peft

# flash-attn requires CUDA at build time; skip gracefully if unavailable
RUN pip install --no-cache-dir flash-attn --no-build-isolation || true

# Set up data directory
RUN mkdir -p /data \
    && chown sandbox:sandbox /data

# Ensure pip cache, HuggingFace cache, and user-install dirs are writable
RUN mkdir -p /home/sandbox/.cache/pip /home/sandbox/.cache/huggingface /home/sandbox/.local \
    && chown -R sandbox:sandbox /home/sandbox

# Switch to non-root user
USER sandbox
ENV HOME=/home/sandbox
ENV PATH="/home/sandbox/.local/bin:${PATH}"
# Point HuggingFace cache to a persistent-friendly location
ENV HF_HOME=/home/sandbox/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/sandbox/.cache/huggingface/transformers
# CUDA and training optimization defaults
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=1
ENV TORCH_CUDNN_V8_API_ENABLED=1
ENV TOKENIZERS_PARALLELISM=true
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /home/sandbox

# Default command - keep container running
CMD ["sleep", "infinity"]
