FROM python:3.12-slim

LABEL org.opencontainers.image.title="onit-sandbox"
LABEL org.opencontainers.image.description="Isolated Python execution sandbox for MCP agents"
LABEL org.opencontainers.image.source="https://github.com/sibyl-oracles/onit-sandbox"
LABEL org.opencontainers.image.licenses="Apache-2.0"

# Install system dependencies and common development toolchain
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    gfortran \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 sandbox

# Install PyTorch ecosystem (CUDA) — largest packages first to maximise
# layer-cache hits when the smaller scientific stack changes.
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio

# Install common scientific packages in system Python
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    scipy \
    pandas \
    scikit-learn \
    sympy \
    pytest

# Install data pipeline and evaluation packages
RUN pip install --no-cache-dir \
    datasets \
    transformers \
    safetensors \
    tensorboard \
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

# Set working directory
WORKDIR /home/sandbox

# Default command - keep container running
CMD ["sleep", "infinity"]
