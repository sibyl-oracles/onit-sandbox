FROM python:3.12-slim

LABEL org.opencontainers.image.title="onit-sandbox"
LABEL org.opencontainers.image.description="Isolated Python execution sandbox for MCP agents"
LABEL org.opencontainers.image.source="https://github.com/sibyl-oracles/onit-sandbox"
LABEL org.opencontainers.image.licenses="Apache-2.0"

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
    pillow

# Set up workspace and data directories
RUN mkdir -p /workspace /data && chown sandbox:sandbox /workspace /data

# Ensure pip cache and user-install dirs are writable by sandbox user
RUN mkdir -p /home/sandbox/.cache/pip /home/sandbox/.local \
    && chown -R sandbox:sandbox /home/sandbox

# Switch to non-root user
USER sandbox
ENV HOME=/home/sandbox
ENV PATH="/home/sandbox/.local/bin:${PATH}"

# Set working directory
WORKDIR /workspace

# Default command - keep container running
CMD ["sleep", "infinity"]
