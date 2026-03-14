FROM python:3.12-slim

# Create non-root user for security
RUN useradd -m -u 1000 sandbox

# Install common scientific packages in system Python
RUN pip install --no-cache-dir \
    numpy \
    matplotlib \
    scipy \
    pandas \
    scikit-learn \
    sympy \
    pytest

# Set up workspace directory
RUN mkdir -p /workspace && chown sandbox:sandbox /workspace

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
