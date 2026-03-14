#!/bin/bash
# Setup script for onit-sandbox

set -e

echo "Setting up onit-sandbox..."

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "Warning: Docker is not installed. The sandbox will not work without Docker."
    echo "Please install Docker from https://docs.docker.com/get-docker/"
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -e ".[dev]"

# Install pre-commit hooks
echo "Installing pre-commit hooks..."
pre-commit install

# Build Docker image
if command -v docker &> /dev/null; then
    echo "Building Docker sandbox image..."
    chmod +x docker/build.sh
    ./docker/build.sh
else
    echo "Skipping Docker image build (Docker not available)"
fi

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To start the sandbox server:"
echo "  onit-sandbox start"
echo ""
echo "To stop the server:"
echo "  onit-sandbox stop"
echo ""
echo "To check server status:"
echo "  onit-sandbox status"
