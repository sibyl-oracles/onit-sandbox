#!/bin/bash
# Build the sandbox Docker image

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Building onit-sandbox Docker image..."
docker build -t onit-sandbox:latest .

echo "Done! Image 'onit-sandbox:latest' is ready."
echo ""
echo "To verify, run:"
echo "  docker run --rm onit-sandbox:latest python -c 'import numpy; print(numpy.__version__)'"
