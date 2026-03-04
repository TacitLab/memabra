#!/usr/bin/env bash
# Memabra Skill - Environment Setup
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Memabra Environment Setup ==="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Install memabra in editable mode
echo "Installing memabra..."
pip install -e "$PROJ_DIR" 2>/dev/null || pip3 install -e "$PROJ_DIR"

# Create data directory
DATA_DIR="$HOME/.memabra"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

echo "=== Setup Complete ==="
