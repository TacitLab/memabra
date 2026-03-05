#!/usr/bin/env bash
# Memabra Skill - Environment Setup
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SKILL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== Memabra Skill Environment Setup ==="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.9+"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Install dependencies
echo "Installing dependencies..."
pip3 install numpy torch scikit-learn pyyaml 2>/dev/null || pip install numpy torch scikit-learn pyyaml

# Create data directory
DATA_DIR="$HOME/.memabra"
mkdir -p "$DATA_DIR"
echo "Data directory: $DATA_DIR"

# Verify import
echo "Verifying memabra import..."
PYTHONPATH="$SKILL_DIR:$PYTHONPATH" python3 -c "from memabra import MemabraAgent; print('OK: MemabraAgent imported successfully')"

echo "=== Setup Complete ==="
