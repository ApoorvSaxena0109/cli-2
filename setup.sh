#!/bin/bash
# =============================================================================
# ARS-VG Analyzer - One-Command Local Setup
# =============================================================================
# Usage: bash setup.sh
#
# This script sets up everything needed to run the ARS-VG Analyzer locally.
# No GPU required. No Google Colab. No disconnections.
# =============================================================================

set -e

echo "============================================================"
echo "  ARS-VG Analyzer - Local Setup"
echo "============================================================"
echo ""

# 1. Check Python version
echo "[1/4] Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "[ERROR] Python 3.8+ is required. Install from https://python.org"
    exit 1
fi

PY_VERSION=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "       Found Python $PY_VERSION"

# 2. Create virtual environment
echo ""
echo "[2/4] Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo "       Created 'venv' directory"
else
    echo "       Virtual environment already exists"
fi

# Activate
source venv/bin/activate
echo "       Activated virtual environment"

# 3. Install dependencies
echo ""
echo "[3/4] Installing dependencies (this may take 2-3 minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "       All dependencies installed"

# 4. Create data directories
echo ""
echo "[4/4] Creating data directories..."
mkdir -p ARS-VG-Analyzer/{input,processed,chromadb,results,graphs}
echo "       Directories ready"

# Done
echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "  To start the analyzer:"
echo ""
echo "    source venv/bin/activate"
echo "    python app.py"
echo ""
echo "  Options:"
echo "    python app.py --share     # Generate public URL for sharing"
echo "    python app.py --port 8080 # Use custom port"
echo "    python app.py --no-ollama # Skip LLM (lightweight mode)"
echo ""
echo "  Optional: Install Ollama for LLM-powered analysis:"
echo "    curl -fsSL https://ollama.com/install.sh | sh"
echo "    ollama pull deepseek-r1:7b"
echo ""
echo "============================================================"
