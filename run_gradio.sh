#!/bin/bash
# =============================================================================
# ARS-VG Analyzer - Gradio Share Mode
# =============================================================================
# Runs the notebook as a Gradio app with share=True
# Gives you a public gradio.live URL (expires after 72 hours)
# Your professor can access the UI from anywhere
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  ARS-VG Analyzer - Gradio Share Mode"
echo "  Public URL via gradio.live (72-hour link)"
echo "============================================================"

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "[*] Installing dependencies..."
pip install -q -r requirements.txt

# Convert notebook to script if not already done
if [ ! -f "ARS_VG_Analyzer.py" ]; then
    echo "[*] Converting notebook to Python script..."
    pip install -q jupyter nbconvert
    jupyter nbconvert --to script ARS_VG_Analyzer.ipynb --output ARS_VG_Analyzer
fi

echo ""
echo "[*] Starting ARS-VG Analyzer with Gradio share=True..."
echo "[*] A public URL will appear below — share it with your professor"
echo "[!] Link expires after 72 hours. Restart this script to get a new one."
echo ""

python ARS_VG_Analyzer.py
