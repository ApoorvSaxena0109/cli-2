#!/bin/bash
# =============================================================================
# ARS-VG Analyzer - Cloudflare Tunnel Mode
# =============================================================================
# Runs the Gradio app locally (no share) and exposes it via Cloudflare Tunnel
# The link stays alive as long as your machine is running (NO 72-hour limit)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

GRADIO_PORT=7860

echo "============================================================"
echo "  ARS-VG Analyzer - Cloudflare Tunnel Mode"
echo "  Permanent URL (no 72-hour expiry)"
echo "============================================================"

# --- Install cloudflared if not present ---
if ! command -v cloudflared &> /dev/null; then
    echo "[*] Installing cloudflared..."
    curl -sL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o /tmp/cloudflared.deb
    sudo dpkg -i /tmp/cloudflared.deb
    rm /tmp/cloudflared.deb
    echo "[OK] cloudflared installed"
fi

# --- Create venv if it doesn't exist ---
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies (show progress - these are large packages)
echo "[*] Installing dependencies (this may take 10-20 min on first run)..."
pip install --progress-bar on -r requirements.txt

# Convert notebook to script if not already done
if [ ! -f "ARS_VG_Analyzer_local.py" ]; then
    echo "[*] Creating local-mode launcher script..."
    pip install jupyter nbconvert
    jupyter nbconvert --to script ARS_VG_Analyzer.ipynb --output ARS_VG_Analyzer_local

    # nbconvert may produce .txt instead of .py — rename if needed
    if [ -f "ARS_VG_Analyzer_local.txt" ] && [ ! -f "ARS_VG_Analyzer_local.py" ]; then
        mv ARS_VG_Analyzer_local.txt ARS_VG_Analyzer_local.py
    fi

    # Patch: change share=True to share=False for local mode
    sed -i 's/share=True/share=False/g' ARS_VG_Analyzer_local.py
    echo "[OK] Created ARS_VG_Analyzer_local.py (share=False)"
fi

echo ""
echo "[*] Starting Gradio app locally on port $GRADIO_PORT..."
echo "[*] Then opening Cloudflare Tunnel..."
echo ""

# Start Gradio app in background
python ARS_VG_Analyzer_local.py &
GRADIO_PID=$!

# Wait for Gradio to start
echo "[*] Waiting for Gradio to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:$GRADIO_PORT > /dev/null 2>&1; then
        echo "[OK] Gradio is running on port $GRADIO_PORT"
        break
    fi
    sleep 2
done

echo ""
echo "============================================================"
echo "  Starting Cloudflare Tunnel..."
echo "  Look for the https://xxxxx.trycloudflare.com URL below"
echo "  Share that URL with your professor!"
echo "  Press Ctrl+C to stop everything"
echo "============================================================"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "[*] Shutting down..."
    kill $GRADIO_PID 2>/dev/null
    exit 0
}
trap cleanup INT TERM

# Start cloudflare tunnel (this blocks and shows the URL)
cloudflared tunnel --url http://localhost:$GRADIO_PORT
