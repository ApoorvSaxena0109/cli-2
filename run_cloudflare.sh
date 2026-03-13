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

echo ""
echo "[*] Starting Gradio app locally on port $GRADIO_PORT..."
echo "[*] Then opening Cloudflare Tunnel..."
echo ""

# Start Gradio app in background (use app.py directly - it has lazy loading
# and proper startup, unlike the notebook which runs all cells sequentially)
python app.py --port $GRADIO_PORT &
GRADIO_PID=$!

# Wait for Gradio to actually be ready before starting the tunnel
echo "[*] Waiting for Gradio to start..."
GRADIO_READY=0
MAX_WAIT=180  # 15 minutes max (180 * 5s = 900s)
for i in $(seq 1 $MAX_WAIT); do
    # Check if the Gradio process is still alive
    if ! kill -0 $GRADIO_PID 2>/dev/null; then
        echo "[ERROR] Gradio process exited unexpectedly. Check the output above for errors."
        exit 1
    fi
    if curl -s -o /dev/null -w '%{http_code}' http://localhost:$GRADIO_PORT 2>/dev/null | grep -q '200\|302'; then
        echo "[OK] Gradio is running on port $GRADIO_PORT"
        GRADIO_READY=1
        break
    fi
    ELAPSED=$((i*5))
    # Print progress every 30 seconds
    if [ $((ELAPSED % 30)) -eq 0 ]; then
        echo "  ... still loading (${ELAPSED}s elapsed)"
    fi
    sleep 5
done

if [ "$GRADIO_READY" -eq 0 ]; then
    echo "[ERROR] Gradio did not start within 15 minutes. Check errors above."
    kill $GRADIO_PID 2>/dev/null
    exit 1
fi

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

# Start cloudflare tunnel (use http2 to avoid QUIC/TLS handshake failures)
cloudflared tunnel --url http://localhost:$GRADIO_PORT --protocol http2
