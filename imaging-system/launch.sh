#!/bin/bash
# launch.sh — Start Plate Imaging System web UI
# Activates venv, starts Flask server, opens Chromium in kiosk mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/venv/bin/activate"

# Activate virtual environment
if [ -f "$VENV" ]; then
    source "$VENV"
else
    echo "venv not found at $VENV — run: python3 -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Kill any existing server instance
pkill -f "python3 server.py" 2>/dev/null

# Start Flask server in background
cd "$SCRIPT_DIR"
python3 server.py &
SERVER_PID=$!

# Wait for server to be ready (up to 10 seconds)
echo "Starting server…"
for i in $(seq 1 20); do
    if curl -s http://localhost:5000 > /dev/null 2>&1; then
        echo "Server ready."
        break
    fi
    sleep 0.5
done

# Open Chromium in kiosk mode
# Pi OS Bookworm uses "chromium"; older releases used "chromium-browser"
CHROMIUM_BIN=""
for candidate in chromium chromium-browser; do
    if command -v "$candidate" &>/dev/null; then
        CHROMIUM_BIN="$candidate"
        break
    fi
done

if [ -z "$CHROMIUM_BIN" ]; then
    echo "Chromium not found — install with: sudo apt install -y chromium"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Ensure Chromium can reach the display when launched from a .desktop shortcut
export DISPLAY="${DISPLAY:-:0}"

"$CHROMIUM_BIN" \
    --kiosk \
    --noerrdialogs \
    --disable-infobars \
    --no-first-run \
    --disable-session-crashed-bubble \
    http://localhost:5000 &

# When Chromium closes, shut down the server
wait
kill $SERVER_PID 2>/dev/null
