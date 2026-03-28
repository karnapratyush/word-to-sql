#!/bin/bash
# Start both FastAPI and Streamlit for local development.
#
# Usage:
#   bash scripts/start_dev.sh
#
# This starts:
#   - FastAPI on http://localhost:8000 (API)
#   - Streamlit on http://localhost:8501 (UI)
#
# Press Ctrl+C to stop both.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== GoComet AI Logistics Assistant ==="
echo ""
echo "Starting FastAPI on http://localhost:8000 ..."
echo "Starting Streamlit on http://localhost:8501 ..."
echo ""
echo "Press Ctrl+C to stop both."
echo ""

# Start FastAPI in background
python run_api.py --port 8000 &
API_PID=$!

# Wait for API to start
sleep 2

# Start Streamlit in foreground (headless skips email prompt)
streamlit run app/Home.py --server.port 8501 --server.headless true

# When Streamlit exits (Ctrl+C), kill API
kill $API_PID 2>/dev/null
