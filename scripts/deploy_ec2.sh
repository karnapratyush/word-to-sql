#!/bin/bash
# Deploy GoComet AI Logistics Assistant on EC2 (Amazon Linux 2023)
#
# Usage:
#   1. SSH into EC2: ssh -i "word-to-sql.pem" ec2-user@ec2-52-90-166-187.compute-1.amazonaws.com
#   2. Copy this script: scp -i "word-to-sql.pem" scripts/deploy_ec2.sh ec2-user@ec2-52-90-166-187.compute-1.amazonaws.com:~/
#   3. Run: bash deploy_ec2.sh
#
# Or run everything in one go after SSH:
#   curl -sSL https://raw.githubusercontent.com/karnapratyush/word-to-sql/main/scripts/deploy_ec2.sh | bash

set -e

echo "=========================================="
echo "  GoComet AI - EC2 Deployment"
echo "=========================================="

# ── Step 1: Install system dependencies ──────────────────────────────
echo ""
echo "[1/8] Installing system dependencies..."
sudo yum update -y
sudo yum install -y git python3.11 python3.11-pip python3.11-devel gcc sqlite-devel

# Use python3.11 (closest to 3.13 on Amazon Linux)
PYTHON=python3.11
PIP="$PYTHON -m pip"

echo "Python version: $($PYTHON --version)"

# ── Step 2: Clone the repo ───────────────────────────────────────────
echo ""
echo "[2/8] Cloning repository..."
cd ~
if [ -d "word-to-sql" ]; then
    echo "  Repo exists, pulling latest..."
    cd word-to-sql
    git pull origin main
else
    git clone https://github.com/karnapratyush/word-to-sql.git
    cd word-to-sql
fi

# ── Step 3: Create virtual environment ───────────────────────────────
echo ""
echo "[3/8] Setting up virtual environment..."
$PYTHON -m venv venv
source venv/bin/activate

# ── Step 4: Install dependencies ─────────────────────────────────────
echo ""
echo "[4/8] Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# ── Step 5: Create .env file ─────────────────────────────────────────
echo ""
echo "[5/8] Configuring environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "  *** IMPORTANT: Edit .env with your API keys ***"
    echo "  Run: nano .env"
    echo "  At minimum set: OPENROUTER_API_KEY"
    echo ""
    read -p "  Press Enter after editing .env (or Ctrl+C to exit and edit later)..."
else
    echo "  .env already exists, skipping..."
fi

# ── Step 6: Initialize database ──────────────────────────────────────
echo ""
echo "[6/8] Initializing database..."
python db/seed_data.py

# ── Step 7: Generate sample documents ────────────────────────────────
echo ""
echo "[7/8] Generating sample documents..."
python db/sample_documents.py

# ── Step 8: Start services ───────────────────────────────────────────
echo ""
echo "[8/8] Starting services..."
echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "  Start the app:"
echo "    cd ~/word-to-sql"
echo "    source venv/bin/activate"
echo ""
echo "  Option A: Both servers (background):"
echo "    nohup python run_api.py --port 8000 > api.log 2>&1 &"
echo "    nohup streamlit run app/Home.py --server.port 8501 --server.headless true --server.address 0.0.0.0 > ui.log 2>&1 &"
echo ""
echo "  Option B: Using screen (recommended):"
echo "    screen -S api -dm bash -c 'source venv/bin/activate && python run_api.py --port 8000'"
echo "    screen -S ui -dm bash -c 'source venv/bin/activate && streamlit run app/Home.py --server.port 8501 --server.headless true --server.address 0.0.0.0'"
echo ""
echo "  Access:"
echo "    API:  http://ec2-52-90-166-187.compute-1.amazonaws.com:8000/docs"
echo "    UI:   http://ec2-52-90-166-187.compute-1.amazonaws.com:8501"
echo ""
echo "  Make sure EC2 security group allows inbound ports 8000 and 8501!"
echo ""
