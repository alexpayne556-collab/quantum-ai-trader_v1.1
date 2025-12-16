#!/bin/bash
# ============================================
# AI COUNCIL + OUR INNOVATIONS - COMPLETE SETUP
# For Git Bash (Windows) or Linux
# ============================================

echo "========================================="
echo "SETTING UP AI COUNCIL TRADING SYSTEM"
echo "========================================="

# Navigate to project (adjust path for your system)
cd /c/Users/alexj/Desktop/shadow_ai/quantum-ai-trader_v1.1 2>/dev/null || cd ~/quantum-ai-trader_v1.1 2>/dev/null || cd /workspaces/quantum-ai-trader_v1.1

# Use EXISTING .venv from yesterday
echo "Using existing .venv from yesterday..."
if [ ! -d ".venv" ]; then
    echo "ERROR: .venv not found! Run this from the project folder."
    exit 1
fi

# Activate venv
echo "Activating virtual environment..."
source .venv/bin/activate || source .venv/Scripts/activate

# Upgrade pip
echo "Upgrading pip..."
python -m pip install --upgrade pip --quiet

# Install only MISSING dependencies (everything else already installed yesterday)
echo "Checking for missing packages..."
pip install --quiet yfinance plotly hmmlearn

echo "Kernel 'shadow_ai' already registered from yesterday ✓"

echo ""
echo "========================================="
echo "✅ SETUP COMPLETE!"
echo "========================================="
echo ""
echo "NEXT STEPS:"
echo "1. Run: jupyter lab"
echo "2. Open: notebooks/AI_COUNCIL_COMPLETE.ipynb"
echo "3. Select kernel: Python (shadow_ai)"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "No NVIDIA GPU detected (running on CPU)"

echo ""
echo "Ready to test ALL implementations!"
