#!/bin/bash
# Quick start script for Jupyter Lab with AI Council notebook

echo "========================================="
echo "AI COUNCIL - STARTING JUPYTER LAB"
echo "========================================="

cd /workspaces/quantum-ai-trader_v1.1

# Activate existing venv
source .venv/bin/activate

# Install missing deps if needed
pip install -q yfinance plotly 2>/dev/null

echo ""
echo "✅ Environment activated"
echo ""
echo "Starting Jupyter Lab..."
echo "Open: notebooks/AI_COUNCIL_MASTER.ipynb"
echo ""
echo "========================================="

# Start Jupyter Lab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''
