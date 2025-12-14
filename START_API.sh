#!/bin/bash

# Quick start script for Underdog API

echo "🚀 Starting Underdog API for Spark Dashboard"
echo ""
echo "Alpha 76 Watchlist: 76 small/mid-cap growth tickers"
echo "Endpoints: http://localhost:5000/api/underdog/"
echo ""
echo "After training models on Colab Pro:"
echo "  1. Download models from Google Drive"
echo "  2. Place in models/underdog_v1/"
echo "  3. Restart this API"
echo ""
echo "Press Ctrl+C to stop"
echo "================================================"
echo ""

cd /workspaces/quantum-ai-trader_v1.1
python underdog_api.py
