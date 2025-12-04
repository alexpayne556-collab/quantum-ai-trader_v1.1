#!/bin/bash
# Run the Quantum AI Trader Local Dashboard

echo "====================================="
echo "🚀 QUANTUM AI TRADER DASHBOARD"
echo "====================================="
echo ""
echo "📊 Model Stats:"
echo "   Win Rate: 84.1%"
echo "   EV/Trade: +3.89%"
echo "   Target: +5% in 3 days"
echo ""
echo "📈 Paper Trading Mode Active"
echo "   Starting Capital: \$10,000"
echo "   Max Positions: 7"
echo ""
echo "🌐 Opening dashboard at http://localhost:5000"
echo ""
echo "====================================="

# Change to script directory
cd "$(dirname "$0")"

# Install dependencies if needed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip install flask yfinance pandas numpy
fi

# Run the app
python3 app.py
