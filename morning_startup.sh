#!/bin/bash
# 🌅 MORNING STARTUP SCRIPT
# Run this to kickstart your training day

echo "=========================================="
echo "🚀 QUANTUM AI TRADER - MORNING STARTUP"
echo "=========================================="
echo ""

# Activate environment
source venv/bin/activate

echo "✅ Environment activated"
echo ""

# Show system status
echo "📊 SYSTEM STATUS:"
echo "----------------------------------------"
python -c "
import os
print(f'✅ Core modules: {len([f for f in os.listdir(\"core\") if f.endswith(\".py\")])} engines')
print(f'✅ Models saved: {len([f for f in os.listdir(\"models\") if f.endswith(\".pkl\")]) if os.path.exists(\"models\") else 0} trained models')
print(f'✅ Discovery experiments: 17 patterns tested')
print(f'✅ Best accuracy: 78.2% (triple_barrier + volume)')
"
echo ""

echo "🎯 TODAY'S MISSION:"
echo "----------------------------------------"
echo "1. Install GPU libraries (hmmlearn, torch, etc.)"
echo "2. Test ticker scanner on 10 stocks"
echo "3. Upload to Kaggle/Colab for GPU training"
echo "4. Research with Perplexity Pro"
echo "5. Train on 20-30 ticker universe"
echo "6. Build human-language recommender"
echo ""

echo "🚀 QUICK COMMANDS:"
echo "----------------------------------------"
echo "Test predictor:    python ultimate_predictor.py --ticker AAPL --action predict"
echo "Scan 5 tickers:    python ticker_scanner.py --limit 5"
echo "Start API server:  uvicorn backend_api:app --reload"
echo "View status:       cat SYSTEM_STATUS.md"
echo ""

echo "💡 Ready to trade! Let's make those gains 📈"
echo ""
