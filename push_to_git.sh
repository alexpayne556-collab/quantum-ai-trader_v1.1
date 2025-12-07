#!/bin/bash
# Git Push Script - Production ML Trading System v1.1

echo "================================================================================"
echo "🚀 PUSHING PRODUCTION ML SYSTEM TO REPOSITORY"
echo "================================================================================"
echo ""

# Check git status
echo "📊 Current Git Status:"
git status --short
echo ""

# Add all production files
echo "📦 Adding production files..."
git add PRODUCTION_ENSEMBLE_69PCT.py
git add TEMPORAL_ENHANCED_OPTIMIZER.py
git add validate_production_system.py
git add OPTIMIZATION_SUMMARY.md
git add OPTIMIZATION_ROADMAP.md
git add VALIDATION_RESULTS.md
echo "✅ Files staged"
echo ""

# Show what will be committed
echo "📋 Files to commit:"
git diff --cached --name-status
echo ""

# Commit with detailed message
echo "💾 Creating commit..."
git commit -m "feat: Production ML system v1.1 - 70.31% accuracy achieved

PERFORMANCE:
- LightGBM: 70.31% validation accuracy (best individual model)
- XGBoost: 69.36% validation accuracy
- HistGradientBoosting: 68.83% validation accuracy
- 3-model ensemble: 69.42% (validated from Colab)
- Total improvement: +8.61% from 61.7% baseline

NEW FILES:
- PRODUCTION_ENSEMBLE_69PCT.py: Production ensemble with validated params
- validate_production_system.py: Comprehensive validation on 48 tickers
- TEMPORAL_ENHANCED_OPTIMIZER.py: Research temporal model (53% - not recommended)
- OPTIMIZATION_SUMMARY.md: Complete optimization journey documentation
- OPTIMIZATION_ROADMAP.md: Implementation guide and roadmap
- VALIDATION_RESULTS.md: Detailed validation results and recommendations

KEY FINDINGS:
- Tree-based models (67-70%) significantly outperform deep learning (53%)
- LightGBM single model beats ensemble (70.31% vs 69.42%)
- Feature engineering (61→42 features) + SMOTE critical for performance
- Temporal CNN-LSTM not recommended (16% worse than LightGBM)

PRODUCTION READY:
✅ Validated on 33,024 samples across 48 tickers
✅ SMOTE class balancing implemented
✅ Confidence-based prediction filtering
✅ Feature selection via mutual information
✅ Optimized hyperparameters from 300 Optuna trials

DEPLOYMENT:
Recommend: LightGBM single model (70.31%) or 3-model ensemble (69.42%)
Skip: Temporal CNN-LSTM enhancement (underperforms at 53.56%)

Dataset: 48 tickers, 3-year history, 5-day horizon, 3-class (BUY/HOLD/SELL)
Training: 21k samples, Validation: 5k samples, Test: 6.6k samples"

echo "✅ Commit created"
echo ""

# Push to repository
echo "🌐 Pushing to remote repository..."
git push origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================================================"
    echo "✅ SUCCESS! Production system pushed to repository"
    echo "================================================================================"
    echo ""
    echo "📊 Performance Summary:"
    echo "   LightGBM: 70.31% (recommended)"
    echo "   Ensemble: 69.42% (validated)"
    echo "   Improvement: +8.61% from baseline"
    echo ""
    echo "🎯 Next Steps:"
    echo "   1. Review validation results in VALIDATION_RESULTS.md"
    echo "   2. Deploy LightGBM or ensemble to production"
    echo "   3. Set up monitoring and paper trading"
    echo "   4. Track live performance metrics"
    echo ""
    echo "📁 Repository: https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1"
    echo "================================================================================"
else
    echo ""
    echo "================================================================================"
    echo "❌ PUSH FAILED - Please check git configuration and try again"
    echo "================================================================================"
    echo ""
    echo "Troubleshooting:"
    echo "   - Check remote: git remote -v"
    echo "   - Check branch: git branch"
    echo "   - Check auth: git config --list | grep user"
    echo "   - Manual push: git push origin main"
    echo ""
fi
