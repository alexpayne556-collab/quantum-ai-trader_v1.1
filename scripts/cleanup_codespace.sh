#!/bin/bash
# ============================================================================
# CLEANUP CODESPACE - REMOVE UNNECESSARY FILES
# ============================================================================
# Removes old logs, caches, temporary files, and redundant code
# Keeps only essential files for production research
# ============================================================================

set -e

echo "========================================"
echo "CLEANING CODESPACE"
echo "========================================"
echo ""

cd /workspaces/quantum-ai-trader_v1.1

# Track space before
BEFORE=$(du -sh . | cut -f1)
echo "Space before: $BEFORE"
echo ""

# 1. Remove old earnings event study caches (we have research_lab now)
echo "[1/8] Removing old earnings caches..."
rm -rf data/earnings_event_study_cache/
rm -rf data/earnings_event_study_runs/
rm -rf data/events/
echo "  ✓ Removed earnings caches"

# 2. Remove daily bars cache (redundant with market_data.db)
echo "[2/8] Removing redundant daily bars cache..."
rm -rf data/daily_bars_cache/
echo "  ✓ Removed daily bars cache"

# 3. Remove old colab files (we're using Shadow PC now)
echo "[3/8] Removing Colab files..."
rm -f COLAB_*.py COLAB_*.ipynb COLAB_*.md
echo "  ✓ Removed Colab files"

# 4. Remove old training notebooks (replaced by research_lab)
echo "[4/8] Removing old training notebooks..."
rm -f AGGRESSIVE_TRAINER.ipynb
rm -f ALPHA76_PRODUCTION_TRAINER.ipynb
rm -f ALPHAGO_*.ipynb
rm -f alpha_discovery_engine.ipynb
rm -f begon.ipynb
rm -f colab_*.ipynb
echo "  ✓ Removed old notebooks"

# 5. Remove redundant strategy files
echo "[5/8] Removing redundant AI recommenders..."
rm -f ai_recommender.py
rm -f ai_recommender_adv.py
rm -f ai_recommender_tuned.py
rm -f ai_pattern_signal_generator.py
echo "  ✓ Removed old recommenders"

# 6. Remove old backtest engines (replaced by research frameworks)
echo "[6/8] Removing old backtest files..."
rm -f backtest_3month.py
rm -f backtest_engine.py
rm -f backtest_validator.py
echo "  ✓ Removed old backtests"

# 7. Remove documentation/roadmap spam
echo "[7/8] Removing redundant documentation..."
rm -f A100_TRAINING_ROADMAP.md
rm -f ACCURACY_IMPROVEMENT_ROADMAP.md
rm -f ADVANCED_CHARTING_SPEC.md
rm -f ADVANCED_WEB_INTERFACE_STRATEGY.md
rm -f ALPHA_76_*.md
rm -f ALPHAGO_VISUAL_COMPLETE.md
rm -f API_KEYS_*.md
rm -f API_SETUP_COMPLETE.md
rm -f API_STATUS_*.md
rm -f BACKEND_*.md
rm -f BASELINE_VALIDATION_COMPLETE.md
rm -f BUILD_COMPLETE*.md BUILD_COMPLETE.txt
rm -f CAPITAL_READY_FIXES_TODO.md
rm -f COLAB_PRO_TRAINING_*.md
rm -f COLAB_QUICK_SETUP.md
rm -f COLAB_READY_TO_RUN.md
rm -f COLAB_TONIGHT_GUIDE.md
rm -f COLAB_TRAINING_*.md
rm -f COMPETITIVE_EDGE.md
rm -f COMPLETE_*.md
rm -f COMPREHENSIVE_SYSTEM_TEST_AND_TRAINER.py
rm -f CONTEXT_AWARE_*.py CONTEXT_AWARE_*.md
rm -f FINANCIAL_PHYSICS_OPUS.md
rm -f HEAVYWEIGHT_PROJECT_STATUS.md
rm -f INDUSTRIAL_PIPELINE_RUNNING.md
rm -f LAB_QUICK_REFERENCE.md
rm -f LAB_RESEARCH.md
rm -f TOMORROW.md
rm -f TOMORROW_START_HERE.md
rm -f TONIGHT_SUMMARY.md
echo "  ✓ Removed redundant docs"

# 8. Remove old scripts replaced by research_lab
echo "[8/8] Removing old scripts..."
rm -f ALPHAGO_AUTO_TUNER*.py
rm -f accuracy_fixes_implementation.py
rm -f alpha_76_*.py
rm -f ALPHA_76_WATCHLIST.py
rm -f analyze_my_portfolio.py
rm -f backup_to_gdrive.py
rm -f chart_engine.py
rm -f competition_dashboard.py
rm -f colab_*.py
echo "  ✓ Removed old scripts"

# Clean Python cache
echo ""
echo "Cleaning Python cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Track space after
echo ""
AFTER=$(du -sh . | cut -f1)
echo "Space after: $AFTER"

echo ""
echo "========================================"
echo "✓ CLEANUP COMPLETE"
echo "========================================"
echo ""
echo "Kept essential files:"
echo "  ✓ research_lab/ (all scientific frameworks)"
echo "  ✓ data/market_data.db (current progress)"
echo "  ✓ data/backups/ (database backups)"
echo "  ✓ data/exports/ (Parquet exports)"
echo "  ✓ scripts/ (production scripts)"
echo "  ✓ SHADOW_PC_*.* (Shadow PC setup)"
echo "  ✓ requirements_download.txt"
echo ""
echo "Removed:"
echo "  ✗ Old Colab notebooks and scripts"
echo "  ✗ Redundant backtesting engines"
echo "  ✗ Old AI recommenders"
echo "  ✗ Earnings event study caches"
echo "  ✗ Documentation spam"
echo ""
