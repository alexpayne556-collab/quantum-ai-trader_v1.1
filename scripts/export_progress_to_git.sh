#!/bin/bash
# ============================================================================
# EXPORT CURRENT PROGRESS TO GIT
# ============================================================================
# Saves current database, backups, and exports to git repo
# So you can pull them on Shadow PC and continue from where you left off
# ============================================================================

set -e

echo "========================================"
echo "EXPORTING PROGRESS TO GIT REPO"
echo "========================================"
echo ""

# Stop download process to safely export database
echo "[1/6] Stopping download process..."
pkill -f industrial_data_pipeline || echo "  (No process running)"
sleep 2
echo "  Stopped"

# Create final backup before export
echo "[2/6] Creating backup..."
cd /workspaces/quantum-ai-trader_v1.1
python3 research_lab/data_cache.py > /dev/null 2>&1 || echo "  Using manual backup"
mkdir -p data/backups
cp data/market_data.db data/backups/market_data_export_$(date +%Y%m%d_%H%M%S).db
echo "  Backup created"

# Export to Parquet (much smaller for git)
echo "[3/6] Exporting to Parquet..."
python3 -c "
from research_lab.data_cache import DataCache
cache = DataCache()
cache.export_to_parquet('data/exports/market_data_export.parquet')
print('  Exported to Parquet')
"

# Get stats
echo "[4/6] Progress stats..."
python3 -c "
import sqlite3
conn = sqlite3.connect('data/market_data.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(DISTINCT ticker), COUNT(*) FROM daily_bars')
tickers, bars = cursor.fetchone()
print(f'  Tickers: {tickers}')
print(f'  Bars: {bars:,}')
conn.close()
"

# Setup Git LFS for large files
echo "[5/6] Setting up Git LFS..."
git lfs install
git lfs track "*.db"
git lfs track "*.parquet"
git add .gitattributes

# Add files to git
echo "[6/6] Pushing to GitHub..."
git add data/market_data.db
git add data/exports/market_data_export.parquet
git add data/backups/*.db 2>/dev/null || true

# Commit and push
git commit -m "Export database progress: $(date '+%Y-%m-%d %H:%M') - Ready for Shadow PC resume"
git push

echo ""
echo "========================================"
echo "✓ EXPORT COMPLETE"
echo "========================================"
echo ""
echo "On Shadow PC, run:"
echo "  git pull"
echo "  .\SHADOW_PC_SETUP.ps1"
echo ""
echo "Download will resume from current progress!"
echo ""
