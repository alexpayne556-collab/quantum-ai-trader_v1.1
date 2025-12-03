# COLAB BACKTEST - Build Confidence Before Risking $500
# ======================================================

"""
COPY-PASTE THIS INTO GOOGLE COLAB
==================================

This will:
1. Mount your Google Drive
2. Install required packages
3. Run ALL backtests
4. Generate confidence report
5. Tell you if you're ready to trade

NO CODING NEEDED - Just run the cells!
"""

# ============================================================================
# CELL 1: Setup & Mount Drive
# ============================================================================

print("🚀 Quantum AI Confidence Builder - Colab Edition")
print("="*70)
print()

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Navigate to your QuantumAI folder
import os
os.chdir('/content/drive/MyDrive/QuantumAI')

print("✅ Google Drive mounted")
print(f"📁 Working directory: {os.getcwd()}")

# ============================================================================
# CELL 2: Install Dependencies
# ============================================================================

print("\n📦 Installing required packages...")
!pip install yfinance pandas numpy nest-asyncio -q

print("✅ Packages installed")

# ============================================================================
# CELL 3: Run Penny Stock Backtest
# ============================================================================

print("\n" + "="*70)
print("📊 RUNNING PENNY STOCK EXPLOSION BACKTEST")
print("="*70)
print()

%run BACKTEST_PENNY_EXPLOSIONS.py

# ============================================================================
# CELL 4: Run Swing Trade Backtest
# ============================================================================

print("\n" + "="*70)
print("📈 RUNNING SWING TRADE BACKTEST")
print("="*70)
print()

%run BACKTEST_SWING_TRADES.py

# ============================================================================
# CELL 5: Generate Combined Report
# ============================================================================

print("\n" + "="*70)
print("🎯 GENERATING COMBINED CONFIDENCE REPORT")
print("="*70)
print()

%run RUN_ALL_BACKTESTS.py

# ============================================================================
# DONE!
# ============================================================================

print("\n✅ ALL BACKTESTS COMPLETE!")
print()
print("📊 Review the results above")
print("🎯 Check your confidence level")
print("📋 Follow the recommended next steps")
print()
print("Files created:")
print("  • penny_stock_backtest_results.csv")
print("  • swing_trade_backtest_results.csv")
print("  • confidence_report_YYYYMMDD_HHMMSS.txt")
print()
print("="*70)

