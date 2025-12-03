"""
COLAB LAUNCHER - One-Click Professional System
===============================================
Paste this entire file into ONE Google Colab cell

Sets up everything:
- Installs dependencies
- Mounts Drive
- Runs daily scanner
- Generates professional report
- Shows top 10 opportunities

ZERO CONFIGURATION NEEDED.
"""

# =============================================================================
# CELL 1: SETUP & MOUNT DRIVE
# =============================================================================

print("🚀 QUANTUM AI PROFESSIONAL TRADING SYSTEM")
print("="*70)
print("Built with Perplexity AI research")
print("Professional-grade signal aggregation")
print("="*70 + "\n")

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.chdir('/content/drive/MyDrive/QuantumAI')
print(f"✅ Drive mounted: {os.getcwd()}\n")

# =============================================================================
# CELL 2: INSTALL DEPENDENCIES
# =============================================================================

print("📦 Installing dependencies...")
!pip install yfinance requests python-dotenv nest-asyncio -q
print("✅ Dependencies installed\n")

# =============================================================================
# CELL 3: ADD MODULES TO PATH
# =============================================================================

import sys
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')
print("✅ Modules path added\n")

# =============================================================================
# CELL 4: RUN DAILY SCANNER
# =============================================================================

print("="*70)
print("🔍 RUNNING DAILY SCANNER")
print("="*70 + "\n")

# Import scanner
from daily_scanner import ProfessionalDailyScanner, Strategy, SWING_TRADE_UNIVERSE

# Initialize
scanner = ProfessionalDailyScanner(
    strategy=Strategy.SWING_TRADE,
    max_workers=4
)

print(f"🎯 Scanning {len(SWING_TRADE_UNIVERSE)} stocks...")
print(f"Strategy: Swing Trade (3-4 month holds)")
print(f"Expected time: ~2 minutes\n")

# Run scan
results = scanner.scan_universe(SWING_TRADE_UNIVERSE, use_modules=False)

# Generate report
scanner.generate_report(results, top_n=10)

# Export
filename = scanner.export_to_csv(results)

print(f"\n✅ SCAN COMPLETE!")
print(f"📊 Results: {len(results)} stocks scored")
print(f"💾 CSV saved: {filename}")
print(f"\n{'='*70}\n")

# =============================================================================
# CELL 5: SHOW DETAILED ANALYSIS OF TOP PICK
# =============================================================================

if results:
    print("="*70)
    print("🔥 DETAILED ANALYSIS OF TOP OPPORTUNITY")
    print("="*70 + "\n")
    
    top_pick = results[0]
    scanner.coordinator.print_detailed_report(top_pick)
    
    print("💡 TRADING PLAN:")
    print(f"   Ticker: {top_pick.ticker}")
    print(f"   AI Score: {top_pick.final_score:.1f}/100")
    print(f"   Recommendation: {top_pick.recommendation.label} {top_pick.recommendation.emoji}")
    print(f"   Confidence: {top_pick.missing_signal_penalty:.0%}")
    print(f"   Strategy: {top_pick.strategy.value.replace('_', ' ').title()}")
    print()
    print("="*70 + "\n")

print("✅ ALL DONE! Review results above.")
print("💡 TIP: Run this daily for fresh opportunities!")

