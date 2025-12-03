"""
SIMPLE DAILY RUN - NO COMPLEXITY
Just runs the 5 core modules
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'backend' / 'modules'))

from master_coordinator_pro_FIXED import MasterCoordinatorProFixed
from production_trading_system import ProductionTradingSystem

print("="*80)
print("QUANTUM AI - DAILY RUN")
print("="*80)

# Run daily
coordinator = MasterCoordinatorProFixed()
signals = coordinator.run_daily()

print(f"\n✅ Generated {len(signals)} safe signals")

if signals:
    # Paper trade
    trading_system = ProductionTradingSystem()
    trading_system.paper_trade(signals)
    
    print(f"\n✅ Paper traded {len(signals)} signals")
    print(f"📁 Signals saved to: data/daily_signals.csv")
    print(f"📁 Trades logged to: logs/paper_trades.txt")
else:
    print("\n⚠️  No signals generated")

print("\n" + "="*80)
print("DONE")
print("="*80)

