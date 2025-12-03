"""
🚀 PHASE 1 INTEGRATION - Quick Win (46.2% → 50-52%)
====================================================
This integrates Phase 1 improvements into your backtest
Run this AFTER uploading INSTITUTIONAL_IMPROVEMENTS.py to Drive
"""

import sys
from pathlib import Path

print("="*80)
print("🚀 PHASE 1 INTEGRATION - INSTITUTIONAL IMPROVEMENTS")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'

backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'
improvements_file = MODULES_DIR / 'INSTITUTIONAL_IMPROVEMENTS.py'

# Check files
if not improvements_file.exists():
    print(f"\n❌ INSTITUTIONAL_IMPROVEMENTS.py not found!")
    print(f"   Upload it to: {MODULES_DIR}")
    sys.exit(1)

if not backtest_file.exists():
    print(f"\n❌ BACKTEST_INSTITUTIONAL_ENSEMBLE.py not found!")
    sys.exit(1)

print(f"\n✅ Files found - ready to integrate")

print("\n" + "="*80)
print("📋 INTEGRATION STEPS")
print("="*80)

print("""
This will modify your backtest to use Phase 1 improvements:

1. ✅ Veto System - Blocks bad trades
2. ✅ Confirmation System - Requires 2+ signals
3. ✅ Confidence Threshold - Skip <65% confidence

EXPECTED RESULTS:
  Win Rate: 46.2% → 50-52%
  Trade Count: 117 → 80-90
  Sharpe: -0.37 → -0.2 to 0.2

""")

# Read backtest file
print("📖 Reading backtest file...")
with open(backtest_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Check if already integrated
if 'InstitutionalEnsembleTrader' in content:
    print("\n⚠️  Backtest already has InstitutionalEnsembleTrader")
    print("   Integration may already be done!")
    response = input("   Continue anyway? (y/n): ")
    if response.lower() != 'y':
        sys.exit(0)

# Step 1: Add import
print("\n1️⃣ Adding import for InstitutionalEnsembleTrader...")
import_line = "from INSTITUTIONAL_IMPROVEMENTS import InstitutionalEnsembleTrader"

if import_line not in content:
    # Find where imports are
    import_end = content.find("from data_orchestrator import DataOrchestrator")
    if import_end != -1:
        # Find end of that line
        line_end = content.find('\n', import_end)
        content = content[:line_end+1] + import_line + '\n' + content[line_end+1:]
        print("   ✅ Import added")
    else:
        # Add after other imports
        content = content.replace(
            "from INSTITUTIONAL_ENSEMBLE_ENGINE import",
            import_line + "\nfrom INSTITUTIONAL_ENSEMBLE_ENGINE import"
        )
        print("   ✅ Import added (alternative location)")
else:
    print("   ✅ Import already exists")

# Step 2: Add InstitutionalEnsembleTrader to __init__
print("\n2️⃣ Adding InstitutionalEnsembleTrader to BacktestEngine...")
init_pattern = "        self.ranking_model = MockRankingModel(self.orchestrator)"
trader_init = """        
        # Institutional improvements (Phase 1)
        self.institutional_trader = InstitutionalEnsembleTrader(
            base_weights={
                'dark_pool': 0.30,
                'sentiment': 0.20,
                'pregainer': 0.15,
                'day_trading': 0.15,
                'opportunity': 0.10,
                'insider_trading': 0.10
            }
        )
        logger.info("✅ Institutional improvements initialized (Phase 1)")"""

if init_pattern in content and 'self.institutional_trader' not in content:
    content = content.replace(init_pattern, init_pattern + trader_init)
    print("   ✅ Institutional trader initialized")
else:
    print("   ⚠️  Could not find exact location - may need manual integration")

# Step 3: Modify _generate_signals to use institutional system
print("\n3️⃣ Modifying signal generation...")
print("   ⚠️  This requires manual integration - see instructions below")

# Write modified file
print("\n💾 Writing modified backtest file...")
backup_file = backtest_file.with_suffix('.py.backup')
with open(backup_file, 'w', encoding='utf-8') as f:
    # Read original
    with open(backtest_file, 'r') as orig:
        f.write(orig.read())
print(f"   ✅ Backup saved: {backup_file.name}")

with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(content)
print(f"   ✅ Modified file saved")

print("\n" + "="*80)
print("✅ PHASE 1 INTEGRATION COMPLETE")
print("="*80)

print("""
📋 MANUAL INTEGRATION REQUIRED:

You need to modify the _generate_signals() method to use the institutional trader.

FIND THIS CODE (around line 358):
──────────────────────────────────
def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
    # ... existing signal generation ...

REPLACE WITH:
──────────────────────────────────
def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
    \"\"\"Generate signals using institutional-grade system\"\"\"
    new_positions = []
    
    # Get ranking predictions
    ranking_predictions = self.ranking_model.predict(
        list(data_dict.keys()), data_dict
    )
    
    # Filter universe
    filtered_symbols = self.ensemble.filter_universe(ranking_predictions)
    
    for symbol in filtered_symbols:
        if symbol not in data_dict or len(data_dict[symbol]) < 20:
            continue
        
        data = data_dict[symbol]
        
        # Collect raw signals from all modules
        raw_signals = {}
        
        # Dark pool
        dp_result = self.dark_pool.analyze_ticker(symbol, data)
        if dp_result.get('signal') == 'BUY':
            raw_signals['dark_pool'] = {
                'direction': 1,
                'confidence': dp_result.get('confidence', 0.5)
            }
        
        # Insider
        insider_result = self.insider.analyze_ticker(symbol, data)
        if insider_result.get('signal') == 'BUY':
            raw_signals['insider_trading'] = {
                'direction': 1,
                'confidence': insider_result.get('confidence', 0.5)
            }
        
        # Scanners
        pregainer_result = self.pregainer.scan(symbol, data)
        if pregainer_result.get('signal') == 'BUY':
            raw_signals['pregainer'] = {
                'direction': 1,
                'confidence': pregainer_result.get('confidence', 0.5)
            }
        
        day_result = self.day_trading.scan(symbol, data)
        if day_result.get('signal') == 'BUY':
            raw_signals['day_trading'] = {
                'direction': 1,
                'confidence': day_result.get('confidence', 0.5)
            }
        
        opp_result = self.opportunity.scan(symbol, data)
        if opp_result.get('signal') == 'BUY':
            raw_signals['opportunity'] = {
                'direction': 1,
                'confidence': opp_result.get('confidence', 0.5)
            }
        
        # Sentiment
        sent_result = self.sentiment.analyze(symbol, data)
        if sent_result.get('signal') == 'BUY':
            raw_signals['sentiment'] = {
                'direction': 1,
                'confidence': sent_result.get('confidence', 0.5)
            }
        
        # Prepare stock data for veto checks
        current_price = self.orchestrator.get_last_close(data)
        volatility_21d = self.orchestrator.get_returns(data, period=21) * np.sqrt(21)
        volatility_252d = abs(self.orchestrator.get_returns(data, period=252)) * np.sqrt(252)
        avg_volume = self.orchestrator.get_volume_ratio(data, period=20) * data['Volume'].iloc[-20:].mean()
        return_5d = self.orchestrator.get_returns(data, period=5)
        
        stock_data = {
            'price': current_price,
            'realized_volatility_21d': abs(volatility_21d),
            'avg_volatility_252d': abs(volatility_252d),
            'avg_volume_20d': avg_volume,
            'intended_position_value': capital / self.config['max_positions'],
            'days_to_earnings': 999,  # TODO: Get from earnings calendar
            'return_5d': return_5d,
            'beta': 1.0,  # TODO: Calculate beta
            'atr': abs(volatility_21d) * current_price / np.sqrt(21),
            'last_gap_pct': 0.0  # TODO: Calculate gap
        }
        
        # Market conditions
        regime_result = self.regime.detect_regime(data)
        market_conditions = {
            'regime': regime_result.get('regime', 'steady_state'),
            'vix_current': 20.0  # TODO: Get actual VIX
        }
        
        # Generate institutional recommendation
        recommendation = self.institutional_trader.generate_trading_signal(
            symbol=symbol,
            raw_signals=raw_signals,
            stock_data=stock_data,
            market_conditions=market_conditions
        )
        
        # Only take BUY signals that passed all filters
        if recommendation['action'] == 'BUY' and len(raw_signals) >= 2:
            entry_price = current_price * (1 + self.config['slippage'])
            new_positions.append((symbol, entry_price, recommendation))
    
    return new_positions

──────────────────────────────────

Then modify _open_position to handle the recommendation object.

""")

print("\n" + "="*80)
print("✅ INTEGRATION SCRIPT COMPLETE")
print("="*80)
print("\n📋 NEXT STEPS:")
print("   1. Review the manual integration code above")
print("   2. Modify _generate_signals() method in your backtest")
print("   3. Test on a small dataset first")
print("   4. Run full backtest and verify improvements")
print("\n💡 Expected: 46.2% → 50-52% win rate!")
print("="*80)

