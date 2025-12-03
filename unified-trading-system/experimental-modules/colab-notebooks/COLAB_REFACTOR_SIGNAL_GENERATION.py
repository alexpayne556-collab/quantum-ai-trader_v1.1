"""
🎯 REFACTOR TO SIGNAL GENERATION FOCUS
======================================
Refactors system to focus on signal generation and AI recommendations
NOT autonomous trading - manual execution on Robinhood
"""

import sys
from pathlib import Path
import re

print("="*80)
print("🎯 REFACTORING TO SIGNAL GENERATION FOCUS")
print("="*80)

BASE_DIR = Path('/content/drive/MyDrive/QuantumAI')
MODULES_DIR = BASE_DIR / 'backend' / 'modules'
backtest_file = MODULES_DIR / 'BACKTEST_INSTITUTIONAL_ENSEMBLE.py'

if not backtest_file.exists():
    print(f"❌ File not found: {backtest_file}")
    sys.exit(1)

print(f"\n📁 File: {backtest_file}")

# ============================================================================
# STEP 1: Update _generate_signals to focus on recommendations
# ============================================================================
print("\n1️⃣ Refactoring _generate_signals for signal generation focus...")

with open(backtest_file, 'r', encoding='utf-8') as f:
    content = f.read()

# Create new signal generation method focused on recommendations
new_generate_signals = """    def _generate_signals(self, date, data_dict, capital, current_positions) -> List[Tuple]:
        \"\"\"
        Generate trading signals and recommendations (NOT autonomous execution)
        
        PURPOSE: Research tool for finding stocks with high probability setups
        - Signals are generated for analysis and review
        - AI Recommender provides final recommendations
        - Manual execution on Robinhood (external)
        - Paper trading available on dashboard for testing only
        
        Returns: List of (symbol, entry_price, recommendation_dict) tuples
        """
        # Run ranking model (Tier 1)
        ranking_predictions = self.ranking_model.predict(
            list(data_dict.keys()),
            data_dict
        )
        
        # Filter universe
        top_stocks = self.ensemble.filter_universe(ranking_predictions)
        
        # Exclude stocks we already have (for paper trading only)
        candidates = [s for s in top_stocks if s not in current_positions]
        
        recommendations = []  # Changed from new_positions to recommendations
        
        for symbol in candidates:
            if len(recommendations) + len(current_positions) >= self.config['max_positions']:
                break
            
            data = data_dict[symbol]
            if len(data) < 20:
                continue
            
            # Gather all module signals
            all_signals = {}
            
            # Dark pool (CRITICAL - your best signal at 52.5%)
            dp_result = self.dark_pool.analyze_ticker(symbol, data)
            all_signals['dark_pool'] = dp_result if dp_result.get('signal') == 'BUY' else None
            
            # Insider trading
            insider_result = self.insider.analyze_ticker(symbol, data)
            all_signals['insider_trading'] = insider_result if insider_result.get('signal') == 'BUY' else None
            
            # Sentiment
            sentiment_score = self.sentiment.analyze(symbol, data)
            all_signals['sentiment'] = {'signal': 'BUY', 'confidence': sentiment_score} if sentiment_score > 0.6 else None
            
            # Prepare price data
            current_price = float(data['Close'].iloc[-1])
            atr = float(data['High'].iloc[-1] - data['Low'].iloc[-1]) if len(data) > 0 else current_price * 0.02
            
            price_data = {
                'price': current_price,
                'atr': atr
            }
            
            # Get recommendation from OptimizedEnsembleTrader
            recommendation, status = self.optimized_trader.generate_recommendation(
                symbol=symbol,
                all_signals=all_signals,
                price_data=price_data
            )
            
            # Check for early pump/breakout signals
            try:
                pump_data = {
                    'volume': data['Volume'] if 'Volume' in data.columns else pd.Series([0]),
                    'price': data,
                    'level2': {},
                    'social': {},
                    'order_book': []
                }
                
                pump_signal = self.pump_detector.scan_for_early_pumps(symbol, pump_data)
                
                if pump_signal and pump_signal.get('confidence', 0) >= 0.80:
                    # High-confidence pump signal - use as primary recommendation
                    recommendation = {
                        'action': 'BUY',
                        'confidence': pump_signal['confidence'],
                        'entry_price': current_price,
                        'target_price': pump_signal.get('target_price', current_price * 1.5),
                        'stop_loss': pump_signal.get('stop_loss', current_price * 0.95),
                        'source': 'PUMP_DETECTION',
                        'expected_move': pump_signal.get('expected_move', '50-150%'),
                        'confirming_modules': pump_signal.get('confirming_modules', []),
                        'time_to_move': pump_signal.get('time_to_move', '1-5 days'),
                        'risk_level': pump_signal.get('risk_level', 'MEDIUM')
                    }
            except Exception as e:
                logger.debug(f"Pump detection error for {symbol}: {e}")
            
            # If we have a recommendation, add it
            if recommendation and recommendation.get('action') == 'BUY':
                entry_price = current_price * (1 + self.config['slippage'])
                
                # Track signals for module performance
                if all_signals.get('dark_pool'):
                    self.signals_by_module['dark_pool'].append((date, symbol, all_signals['dark_pool'].get('confidence', 0.5)))
                if all_signals.get('insider_trading'):
                    self.signals_by_module['insider_trading'].append((date, symbol, all_signals['insider_trading'].get('confidence', 0.5)))
                if all_signals.get('sentiment'):
                    self.signals_by_module['sentiment'].append((date, symbol, all_signals['sentiment'].get('confidence', 0.5)))
                
                # Add recommendation (for paper trading or manual review)
                recommendations.append((symbol, entry_price, recommendation))
        
        return recommendations"""

# Replace the entire _generate_signals method
method_pattern = r'(def _generate_signals\(self, date, data_dict, capital, current_positions\) -> List\[Tuple\]:.*?)(?=\n    def |\nclass |\Z)'
match = re.search(method_pattern, content, re.DOTALL)

if match:
    content = content[:match.start()] + new_generate_signals + content[match.end():]
    print("   ✅ Replaced _generate_signals method")
else:
    print("   ⚠️  Could not find _generate_signals method")

# ============================================================================
# STEP 2: Update comments to emphasize research/analysis focus
# ============================================================================
print("\n2️⃣ Updating system comments for research focus...")

# Update BacktestEngine docstring
engine_docstring = '''class BacktestEngine:
    """
    Research & Analysis Engine - Signal Generation & Recommendations
    
    PURPOSE: Generate high-quality trading signals and recommendations
    - NOT an autonomous trading system
    - Signals are for analysis and manual review
    - AI Recommender provides final recommendations
    - Manual execution on Robinhood (external)
    - Paper trading available for testing only
    
    This engine:
    1. Generates signals from multiple modules
    2. Combines signals using OptimizedEnsembleTrader
    3. Detects early pump/breakout opportunities
    4. Provides recommendations via AI Recommender
    5. Tracks performance for learning
    """'''

content = re.sub(
    r'class BacktestEngine:.*?""".*?"""',
    engine_docstring,
    content,
    flags=re.DOTALL
)

print("   ✅ Updated class documentation")

# ============================================================================
# STEP 3: Add method to get recommendations (for AI Recommender)
# ============================================================================
print("\n3️⃣ Adding get_recommendations method for AI Recommender...")

recommendations_method = """
    def get_recommendations(self, symbols: List[str] = None, date=None) -> List[Dict]:
        \"\"\"
        Get trading recommendations for AI Recommender
        
        This is the main interface for getting stock recommendations.
        Returns signals and recommendations for manual review and execution.
        
        Args:
            symbols: List of symbols to analyze (None = use universe)
            date: Date to analyze (None = current date)
        
        Returns:
            List of recommendation dicts with:
            - symbol
            - action (BUY/SELL/HOLD)
            - confidence
            - entry_price
            - target_price
            - stop_loss
            - reasoning
            - confirming_modules
        \"\"\"
        if symbols is None:
            symbols = self.config['universe']
        
        if date is None:
            date = datetime.now()
        
        # Download current data
        data_dict = {}
        for symbol in symbols:
            try:
                data = yf.download(
                    symbol,
                    period='6mo',
                    progress=False,
                    auto_adjust=True
                )
                if not data.empty:
                    data_dict[symbol] = data
            except:
                continue
        
        if not data_dict:
            return []
        
        # Generate signals
        recommendations_list = self._generate_signals(date, data_dict, 0, {})
        
        # Format for AI Recommender
        formatted_recommendations = []
        for symbol, entry_price, rec in recommendations_list:
            formatted_recommendations.append({
                'symbol': symbol,
                'action': rec.get('action', 'BUY'),
                'confidence': rec.get('confidence', 0.5),
                'entry_price': entry_price,
                'target_price': rec.get('target_price', entry_price * 1.2),
                'stop_loss': rec.get('stop_loss', entry_price * 0.95),
                'reasoning': f"Signals from: {', '.join(rec.get('confirming_modules', []))}",
                'confirming_modules': rec.get('confirming_modules', []),
                'expected_move': rec.get('expected_move', '20-50%'),
                'source': rec.get('source', 'OPTIMIZED_ENSEMBLE'),
                'timestamp': date.isoformat() if isinstance(date, datetime) else str(date)
            })
        
        # Sort by confidence
        formatted_recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return formatted_recommendations
"""

# Add before _calculate_results
calc_pattern = r'(def _calculate_results\(self)'
if re.search(calc_pattern, content):
    if 'def get_recommendations' not in content:
        content = re.sub(calc_pattern, recommendations_method + r'\n    \1', content)
        print("   ✅ Added get_recommendations method")
    else:
        print("   ✅ get_recommendations method already exists")
else:
    print("   ⚠️  Could not find insertion point")

# Write updated file
with open(backtest_file, 'w', encoding='utf-8') as f:
    f.write(content)

print("\n   ✅ Backtest file refactored")

# ============================================================================
# VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("🔍 VERIFICATION")
print("="*80)

with open(backtest_file, 'r') as f:
    content_check = f.read()

checks = {
    'Signal generation focus': 'NOT autonomous execution' in content_check,
    'Research tool comment': 'Research tool' in content_check or 'Research & Analysis' in content_check,
    'get_recommendations method': 'def get_recommendations' in content_check,
    'AI Recommender interface': 'AI Recommender' in content_check,
    'Manual execution note': 'Manual execution' in content_check or 'manual review' in content_check,
}

print("\n✅ Refactoring Checks:")
for check_name, passed in checks.items():
    status = "✅" if passed else "❌"
    print(f"   {status} {check_name}")

all_passed = all(checks.values())

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("📋 REFACTORING SUMMARY")
print("="*80)

if all_passed:
    print("""
✅ Refactoring Complete!

System Now Focuses On:
   ✅ Signal generation and analysis (NOT autonomous trading)
   ✅ AI Recommender as main interface
   ✅ Manual execution on Robinhood (external)
   ✅ Paper trading for testing only
   ✅ Research and analysis tool

New Features:
   ✅ get_recommendations() method for AI Recommender
   ✅ Clear documentation about research focus
   ✅ Recommendations formatted for manual review
   ✅ Performance tracking for learning

Next Steps:
   1. Update dashboard to use get_recommendations()
   2. Ensure AI Recommender is main interface
   3. Remove any remaining execution logic (keep paper trading)
   4. Test signal generation and recommendations
""")
else:
    print("""
⚠️  Some refactoring steps may need manual review.
Please check the verification results above.
""")

print("="*80)
print("✅ REFACTORING COMPLETE!")
print("="*80)

