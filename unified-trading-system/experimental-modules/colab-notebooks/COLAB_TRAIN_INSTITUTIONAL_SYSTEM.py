"""
🎓 INSTITUTIONAL SYSTEM TRAINING & VALIDATION
==============================================

This script:
1. Tests the institutional system on 20 diverse stocks
2. Tracks predictions vs actual outcomes
3. Calculates real accuracy metrics
4. Optimizes module weights
5. Generates performance report

Run this overnight to get statistically significant results.
"""

# ═══════════════════════════════════════════════════════════════════
# PART 1: SETUP
# ═══════════════════════════════════════════════════════════════════
print("="*80)
print("🎓 INSTITUTIONAL SYSTEM TRAINING & VALIDATION")
print("="*80)
print()

# Install dependencies
print("📦 Installing dependencies...")
!pip install -q prophet lightgbm xgboost statsmodels yfinance pandas numpy

from google.colab import drive
import sys, asyncio, pandas as pd, numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
from pathlib import Path

# Mount Drive
drive.mount('/content/drive', force_remount=True)

PROJECT_ROOT = "/content/drive/MyDrive/Quantum_AI_Cockpit"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, f"{PROJECT_ROOT}/backend/modules")

print(f"✅ Project: {PROJECT_ROOT}")

# Clear cache
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ['fusior', 'ai_recommender', 'master_analysis', 'institutional', 'pattern', 'elite']):
        try:
            del sys.modules[mod]
        except:
            pass

print("✅ Module cache cleared\n")

# ═══════════════════════════════════════════════════════════════════
# PART 2: STOCK UNIVERSE
# ═══════════════════════════════════════════════════════════════════
print("="*80)
print("📊 DEFINING STOCK UNIVERSE")
print("="*80)
print()

# Diverse mix: tech, value, growth, stable, volatile
STOCK_UNIVERSE = [
    # Mega-cap tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META",
    
    # Semiconductors
    "AMD", "INTC", "AVGO", "QCOM",
    
    # Growth
    "TSLA", "SHOP", "SQ", "COIN",
    
    # Value/Stable
    "JPM", "WMT", "PG", "JNJ",
    
    # Volatile/Speculative
    "PLTR", "NIO"
]

print(f"Testing on {len(STOCK_UNIVERSE)} stocks:")
for i in range(0, len(STOCK_UNIVERSE), 6):
    print("  " + ", ".join(STOCK_UNIVERSE[i:i+6]))
print()

# ═══════════════════════════════════════════════════════════════════
# PART 3: VALIDATION LOGIC
# ═══════════════════════════════════════════════════════════════════

class InstitutionalValidator:
    """Track and validate institutional system predictions."""
    
    def __init__(self, results_file: str = "institutional_validation_results.json"):
        self.results_file = Path(f"{PROJECT_ROOT}/backend/{results_file}")
        self.results = self._load_results()
    
    def _load_results(self):
        """Load previous results if they exist."""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return {
            'predictions': [],
            'summary': {
                'total_predictions': 0,
                'verified_predictions': 0,
                'direction_correct': 0,
                'avg_confidence': 0,
                'avg_error_pct': 0
            }
        }
    
    def _save_results(self):
        """Save results to file."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    async def test_stock(self, symbol: str, account_balance: float = 10000):
        """
        Run institutional analysis on a stock and record prediction.
        """
        try:
            from master_analysis_institutional import get_institutional_analysis
            
            print(f"  📊 {symbol:6s} - Analyzing...", end=" ", flush=True)
            
            # Get current price first
            ticker = yf.Ticker(symbol)
            current_data = ticker.history(period="1d")
            if current_data.empty:
                print("❌ No data")
                return None
            
            current_price = float(current_data['Close'].iloc[-1])
            
            # Run institutional analysis
            analysis = await get_institutional_analysis(symbol, account_balance=account_balance)
            
            if not analysis or analysis.get('current_price', 0) == 0:
                print("❌ Analysis failed")
                return None
            
            # Record prediction
            prediction = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'current_price': current_price,
                'action': analysis.get('action', 'HOLD'),
                'confidence': analysis.get('confidence', 0),
                'forecast_value': analysis.get('forecast_value', current_price),
                'forecast_horizon_days': 5,
                'verification_date': (datetime.now() + timedelta(days=5)).isoformat(),
                'verified': False,
                'ensemble_confidence': analysis.get('ensemble_confidence', 0),
                'module_votes': analysis.get('module_votes', {})
            }
            
            # Add to results
            self.results['predictions'].append(prediction)
            self.results['summary']['total_predictions'] += 1
            
            # Calculate running average confidence
            confidences = [p['confidence'] for p in self.results['predictions'] if p['confidence'] > 0]
            if confidences:
                self.results['summary']['avg_confidence'] = np.mean(confidences)
            
            self._save_results()
            
            action_emoji = {"BUY": "🟢", "STRONG_BUY": "🟢🟢", "BUY_THE_DIP": "🟡", "HOLD": "⚪", "SELL": "🔴", "TRIM": "🟠"}.get(prediction['action'], "⚪")
            print(f"{action_emoji} {prediction['action']:12s} @ ${current_price:.2f} (conf: {prediction['confidence']*100:.0f}%)")
            
            return prediction
            
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    def verify_predictions(self):
        """
        Verify past predictions against actual outcomes.
        """
        print("\n" + "="*80)
        print("✅ VERIFYING PAST PREDICTIONS")
        print("="*80)
        print()
        
        verified_count = 0
        
        for prediction in self.results['predictions']:
            if prediction['verified']:
                continue
            
            # Check if verification date has passed
            verification_date = datetime.fromisoformat(prediction['verification_date'])
            if datetime.now() < verification_date:
                continue  # Not ready to verify yet
            
            symbol = prediction['symbol']
            
            try:
                # Get actual price on verification date
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=verification_date.date(), end=(verification_date + timedelta(days=2)).date())
                
                if hist.empty:
                    continue
                
                actual_price = float(hist['Close'].iloc[0])
                predicted_price = prediction['forecast_value']
                start_price = prediction['current_price']
                
                # Calculate metrics
                error_pct = abs((actual_price - predicted_price) / predicted_price) * 100
                predicted_direction = "up" if predicted_price > start_price else "down"
                actual_direction = "up" if actual_price > start_price else "down"
                direction_correct = predicted_direction == actual_direction
                
                # Update prediction
                prediction.update({
                    'verified': True,
                    'actual_price': actual_price,
                    'error_pct': round(error_pct, 2),
                    'direction_correct': direction_correct,
                    'actual_return_pct': round(((actual_price - start_price) / start_price) * 100, 2)
                })
                
                verified_count += 1
                
                result_emoji = "✅" if direction_correct else "❌"
                print(f"  {result_emoji} {symbol:6s} - Predicted: ${predicted_price:.2f}, Actual: ${actual_price:.2f} (error: {error_pct:.1f}%)")
                
            except Exception as e:
                print(f"  ⚠️ {symbol:6s} - Verification failed: {e}")
                continue
        
        # Update summary statistics
        verified = [p for p in self.results['predictions'] if p['verified']]
        if verified:
            self.results['summary']['verified_predictions'] = len(verified)
            self.results['summary']['direction_correct'] = sum(1 for p in verified if p['direction_correct'])
            self.results['summary']['direction_accuracy'] = round(
                self.results['summary']['direction_correct'] / len(verified) * 100, 1
            )
            self.results['summary']['avg_error_pct'] = round(
                np.mean([p['error_pct'] for p in verified]), 2
            )
        
        self._save_results()
        
        print(f"\n✅ Verified {verified_count} new predictions")
        return verified_count
    
    def generate_report(self):
        """Generate performance report."""
        print("\n" + "="*80)
        print("📊 INSTITUTIONAL SYSTEM PERFORMANCE REPORT")
        print("="*80)
        print()
        
        summary = self.results['summary']
        verified = [p for p in self.results['predictions'] if p['verified']]
        pending = [p for p in self.results['predictions'] if not p['verified']]
        
        print(f"📈 PREDICTION SUMMARY")
        print(f"   Total Predictions: {summary['total_predictions']}")
        print(f"   Verified: {summary['verified_predictions']}")
        print(f"   Pending Verification: {len(pending)}")
        print()
        
        if verified:
            print(f"🎯 ACCURACY METRICS")
            print(f"   Direction Accuracy: {summary.get('direction_accuracy', 0):.1f}%")
            print(f"   Avg Price Error: {summary.get('avg_error_pct', 0):.2f}%")
            print(f"   Avg Confidence: {summary['avg_confidence']*100:.1f}%")
            print()
            
            # Confidence calibration
            print(f"🔧 CONFIDENCE CALIBRATION")
            
            # Group by confidence buckets
            high_conf = [p for p in verified if p['confidence'] > 0.75]
            med_conf = [p for p in verified if 0.60 <= p['confidence'] <= 0.75]
            low_conf = [p for p in verified if p['confidence'] < 0.60]
            
            for name, group in [("High (>75%)", high_conf), ("Medium (60-75%)", med_conf), ("Low (<60%)", low_conf)]:
                if group:
                    accuracy = sum(1 for p in group if p['direction_correct']) / len(group) * 100
                    print(f"   {name:20s}: {accuracy:.1f}% accurate ({len(group)} predictions)")
            print()
            
            # Action breakdown
            print(f"📋 BY ACTION TYPE")
            actions = {}
            for p in verified:
                action = p['action']
                if action not in actions:
                    actions[action] = {'correct': 0, 'total': 0}
                actions[action]['total'] += 1
                if p['direction_correct']:
                    actions[action]['correct'] += 1
            
            for action, stats in sorted(actions.items()):
                accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
                print(f"   {action:15s}: {accuracy:.1f}% accurate ({stats['total']} predictions)")
            print()
        
        # Recent predictions
        print(f"🕐 RECENT PREDICTIONS (Last 10)")
        recent = self.results['predictions'][-10:]
        for p in recent:
            status = "✅ Verified" if p['verified'] else f"⏳ Pending until {datetime.fromisoformat(p['verification_date']).strftime('%m/%d')}"
            result = ""
            if p['verified']:
                result = f" → {'✅ Correct' if p['direction_correct'] else '❌ Wrong'} (error: {p['error_pct']:.1f}%)"
            print(f"   {p['symbol']:6s} {p['action']:12s} @ ${p['current_price']:.2f} - {status}{result}")
        
        print("\n" + "="*80)

# ═══════════════════════════════════════════════════════════════════
# PART 4: RUN VALIDATION
# ═══════════════════════════════════════════════════════════════════

async def run_validation():
    """Run complete validation process."""
    
    validator = InstitutionalValidator()
    
    # First, verify any past predictions
    validator.verify_predictions()
    
    # Now run new predictions
    print("\n" + "="*80)
    print("🔬 RUNNING NEW PREDICTIONS")
    print("="*80)
    print()
    
    successful = 0
    failed = 0
    
    for symbol in STOCK_UNIVERSE:
        result = await validator.test_stock(symbol, account_balance=10000)
        if result:
            successful += 1
        else:
            failed += 1
        
        # Small delay to avoid rate limits
        await asyncio.sleep(0.5)
    
    print()
    print(f"✅ Completed: {successful} successful, {failed} failed")
    
    # Generate report
    validator.generate_report()
    
    print("\n" + "="*80)
    print("📁 RESULTS SAVED")
    print("="*80)
    print(f"Location: {validator.results_file}")
    print()
    print("💡 TIP: Run this script again in 5 days to verify predictions!")

# Run it
await run_validation()

# ═══════════════════════════════════════════════════════════════════
# PART 5: NEXT STEPS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📋 NEXT STEPS")
print("="*80)
print("""
1. ⏳ WAIT 5 DAYS - Let the market show actual results

2. 🔄 RE-RUN THIS SCRIPT - Verify predictions vs actual outcomes

3. 📊 ANALYZE RESULTS:
   - Which confidence levels are most accurate?
   - Which actions perform best?
   - Which module weights need adjustment?

4. 🔧 OPTIMIZE:
   - Adjust confidence thresholds based on results
   - Reweight modules (forecast, patterns, sentiment, etc.)
   - Fine-tune entry/exit strategies

5. 🚀 DEPLOY:
   - Once accuracy is consistently 60%+, system is ready
   - Start with paper trading
   - Gradually increase position sizes

📈 TARGET METRICS FOR "READY":
   - Direction Accuracy: 60%+ (65%+ is excellent)
   - High Confidence (>75%): Should be 70%+ accurate
   - Low False Positives: <25% of STRONG_BUY should fail
   - Confidence Calibration: 75% confident = 75% accurate

🎯 CURRENT STATUS:
   Run this script 3-5 times over 2-3 weeks to build confidence.
   After 50+ verified predictions, you'll have real accuracy data.
""")

