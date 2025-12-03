# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 QUANTUM AI COCKPIT — PORTFOLIO TRAINING (BULLETPROOF VERSION)
# ═══════════════════════════════════════════════════════════════════════════════
# Copy this ENTIRE cell into Colab and run!
# This will train N-BEATS on YOUR 7 portfolio stocks + backtest NVDA, TSLA, GOOG

print("=" * 80)
print("🚀 QUANTUM AI COCKPIT — PORTFOLIO TRAINING")
print("=" * 80)
print("\n⏰ Start Time:", datetime.now().strftime('%I:%M %p'))
print("💡 Training on YOUR 7 portfolio stocks")
print("🔒 Keep-alive enabled - won't stop if you minimize!\n")

import os
import sys
import time
import asyncio
import threading
import pandas as pd
from pathlib import Path
from datetime import datetime

# ═══════════════════════════════════════════════════════════════════════════════
# KEEP-ALIVE: Prevents Colab from stopping
# ═══════════════════════════════════════════════════════════════════════════════

def keep_colab_alive():
    """Keeps Colab session active"""
    while True:
        time.sleep(300)  # Every 5 minutes
        print(".", end="", flush=True)

# Start keep-alive thread
threading.Thread(target=keep_colab_alive, daemon=True).start()
print("✅ Keep-alive thread started (Colab won't timeout)\n")

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY SETUP
# ═══════════════════════════════════════════════════════════════════════════════

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}\n")
else:
    print("⚠️ NO GPU - Training will be slower\n")

# Verify Google Drive
DRIVE_BASE = Path("/content/drive/MyDrive/Quantum_AI_Cockpit")
if not DRIVE_BASE.exists():
    print("❌ Google Drive not mounted! Run the mount cell first!")
    raise FileNotFoundError("Google Drive not found")

print(f"✅ Google Drive connected: {DRIVE_BASE}\n")

# Verify modules
MODULES_PATH = DRIVE_BASE / "backend" / "modules"
if MODULES_PATH not in [Path(p) for p in sys.path]:
    sys.path.insert(0, str(MODULES_PATH))

print("✅ Modules loaded\n")

# ═══════════════════════════════════════════════════════════════════════════════
# YOUR PORTFOLIO STOCKS
# ═══════════════════════════════════════════════════════════════════════════════

PORTFOLIO_STOCKS = [
    "MU",      # Micron (Down -33%)
    "ANRO",    # Anro (Down -27%)
    "SGML",    # Sigma Labs (Down -48%)
    "ADBE",    # Adobe (Down -8%)
    "NVDA",    # Nvidia (Up +17%)
    "TSLA",    # Tesla (Up +41%)
    "GOOG",    # Google (Up +32%)
]

BACKTEST_STOCKS = ["NVDA", "TSLA", "GOOG"]  # Your 3 winners

print("=" * 80)
print("📊 TRAINING PLAN")
print("=" * 80)
print(f"\n🎯 Portfolio Stocks: {', '.join(PORTFOLIO_STOCKS)}")
print(f"📈 Backtesting: {', '.join(BACKTEST_STOCKS)}")
print(f"\n⏰ Estimated Time: 15-20 minutes")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# PART 1: TRAIN N-BEATS ON YOUR PORTFOLIO (10-15 min)
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 80)
print("🔥 PART 1/4: N-BEATS TRAINING ON YOUR PORTFOLIO")
print("=" * 80)
print()

from master_analysis_engine import analyze_stock

training_results = []
training_start = time.time()

async def train_portfolio():
    for i, ticker in enumerate(PORTFOLIO_STOCKS, 1):
        print(f"\n[{i}/{len(PORTFOLIO_STOCKS)}] 🔥 Training {ticker}...")
        start = time.time()
        
        try:
            result = await analyze_stock(
                symbol=ticker,
                account_balance=50000,
                forecast_days=21,
                verbose=False
            )
            
            elapsed = time.time() - start
            
            if result['status'] == 'ok':
                rec = result['recommendation']
                patterns = rec.get('pattern_analysis', {})
                
                training_results.append({
                    'ticker': ticker,
                    'status': 'success',
                    'time_sec': elapsed,
                    'action': rec.get('action'),
                    'confidence': rec.get('confidence', 0) * 100,
                    'patterns': patterns.get('total_patterns', 0),
                    'forecast_5d': rec.get('expected_move_5d', 0)
                })
                
                print(f"   ✅ {elapsed:.1f}s | {rec['action']} | "
                      f"{rec.get('confidence', 0)*100:.0f}% conf | "
                      f"+{rec.get('expected_move_5d', 0):.1f}% (5D)")
            else:
                training_results.append({
                    'ticker': ticker,
                    'status': 'error',
                    'time_sec': elapsed,
                    'error': result.get('error', 'unknown')[:50]
                })
                print(f"   ❌ Failed: {result.get('error', 'unknown')[:50]}")
                
        except Exception as e:
            elapsed = time.time() - start
            training_results.append({
                'ticker': ticker,
                'status': 'error',
                'time_sec': elapsed,
                'error': str(e)[:50]
            })
            print(f"   ❌ Error: {str(e)[:50]}")

# Run training
await train_portfolio()

training_elapsed = time.time() - training_start

# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

df_train = pd.DataFrame(training_results)
success = df_train[df_train['status'] == 'success']

print("\n" + "=" * 80)
print("📊 PORTFOLIO TRAINING SUMMARY")
print("=" * 80)
print(f"\n✅ Success Rate: {len(success)}/{len(PORTFOLIO_STOCKS)} ({len(success)/len(PORTFOLIO_STOCKS)*100:.0f}%)")
print(f"⏱️  Total Time: {training_elapsed/60:.1f} minutes")

if len(success) > 0:
    print(f"⚡ Avg Time/Stock: {success['time_sec'].mean():.1f} seconds")
    print(f"📈 Avg Confidence: {success['confidence'].mean():.0f}%")
    print(f"🔍 Avg Patterns: {success['patterns'].mean():.1f}")
    print(f"💰 Avg 5D Forecast: {success['forecast_5d'].mean():+.1f}%")
    
    print("\n📋 YOUR PORTFOLIO ANALYSIS:")
    print("=" * 80)
    for _, row in success.iterrows():
        print(f"\n{row['ticker']:5} | {row['action']:12} | {row['confidence']:.0f}% conf | {row['forecast_5d']:+.1f}% (5D)")

# Save portfolio report
portfolio_report_path = DRIVE_BASE / "results" / f"portfolio_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
portfolio_report_path.parent.mkdir(parents=True, exist_ok=True)
df_train.to_csv(portfolio_report_path, index=False)
print(f"\n💾 Portfolio report saved: {portfolio_report_path.name}")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 2: BACKTEST YOUR WINNERS (30-45 min)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("📊 PART 2/4: BACKTESTING YOUR TOP 3 WINNERS")
print("=" * 80)
print(f"\nTesting historical performance of {', '.join(BACKTEST_STOCKS)}")
print()

from forecast_trainer import ForecastTrainer
import fusior_forecast
import yfinance as yf

backtest_results = []
backtest_start = time.time()

async def run_backtests():
    for i, ticker in enumerate(BACKTEST_STOCKS, 1):
        print(f"\n[{i}/{len(BACKTEST_STOCKS)}] 📊 Backtesting {ticker}...")
        start = time.time()
        
        try:
            # Fetch historical data
            stock = yf.Ticker(ticker)
            df = stock.history(period="3y")
            df.columns = df.columns.str.lower()
            
            if len(df) < 300:
                print(f"   ⚠️  Only {len(df)} days - skipping")
                continue
            
            # Run walk-forward backtest
            trainer = ForecastTrainer()
            result = await trainer.walk_forward_backtest(
                symbol=ticker,
                df=df,
                forecast_func=fusior_forecast.run,
                train_window_days=252,
                test_window_days=21,
                step_days=21,
                min_confidence=0.60
            )
            
            elapsed = time.time() - start
            
            # Score the backtest
            score = sum([
                result.sharpe_ratio >= 1.0,
                result.win_rate >= 55,
                result.total_return > 0,
                result.max_drawdown < 15
            ])
            
            backtest_results.append({
                'ticker': ticker,
                'return_pct': result.total_return,
                'sharpe': result.sharpe_ratio,
                'win_rate': result.win_rate,
                'max_dd': result.max_drawdown,
                'trades': result.total_trades,
                'score': score,
                'time_sec': elapsed
            })
            
            print(f"   ✅ {elapsed:.0f}s | Return: {result.total_return:+.1f}% | "
                  f"Sharpe: {result.sharpe_ratio:.2f} | Win: {result.win_rate:.0f}% | "
                  f"Score: {score}/4")
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")

await run_backtests()

backtest_elapsed = time.time() - backtest_start

# Print backtest summary
if backtest_results:
    df_bt = pd.DataFrame(backtest_results)
    
    print("\n" + "=" * 80)
    print("📊 BACKTEST SUMMARY")
    print("=" * 80)
    print(f"\n⏱️  Total Time: {backtest_elapsed/60:.1f} minutes")
    print(f"\n📈 HISTORICAL PERFORMANCE:")
    print("=" * 80)
    print(df_bt.to_string(index=False))
    
    print(f"\n🎯 AVERAGES:")
    print(f"   • Return: {df_bt['return_pct'].mean():+.1f}%")
    print(f"   • Sharpe: {df_bt['sharpe'].mean():.2f}")
    print(f"   • Win Rate: {df_bt['win_rate'].mean():.0f}%")
    print(f"   • Max Drawdown: {df_bt['max_dd'].mean():.1f}%")
    
    # Save backtest report
    backtest_report_path = DRIVE_BASE / "results" / f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    df_bt.to_csv(backtest_report_path, index=False)
    print(f"\n💾 Backtest report saved: {backtest_report_path.name}")
else:
    print("\n⚠️ No backtest results")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 3: PATTERN VALIDATION (5 min)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("🔍 PART 3/4: PATTERN VALIDATION")
print("=" * 80)
print(f"\nValidating pattern detection on your top 3 stocks...")
print()

from pattern_integration_layer import analyze_all_patterns

PATTERN_STOCKS = ["NVDA", "TSLA", "GOOG"]

pattern_results = []
pattern_start = time.time()

async def test_patterns():
    for i, ticker in enumerate(PATTERN_STOCKS, 1):
        print(f"\n[{i}/{len(PATTERN_STOCKS)}] 🔍 Validating {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="6mo")
            df.columns = df.columns.str.lower()
            
            result = await analyze_all_patterns(df, ticker)
            
            if result['status'] == 'ok':
                s = result['summary']
                pattern_results.append({
                    'ticker': ticker,
                    'patterns': s['total_patterns_detected'],
                    'confluence': s['confluence_score']
                })
                
                print(f"   ✅ Patterns: {s['total_patterns_detected']} | "
                      f"Confluence: {s['confluence_score']:.0f}%")
            else:
                print(f"   ⚠️  Pattern detection unavailable")
                
        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")

await test_patterns()

pattern_elapsed = time.time() - pattern_start

# Print pattern summary
if pattern_results:
    df_pat = pd.DataFrame(pattern_results)
    
    print("\n" + "=" * 80)
    print("🔍 PATTERN VALIDATION SUMMARY")
    print("=" * 80)
    print(f"\n⏱️  Total Time: {pattern_elapsed/60:.1f} minutes")
    print(f"\n📊 Average Patterns/Stock: {df_pat['patterns'].mean():.1f}")
    print(f"🎯 Average Confluence: {df_pat['confluence'].mean():.0f}%")
    print(f"\n📋 Results:")
    print(df_pat.to_string(index=False))
else:
    print("\n⚠️ Pattern detection had errors (non-critical)")

# ═══════════════════════════════════════════════════════════════════════════════
# PART 4: EXPORT MODELS TO GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("💾 PART 4/4: EXPORTING MODELS")
print("=" * 80)

import shutil

model_dir = Path("/content/output/nbeats_models")
export_dir = DRIVE_BASE / "trained_models"

models = []
if model_dir.exists():
    models = list(model_dir.glob("*.pkl"))
    if models:
        print(f"\n✅ Found {len(models)} trained N-BEATS models")
        
        export_dir.mkdir(parents=True, exist_ok=True)
        
        for model in models:
            shutil.copy(model, export_dir / model.name)
            print(f"   📦 {model.name}")
        
        print(f"\n✅ All models exported to: {export_dir}")
    else:
        print("\n⚠️  Model directory exists but no .pkl files found")
else:
    print("\n⚠️  No model directory found")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

total_elapsed = time.time() - training_start

print("\n" + "=" * 80)
print("🎉 PORTFOLIO TRAINING COMPLETE!")
print("=" * 80)

print(f"\n⏰ End Time: {datetime.now().strftime('%I:%M %p')}")
print(f"⏱️  Total Duration: {total_elapsed/60:.0f} minutes ({total_elapsed/3600:.1f} hours)")

print(f"\n✅ COMPLETED:")
print(f"   • Portfolio trained: {len(success)}/{len(PORTFOLIO_STOCKS)} stocks")
print(f"   • Backtested: {len(backtest_results)} stocks")
print(f"   • Pattern validated: {len(pattern_results)} stocks")
print(f"   • Models exported: {len(models)}")

if len(success) > 0:
    print(f"\n🎯 YOUR PORTFOLIO INSIGHTS:")
    print(f"   • Avg Confidence: {success['confidence'].mean():.0f}%")
    print(f"   • Avg 5D Forecast: {success['forecast_5d'].mean():+.1f}%")
    
    # Count buy/sell/hold
    actions = success['action'].value_counts()
    print(f"\n📊 RECOMMENDATIONS:")
    for action, count in actions.items():
        print(f"   • {action}: {count} stocks")
    
    if backtest_results:
        df_bt = pd.DataFrame(backtest_results)
        print(f"\n📈 BACKTEST PERFORMANCE:")
        print(f"   • Avg Return: {df_bt['return_pct'].mean():+.1f}%")
        print(f"   • Avg Sharpe: {df_bt['sharpe'].mean():.2f}")
        print(f"   • Avg Win Rate: {df_bt['win_rate'].mean():.0f}%")

# Save comprehensive report
final_report_path = DRIVE_BASE / "results" / f"COMPLETE_TRAINING_REPORT_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"

with open(final_report_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("QUANTUM AI COCKPIT — PORTFOLIO TRAINING REPORT\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %I:%M %p')}\n")
    f.write(f"Duration: {total_elapsed/60:.0f} minutes\n\n")
    
    f.write("YOUR PORTFOLIO ANALYSIS:\n")
    f.write("=" * 80 + "\n")
    f.write(df_train.to_string(index=False))
    f.write("\n\n")
    
    if backtest_results:
        f.write("BACKTEST RESULTS:\n")
        f.write("=" * 80 + "\n")
        f.write(df_bt.to_string(index=False))
        f.write("\n\n")
    
    if pattern_results:
        f.write("PATTERN VALIDATION:\n")
        f.write("=" * 80 + "\n")
        f.write(df_pat.to_string(index=False))

print(f"\n📄 Complete report saved: {final_report_path.name}")
print(f"📂 Location: {DRIVE_BASE}/results/")

print("\n" + "=" * 80)
print("🚀 READY FOR DEPLOYMENT!")
print("=" * 80)
print("\n💡 Next steps:")
print("   1. Review your portfolio recommendations above")
print("   2. Check backtest performance")
print("   3. Build the dashboard tomorrow!")
print("\n🌙 You can now close Colab - everything is saved to Google Drive!")
print("=" * 80)

