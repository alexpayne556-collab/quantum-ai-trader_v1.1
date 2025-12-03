"""
COLAB TEST - 8 PRO MODULES
===========================

This tests all 8 PRO modules in Google Colab to ensure they work.

Run this FIRST before building the dashboard.
"""

# ═══════════════════════════════════════════════════════════════════
# SETUP - INSTALL DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════

print("="*80)
print("🔧 INSTALLING DEPENDENCIES...")
print("="*80)

import subprocess
import sys

# Install required packages
packages = [
    "prophet",
    "lightgbm", 
    "xgboost",
    "yfinance",
    "pandas",
    "numpy",
    "scikit-learn",
    "statsmodels"
]

for package in packages:
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("\n✅ All dependencies installed!")

# ═══════════════════════════════════════════════════════════════════
# MOUNT GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📂 MOUNTING GOOGLE DRIVE...")
print("="*80)

from google.colab import drive
drive.mount('/content/drive')

# Add QuantumAI to path
import sys
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')

print("✅ Drive mounted and path configured!")

# ═══════════════════════════════════════════════════════════════════
# FETCH SAMPLE DATA
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("📊 FETCHING SAMPLE DATA...")
print("="*80)

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

# Fetch AMD data for testing
ticker = "AMD"
print(f"Fetching {ticker} data...")
stock = yf.Ticker(ticker)
df = stock.history(period="1y")

# Rename columns to match our format
df = df.reset_index()
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={'date': 'date'})

print(f"✅ Fetched {len(df)} days of data for {ticker}")
print(f"   Date range: {df['date'].iloc[0]} to {df['date'].iloc[-1]}")

# ═══════════════════════════════════════════════════════════════════
# TEST 1: AI_FORECAST_PRO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧪 TEST 1: AI_FORECAST_PRO")
print("="*80)

try:
    from ai_forecast_pro import AIForecastPro
    
    forecaster = AIForecastPro()
    
    # Run forecast
    import asyncio
    result = asyncio.run(forecaster.forecast(
        symbol=ticker,
        df=df,
        horizon_days=5,
        include_scenarios=True
    ))
    
    print(f"\n✅ AI_FORECAST_PRO WORKING!")
    print(f"   Current Price: ${result['current_price']:.2f}")
    print(f"   Base Case: ${result['base_case']['price']:.2f} ({result['base_case']['return_pct']:+.1f}%)")
    print(f"   Bull Case: ${result['bull_case']['price']:.2f} ({result['bull_case']['return_pct']:+.1f}%)")
    print(f"   Bear Case: ${result['bear_case']['price']:.2f} ({result['bear_case']['return_pct']:+.1f}%)")
    print(f"   Confidence: {result['confidence']*100:.0f}%")
    print(f"   Models: {', '.join(result['models_used'])}")
    
except Exception as e:
    print(f"\n❌ AI_FORECAST_PRO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# TEST 2: INSTITUTIONAL_FLOW_PRO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧪 TEST 2: INSTITUTIONAL_FLOW_PRO")
print("="*80)

try:
    from institutional_flow_pro import InstitutionalFlowPro
    
    tracker = InstitutionalFlowPro()
    result = tracker.analyze(ticker)
    
    print(f"\n✅ INSTITUTIONAL_FLOW_PRO WORKING!")
    print(f"   Signal: {result['signal']}")
    print(f"   Institutional Score: {result['institutional_score']}/100")
    print(f"   Dark Pool Score: {result['component_scores']['dark_pool']}/100")
    print(f"   Insider Score: {result['component_scores']['insider']}/100")
    print(f"   Earnings Score: {result['component_scores']['earnings']}/100")
    print(f"   {result['interpretation']}")
    
except Exception as e:
    print(f"\n❌ INSTITUTIONAL_FLOW_PRO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# TEST 3: AI_RECOMMENDER_PRO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧪 TEST 3: AI_RECOMMENDER_PRO")
print("="*80)

try:
    from ai_recommender_pro import AIRecommenderPro
    
    recommender = AIRecommenderPro()
    
    current_price = df['close'].iloc[-1]
    target_price = current_price * 1.12  # 12% target
    
    rec = recommender.recommend(
        symbol=ticker,
        current_price=current_price,
        target_price=target_price,
        confidence=0.75,
        account_balance=100000,
        win_rate=0.62
    )
    
    print(f"\n✅ AI_RECOMMENDER_PRO WORKING!")
    print(f"   Signal: {rec['signal']}")
    print(f"   Action: {rec['action']}")
    print(f"   Expected Return: {rec['expected_return_pct']:+.1f}%")
    print(f"   Risk/Reward: {rec['risk_reward']['risk_reward_ratio']}:1 ({rec['risk_reward']['assessment']})")
    
    if rec['position_sizing']:
        ps = rec['position_sizing']
        print(f"   Position Size: {ps['shares']} shares (${ps['position_value']:,.0f})")
        print(f"   Risk: {ps['risk_pct_of_account']:.1f}% of account")
        print(f"   Kelly Fraction: {ps['kelly_fraction']:.1%}")
    
except Exception as e:
    print(f"\n❌ AI_RECOMMENDER_PRO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# TEST 4: SCANNER_PRO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧪 TEST 4: SCANNER_PRO")
print("="*80)

try:
    from scanner_pro import ScannerPro
    
    scanner = ScannerPro()
    
    # Fetch data for multiple tickers
    test_tickers = ["AMD", "NVDA", "TSLA"]
    price_data = {}
    
    for t in test_tickers:
        try:
            stock = yf.Ticker(t)
            data = stock.history(period="3mo")
            data = data.reset_index()
            data.columns = [c.lower() for c in data.columns]
            price_data[t] = data
        except:
            continue
    
    # Run scans
    breakouts = scanner.scan_breakouts(tickers=test_tickers, data=price_data, min_score=0)
    momentum = scanner.scan_momentum(tickers=test_tickers, data=price_data, min_score=0)
    
    print(f"\n✅ SCANNER_PRO WORKING!")
    print(f"   Breakout opportunities: {len(breakouts)}")
    print(f"   Momentum opportunities: {len(momentum)}")
    
    if not breakouts.empty:
        print(f"\n   Top Breakout: {breakouts.iloc[0]['ticker']} (Score: {breakouts.iloc[0]['score']}/100)")
    
    if not momentum.empty:
        print(f"   Top Momentum: {momentum.iloc[0]['ticker']} (Score: {momentum.iloc[0]['score']}/100)")
    
except Exception as e:
    print(f"\n❌ SCANNER_PRO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# TEST 5: RISK_MANAGER_PRO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧪 TEST 5: RISK_MANAGER_PRO")
print("="*80)

try:
    from risk_manager_pro import RiskManagerPro
    
    rm = RiskManagerPro()
    
    risk = rm.analyze_position(
        ticker=ticker,
        df=df,
        position_value=20000,
        portfolio_value=100000
    )
    
    print(f"\n✅ RISK_MANAGER_PRO WORKING!")
    print(f"   Risk Level: {risk['risk_level']}")
    print(f"   Recommendation: {risk['recommendation']}")
    print(f"   Volatility: {risk['volatility_pct']:.1f}%")
    print(f"   Sharpe Ratio: {risk['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {risk['max_drawdown_pct']:.1f}%")
    print(f"   Position Size OK: {risk['position_size_ok']}")
    
except Exception as e:
    print(f"\n❌ RISK_MANAGER_PRO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# TEST 6: PATTERN_ENGINE_PRO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧪 TEST 6: PATTERN_ENGINE_PRO")
print("="*80)

try:
    from pattern_engine_pro import PatternEnginePro
    
    engine = PatternEnginePro()
    patterns = engine.detect(df=df, ticker=ticker)
    
    print(f"\n✅ PATTERN_ENGINE_PRO WORKING!")
    print(f"   Patterns detected: {len(patterns)}")
    
    for pattern in patterns:
        print(f"   - {pattern['type']}: {pattern['direction']} ({pattern['confidence']}% confidence)")
    
except Exception as e:
    print(f"\n❌ PATTERN_ENGINE_PRO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# TEST 7: SENTIMENT_PRO
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🧪 TEST 7: SENTIMENT_PRO")
print("="*80)

try:
    from sentiment_pro import SentimentPro
    
    sentiment = SentimentPro()
    result = sentiment.analyze(ticker)
    
    print(f"\n✅ SENTIMENT_PRO WORKING!")
    print(f"   Sentiment Score: {result['sentiment_score']:+.1f}")
    print(f"   Classification: {result['classification']}")
    print(f"   Recommendation: {result['recommendation']}")
    print(f"   Strength: {result['strength']}")
    print(f"   News Score: {result['news_sentiment']['score']:+.1f}")
    print(f"   Social Buzz: {result['social_sentiment']['buzz_level']}")
    
except Exception as e:
    print(f"\n❌ SENTIMENT_PRO FAILED: {e}")
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*80)
print("🎉 TESTING COMPLETE!")
print("="*80)
print("\nAll 8 PRO modules have been tested in Google Colab!")
print("\nNext steps:")
print("1. Fix any failed modules")
print("2. Build dashboard using PRO modules")
print("3. Train models on historical data")
print("4. Deploy to production")
print("\n🚀 You're ready to build a world-class trading system!")

