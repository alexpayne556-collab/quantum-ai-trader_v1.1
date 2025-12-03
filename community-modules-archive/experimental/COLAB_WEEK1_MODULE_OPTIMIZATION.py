"""
🚀 WEEK 1-2: MODULE OPTIMIZATION FOR EARLY DETECTION
====================================================
Based on Perplexity's exact recommendations
Enhance existing 7 modules for early detection capabilities

Priority order:
1. AI Forecast Pro (1-3 day predictions)
2. Institutional Flow Pro (early accumulation)
3. Sentiment Pro (sentiment acceleration)
4. Scanner Pro (pre-breakout detection)
5. Pattern Engine Pro (forming patterns)
"""

# ============================================================================
# CELL 1: SETUP
# ============================================================================

print("🚀 WEEK 1-2: MODULE OPTIMIZATION FOR EARLY DETECTION")
print("="*70)

from google.colab import drive
drive.mount('/content/drive')

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up paths
possible_paths = [
    '/content/drive/MyDrive/Quantum_AI_Cockpit',
    '/content/drive/MyDrive/QuantumAI',
]

PROJECT_PATH = None
for path in possible_paths:
    if os.path.exists(path) and os.path.isdir(path):
        if os.path.exists(os.path.join(path, 'backend', 'modules')):
            PROJECT_PATH = path
            break

if PROJECT_PATH is None:
    PROJECT_PATH = possible_paths[0]

sys.path.insert(0, PROJECT_PATH)
sys.path.insert(0, os.path.join(PROJECT_PATH, 'backend', 'modules'))
os.chdir(PROJECT_PATH)

print(f"✅ Project path: {PROJECT_PATH}")

# Install dependencies
!pip install -q yfinance pandas numpy scipy scikit-learn xgboost prophet plotly ta

print("✅ Dependencies installed")

# ============================================================================
# CELL 2: ENHANCE AI FORECAST PRO - EARLY DETECTION
# ============================================================================

print("\n" + "="*70)
print("MODULE 1: AI FORECAST PRO - 1-3 DAY PREDICTIONS")
print("="*70)
print("Perplexity Rating: ★★★★★ (Best for early detection)")
print("="*70)

def enhance_forecast_for_early_detection():
    """
    Enhance AI Forecast Pro for 1-3 day early detection
    Perplexity recommendations:
    - Focus on direction accuracy > magnitude
    - Add features: volume acceleration, volatility compression, sentiment shifts
    - Target: 57%+ direction accuracy
    """
    
    print("\n📊 Step 1: Loading AI Forecast Pro module...")
    
    try:
        import importlib.util
        spec_path = os.path.join(PROJECT_PATH, 'backend', 'modules', 'ai_forecast_pro.py')
        
        if os.path.exists(spec_path):
            spec = importlib.util.spec_from_file_location("ai_forecast_pro", spec_path)
            forecast_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(forecast_module)
            AIForecastPro = forecast_module.AIForecastPro
            print("   ✅ Module loaded")
        else:
            print("   ⚠️  Module not found - will create enhancement template")
            AIForecastPro = None
    except Exception as e:
        print(f"   ⚠️  Error: {e}")
        AIForecastPro = None
    
    # Enhancement code (to add to module)
    enhancement_code = '''
# ============================================================================
# EARLY DETECTION ENHANCEMENTS (Add to ai_forecast_pro.py)
# ============================================================================

def calculate_volume_acceleration(self, df, periods=[5, 10, 20]):
    """
    Detect volume acceleration (early signal)
    """
    volume_acceleration = {}
    
    for period in periods:
        recent_volume = df['volume'].tail(period).mean()
        previous_volume = df['volume'].tail(period*2).head(period).mean()
        
        if previous_volume > 0:
            acceleration = (recent_volume - previous_volume) / previous_volume
            volume_acceleration[f'{period}d'] = acceleration
    
    return volume_acceleration

def detect_volatility_compression(self, df):
    """
    Detect volatility squeeze (about to explode)
    """
    # Calculate ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    
    # Current ATR vs 20-day average
    current_atr = atr.iloc[-1]
    avg_atr = atr.tail(20).mean()
    
    # Compression ratio (lower = more compressed)
    compression_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
    
    return {
        'compression_ratio': compression_ratio,
        'is_squeeze': compression_ratio < 0.7,  # 30%+ below average
        'score': max(0, (1 - compression_ratio) * 100)  # Higher score = more compressed
    }

def predict_early_moves(self, ticker, horizon_days=3):
    """
    Predict price moves 1-3 days ahead (optimized for early detection)
    """
    # Get data
    df = self.get_data(ticker)
    
    if df is None or len(df) < 50:
        return {'error': 'Insufficient data'}
    
    # Calculate early detection features
    features = {
        'volume_acceleration': self.calculate_volume_acceleration(df),
        'volatility_compression': self.detect_volatility_compression(df),
        'price_momentum': self.calculate_momentum(df, periods=[5, 10, 20]),
    }
    
    # Run forecasts (existing Prophet/ARIMA)
    prophet_forecast = self.prophet_forecast(df, horizon_days)
    arima_forecast = self.arima_forecast(df, horizon_days)
    
    # XGBoost for short-term (best for 1-3 day)
    xgb_forecast = self.xgboost_forecast(df, features, horizon_days)
    
    # Weighted ensemble (XGBoost highest weight for short-term)
    ensemble_forecast = (
        0.20 * prophet_forecast['forecast'] +
        0.15 * arima_forecast['forecast'] +
        0.40 * xgb_forecast['forecast'] +  # Highest weight for short-term
        0.25 * self.calculate_trend_forecast(df, horizon_days)
    )
    
    # Calculate early detection score
    early_score = self.calculate_early_score(features, ensemble_forecast)
    
    return {
        'predicted_direction': 'UP' if ensemble_forecast > 0 else 'DOWN',
        'predicted_magnitude': abs(ensemble_forecast),
        'confidence': self.calculate_confidence([prophet_forecast, arima_forecast, xgb_forecast]),
        'horizon_days': horizon_days,
        'early_detection_score': early_score,
        'features': features
    }

def calculate_early_score(self, features, forecast):
    """
    Score how early we're detecting this move (0-100)
    """
    early_signals = {
        'volume_acceleration': features['volume_acceleration'].get('5d', 0) > 0.5,  # 50%+ acceleration
        'volatility_squeeze': features['volatility_compression']['is_squeeze'],
        'momentum_positive': features['price_momentum']['5d'] > 0,
    }
    
    early_signal_count = sum(early_signals.values())
    forecast_magnitude = abs(forecast)
    
    early_score = (
        (early_signal_count / 3) * 60 +  # Max 60 points from signals
        min(forecast_magnitude * 100, 40)  # Max 40 points from magnitude
    )
    
    return min(100, early_score)
'''
    
    print("\n📝 Enhancement Code Generated:")
    print("   → Add these methods to ai_forecast_pro.py")
    print("   → Focus: 1-3 day predictions with early detection features")
    print("   → Target: 57%+ direction accuracy")
    
    # Save enhancement code
    enhancement_file = os.path.join(PROJECT_PATH, 'backend', 'modules', 'ai_forecast_pro_enhancements.py')
    with open(enhancement_file, 'w') as f:
        f.write(enhancement_code)
    
    print(f"\n✅ Enhancement code saved to: {enhancement_file}")
    
    return enhancement_file

# Run enhancement
forecast_enhancement = enhance_forecast_for_early_detection()

# ============================================================================
# CELL 3: ENHANCE INSTITUTIONAL FLOW PRO - EARLY ACCUMULATION
# ============================================================================

print("\n" + "="*70)
print("MODULE 2: INSTITUTIONAL FLOW PRO - EARLY ACCUMULATION")
print("="*70)
print("Perplexity Rating: ★★★★☆ (Very Good for early detection)")
print("="*70)

def enhance_institutional_for_early_detection():
    """
    Enhance Institutional Flow Pro for early accumulation detection
    Perplexity recommendations:
    - Add price/volume divergence detection (KEY signal)
    - Weight recent flow heavily (last 2 days = 70% weight)
    - Target: Correlation >0.30 with 2-day forward returns
    """
    
    enhancement_code = '''
# ============================================================================
# EARLY ACCUMULATION DETECTION (Add to institutional_flow_pro.py)
# ============================================================================

def detect_price_volume_divergence(self, ticker, lookback_days=5):
    """
    Price/Volume divergence is THE BEST early signal
    Institutions buying while price is flat/down = accumulation
    """
    # Get price and dark pool data
    price_data = self.get_price_data(ticker, lookback_days)
    dp_data = self.get_dark_pool_data(ticker, lookback_days)
    
    if len(price_data) < 3 or len(dp_data) < 3:
        return {'detected': False}
    
    recent_3days_price = price_data[-3:]
    recent_3days_dp = dp_data[-3:]
    
    # Price trend
    price_change = (recent_3days_price[-1]['close'] - recent_3days_price[0]['close']) / recent_3days_price[0]['close']
    
    # Dark pool volume trend
    dp_volume_start = recent_3days_dp[0].get('dp_volume', 0)
    dp_volume_end = recent_3days_dp[-1].get('dp_volume', 0)
    
    if dp_volume_start > 0:
        dp_volume_trend = (dp_volume_end - dp_volume_start) / dp_volume_start
    else:
        dp_volume_trend = 0
    
    # Divergence: Price flat/down (-5% to +2%) but DP volume up (+30%+)
    divergence_detected = (price_change < 0.02 and dp_volume_trend > 0.30)
    
    return {
        'detected': divergence_detected,
        'price_change': price_change,
        'volume_change': dp_volume_trend,
        'strength': dp_volume_trend if divergence_detected else 0,
        'early_signal_score': 25 if divergence_detected else 0  # 25 points for divergence
    }

def detect_early_accumulation(self, ticker, lookback_days=5):
    """
    Detect institutional accumulation BEFORE price moves
    """
    dark_pool_data = self.get_dark_pool_data(ticker, lookback_days)
    
    if len(dark_pool_data) < 3:
        return {'early_accumulation_score': 0}
    
    # Key early detection signals
    signals = {
        'dark_pool_surge': self.detect_dp_volume_surge(dark_pool_data),
        'consecutive_accumulation': self.detect_consecutive_days(dark_pool_data),
        'price_volume_divergence': self.detect_price_volume_divergence(ticker, lookback_days),
        'large_block_trades': self.detect_block_trades(dark_pool_data),
    }
    
    # Early accumulation scoring
    early_score = 0
    
    # 1. Dark pool surge (30 points)
    dp_ratio = signals['dark_pool_surge'].get('ratio', 0)
    if dp_ratio > 0.40:  # DP > 40% total volume
        early_score += 30
    elif dp_ratio > 0.30:
        early_score += 20
    
    # 2. Consecutive days (25 points)
    consecutive_days = signals['consecutive_accumulation'].get('days', 0)
    if consecutive_days >= 3:
        early_score += 25
    elif consecutive_days >= 2:
        early_score += 15
    
    # 3. Price/Volume divergence (25 points) - CRITICAL
    if signals['price_volume_divergence']['detected']:
        early_score += 25
    
    # 4. Large block trades (20 points)
    block_count = signals['large_block_trades'].get('count', 0)
    if block_count >= 3:
        early_score += 20
    
    # Weight recent flow MORE heavily (exponential decay)
    recent_weight = self.calculate_recent_weight(dark_pool_data)
    early_score *= recent_weight
    
    return {
        'early_accumulation_score': min(100, early_score),
        'confidence': self.calculate_confidence(signals),
        'estimated_lag': 1.5,  # 1-2 days typical lag
        'signals': signals
    }

def calculate_recent_weight(self, dp_data):
    """
    Weight recent flow heavily (last 2 days = 70% weight)
    """
    if len(dp_data) < 2:
        return 1.0
    
    # Last 2 days get 70% weight, older gets 30%
    recent_weight = 0.70
    older_weight = 0.30
    
    return recent_weight + (older_weight * 0.5)  # Decay factor
'''
    
    print("\n📝 Enhancement Code Generated:")
    print("   → Add price/volume divergence detection (KEY signal)")
    print("   → Weight recent flow heavily (last 2 days = 70%)")
    print("   → Target: Correlation >0.30 with 2-day forward returns")
    
    enhancement_file = os.path.join(PROJECT_PATH, 'backend', 'modules', 'institutional_flow_pro_enhancements.py')
    with open(enhancement_file, 'w') as f:
        f.write(enhancement_code)
    
    print(f"\n✅ Enhancement code saved to: {enhancement_file}")
    
    return enhancement_file

# Run enhancement
institutional_enhancement = enhance_institutional_for_early_detection()

# ============================================================================
# CELL 4: ENHANCE SENTIMENT PRO - ACCELERATION DETECTION
# ============================================================================

print("\n" + "="*70)
print("MODULE 3: SENTIMENT PRO - ACCELERATION DETECTION")
print("="*70)
print("Perplexity Rating: ★★★★☆ (Very Good for early detection)")
print("="*70)

def enhance_sentiment_for_acceleration():
    """
    Enhance Sentiment Pro for sentiment acceleration detection
    Perplexity recommendations:
    - Detect sentiment acceleration (improving fast = early sign)
    - Time decay: Last 24 hours = 80% weight
    - Target: Correlation >0.25 with next-day returns
    """
    
    enhancement_code = '''
# ============================================================================
# SENTIMENT ACCELERATION DETECTION (Add to sentiment_pro.py)
# ============================================================================

def detect_sentiment_shift(self, ticker, lookback_hours=72):
    """
    Detect sentiment acceleration (early warning signal)
    Sentiment improving rapidly = early sign of price move
    """
    sentiment_history = self.get_sentiment_history(ticker, lookback_hours)
    
    if len(sentiment_history) < 24:
        return {'early_detection_score': 0}
    
    # Calculate sentiment trend
    recent_24h = sentiment_history[-24:]  # Last 24 hours
    previous_48h = sentiment_history[-72:-24] if len(sentiment_history) >= 72 else sentiment_history[:-24]  # Previous 48 hours
    
    recent_sentiment = np.mean([s.get('score', 50) for s in recent_24h])
    previous_sentiment = np.mean([s.get('score', 50) for s in previous_48h]) if len(previous_48h) > 0 else 50
    
    # Sentiment acceleration
    sentiment_change = recent_sentiment - previous_sentiment
    sentiment_acceleration = sentiment_change / previous_sentiment if previous_sentiment > 0 else 0
    
    # Early detection signals
    early_signals = {
        'sentiment_improving': sentiment_change > 5,  # +5 points improvement
        'acceleration_detected': sentiment_acceleration > 0.15,  # 15%+ acceleration
        'high_volume': len(recent_24h) > len(previous_48h) / 2,  # Increasing mentions
        'source_consensus': self.check_source_consensus(recent_24h) > 0.75,  # 75%+ agree
    }
    
    # Calculate early detection score
    early_score = 0
    
    if early_signals['sentiment_improving']:
        early_score += 25
    
    if early_signals['acceleration_detected']:
        early_score += 30  # Acceleration is KEY
    
    if early_signals['high_volume']:
        early_score += 25
    
    if early_signals['source_consensus']:
        early_score += 20
    
    return {
        'early_detection_score': min(100, early_score),
        'sentiment_change': sentiment_change,
        'acceleration': sentiment_acceleration,
        'current_sentiment': recent_sentiment,
        'signals': early_signals,
        'estimated_price_lag': 24 if sentiment_acceleration > 0.20 else 48  # 12-48 hours
    }

def apply_time_decay(self, sentiment_score, hours_old):
    """
    Weight recent sentiment HEAVILY
    Last 24 hours = 80% weight, 24-48 hours = 15%, older = 5%
    """
    TIME_DECAY_HALFLIFE = 24  # hours
    
    if hours_old <= 24:
        weight = 0.80  # 80% weight for last 24 hours
    elif hours_old <= 48:
        weight = 0.15  # 15% weight for 24-48 hours
    else:
        weight = 0.05  # 5% weight for older
    
    decayed_score = sentiment_score * weight
    
    return decayed_score
'''
    
    print("\n📝 Enhancement Code Generated:")
    print("   → Add sentiment acceleration detection")
    print("   → Time decay: Last 24h = 80% weight")
    print("   → Target: Correlation >0.25 with next-day returns")
    
    enhancement_file = os.path.join(PROJECT_PATH, 'backend', 'modules', 'sentiment_pro_enhancements.py')
    with open(enhancement_file, 'w') as f:
        f.write(enhancement_code)
    
    print(f"\n✅ Enhancement code saved to: {enhancement_file}")
    
    return enhancement_file

# Run enhancement
sentiment_enhancement = enhance_sentiment_for_acceleration()

# ============================================================================
# CELL 5: ENHANCE SCANNER PRO - PRE-BREAKOUT DETECTION
# ============================================================================

print("\n" + "="*70)
print("MODULE 4: SCANNER PRO - PRE-BREAKOUT DETECTION")
print("="*70)
print("Perplexity Rating: ★★★☆☆ (Good for early detection)")
print("="*70)

def enhance_scanner_for_pre_breakout():
    """
    Enhance Scanner Pro for pre-breakout detection
    Perplexity recommendations:
    - Detect volume acceleration + price consolidation
    - Lookback: 20 days for volume average, 5 days for recent
    - Target: 60%+ of volume accelerations lead to 5-day positive returns
    """
    
    enhancement_code = '''
# ============================================================================
# PRE-BREAKOUT DETECTION (Add to scanner_pro.py)
# ============================================================================

def detect_pre_breakout(self, ticker, lookback_days=20):
    """
    Scan for stocks ABOUT TO break out
    Volume precedes price (volume up but price not yet moving)
    """
    price_data = self.get_price_data(ticker, lookback_days)
    
    if len(price_data) < 20:
        return {'pre_breakout_score': 0}
    
    # Pre-breakout signals
    signals = {
        'volume_acceleration': self.detect_volume_acceleration(price_data),
        'consolidation': self.detect_consolidation(price_data),
        'at_resistance': self.check_resistance_proximity(price_data),
        'volatility_compression': self.detect_volatility_squeeze(price_data),
    }
    
    pre_breakout_score = 0
    
    # 1. Volume acceleration (30 points)
    if signals['volume_acceleration']['detected']:
        pre_breakout_score += 30
    
    # 2. Consolidation (25 points)
    if signals['consolidation']['detected']:
        pre_breakout_score += 25
    
    # 3. At resistance (25 points)
    if signals['at_resistance']['distance'] < 0.03:  # Within 3% of resistance
        pre_breakout_score += 25
    
    # 4. Volatility compression (20 points)
    if signals['volatility_compression']['detected']:
        pre_breakout_score += 20
    
    return {
        'pre_breakout_score': min(100, pre_breakout_score),
        'estimated_breakout_days': self.estimate_breakout_timing(signals),  # 1-3 days
        'signals': signals,
        'confidence': self.calculate_confidence(signals)
    }

def detect_volume_acceleration(self, price_data):
    """
    Volume increasing but price not yet moving (KEY early signal)
    """
    if len(price_data) < 20:
        return {'detected': False}
    
    recent_5days = price_data[-5:]
    previous_15days = price_data[-20:-5]
    
    # Volume trend
    recent_volume = np.mean([d.get('volume', 0) for d in recent_5days])
    previous_volume = np.mean([d.get('volume', 0) for d in previous_15days])
    
    if previous_volume > 0:
        volume_ratio = recent_volume / previous_volume
    else:
        volume_ratio = 1.0
    
    # Price trend
    recent_price_change = (recent_5days[-1].get('close', 0) - recent_5days[0].get('close', 0)) / recent_5days[0].get('close', 1)
    
    # Volume acceleration detected if:
    # - Volume up 50%+ but price flat (-2% to +5%)
    detected = (volume_ratio > 1.50 and -0.02 < recent_price_change < 0.05)
    
    return {
        'detected': detected,
        'volume_ratio': volume_ratio,
        'price_change': recent_price_change,
        'strength': volume_ratio if detected else 0
    }
'''
    
    print("\n📝 Enhancement Code Generated:")
    print("   → Add pre-breakout detection")
    print("   → Volume acceleration + price consolidation")
    print("   → Target: 60%+ success rate")
    
    enhancement_file = os.path.join(PROJECT_PATH, 'backend', 'modules', 'scanner_pro_enhancements.py')
    with open(enhancement_file, 'w') as f:
        f.write(enhancement_code)
    
    print(f"\n✅ Enhancement code saved to: {enhancement_file}")
    
    return enhancement_file

# Run enhancement
scanner_enhancement = enhance_scanner_for_pre_breakout()

# ============================================================================
# CELL 6: SUMMARY & NEXT STEPS
# ============================================================================

print("\n" + "="*70)
print("✅ WEEK 1 MODULE ENHANCEMENTS COMPLETE")
print("="*70)

print("\n📋 Enhancement Files Created:")
print(f"   1. {forecast_enhancement}")
print(f"   2. {institutional_enhancement}")
print(f"   3. {sentiment_enhancement}")
print(f"   4. {scanner_enhancement}")

print("\n📋 Next Steps:")
print("   1. Review enhancement code in each file")
print("   2. Integrate enhancements into original modules")
print("   3. Test each enhanced module individually")
print("   4. Validate early detection accuracy")
print("   5. Continue to Week 2: Pattern Engine & Risk Manager enhancements")

print("\n🎯 Perplexity's Priority Order:")
print("   ✅ AI Forecast Pro (1-3 day predictions) - DONE")
print("   ✅ Institutional Flow Pro (early accumulation) - DONE")
print("   ✅ Sentiment Pro (acceleration detection) - DONE")
print("   ✅ Scanner Pro (pre-breakout) - DONE")
print("   ⏳ Pattern Engine Pro (forming patterns) - NEXT")
print("   ⏳ Risk Manager Pro (calibration) - NEXT")

print("\n💡 Remember:")
print("   - Focus on QUALITY over quantity")
print("   - Optimize existing modules FIRST")
print("   - Don't add new modules yet")
print("   - Validate each enhancement")

print("\n🚀 Ready for Week 2!")

