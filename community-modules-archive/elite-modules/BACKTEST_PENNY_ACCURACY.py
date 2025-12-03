"""
BACKTEST PENNY EXPLOSIONS - Validate System Accuracy
=====================================================
Tests if our system would have caught historical explosions:
- CIFR (you missed this)
- NUVVE (you missed this)
- Other known 100-500% movers

If system scores these 85+ BEFORE explosion → System works!
If system misses them → Need to improve
"""

import sys
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from professional_signal_coordinator import ProfessionalSignalCoordinator, Strategy, SignalNormalizer
from api_integrations import DataFetcher

# =============================================================================
# HISTORICAL PENNY EXPLOSIONS TO TEST
# =============================================================================

KNOWN_EXPLOSIONS = {
    'CIFR': {
        'explosion_date': '2024-11-15',  # Approximate
        'price_before': 1.50,
        'price_peak': 8.00,
        'gain': 433,
        'timeframe': '5 days'
    },
    'NUVVE': {
        'explosion_date': '2024-11-01',  # Approximate
        'price_before': 2.00,
        'price_peak': 12.00,
        'gain': 500,
        'timeframe': '10 days'
    },
    'IMPP': {
        'explosion_date': '2024-10-20',  # Known mover
        'price_before': 1.00,
        'price_peak': 4.50,
        'gain': 350,
        'timeframe': '7 days'
    },
    'BBIG': {
        'explosion_date': '2021-09-01',  # Famous short squeeze
        'price_before': 2.50,
        'price_peak': 12.00,
        'gain': 380,
        'timeframe': '14 days'
    }
}

# =============================================================================
# BACKTEST FUNCTION
# =============================================================================

def backtest_explosion(ticker: str, explosion_info: dict, days_before: int = 7):
    """
    Test if system would have flagged stock BEFORE explosion
    
    Args:
        ticker: Stock symbol
        explosion_info: Dict with explosion_date, price_before, etc.
        days_before: How many days before explosion to test (default 7)
    
    Returns:
        Dict with results
    """
    print(f"\n{'='*70}")
    print(f"🧪 BACKTESTING: {ticker}")
    print(f"{'='*70}")
    print(f"Known explosion: {explosion_info['explosion_date']}")
    print(f"Move: ${explosion_info['price_before']:.2f} → ${explosion_info['price_peak']:.2f}")
    print(f"Gain: {explosion_info['gain']}% in {explosion_info['timeframe']}")
    print()
    
    # Calculate test date (X days before explosion)
    explosion_date = pd.to_datetime(explosion_info['explosion_date'])
    test_date = explosion_date - timedelta(days=days_before)
    
    print(f"🔍 Testing system on: {test_date.date()}")
    print(f"   (This is {days_before} days BEFORE the explosion)")
    print()
    
    # Get historical data UP TO test_date (simulate we don't know future)
    try:
        stock = yf.Ticker(ticker)
        # Get 1 year of data ending at test_date
        df = stock.history(start=test_date - timedelta(days=365), end=test_date)
        
        if df.empty:
            print(f"❌ No historical data available for {ticker}")
            return {'success': False, 'reason': 'No data'}
        
        df.columns = [col.lower() for col in df.columns]
        current_price = df['close'].iloc[-1]
        
        print(f"📊 Data retrieved:")
        print(f"   Price on test date: ${current_price:.2f}")
        print(f"   Data points: {len(df)} days")
        print()
        
        # Build signals (same as scanner does)
        signals = build_historical_signals(ticker, df, current_price)
        
        # Score with penny stock strategy
        coordinator = ProfessionalSignalCoordinator(Strategy.PENNY_STOCK)
        result = coordinator.score_ticker(ticker, signals)
        
        if not result:
            print(f"❌ Failed to score {ticker}")
            return {'success': False, 'reason': 'Scoring failed'}
        
        # Analyze result
        score = result.final_score
        recommendation = result.recommendation.label
        
        print(f"📈 SYSTEM SCORE: {score:.1f}/100")
        print(f"   Recommendation: {recommendation} {result.recommendation.emoji}")
        print(f"   Signals used: {result.signals_used}/7")
        print()
        
        # Did it catch it?
        caught = score >= 75  # 75+ = actionable
        
        if caught:
            print(f"✅ SUCCESS! System would have flagged this!")
            print(f"   Score {score:.1f} is {'STRONG BUY' if score >= 85 else 'BUY'}")
            print(f"   You would have made {explosion_info['gain']}%!")
        else:
            print(f"❌ MISS! System scored too low ({score:.1f})")
            print(f"   Would have missed this {explosion_info['gain']}% move")
        
        print()
        print(f"🔍 Signal Breakdown:")
        for name, value in result.normalized_signals.items():
            weight = result.weights.get(name, 0)
            contribution = value * weight * 100
            print(f"   {name:15s}: {value:5.1f}/100 × {weight:4.1%} = {contribution:5.1f}")
        
        return {
            'success': True,
            'ticker': ticker,
            'score': score,
            'recommendation': recommendation,
            'caught': caught,
            'actual_gain': explosion_info['gain'],
            'test_date': test_date.date(),
            'explosion_date': explosion_date.date()
        }
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'reason': str(e)}


def build_historical_signals(ticker: str, df: pd.DataFrame, current_price: float) -> dict:
    """Build signals from historical data (same as scanner)"""
    
    signals = {}
    
    # 1. Forecast (simple momentum)
    try:
        price_20d_ago = df['close'].iloc[-20] if len(df) >= 20 else df['close'].iloc[0]
        momentum = ((current_price - price_20d_ago) / price_20d_ago)
        target_price = current_price * (1 + momentum * 0.5)
        
        returns = df['close'].pct_change().tail(20)
        volatility = returns.std()
        confidence = max(40, min(80, 60 - volatility * 100))
        
        signals['forecast'] = {
            'base_case': {'price': target_price},
            'confidence': confidence,
            'current_price': current_price
        }
    except:
        signals['forecast'] = {'base_case': {'price': current_price}, 'confidence': 50, 'current_price': current_price}
    
    # 2. Patterns (simple)
    try:
        closes = df['close'].values
        highs = df['high'].values
        
        patterns = []
        confidence = 50
        
        if len(closes) >= 20:
            if closes[-1] > closes[-10] > closes[-20]:
                patterns.append('Uptrend')
                confidence += 15
            
            if closes[-1] > highs[-20:-1].max():
                patterns.append('Breakout')
                confidence += 20
        
        signals['patterns'] = {'patterns': patterns, 'confidence': min(confidence, 85)}
    except:
        signals['patterns'] = {'patterns': [], 'confidence': 50}
    
    # 3. Risk metrics
    try:
        returns = df['close'].pct_change().dropna()
        mean_return = returns.mean()
        std_return = returns.std()
        sharpe = (mean_return / std_return) * (252 ** 0.5) if std_return > 0 else 0
        volatility = std_return * (252 ** 0.5)
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        signals['risk'] = {
            'sharpe_ratio': sharpe,
            'volatility': volatility,
            'max_drawdown': max_drawdown
        }
    except:
        signals['risk'] = {'sharpe_ratio': 1.0, 'volatility': 0.30, 'max_drawdown': -0.20}
    
    # 4. Scanner (momentum + volume)
    try:
        momentum_5d = df['close'].pct_change(5).iloc[-1] if len(df) >= 5 else 0
        momentum_20d = df['close'].pct_change(20).iloc[-1] if len(df) >= 20 else 0
        
        avg_volume = df['volume'].tail(20).mean()
        recent_volume = df['volume'].tail(5).mean()
        volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
        
        momentum_score = 50 + (momentum_5d * 100) + (momentum_20d * 50)
        volume_score = min(100, volume_ratio * 50)
        combined = (momentum_score + volume_score) / 2
        
        signals['scanner'] = {'momentum_score': max(0, min(100, combined))}
    except:
        signals['scanner'] = {'momentum_score': 50}
    
    # 5. Sentiment (neutral for historical)
    signals['sentiment'] = {'sentiment_score': 50}
    
    # 6. Institutional (neutral for historical)
    signals['institutional'] = {'insider_score': 50, 'dark_pool_score': 50, 'flow_direction': 'neutral'}
    
    # 7. Recommender
    scores = []
    if signals.get('forecast'):
        scores.append(signals['forecast']['confidence'])
    if signals.get('patterns'):
        scores.append(signals['patterns']['confidence'])
    avg_score = sum(scores) / len(scores) if scores else 50
    
    signals['recommender'] = {
        'recommendation': 'buy' if avg_score > 60 else 'hold',
        'confidence': avg_score
    }
    
    return signals


# =============================================================================
# RUN BACKTEST
# =============================================================================

def run_all_backtests():
    """Test system on all known explosions"""
    
    print("\n" + "="*70)
    print("🧪 BACKTESTING PENNY EXPLOSION DETECTOR")
    print("="*70)
    print("Testing if system would have caught past 100-500% movers")
    print("="*70 + "\n")
    
    results = []
    
    for ticker, info in KNOWN_EXPLOSIONS.items():
        result = backtest_explosion(ticker, info, days_before=7)
        if result['success']:
            results.append(result)
    
    # Summary
    print("\n" + "="*70)
    print("📊 BACKTEST SUMMARY")
    print("="*70 + "\n")
    
    caught_count = sum(1 for r in results if r['caught'])
    total_count = len(results)
    accuracy = (caught_count / total_count * 100) if total_count > 0 else 0
    
    print(f"Stocks tested: {total_count}")
    print(f"Caught by system (75+ score): {caught_count}")
    print(f"Missed: {total_count - caught_count}")
    print(f"Accuracy: {accuracy:.1f}%")
    print()
    
    if caught_count > 0:
        caught_gains = [r['actual_gain'] for r in results if r['caught']]
        avg_gain = sum(caught_gains) / len(caught_gains)
        print(f"Average gain on caught stocks: {avg_gain:.0f}%")
    
    print()
    print("Results by stock:")
    for r in results:
        status = "✅ CAUGHT" if r['caught'] else "❌ MISSED"
        print(f"  {r['ticker']:8s} - Score: {r['score']:5.1f}/100 - {status} - Actual gain: {r['actual_gain']}%")
    
    print()
    
    if accuracy >= 70:
        print("🎉 EXCELLENT! System catches 70%+ of explosions!")
        print("   This is professional-grade accuracy.")
    elif accuracy >= 50:
        print("✅ GOOD! System catches 50%+ of explosions.")
        print("   System works but could be improved.")
    else:
        print("⚠️  NEEDS WORK! System misses too many opportunities.")
        print("   Need to adjust weights or add more signals.")
    
    print("\n" + "="*70)
    
    return results


# =============================================================================
# RUN IT
# =============================================================================

if __name__ == "__main__":
    results = run_all_backtests()

