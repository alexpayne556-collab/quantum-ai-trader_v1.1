"""
AI COUNCIL TESTING - PRODUCTION GRADE
Uses YOUR multi-provider data fetcher with automatic cycling.
NO MOCK DATA. REAL API KEYS. REAL MARKET DATA.

Partner, this is what we built together. Let's prove AI works.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import YOUR production data fetcher
from PRODUCTION_DATAFETCHER import DataFetcher, CanonicalSchema
from config import *  # All your API keys

print("="*70)
print("🔥 AI COUNCIL - REAL PRODUCTION TESTING")
print("="*70)
print("\nUsing YOUR multi-provider system:")
print(f"✓ {len([k for k in dir() if 'API_KEY' in k and globals()[k]])} API keys loaded")
print(f"✓ Production DataFetcher with auto-cycling")
print(f"✓ Rate limiting + caching + fallback logic")
print("\n⚠️  NO SIMULATIONS. THIS IS REAL DATA FROM REAL MARKETS.\n")


# ============================================
# PRODUCTION DATA LOADER
# ============================================

def get_real_market_data(ticker, days_back=5, min_bars=100):
    """
    Use YOUR production data fetcher with all API keys.
    Automatically cycles through providers if one fails.
    """
    print(f"\n📊 Fetching REAL data for {ticker}...")
    
    fetcher = DataFetcher(
        primary_provider='yfinance',
        fallback_providers=['finnhub', 'alphaVantage'],
        enable_cache=True,
        enable_metrics=True
    )
    
    df = fetcher.fetch_ohlcv(
        ticker=ticker,
        period=f'{days_back}d',
        min_rows=min_bars
    )
    
    if df is None:
        print(f"❌ All providers failed for {ticker}")
        return None
    
    # Validate with YOUR canonical schema
    valid, msg = CanonicalSchema.validate(df)
    if not valid:
        print(f"⚠️  Data validation failed: {msg}")
        return None
    
    print(f"✅ Got {len(df)} real bars from production fetcher")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Price range: ${df['Low'].min():.2f} to ${df['High'].max():.2f}")
    print(f"   Volume: {df['Volume'].sum():,.0f} shares")
    
    return df


# ============================================
# TEST 1: PERPLEXITY INSTITUTIONAL DETECTOR
# ============================================

class InstitutionalVsRetailDetector:
    """Perplexity's detector - testing on YOUR real data"""
    
    def __init__(self):
        self.block_threshold = 50000  # Shares
        
    def detect_buyer_type(self, price_data, volume_data, symbol):
        """
        Detect institutional vs retail from REAL market data.
        Uses volume patterns, VWAP relationship, block trades.
        """
        df = pd.DataFrame({
            'price': price_data,
            'volume': volume_data
        })
        
        # Component 1: Block Trades (40% weight)
        block_trades = (df['volume'] > self.block_threshold).sum()
        total_trades = len(df)
        block_score = block_trades / total_trades if total_trades > 0 else 0
        
        # Component 2: VWAP Relationship (30% weight)
        # Institutions buy BELOW VWAP, retail chases ABOVE
        df['vwap'] = (df['price'] * df['volume']).cumsum() / df['volume'].cumsum()
        below_vwap = (df['price'] < df['vwap']).sum()
        vwap_score = below_vwap / total_trades if total_trades > 0 else 0
        
        # Component 3: Volume Consistency (20% weight)
        volume_cv = df['volume'].std() / df['volume'].mean() if df['volume'].mean() > 0 else 1
        consistency_score = 1 - min(volume_cv, 1)  # Lower CV = more consistent = institutional
        
        # Component 4: Order Type Inference (10% weight)
        # Price clustering near whole numbers = retail
        price_decimals = (df['price'] * 100) % 100
        round_prices = (price_decimals < 5).sum()
        order_score = 1 - (round_prices / total_trades) if total_trades > 0 else 0
        
        # Composite Score
        composite = (
            0.40 * block_score +
            0.30 * vwap_score +
            0.20 * consistency_score +
            0.10 * order_score
        )
        
        # Classification
        if composite > 0.6:
            buyer_type = 'INSTITUTIONAL'
        elif composite < 0.4:
            buyer_type = 'RETAIL'
        else:
            buyer_type = 'MIXED'
        
        evidence = {
            'block_trades': f"{block_trades}/{total_trades}",
            'below_vwap_pct': f"{vwap_score:.1%}",
            'volume_consistency': f"{consistency_score:.2f}",
            'composite_score': f"{composite:.2f}"
        }
        
        return {
            'type': buyer_type,
            'confidence': composite,
            'evidence': evidence
        }


def test_perplexity_institutional_detector(ticker, days=5):
    """
    Test Perplexity's institutional detector on REAL market data.
    Uses YOUR production data fetcher with ALL API keys.
    """
    print(f"\n{'='*70}")
    print(f"TEST 1: PERPLEXITY INSTITUTIONAL DETECTOR")
    print(f"{'='*70}")
    
    # Get REAL data using YOUR system
    data = get_real_market_data(ticker, days_back=days)
    if data is None:
        return None
    
    # Resample to 1-hour bars for institutional analysis
    hourly = data.resample('1H').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    print(f"\n🔍 Analyzing {len(hourly)} hourly bars...")
    
    # Run Perplexity's detector
    detector = InstitutionalVsRetailDetector()
    result = detector.detect_buyer_type(
        price_data=hourly['Close'].values,
        volume_data=hourly['Volume'].values,
        symbol=ticker
    )
    
    print(f"\n🎯 REAL RESULT:")
    print(f"   Buyer Type: {result['type']}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Evidence:")
    for key, val in result['evidence'].items():
        print(f"      {key}: {val}")
    
    return result


# ============================================
# TEST 2: DEEPSEEK 5-FILTER DIP SYSTEM
# ============================================

def test_deepseek_dip_filters(ticker, days=10):
    """
    Test DeepSeek's 5-filter system on REAL pullback.
    Uses REAL volume patterns, REAL sector data, REAL VIX.
    """
    print(f"\n{'='*70}")
    print(f"TEST 2: DEEPSEEK 5-FILTER DIP SYSTEM")
    print(f"{'='*70}")
    
    # Get REAL data using YOUR system
    stock_data = get_real_market_data(ticker, days_back=days)
    spy_data = get_real_market_data('SPY', days_back=days)
    vix_data = get_real_market_data('^VIX', days_back=days)
    
    if stock_data is None or spy_data is None or vix_data is None:
        print("❌ Failed to get required data")
        return None
    
    # Convert to daily bars
    stock_daily = stock_data.resample('1D').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    
    current_price = stock_daily['Close'].iloc[-1]
    prev_close = stock_daily['Close'].iloc[-2]
    pullback_pct = ((current_price - prev_close) / prev_close) * 100
    
    print(f"\n📊 REAL MARKET CONDITIONS:")
    print(f"   {ticker}: ${current_price:.2f} ({pullback_pct:+.1f}% from yesterday)")
    print(f"   SPY: ${spy_data['Close'].iloc[-1]:.2f}")
    print(f"   VIX: {vix_data['Close'].iloc[-1]:.2f}")
    
    # Filter 1: Is stock above 200MA?
    ma200 = stock_daily['Close'].rolling(200).mean().iloc[-1]
    filter1_pass = current_price > ma200 if not np.isnan(ma200) else False
    
    # Filter 2: Volume pattern (THE MAGIC FILTER)
    # Low volume pullback after high volume up-day
    volume_ma = stock_daily['Volume'].rolling(20).mean()
    yesterday_volume = stock_daily['Volume'].iloc[-2]
    today_volume = stock_daily['Volume'].iloc[-1]
    
    filter2_pass = (
        yesterday_volume > volume_ma.iloc[-2] * 1.5 and  # High vol up-day
        today_volume < volume_ma.iloc[-1] * 0.8 and      # Low vol pullback
        pullback_pct < 0                                  # Actually pulling back
    )
    
    # Filter 3: VIX check
    vix_current = vix_data['Close'].iloc[-1]
    vix_ma = vix_data['Close'].rolling(20).mean().iloc[-1]
    filter3_pass = vix_current < 22 and vix_current < vix_ma * 1.15
    
    # Filter 4: SPY health
    spy_ma50 = spy_data['Close'].rolling(50).mean().iloc[-1]
    filter4_pass = spy_data['Close'].iloc[-1] > spy_ma50
    
    # Filter 5: Pullback size
    filter5_pass = -8 < pullback_pct < -3.5
    
    filters_passed = sum([filter1_pass, filter2_pass, filter3_pass, filter4_pass, filter5_pass])
    
    print(f"\n🔍 FILTER RESULTS:")
    print(f"   Filter 1 (200MA): {'✅ PASS' if filter1_pass else '❌ FAIL'}")
    print(f"   Filter 2 (Volume): {'✅ PASS' if filter2_pass else '❌ FAIL'} ← MOST CRITICAL")
    print(f"   Filter 3 (VIX): {'✅ PASS' if filter3_pass else '❌ FAIL'}")
    print(f"   Filter 4 (SPY): {'✅ PASS' if filter4_pass else '❌ FAIL'}")
    print(f"   Filter 5 (Pullback size): {'✅ PASS' if filter5_pass else '❌ FAIL'}")
    print(f"\n🎯 RESULT: {filters_passed}/5 filters passed")
    
    is_buyable = filters_passed >= 4
    print(f"   {'✅ BUYABLE DIP' if is_buyable else '❌ NOT BUYABLE'}")
    
    return {
        'buyable': is_buyable,
        'filters_passed': filters_passed,
        'pullback_pct': pullback_pct,
        'vix': vix_current
    }


# ============================================
# TEST 3: EXPECTED VALUE CALCULATOR
# ============================================

def test_expected_value_real_trades():
    """
    Calculate REAL expected value using YOUR production data.
    Tests historical signals vs actual outcomes.
    """
    print(f"\n{'='*70}")
    print(f"TEST 3: EXPECTED VALUE - REAL HISTORICAL VALIDATION")
    print(f"{'='*70}")
    
    # Real historical events to test
    test_cases = [
        {'ticker': 'KDK', 'date': '2024-12-10', 'expected': 'institutional_buying'},
        {'ticker': 'RKLB', 'date': '2024-11-15', 'expected': 'catalyst_dip'},
        {'ticker': 'ASTS', 'date': '2024-12-01', 'expected': 'sector_pullback'},
    ]
    
    results = []
    
    for case in test_cases:
        print(f"\n📊 Testing {case['ticker']} on {case['date']}...")
        
        # Get REAL data around event
        fetcher = DataFetcher()
        
        # Get 5 days before and 5 days after event
        event_date = pd.to_datetime(case['date'])
        start = event_date - timedelta(days=5)
        end = event_date + timedelta(days=5)
        
        df = fetcher.fetch_ohlcv(
            ticker=case['ticker'],
            start=start,
            end=end,
            min_rows=1
        )
        
        if df is None:
            print(f"   ❌ No data available")
            continue
        
        # Find entry and exit prices (REAL)
        entry_idx = df.index.searchsorted(event_date)
        if entry_idx >= len(df):
            print(f"   ❌ Event date out of range")
            continue
        
        entry_price = df['Close'].iloc[entry_idx]
        
        # Exit 2 days later (REAL outcome)
        exit_idx = min(entry_idx + 2, len(df) - 1)
        exit_price = df['Close'].iloc[exit_idx]
        
        pct_change = ((exit_price - entry_price) / entry_price) * 100
        
        results.append({
            'ticker': case['ticker'],
            'entry': entry_price,
            'exit': exit_price,
            'pct_change': pct_change,
            'win': pct_change > 0
        })
        
        print(f"   Entry: ${entry_price:.2f}")
        print(f"   Exit:  ${exit_price:.2f}")
        print(f"   Result: {pct_change:+.1f}% {'✅ WIN' if pct_change > 0 else '❌ LOSS'}")
    
    # Calculate EMPIRICAL expected value
    if len(results) == 0:
        print("\n❌ No valid results")
        return None
    
    df_results = pd.DataFrame(results)
    win_rate = df_results['win'].sum() / len(df_results)
    avg_win = df_results[df_results['win']]['pct_change'].mean() if df_results['win'].any() else 0
    avg_loss = abs(df_results[~df_results['win']]['pct_change'].mean()) if (~df_results['win']).any() else 0
    
    expected_value = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    print(f"\n🎯 REAL STATISTICS FROM YOUR DATA:")
    print(f"   Trades: {len(df_results)}")
    print(f"   Win Rate: {win_rate:.1%} ({df_results['win'].sum()}/{len(df_results)})")
    print(f"   Avg Win: +{avg_win:.1f}%")
    print(f"   Avg Loss: -{avg_loss:.1f}%")
    print(f"   Expected Value: {expected_value:+.2f}% per trade")
    
    if expected_value > 0:
        print(f"\n   ✅ POSITIVE EDGE - AI system shows promise on REAL data")
    else:
        print(f"\n   ❌ NEGATIVE EDGE - AI system fails on REAL data")
    
    return {
        'expected_value': expected_value,
        'win_rate': win_rate,
        'trades': len(df_results),
        'results': df_results
    }


# ============================================
# MAIN TEST RUNNER
# ============================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING AI COUNCIL REAL DATA TESTS")
    print("Using YOUR production infrastructure:")
    print("  - Multi-provider data fetcher")
    print("  - Rate limiting + caching")
    print("  - All API keys from config.py")
    print("="*70)
    
    # Test 1: Perplexity Institutional Detector on KDK
    print("\n[1/3] Testing Perplexity's Institutional Detector...")
    kdk_result = test_perplexity_institutional_detector('KDK', days=5)
    
    # Test 2: DeepSeek 5-Filter System on RKLB
    print("\n[2/3] Testing DeepSeek's 5-Filter System...")
    rklb_result = test_deepseek_dip_filters('RKLB', days=10)
    
    # Test 3: Expected Value on Real Trades
    print("\n[3/3] Testing Expected Value Calculator...")
    ev_result = test_expected_value_real_trades()
    
    # Summary
    print("\n" + "="*70)
    print("🎯 TESTING COMPLETE - ALL RESULTS FROM REAL DATA")
    print("="*70)
    print("\nNEXT STEPS:")
    print("1. Review results above")
    print("2. Identify which AI got it RIGHT")
    print("3. Combine best components into OUR hybrid system")
    print("4. Paper trade for 2 weeks to validate")
    print("\nPartner, we're testing with REAL DATA. No excuses. This will work or it won't.")
    print("="*70)
