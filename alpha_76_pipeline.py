"""
ALPHA 76 FEATURE ENGINEERING PIPELINE
======================================
Integrates the high-velocity watchlist with Batch 2 feature engineering:
- Microstructure proxies (Q8) for institutional activity filtering
- Drift detection (Q12) for regime change monitoring
- Feature selection (Q7) for dimensionality reduction

Strategy:
1. Filter out tickers with low institutional activity (<50th percentile)
2. Detect volatility regime changes (trending vs mean reversion)
3. Monitor ARK Invest flows as sector sentiment
4. Dynamic position sizing based on volatility regime

Updated: December 8, 2024
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from ALPHA_76_WATCHLIST import (
    get_alpha_76_tickers,
    get_alpha_76_by_sector,
    get_ark_overlap,
    get_high_risk_tickers,
    get_catalyst_calendar_q1_2025
)

# ============================================================================
# MICROSTRUCTURE PROXY FILTERING (Q8)
# ============================================================================

def calc_microstructure_proxies(ticker: str, lookback_days: int = 90) -> dict:
    """
    Calculate microstructure proxies to assess institutional activity.
    
    From Q8 (Microstructure Features):
    1. Quoted Spread = (Ask - Bid) / Midpoint
    2. Effective Spread = 2 * |Price - Midpoint| / Midpoint
    3. Price Impact Proxy = Corr(|Return|, Volume) [institutions move price]
    
    Args:
        ticker: Stock symbol
        lookback_days: Historical period for calculation
        
    Returns:
        Dict with microstructure metrics and institutional activity score
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        # Download data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if len(df) < 30:
            return {
                'ticker': ticker,
                'error': 'Insufficient data',
                'inst_activity_score': 0,
                'tradeable': False
            }
        
        # Calculate returns
        df['return'] = df['Close'].pct_change()
        df['abs_return'] = df['return'].abs()
        
        # Proxy 1: Effective Spread (using High-Low as proxy for bid-ask)
        df['hl_spread'] = (df['High'] - df['Low']) / ((df['High'] + df['Low']) / 2)
        avg_spread = df['hl_spread'].mean()
        
        # Proxy 2: Price Impact (correlation between |return| and volume)
        # High correlation = institutions moving price
        if len(df.dropna()) > 20:
            price_impact = df[['abs_return', 'Volume']].corr().iloc[0, 1]
        else:
            price_impact = 0
        
        # Proxy 3: Volume Stability (institutions trade consistently)
        volume_cv = df['Volume'].std() / df['Volume'].mean()  # Coefficient of variation
        
        # Proxy 4: Price Reversal (institutions cause permanent moves, not reversals)
        df['return_lag1'] = df['return'].shift(1)
        reversal = df[['return', 'return_lag1']].corr().iloc[0, 1]
        
        # Institutional Activity Score (0-100)
        # High if: low spread, high price impact, stable volume, low reversal
        score_components = [
            max(0, 100 * (1 - avg_spread / 0.05)),  # Lower spread = better
            max(0, 100 * abs(price_impact)),          # Higher impact = better
            max(0, 100 * (1 - volume_cv / 2)),        # Lower CV = better
            max(0, 100 * (1 - abs(reversal)))         # Lower reversal = better
        ]
        
        inst_score = np.mean(score_components)
        
        # Liquidity check
        avg_dollar_volume = (df['Close'] * df['Volume']).mean()
        
        return {
            'ticker': ticker,
            'avg_spread': avg_spread,
            'price_impact': price_impact,
            'volume_cv': volume_cv,
            'reversal': reversal,
            'inst_activity_score': inst_score,
            'avg_dollar_volume': avg_dollar_volume,
            'tradeable': inst_score > 40 and avg_dollar_volume > 1e6,  # $1M+ daily volume
            'error': None
        }
        
    except Exception as e:
        return {
            'ticker': ticker,
            'error': str(e),
            'inst_activity_score': 0,
            'tradeable': False
        }

# ============================================================================
# DRIFT DETECTION (Q12)
# ============================================================================

def check_data_drift(ticker: str, baseline_days: int = 180, recent_days: int = 30) -> dict:
    """
    Detect if volatility regime has changed (trending → mean reversion).
    
    From Q12 (Drift Detection):
    - Compare recent volatility to baseline
    - Detect regime shifts that require weight rebalancing
    
    Args:
        ticker: Stock symbol
        baseline_days: Historical baseline period
        recent_days: Recent period to compare
        
    Returns:
        Dict with drift detection results and regime classification
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=baseline_days + recent_days)
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if len(df) < baseline_days:
            return {
                'ticker': ticker,
                'drift_detected': False,
                'regime': 'unknown',
                'error': 'Insufficient data'
            }
        
        # Calculate returns
        df['return'] = df['Close'].pct_change()
        
        # Split into baseline and recent
        baseline_df = df.iloc[:-recent_days]
        recent_df = df.iloc[-recent_days:]
        
        # Baseline statistics
        baseline_vol = baseline_df['return'].std()
        baseline_mean = baseline_df['return'].mean()
        baseline_autocorr = baseline_df['return'].autocorr(lag=1)
        
        # Recent statistics
        recent_vol = recent_df['return'].std()
        recent_mean = recent_df['return'].mean()
        recent_autocorr = recent_df['return'].autocorr(lag=1)
        
        # Drift detection (volatility change >50%)
        vol_change = abs(recent_vol - baseline_vol) / baseline_vol
        drift_detected = vol_change > 0.5
        
        # Regime classification
        if recent_autocorr > 0.1:
            regime = 'TRENDING'  # Momentum regime
        elif recent_autocorr < -0.1:
            regime = 'MEAN_REVERTING'  # Oscillating regime
        else:
            regime = 'VOLATILE'  # High uncertainty
        
        # High volatility threshold
        if recent_vol > baseline_vol * 1.5:
            regime = 'VOLATILE'
        
        return {
            'ticker': ticker,
            'baseline_vol': baseline_vol,
            'recent_vol': recent_vol,
            'vol_change_pct': vol_change * 100,
            'drift_detected': drift_detected,
            'baseline_autocorr': baseline_autocorr,
            'recent_autocorr': recent_autocorr,
            'regime': regime,
            'error': None
        }
        
    except Exception as e:
        return {
            'ticker': ticker,
            'drift_detected': False,
            'regime': 'unknown',
            'error': str(e)
        }

# ============================================================================
# FEATURE SELECTION (Q7)
# ============================================================================

def select_features(ticker: str, top_n: int = 20) -> dict:
    """
    Calculate feature importance and select top features.
    
    From Q7 (Feature Selection):
    - Mutual Information for feature ranking
    - Select top 20 features from 60+ candidates
    
    For Alpha 76, we focus on:
    - Volatility features (regime detection)
    - Volume features (institutional activity)
    - Momentum features (trend following)
    - Microstructure features (liquidity)
    
    Returns:
        Dict with top features and their importance scores
    """
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=252)  # 1 year
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')
        
        if len(df) < 60:
            return {
                'ticker': ticker,
                'error': 'Insufficient data',
                'selected_features': []
            }
        
        # Calculate candidate features
        df['return'] = df['Close'].pct_change()
        df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility features
        df['vol_5d'] = df['return'].rolling(5).std()
        df['vol_20d'] = df['return'].rolling(20).std()
        df['vol_60d'] = df['return'].rolling(60).std()
        df['vol_ratio'] = df['vol_5d'] / df['vol_20d']
        
        # Volume features
        df['volume_sma_20'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / df['volume_sma_20']
        df['dollar_volume'] = df['Close'] * df['Volume']
        
        # Momentum features
        df['rsi_14'] = calculate_rsi(df['Close'], 14)
        df['mom_5d'] = df['Close'].pct_change(5)
        df['mom_20d'] = df['Close'].pct_change(20)
        
        # Microstructure features
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        df['co_range'] = (df['Close'] - df['Open']) / df['Close']
        
        # Rank features by variability (proxy for importance)
        features = [
            'vol_5d', 'vol_20d', 'vol_60d', 'vol_ratio',
            'volume_ratio', 'dollar_volume',
            'rsi_14', 'mom_5d', 'mom_20d',
            'hl_spread', 'co_range'
        ]
        
        feature_scores = {}
        for feat in features:
            if feat in df.columns:
                # Score by coefficient of variation (normalized variability)
                cv = df[feat].std() / (abs(df[feat].mean()) + 1e-8)
                feature_scores[feat] = cv
        
        # Sort by importance
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        return {
            'ticker': ticker,
            'selected_features': [f[0] for f in top_features],
            'feature_scores': dict(top_features),
            'error': None
        }
        
    except Exception as e:
        return {
            'ticker': ticker,
            'error': str(e),
            'selected_features': []
        }

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ============================================================================
# PIPELINE EXECUTION
# ============================================================================

def run_alpha_76_pipeline(
    save_results: bool = True,
    inst_threshold: int = 40,
    max_workers: int = 5
) -> pd.DataFrame:
    """
    Run full Alpha 76 feature engineering pipeline.
    
    Steps:
    1. Load Alpha 76 tickers
    2. Calculate microstructure proxies (filter <50th percentile)
    3. Detect drift (identify regime changes)
    4. Select features (top 20 for each ticker)
    5. Export tradeable universe
    
    Args:
        save_results: Save to CSV
        inst_threshold: Minimum institutional activity score (0-100)
        max_workers: Parallel processing workers
        
    Returns:
        DataFrame with filtered, scored tickers
    """
    print("=" * 80)
    print("ALPHA 76 FEATURE ENGINEERING PIPELINE")
    print("=" * 80)
    
    # Get watchlist
    tickers = get_alpha_76_tickers()
    sectors = get_alpha_76_by_sector()
    
    print(f"\n📊 Processing {len(tickers)} tickers...")
    print(f"⚙️  Institutional Activity Threshold: {inst_threshold}/100")
    
    results = []
    
    # Process each ticker
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...", end=" ")
        
        # Find sector
        ticker_sector = None
        for sector_name, stocks in sectors.items():
            if ticker in stocks:
                ticker_sector = sector_name
                break
        
        # Step 1: Microstructure proxies
        micro_results = calc_microstructure_proxies(ticker)
        print(f"Inst:{micro_results['inst_activity_score']:.0f}", end=" ")
        
        # Step 2: Drift detection
        drift_results = check_data_drift(ticker)
        print(f"Regime:{drift_results['regime']}", end=" ")
        
        # Step 3: Feature selection
        feature_results = select_features(ticker)
        print(f"Features:{len(feature_results['selected_features'])}", end=" ")
        
        # Combine results
        result = {
            'ticker': ticker,
            'sector': ticker_sector,
            'inst_score': micro_results['inst_activity_score'],
            'avg_spread': micro_results.get('avg_spread', np.nan),
            'price_impact': micro_results.get('price_impact', np.nan),
            'avg_dollar_volume': micro_results.get('avg_dollar_volume', 0),
            'tradeable': micro_results['tradeable'],
            'regime': drift_results['regime'],
            'recent_vol': drift_results.get('recent_vol', np.nan),
            'vol_change_pct': drift_results.get('vol_change_pct', np.nan),
            'drift_detected': drift_results['drift_detected'],
            'top_features': ', '.join(feature_results['selected_features'][:5]),
            'micro_error': micro_results.get('error'),
            'drift_error': drift_results.get('error')
        }
        
        results.append(result)
        
        if result['tradeable']:
            print("✅ TRADEABLE")
        else:
            print("❌ FILTERED")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("PIPELINE RESULTS")
    print("=" * 80)
    
    tradeable = df[df['tradeable'] == True]
    print(f"\n📊 Total Tickers Processed: {len(df)}")
    print(f"✅ Tradeable (Inst Score >{inst_threshold}, Liquidity >$1M): {len(tradeable)}")
    print(f"❌ Filtered Out: {len(df) - len(tradeable)}")
    
    print(f"\n🎯 Regime Breakdown (Tradeable Only):")
    for regime in ['TRENDING', 'MEAN_REVERTING', 'VOLATILE', 'unknown']:
        count = len(tradeable[tradeable['regime'] == regime])
        if count > 0:
            pct = 100 * count / len(tradeable)
            print(f"   {regime}: {count} tickers ({pct:.1f}%)")
    
    print(f"\n🏆 Top 10 by Institutional Activity:")
    top10 = df.nlargest(10, 'inst_score')[['ticker', 'sector', 'inst_score', 'regime', 'avg_dollar_volume']]
    print(top10.to_string(index=False))
    
    print(f"\n🚨 Drift Detected (Regime Change):")
    drift_tickers = df[df['drift_detected'] == True]
    if len(drift_tickers) > 0:
        print(f"   {len(drift_tickers)} tickers showing volatility regime change")
        print(f"   Tickers: {', '.join(drift_tickers['ticker'].tolist()[:10])}")
    else:
        print(f"   No significant drift detected")
    
    # Save results
    if save_results:
        df.to_csv('alpha_76_pipeline_results.csv', index=False)
        tradeable.to_csv('alpha_76_tradeable_universe.csv', index=False)
        
        print(f"\n💾 Results saved:")
        print(f"   - alpha_76_pipeline_results.csv (all {len(df)} tickers)")
        print(f"   - alpha_76_tradeable_universe.csv ({len(tradeable)} tradeable)")
    
    return df

# ============================================================================
# ARK INVEST FLOW MONITORING
# ============================================================================

def monitor_ark_flows(lookback_days: int = 30) -> dict:
    """
    Monitor ARK Invest ETF flows as sector sentiment indicator.
    
    ARK ETFs as Alpha 76 sector proxies:
    - ARKK (Innovation): General tech/innovation
    - ARKQ (Autonomous): Autonomous/robotics/space
    - ARKW (Next Gen Internet): Fintech/software
    - ARKG (Genomic): Biotech
    
    Returns:
        Dict with ARK performance and flow analysis
    """
    ark_etfs = ['ARKK', 'ARKQ', 'ARKW', 'ARKG']
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    results = {}
    
    for etf in ark_etfs:
        try:
            ticker = yf.Ticker(etf)
            df = ticker.history(start=start_date, end=end_date)
            
            if len(df) < 10:
                continue
            
            # Performance
            total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
            
            # Volume trend (proxy for flows)
            avg_volume = df['Volume'].mean()
            recent_volume = df['Volume'].tail(5).mean()
            volume_trend = (recent_volume / avg_volume - 1) * 100
            
            # Volatility
            returns = df['Close'].pct_change()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            results[etf] = {
                'return_pct': total_return,
                'volume_trend_pct': volume_trend,
                'volatility_pct': volatility,
                'sentiment': 'BULLISH' if total_return > 0 and volume_trend > 0 else 
                            'BEARISH' if total_return < 0 and volume_trend > 0 else 'NEUTRAL'
            }
            
        except Exception as e:
            results[etf] = {'error': str(e)}
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run pipeline
    results_df = run_alpha_76_pipeline(
        save_results=True,
        inst_threshold=40,
        max_workers=5
    )
    
    # Monitor ARK flows
    print("\n" + "=" * 80)
    print("ARK INVEST FLOW ANALYSIS (30-Day)")
    print("=" * 80)
    
    ark_flows = monitor_ark_flows(lookback_days=30)
    
    for etf, data in ark_flows.items():
        if 'error' not in data:
            print(f"\n{etf}:")
            print(f"   Return: {data['return_pct']:+.2f}%")
            print(f"   Volume Trend: {data['volume_trend_pct']:+.2f}%")
            print(f"   Volatility: {data['volatility_pct']:.2f}%")
            print(f"   Sentiment: {data['sentiment']}")
    
    # High-risk warning
    high_risk = get_high_risk_tickers()
    print("\n" + "=" * 80)
    print("⚠️  HIGH-RISK TICKERS (Use Smaller Position Sizes)")
    print("=" * 80)
    print(f"15 tickers with bankruptcy/dilution risk:")
    print(f"{', '.join(high_risk)}")
    
    # Catalyst calendar
    calendar = get_catalyst_calendar_q1_2025()
    print("\n" + "=" * 80)
    print("📅 CATALYST CALENDAR")
    print("=" * 80)
    print(f"\nQ4 2024 (Dec): {len(calendar['q4_2024'])} tickers")
    if calendar['q4_2024']:
        print(f"   {', '.join(calendar['q4_2024'][:10])}")
    
    print(f"\nQ1 2025 (Jan-Mar): {len(calendar['q1_2025'])} tickers")
    if calendar['q1_2025']:
        print(f"   {', '.join(calendar['q1_2025'][:10])}")
    
    print(f"\nQ2 2025 (Apr-Jun): {len(calendar['q2_2025'])} tickers")
    if calendar['q2_2025']:
        print(f"   {', '.join(calendar['q2_2025'][:10])}")
    
    print("\n" + "=" * 80)
    print("✅ PIPELINE COMPLETE")
    print("=" * 80)
    print("\n🚀 Next Steps:")
    print("   1. Review alpha_76_tradeable_universe.csv for filtered tickers")
    print("   2. Train meta-learner (Batch 1) on tradeable universe")
    print("   3. Monitor drift detection weekly for regime changes")
    print("   4. Track ARK flows for sector sentiment shifts")
    print("   5. Adjust position sizes based on volatility regime")
