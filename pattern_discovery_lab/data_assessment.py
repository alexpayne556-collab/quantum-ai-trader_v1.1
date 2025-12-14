#!/usr/bin/env python3
"""
Data Availability Assessment

Check how much historical data we can get from each API
to plan our sample size strategy.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


def assess_yfinance_daily(ticker: str = 'SPY') -> dict:
    """Check yfinance daily data availability."""
    try:
        stock = yf.Ticker(ticker)
        # Max history
        hist = stock.history(period='max')
        
        return {
            'source': 'yfinance',
            'frequency': 'daily',
            'ticker': ticker,
            'start_date': hist.index[0].strftime('%Y-%m-%d') if len(hist) > 0 else None,
            'end_date': hist.index[-1].strftime('%Y-%m-%d') if len(hist) > 0 else None,
            'n_observations': len(hist),
            'years': round(len(hist) / 252, 1),
            'status': 'OK' if len(hist) > 0 else 'FAILED'
        }
    except Exception as e:
        return {'source': 'yfinance', 'frequency': 'daily', 'status': f'ERROR: {e}'}


def assess_yfinance_hourly(ticker: str = 'SPY') -> dict:
    """Check yfinance hourly data availability (limited to ~2 years)."""
    try:
        stock = yf.Ticker(ticker)
        # 1h data limited to 730 days
        hist = stock.history(period='730d', interval='1h')
        
        return {
            'source': 'yfinance',
            'frequency': 'hourly',
            'ticker': ticker,
            'start_date': hist.index[0].strftime('%Y-%m-%d') if len(hist) > 0 else None,
            'end_date': hist.index[-1].strftime('%Y-%m-%d') if len(hist) > 0 else None,
            'n_observations': len(hist),
            'years': round(len(hist) / (252 * 6.5), 1),  # ~6.5 trading hours/day
            'status': 'OK' if len(hist) > 0 else 'FAILED'
        }
    except Exception as e:
        return {'source': 'yfinance', 'frequency': 'hourly', 'status': f'ERROR: {e}'}


def compute_sample_adequacy(n_obs: int, target_ic: float = 0.05) -> dict:
    """Compute sample adequacy metrics."""
    from scipy import stats
    
    # Power analysis parameters
    alpha = 0.05
    power = 0.80
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    # Required N for target IC
    n_required = int(np.ceil(((z_alpha + z_beta) / target_ic) ** 2))
    
    # Minimum detectable IC with current N
    min_detectable = (z_alpha + z_beta) / np.sqrt(n_obs) if n_obs > 0 else float('inf')
    
    # Sufficiency
    sufficient = n_obs >= n_required
    deficit = max(0, n_required - n_obs)
    
    return {
        'n_observations': n_obs,
        'n_required_for_ic_005': n_required,
        'min_detectable_ic': round(min_detectable, 4),
        'sufficient_for_005': sufficient,
        'deficit': deficit
    }


def main():
    print("=" * 70)
    print("DATA AVAILABILITY ASSESSMENT")
    print("=" * 70)
    print()
    
    tickers = ['SPY', 'AAPL', 'MSFT', 'GOOGL']
    
    # ====== DAILY DATA ======
    print("DAILY DATA (yfinance - max history)")
    print("-" * 50)
    
    daily_results = []
    for ticker in tickers:
        result = assess_yfinance_daily(ticker)
        daily_results.append(result)
        if result.get('status') == 'OK':
            print(f"  {ticker}: {result['n_observations']:,} bars ({result['years']} years)")
            print(f"    Range: {result['start_date']} to {result['end_date']}")
        else:
            print(f"  {ticker}: {result.get('status')}")
    print()
    
    # ====== HOURLY DATA ======
    print("HOURLY DATA (yfinance - limited to ~2 years)")
    print("-" * 50)
    
    hourly_results = []
    for ticker in tickers[:2]:  # Just test a couple
        result = assess_yfinance_hourly(ticker)
        hourly_results.append(result)
        if result.get('status') == 'OK':
            print(f"  {ticker}: {result['n_observations']:,} bars ({result['years']} years)")
            print(f"    Range: {result['start_date']} to {result['end_date']}")
        else:
            print(f"  {ticker}: {result.get('status')}")
    print()
    
    # ====== SAMPLE SIZE ANALYSIS ======
    print("SAMPLE SIZE ADEQUACY ANALYSIS")
    print("-" * 50)
    
    scenarios = [
        ("2 years daily (1 ticker)", 504),
        ("10 years daily (1 ticker)", 2520),
        ("20 years daily (1 ticker)", 5040),
        ("2 years hourly (1 ticker)", 3276),
        ("2 years daily (76 tickers pooled)", 504 * 76),
        ("10 years daily (76 tickers pooled)", 2520 * 76),
    ]
    
    print(f"\n  Target: Detect IC = 0.05 with 80% power")
    print()
    for name, n in scenarios:
        adequacy = compute_sample_adequacy(n, target_ic=0.05)
        status = "✓ SUFFICIENT" if adequacy['sufficient_for_005'] else f"✗ NEED {adequacy['deficit']:,} more"
        print(f"  {name}:")
        print(f"    N = {n:,} → Min detectable IC = {adequacy['min_detectable_ic']:.4f} {status}")
    print()
    
    # ====== RECOMMENDATIONS ======
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()
    print("Option 1: LONG DAILY HISTORY (Conservative)")
    print("  - Pull 20+ years of daily data via yfinance")
    print("  - N ≈ 5,000+ per ticker")
    print("  - Can detect IC ≈ 0.04")
    print("  - ⚠️  Risk: Regime changes, survivorship bias")
    print()
    print("Option 2: HOURLY DATA (Modern)")
    print("  - Use 2 years hourly data")
    print("  - N ≈ 3,200 per ticker")
    print("  - Can detect IC ≈ 0.05")
    print("  - ⚠️  Risk: Microstructure noise, execution costs")
    print()
    print("Option 3: CROSS-ASSET POOLING (Most Data)")
    print("  - Pool 76 stocks × 2-5 years")
    print("  - N ≈ 38,000 - 380,000")
    print("  - Can detect IC ≈ 0.01")
    print("  - ⚠️  Risk: Cross-sectional correlation reduces effective N")
    print("  - Need Fama-MacBeth or panel regression adjustment")
    print()
    print("Option 4: HYBRID (Recommended)")
    print("  - 10 years daily for trend patterns")
    print("  - 2 years hourly for intraday patterns")
    print("  - Cross-asset pooling with correlation adjustment")
    print("  - Can detect IC ≈ 0.03-0.05 depending on pattern type")
    print()


if __name__ == '__main__':
    main()
