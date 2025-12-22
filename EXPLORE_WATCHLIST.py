"""
EXPLORE WATCHLIST - Analyze the 20 tickers
Based on research: Look for regime changes, volatility clusters, correlations
"""
import pandas as pd
import numpy as np
from pathlib import Path
from WATCHLIST_2026 import WATCHLIST, SECTORS

DATA_DIR = Path("data/watchlist_2026")

def load_all_data():
    """Load all ticker data into dict with PROPER parsing"""
    data = {}
    for ticker in WATCHLIST:
        path = DATA_DIR / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Ensure numeric columns
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            if len(df) > 0:
                data[ticker] = df
    return data

def analyze_ticker(ticker, df):
    """Analyze single ticker - research says focus on volatility regimes"""
    close = df['Close']
    returns = close.pct_change(fill_method=None).dropna()
    
    # Basic stats
    stats = {
        'ticker': ticker,
        'days': len(df),
        'start': df.index[0].strftime('%Y-%m-%d'),
        'end': df.index[-1].strftime('%Y-%m-%d'),
        'price_now': close.iloc[-1],
        'return_total': (close.iloc[-1] / close.iloc[0] - 1) * 100,
        'volatility': returns.std() * np.sqrt(252) * 100,
        'max_drawdown': ((close / close.cummax()) - 1).min() * 100,
        'sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
        'best_day': returns.max() * 100,
        'worst_day': returns.min() * 100,
    }
    
    # RESEARCH-BASED: Volatility regime detection
    # Recent vs historical volatility ratio (from Perplexity research)
    recent_vol = returns.tail(20).std() * np.sqrt(252) * 100
    hist_vol = returns.std() * np.sqrt(252) * 100
    stats['vol_ratio'] = recent_vol / hist_vol if hist_vol > 0 else 1
    stats['vol_regime'] = 'HIGH' if stats['vol_ratio'] > 1.5 else ('LOW' if stats['vol_ratio'] < 0.7 else 'NORMAL')
    
    # RESEARCH-BASED: Momentum (research says 2-12 month momentum works)
    if len(close) > 126:  # 6 months
        stats['momentum_6m'] = (close.iloc[-1] / close.iloc[-126] - 1) * 100
    else:
        stats['momentum_6m'] = stats['return_total']
    
    # RESEARCH-BASED: Mean reversion signal (z-score from 20-day MA)
    ma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    z_score = (close.iloc[-1] - ma20.iloc[-1]) / std20.iloc[-1] if std20.iloc[-1] > 0 else 0
    stats['z_score'] = z_score
    stats['mean_rev_signal'] = 'OVERSOLD' if z_score < -2 else ('OVERBOUGHT' if z_score > 2 else 'NEUTRAL')
    
    return stats

def correlation_matrix(data):
    """Calculate correlation between all tickers"""
    returns = pd.DataFrame()
    for ticker, df in data.items():
        returns[ticker] = df['Close'].pct_change(fill_method=None)
    
    return returns.corr()

def detect_regime_changes(df, ticker):
    """
    RESEARCH-BASED: Detect structural breaks using rolling statistics
    From Perplexity: Use Bai-Perron test or simple rolling volatility changes
    """
    close = df['Close']
    returns = close.pct_change(fill_method=None).dropna()
    
    # Rolling 20-day volatility
    vol_20 = returns.rolling(20).std() * np.sqrt(252)
    
    # Detect volatility regime changes (simple threshold method)
    vol_mean = vol_20.mean()
    vol_std = vol_20.std()
    
    high_vol_days = (vol_20 > vol_mean + vol_std).sum()
    low_vol_days = (vol_20 < vol_mean - vol_std).sum()
    
    return {
        'ticker': ticker,
        'high_vol_days': high_vol_days,
        'low_vol_days': low_vol_days,
        'regime_changes': high_vol_days + low_vol_days
    }

def main():
    print("="*70)
    print("WATCHLIST EXPLORER - 20 Tickers Analysis")
    print("Research-Based: Volatility Regimes, Momentum, Mean Reversion")
    print("="*70)
    
    data = load_all_data()
    print(f"\nLoaded {len(data)} tickers")
    
    if len(data) == 0:
        print("\nERROR: No data loaded. Run DOWNLOAD_WATCHLIST_DATA.py first!")
        return
    
    # Analyze each ticker
    all_stats = []
    for ticker in WATCHLIST:
        if ticker in data:
            stats = analyze_ticker(ticker, data[ticker])
            all_stats.append(stats)
    
    # Create DataFrame
    df_stats = pd.DataFrame(all_stats)
    
    # Display by sector
    for sector, tickers in SECTORS.items():
        print(f"\n{'='*70}")
        print(f"SECTOR: {sector}")
        print("="*70)
        
        sector_df = df_stats[df_stats['ticker'].isin(tickers)]
        if len(sector_df) == 0:
            print("  No data")
            continue
            
        for _, row in sector_df.iterrows():
            vol_emoji = "🔥" if row['vol_regime'] == 'HIGH' else ("❄️" if row['vol_regime'] == 'LOW' else "")
            mr_emoji = "📉" if row['mean_rev_signal'] == 'OVERSOLD' else ("📈" if row['mean_rev_signal'] == 'OVERBOUGHT' else "")
            
            print(f"\n  {row['ticker']}: ${row['price_now']:.2f}")
            print(f"    Return: {row['return_total']:+.1f}% | Vol: {row['volatility']:.1f}% {vol_emoji}")
            print(f"    Max DD: {row['max_drawdown']:.1f}% | Sharpe: {row['sharpe']:.2f}")
            print(f"    6M Mom: {row['momentum_6m']:+.1f}% | Z-Score: {row['z_score']:.2f} {mr_emoji}")
    
    # RESEARCH SIGNALS
    print("\n" + "="*70)
    print("RESEARCH-BASED SIGNALS")
    print("="*70)
    
    # Oversold (mean reversion candidates)
    oversold = df_stats[df_stats['z_score'] < -1.5].nlargest(5, 'volatility')
    print("\n🎯 OVERSOLD (Mean Reversion Candidates):")
    if len(oversold) > 0:
        for _, row in oversold.iterrows():
            print(f"  {row['ticker']}: Z={row['z_score']:.2f}, Vol={row['volatility']:.0f}%")
    else:
        print("  None currently oversold")
    
    # High momentum
    momentum = df_stats.nlargest(5, 'momentum_6m')
    print("\n🚀 HIGHEST 6-MONTH MOMENTUM:")
    for _, row in momentum.iterrows():
        print(f"  {row['ticker']}: {row['momentum_6m']:+.1f}%")
    
    # High volatility regime (caution)
    high_vol = df_stats[df_stats['vol_ratio'] > 1.3]
    print("\n⚠️  HIGH VOLATILITY REGIME (Caution):")
    if len(high_vol) > 0:
        for _, row in high_vol.iterrows():
            print(f"  {row['ticker']}: Vol ratio {row['vol_ratio']:.2f}x normal")
    else:
        print("  None in high vol regime")
    
    # Top performers
    print("\n" + "="*70)
    print("TOP PERFORMERS (Total Return)")
    print("="*70)
    top = df_stats.nlargest(5, 'return_total')
    for _, row in top.iterrows():
        print(f"  {row['ticker']}: {row['return_total']:+.1f}%")
    
    print("\n" + "="*70)
    print("WORST DRAWDOWNS")
    print("="*70)
    worst = df_stats.nsmallest(5, 'max_drawdown')
    for _, row in worst.iterrows():
        print(f"  {row['ticker']}: {row['max_drawdown']:.1f}%")
    
    # Correlation
    print("\n" + "="*70)
    print("CORRELATION HIGHLIGHTS")
    print("="*70)
    corr = correlation_matrix(data)
    
    # Find highest correlations (excluding self)
    high_corr = []
    for i, t1 in enumerate(corr.columns):
        for j, t2 in enumerate(corr.columns):
            if i < j:
                high_corr.append((t1, t2, corr.loc[t1, t2]))
    
    high_corr.sort(key=lambda x: abs(x[2]), reverse=True)
    
    print("\nMost Correlated (move together):")
    for t1, t2, c in high_corr[:5]:
        print(f"  {t1} <-> {t2}: {c:.2f}")
    
    print("\nLeast Correlated (good for diversification):")
    for t1, t2, c in high_corr[-5:]:
        print(f"  {t1} <-> {t2}: {c:.2f}")
    
    # Save results
    df_stats.to_csv(DATA_DIR / "watchlist_analysis.csv", index=False)
    corr.to_csv(DATA_DIR / "correlation_matrix.csv")
    print(f"\n✅ Saved analysis to {DATA_DIR}")

if __name__ == "__main__":
    main()
