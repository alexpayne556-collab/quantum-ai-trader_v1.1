"""
EXPLORE WATCHLIST - Analyze the 20 tickers
"""
import pandas as pd
import numpy as np
from pathlib import Path
from WATCHLIST_2026 import WATCHLIST, SECTORS

DATA_DIR = Path("data/watchlist_2026")

def load_all_data():
    """Load all ticker data into dict"""
    data = {}
    for ticker in WATCHLIST:
        path = DATA_DIR / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            # Handle multi-level columns from yfinance
            if isinstance(df.columns[0], tuple):
                df.columns = [c[0] for c in df.columns]
            data[ticker] = df
    return data

def analyze_ticker(ticker, df):
    """Analyze single ticker"""
    # Handle column names (could be 'Close' or 'close')
    close_col = 'Close' if 'Close' in df.columns else 'close'
    
    close = df[close_col]
    returns = close.pct_change().dropna()
    
    stats = {
        'ticker': ticker,
        'days': len(df),
        'start': df.index[0].strftime('%Y-%m-%d'),
        'end': df.index[-1].strftime('%Y-%m-%d'),
        'price_now': close.iloc[-1],
        'price_1y_ago': close.iloc[0] if len(close) > 252 else close.iloc[0],
        'return_total': (close.iloc[-1] / close.iloc[0] - 1) * 100,
        'volatility': returns.std() * np.sqrt(252) * 100,
        'max_drawdown': ((close / close.cummax()) - 1).min() * 100,
        'sharpe': (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0,
        'best_day': returns.max() * 100,
        'worst_day': returns.min() * 100,
    }
    return stats

def correlation_matrix(data):
    """Calculate correlation between all tickers"""
    returns = pd.DataFrame()
    for ticker, df in data.items():
        close_col = 'Close' if 'Close' in df.columns else 'close'
        returns[ticker] = df[close_col].pct_change()
    
    return returns.corr()

def main():
    print("="*70)
    print("WATCHLIST EXPLORER - 20 Tickers Analysis")
    print("="*70)
    
    data = load_all_data()
    print(f"\nLoaded {len(data)} tickers")
    
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
            print(f"\n  {row['ticker']}:")
            print(f"    Price: ${row['price_now']:.2f}")
            print(f"    Total Return: {row['return_total']:+.1f}%")
            print(f"    Volatility: {row['volatility']:.1f}%")
            print(f"    Max Drawdown: {row['max_drawdown']:.1f}%")
            print(f"    Sharpe: {row['sharpe']:.2f}")
            print(f"    Best Day: {row['best_day']:+.1f}%")
            print(f"    Worst Day: {row['worst_day']:.1f}%")
    
    # Top movers
    print("\n" + "="*70)
    print("TOP PERFORMERS (Total Return)")
    print("="*70)
    top = df_stats.nlargest(5, 'return_total')
    for _, row in top.iterrows():
        print(f"  {row['ticker']}: {row['return_total']:+.1f}%")
    
    print("\n" + "="*70)
    print("MOST VOLATILE (Annualized)")
    print("="*70)
    volatile = df_stats.nlargest(5, 'volatility')
    for _, row in volatile.iterrows():
        print(f"  {row['ticker']}: {row['volatility']:.1f}%")
    
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
    
    print("\nMost Correlated Pairs:")
    for t1, t2, c in high_corr[:5]:
        print(f"  {t1} <-> {t2}: {c:.2f}")
    
    print("\nLeast Correlated Pairs (good for diversification):")
    for t1, t2, c in high_corr[-5:]:
        print(f"  {t1} <-> {t2}: {c:.2f}")
    
    # Save results
    df_stats.to_csv(DATA_DIR / "watchlist_analysis.csv", index=False)
    corr.to_csv(DATA_DIR / "correlation_matrix.csv")
    print(f"\nSaved analysis to {DATA_DIR}")

if __name__ == "__main__":
    main()
