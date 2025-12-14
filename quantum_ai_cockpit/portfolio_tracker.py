"""Portfolio tracker for monitoring current holdings."""
from __future__ import annotations
import yfinance as yf
import pandas as pd
from typing import List, Dict
from .signals import calculate_signals, score_setup


def get_current_price(ticker: str) -> float:
    """Fetch current price for a ticker."""
    data = yf.download(ticker, period="1d", progress=False)
    if data.empty:
        raise ValueError(f"Could not fetch price for {ticker}")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return float(data['Close'].iloc[-1])


def track_position(ticker: str, shares: float, cost_basis: float) -> dict:
    """Track a single position with P&L calculations."""
    current_price = get_current_price(ticker)
    market_value = shares * current_price
    cost_value = shares * cost_basis
    pnl_dollars = market_value - cost_value
    pnl_percent = ((current_price - cost_basis) / cost_basis) * 100
    
    return {
        'ticker': ticker,
        'shares': shares,
        'cost_basis': round(cost_basis, 2),
        'current_price': round(current_price, 2),
        'market_value': round(market_value, 2),
        'cost_value': round(cost_value, 2),
        'pnl_dollars': round(pnl_dollars, 2),
        'pnl_percent': round(pnl_percent, 2),
        'status': 'PROFIT' if pnl_dollars > 0 else 'LOSS'
    }


def track_portfolio(holdings: List[Dict]) -> pd.DataFrame:
    """
    Track entire portfolio.
    
    holdings format: [
        {'ticker': 'APLD', 'shares': 2.99, 'cost_basis': 20.00},
        {'ticker': 'MU', 'shares': 1.09, 'cost_basis': 95.00},
    ]
    """
    results = []
    for h in holdings:
        try:
            position = track_position(h['ticker'], h['shares'], h['cost_basis'])
            results.append(position)
        except Exception as e:
            print(f"Error tracking {h['ticker']}: {e}")
    
    df = pd.DataFrame(results)
    return df


def portfolio_summary(df: pd.DataFrame) -> dict:
    """Calculate portfolio-level summary stats."""
    if df.empty:
        return {}
    
    total_value = df['market_value'].sum()
    total_cost = df['cost_value'].sum()
    total_pnl = df['pnl_dollars'].sum()
    total_pnl_pct = ((total_value - total_cost) / total_cost) * 100
    
    return {
        'total_positions': len(df),
        'total_market_value': round(total_value, 2),
        'total_cost_basis': round(total_cost, 2),
        'total_pnl_dollars': round(total_pnl, 2),
        'total_pnl_percent': round(total_pnl_pct, 2),
        'winners': len(df[df['pnl_dollars'] > 0]),
        'losers': len(df[df['pnl_dollars'] < 0])
    }


def print_portfolio(holdings: List[Dict]) -> None:
    """Pretty print portfolio status."""
    df = track_portfolio(holdings)
    summary = portfolio_summary(df)
    
    print("=" * 60)
    print("📊 PORTFOLIO TRACKER")
    print("=" * 60)
    
    for _, row in df.iterrows():
        emoji = "🟢" if row['pnl_dollars'] > 0 else "🔴"
        print(f"\n{emoji} {row['ticker']}")
        print(f"   Shares: {row['shares']}")
        print(f"   Cost: ${row['cost_basis']} → Current: ${row['current_price']}")
        print(f"   Value: ${row['market_value']}")
        print(f"   P&L: ${row['pnl_dollars']:+.2f} ({row['pnl_percent']:+.1f}%)")
    
    print("\n" + "=" * 60)
    print("📈 SUMMARY")
    print("=" * 60)
    print(f"   Total Value: ${summary['total_market_value']}")
    print(f"   Total P&L: ${summary['total_pnl_dollars']:+.2f} ({summary['total_pnl_percent']:+.1f}%)")
    print(f"   Winners: {summary['winners']} | Losers: {summary['losers']}")


if __name__ == "__main__":
    # Your current holdings from screenshot
    my_holdings = [
        {'ticker': 'MU', 'shares': 1.09, 'cost_basis': 95.00},
        {'ticker': 'APLD', 'shares': 2.99, 'cost_basis': 20.00},
        {'ticker': 'HOOD', 'shares': 0.35, 'cost_basis': 40.00},
        {'ticker': 'NVDA', 'shares': 0.18, 'cost_basis': 135.00},
        {'ticker': 'ORCL', 'shares': 0.12, 'cost_basis': 165.00},
    ]
    
    print_portfolio(my_holdings)
