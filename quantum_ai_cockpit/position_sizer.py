"""Position sizing calculator for Quantum AI Cockpit."""
from __future__ import annotations
import yfinance as yf
from .config import STOP_LOSS_PCT, TARGET_PCT


def calculate_position_size(account_balance: float, stock_price: float, stop_loss_price: float) -> dict:
    max_risk = account_balance * 0.10
    risk_per_share = stock_price - stop_loss_price
    
    if risk_per_share <= 0:
        return {'shares_to_buy': 0, 'dollar_risk': 0, 'position_value': 0}
    
    shares = int(max_risk / risk_per_share)
    return {
        'shares_to_buy': shares,
        'dollar_risk': round(shares * risk_per_share, 2),
        'position_value': round(shares * stock_price, 2)
    }


def get_trade_plan(ticker: str, account_balance: float) -> dict:
    data = yf.download(ticker, period="1d", progress=False)
    if data.empty:
        raise ValueError(f"Could not fetch price for {ticker}")
    
    price = float(data['Close'].iloc[-1])
    stop_loss = round(price * (1 - STOP_LOSS_PCT), 2)
    target = round(price * (1 + TARGET_PCT), 2)
    
    position = calculate_position_size(account_balance, price, stop_loss)
    
    return {
        'ticker': ticker,
        'entry_price': round(price, 2),
        'stop_loss': stop_loss,
        'target': target,
        'shares_to_buy': position['shares_to_buy'],
        'position_value': position['position_value'],
        'dollar_risk': position['dollar_risk'],
        'potential_profit': round(position['shares_to_buy'] * (target - price), 2)
    }
