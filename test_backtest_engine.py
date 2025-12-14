"""
Test Backtest Engine with real historical data and research-backed metrics
Uses yFinance data with TA-Lib indicators
"""
import asyncio
import yfinance as yf
import pandas as pd
import talib
from datetime import datetime, timedelta
from backtest_engine import BacktestEngine, BacktestConfig
from watchlist_scanner import WatchlistScanner, TechnicalIndicators
import numpy as np

def generate_signals_from_historical(ticker: str, start_date: str, end_date: str):
    """Generate buy/sell signals from historical data using research-backed indicators"""
    # Fetch data
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    
    if len(df) < 100:
        return None
    
    # Flatten multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Use TA-Lib for reliable indicator calculation
    close = df['Close'].values
    
    df['RSI_9'] = talib.RSI(close, timeperiod=9)
    df['RSI_14'] = talib.RSI(close, timeperiod=14)
    
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=5, slowperiod=13, signalperiod=1)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    
    df['BB_upper'], df['BB_mid'], df['BB_lower'] = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
    
    df = df.dropna()
    
    signals = []
    position_open = False
    
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # BUY Signal: RSI oversold + MACD crossover bullish
        if not position_open:
            if (row['RSI_9'] < 40 or row['RSI_14'] < 40) and \
               row['MACD'] > row['MACD_signal'] and \
               prev_row['MACD'] <= prev_row['MACD_signal']:
                signals.append({
                    'date': row.name,
                    'type': 'BUY',
                    'price': float(row['Close']),
                    'rsi_9': row['RSI_9'],
                    'rsi_14': row['RSI_14'],
                    'macd': 'BULLISH'
                })
                position_open = True
        
        # SELL Signal: RSI overbought + MACD crossover bearish
        elif position_open:
            if (row['RSI_9'] > 60 or row['RSI_14'] > 60) and \
               row['MACD'] < row['MACD_signal'] and \
               prev_row['MACD'] >= prev_row['MACD_signal']:
                signals.append({
                    'date': row.name,
                    'type': 'SELL',
                    'price': float(row['Close']),
                    'rsi_9': row['RSI_9'],
                    'rsi_14': row['RSI_14'],
                    'macd': 'BEARISH'
                })
                position_open = False
    
    return signals, df

def run_backtest(ticker: str, start_date: str, end_date: str):
    """Run backtest on single ticker"""
    print(f"\n{'='*80}")
    print(f"Running Backtest: {ticker}")
    print(f"Period: {start_date} to {end_date}")
    print(f"{'='*80}")
    
    # Generate signals
    result = generate_signals_from_historical(ticker, start_date, end_date)
    if result is None:
        print(f"❌ Insufficient data for {ticker}")
        return None
    
    signals, df = result
    
    print(f"\n📊 Generated {len(signals)} signals")
    
    if len(signals) < 2:
        print(f"⚠️ Not enough signals for backtest")
        return None
    
    # Initialize backtest engine
    config = BacktestConfig(
        initial_capital=500.0,  # Research: <$1k account
        risk_per_trade=0.005,  # 0.5% risk per trade
        commission_rate=0.0001,
        slippage_bps=2.0,
        max_position_size=0.10,
        daily_loss_cap=0.03
    )
    
    engine = BacktestEngine(config)
    
    # Run backtest
    position_open = False
    trades_executed = 0
    
    for signal in signals:
        if signal['type'] == 'BUY' and not position_open:
            # Enter position
            trade = engine.enter_position(
                ticker=ticker,
                date=signal['date'],
                price=signal['price'],
                signal='BUY',
                stop_loss_pct=0.02  # 2% stop loss (research guideline)
            )
            if trade:
                position_open = True
                trades_executed += 1
                print(f"✅ BUY {ticker} @ ${signal['price']:.2f} on {signal['date'].date()}")
        
        elif signal['type'] == 'SELL' and position_open:
            # Exit position
            trade = engine.exit_position(
                ticker=ticker,
                date=signal['date'],
                price=signal['price']
            )
            if trade:
                position_open = False
                print(f"✅ SELL {ticker} @ ${signal['price']:.2f} on {signal['date'].date()} | P&L: ${trade.pnl:.2f} ({trade.pnl_pct:.2f}%)")
    
    # Close any open positions at end
    if position_open and ticker in engine.positions:
        last_price = df['Close'].iloc[-1]
        last_date = df.index[-1]
        trade = engine.exit_position(ticker, last_date, last_price)
        print(f"🔚 CLOSE {ticker} @ ${last_price:.2f} (end of backtest) | P&L: ${trade.pnl:.2f}")
    
    # Calculate metrics
    results = engine.calculate_metrics()
    
    if results is None:
        print(f"⚠️ No completed trades")
        return None
    
    # Monte Carlo simulation
    monte_carlo = engine.monte_carlo_simulation(results, num_simulations=1000)
    
    # Print results
    engine.print_results(results, monte_carlo)
    
    return results

def run_portfolio_backtest():
    """Run backtest on multiple tickers from config"""
    print("\n" + "="*80)
    print("PORTFOLIO BACKTEST - Research-Backed Strategy")
    print("Based on research.md: <$1k account, 0.5% risk per trade")
    print("="*80)
    
    # Test on 5 high-liquidity tickers
    tickers = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA']
    start_date = '2024-01-01'
    end_date = '2025-11-30'
    
    all_results = {}
    
    for ticker in tickers:
        try:
            results = run_backtest(ticker, start_date, end_date)
            if results:
                all_results[ticker] = results
        except Exception as e:
            print(f"❌ Error backtesting {ticker}: {e}")
    
    # Portfolio summary
    print("\n" + "="*80)
    print("PORTFOLIO SUMMARY")
    print("="*80)
    
    for ticker, results in all_results.items():
        print(f"\n{ticker}:")
        print(f"  Trades: {results.total_trades} | Win Rate: {results.win_rate*100:.1f}%")
        print(f"  Return: {results.total_return_pct:.2f}% | Sharpe: {results.sharpe_ratio:.3f}")
        print(f"  Max DD: {results.max_drawdown_pct:.2f}% | Profit Factor: {results.profit_factor:.2f}")
    
    # Aggregate stats
    if all_results:
        avg_return = np.mean([r.total_return_pct for r in all_results.values()])
        avg_sharpe = np.mean([r.sharpe_ratio for r in all_results.values()])
        avg_win_rate = np.mean([r.win_rate for r in all_results.values()])
        total_trades = sum([r.total_trades for r in all_results.values()])
        
        print(f"\n📈 Portfolio Averages:")
        print(f"   Average Return: {avg_return:.2f}%")
        print(f"   Average Sharpe: {avg_sharpe:.3f}")
        print(f"   Average Win Rate: {avg_win_rate*100:.1f}%")
        print(f"   Total Trades: {total_trades}")
    
    print("\n" + "="*80)
    print("✅ Portfolio backtest completed!")

if __name__ == "__main__":
    run_portfolio_backtest()
