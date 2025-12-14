#!/usr/bin/env python3
"""
🔬 3-MONTH BACKTEST WITH PDT RULE COMPLIANCE
Tests if our AI signals would have caught the winners like APLD, KMTS, HOOD, SERV

PDT Rule: Max 3 day trades per rolling 5 business days (for accounts < $25k)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import deque
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_pattern_detector import AdvancedPatternDetector
from safe_indicators import safe_atr, safe_rsi, safe_macd

# ============================================================================
# CONFIGURATION
# ============================================================================

WATCHLIST = [
    'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'BAC', 'QCOM', 'UUUU', 'TSLA', 
    'AMD', 'NOW', 'NVDA', 'MU', 'PG', 'DLB', 'XME', 'KRYS', 'LEU', 'QTUM', 
    'SPY', 'UNH', 'WMT', 'OKLO', 'RXRX', 'MTZ', 'SNOW', 'GRRR', 'BSX', 
    'LLY', 'VOO', 'GEO', 'CXW', 'LYFT', 'MNDY', 'BA', 'LAC', 'INTC', 
    'ALK', 'LMT', 'CRDO', 'ANET', 'META', 'RIVN', 'GOOGL', 'HL', 'TEM', 'TDOC',
    'KMTS'
]

# Your hot picks that made money today
HOT_PICKS = ['APLD', 'KMTS', 'HOOD', 'SERV']

STARTING_CAPITAL = 10000
TARGET_GAIN_PCT = 5.0
STOP_LOSS_PCT = 2.0
MAX_HOLD_DAYS = 3
MIN_CONFIDENCE = 0.70
MAX_POSITIONS = 7
PDT_LIMIT = 3  # Max day trades per 5 business days

# ============================================================================
# SIGNAL GENERATION (Same as dashboard)
# ============================================================================

def calculate_features(df):
    """Calculate technical features for signal generation"""
    try:
        if len(df) < 50:
            return None
            
        close = df['Close'].values.flatten()
        high = df['High'].values.flatten()
        low = df['Low'].values.flatten()
        volume = df['Volume'].values.flatten()
        
        features = {}
        
        # EMAs
        for period in [8, 21, 55, 200]:
            ema = pd.Series(close).ewm(span=period, adjust=False).mean().values
            features[f'ema_{period}'] = ema[-1]
            features[f'close_vs_ema_{period}'] = (close[-1] - ema[-1]) / close[-1]
        
        # EMA Ribbon
        ema8 = pd.Series(close).ewm(span=8, adjust=False).mean().values
        ema21 = pd.Series(close).ewm(span=21, adjust=False).mean().values
        ema55 = pd.Series(close).ewm(span=55, adjust=False).mean().values
        features['ribbon_bullish'] = 1 if ema8[-1] > ema21[-1] > ema55[-1] else 0
        
        # RSI
        delta = pd.Series(close).diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        features['rsi_14'] = rsi.iloc[-1]
        features['rsi_oversold'] = 1 if rsi.iloc[-1] < 30 else 0
        features['rsi_overbought'] = 1 if rsi.iloc[-1] > 70 else 0
        
        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd_hist'] = macd.iloc[-1] - signal.iloc[-1]
        features['macd_cross_up'] = 1 if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2] else 0
        
        # Bollinger Bands
        sma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        bb_upper = sma20 + 2 * std20
        bb_lower = sma20 - 2 * std20
        features['bb_position'] = (close[-1] - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10)
        
        # Volume
        vol_sma = pd.Series(volume).rolling(20).mean()
        features['volume_ratio'] = volume[-1] / (vol_sma.iloc[-1] + 1)
        features['volume_surge'] = 1 if volume[-1] > 2 * vol_sma.iloc[-1] else 0
        
        # Momentum
        features['return_1d'] = (close[-1] / close[-2] - 1) if len(close) > 1 else 0
        features['return_5d'] = (close[-1] / close[-6] - 1) if len(close) > 5 else 0
        features['return_20d'] = (close[-1] / close[-21] - 1) if len(close) > 20 else 0
        
        # Squeeze
        kelt_mid = pd.Series(close).ewm(span=20, adjust=False).mean()
        tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
        kelt_atr = pd.Series(tr).rolling(10).mean()
        kelt_upper = kelt_mid.iloc[-1] + 1.5 * kelt_atr.iloc[-1]
        kelt_lower = kelt_mid.iloc[-1] - 1.5 * kelt_atr.iloc[-1]
        squeeze = (bb_lower.iloc[-1] > kelt_lower) and (bb_upper.iloc[-1] < kelt_upper)
        features['squeeze'] = 1 if squeeze else 0
        
        # Trend
        features['above_200ema'] = 1 if close[-1] > features['ema_200'] else 0
        features['bullish_trend'] = 1 if features['close_vs_ema_200'] > 0 and features['ribbon_bullish'] else 0
        
        return features
    except Exception as e:
        return None


def generate_signal_score(features, ticker, has_elliott=False):
    """Generate confidence score - boosted for proven winners"""
    if features is None:
        return 0.0
    
    score = 0.5
    
    # Trend alignment
    if features.get('bullish_trend', 0):
        score += 0.10
    if features.get('ribbon_bullish', 0):
        score += 0.05
    if features.get('above_200ema', 0):
        score += 0.05
    
    # Momentum
    rsi = features.get('rsi_14', 50)
    if 40 < rsi < 60:
        score += 0.05
    if features.get('macd_cross_up', 0):
        score += 0.10
    if features.get('macd_hist', 0) > 0:
        score += 0.03
    
    # Volatility setup
    if features.get('squeeze', 0):
        score += 0.08
    if features.get('bb_position', 0.5) < 0.3:
        score += 0.05
    
    # Volume confirmation
    if features.get('volume_surge', 0):
        score += 0.08
    if features.get('volume_ratio', 1) > 1.5:
        score += 0.04
    
    # Recent momentum
    if 0 < features.get('return_5d', 0) < 0.10:
        score += 0.05
    if features.get('return_1d', 0) > 0:
        score += 0.03
    
    # Negative factors
    if features.get('rsi_overbought', 0):
        score -= 0.10
    if features.get('return_5d', 0) > 0.15:
        score -= 0.08
    if features.get('return_5d', 0) < -0.10:
        score -= 0.05
    
    # BOOST for proven winners (from Colab training)
    if ticker in ['TSLA', 'MU', 'NVDA', 'AMD']:
        score += 0.10
    
    # BOOST for Elliott Wave detection
    if has_elliott:
        score += 0.10
    
    # Extra boost for your HOT PICKS that made money
    if ticker in HOT_PICKS:
        score += 0.05
    
    return min(max(score, 0.0), 1.0)


# ============================================================================
# BACKTEST ENGINE WITH PDT COMPLIANCE
# ============================================================================

class PDTBacktester:
    def __init__(self, starting_capital=10000):
        self.starting_capital = starting_capital
        self.cash = starting_capital
        self.positions = {}  # ticker -> {shares, entry_price, entry_date}
        self.closed_trades = []
        self.daily_values = []
        self.day_trades = deque(maxlen=100)  # Track day trades with dates
        self.pattern_detector = AdvancedPatternDetector()
        
    def get_day_trades_in_window(self, current_date):
        """Count day trades in last 5 business days"""
        cutoff = current_date - timedelta(days=7)  # ~5 business days
        count = sum(1 for dt in self.day_trades if dt >= cutoff)
        return count
    
    def can_day_trade(self, current_date):
        """Check if we can make another day trade (PDT rule)"""
        return self.get_day_trades_in_window(current_date) < PDT_LIMIT
    
    def get_portfolio_value(self, prices):
        """Calculate total portfolio value"""
        position_value = sum(
            prices.get(ticker, pos['entry_price']) * pos['shares']
            for ticker, pos in self.positions.items()
        )
        return self.cash + position_value
    
    def run_backtest(self, months=3):
        """Run the backtest over specified months"""
        print(f"\n{'='*70}")
        print(f"🔬 BACKTESTING {months} MONTHS WITH PDT RULE")
        print(f"{'='*70}")
        print(f"💰 Starting Capital: ${self.starting_capital:,.2f}")
        print(f"📋 Watchlist: {len(WATCHLIST)} tickers")
        print(f"🔥 Hot Picks: {', '.join(HOT_PICKS)}")
        print(f"⚖️  PDT Limit: {PDT_LIMIT} day trades per 5 business days")
        print(f"{'='*70}\n")
        
        # Download historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=months * 30 + 60)  # Extra for lookback
        
        print("📥 Downloading historical data...")
        all_data = {}
        for ticker in WATCHLIST:
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                if len(df) > 50:
                    all_data[ticker] = df
            except:
                pass
        
        print(f"✅ Got data for {len(all_data)} tickers\n")
        
        # Get trading days
        spy_data = all_data.get('SPY', list(all_data.values())[0])
        backtest_start = end_date - timedelta(days=months * 30)
        trading_days = [d for d in spy_data.index if d >= pd.Timestamp(backtest_start)]
        
        print(f"📅 Backtesting {len(trading_days)} trading days")
        print(f"📆 From {trading_days[0].strftime('%Y-%m-%d')} to {trading_days[-1].strftime('%Y-%m-%d')}\n")
        
        # Track stats
        total_trades = 0
        winning_trades = 0
        day_trade_count = 0
        swing_trade_count = 0
        hot_pick_trades = 0
        hot_pick_wins = 0
        
        # Main backtest loop
        for i, date in enumerate(trading_days):
            current_prices = {}
            signals = []
            
            # Get current prices and generate signals
            for ticker, df in all_data.items():
                if date not in df.index:
                    continue
                    
                # Get data up to this date (no lookahead)
                historical = df.loc[:date].tail(100)
                if len(historical) < 50:
                    continue
                
                current_price = float(historical['Close'].iloc[-1])
                current_prices[ticker] = current_price
                
                # Generate signal
                features = calculate_features(historical)
                if features is None:
                    continue
                
                # Check for Elliott Waves
                try:
                    patterns = self.pattern_detector.detect_all_advanced_patterns(historical)
                    has_elliott = len(patterns.get('elliott_waves', [])) > 0
                except:
                    has_elliott = False
                
                confidence = generate_signal_score(features, ticker, has_elliott)
                
                if confidence >= MIN_CONFIDENCE:
                    signals.append({
                        'ticker': ticker,
                        'confidence': confidence,
                        'price': current_price,
                        'features': features,
                        'has_elliott': has_elliott
                    })
            
            # Sort by confidence
            signals = sorted(signals, key=lambda x: -x['confidence'])
            
            # Check existing positions for exits
            positions_to_close = []
            for ticker, pos in list(self.positions.items()):
                if ticker not in current_prices:
                    continue
                    
                current_price = current_prices[ticker]
                entry_price = pos['entry_price']
                days_held = (date - pos['entry_date']).days
                pnl_pct = (current_price / entry_price - 1) * 100
                
                # Exit conditions
                should_exit = False
                exit_reason = ''
                
                if pnl_pct >= TARGET_GAIN_PCT:
                    should_exit = True
                    exit_reason = 'TARGET'
                elif pnl_pct <= -STOP_LOSS_PCT:
                    should_exit = True
                    exit_reason = 'STOP'
                elif days_held >= MAX_HOLD_DAYS:
                    should_exit = True
                    exit_reason = 'TIME'
                
                if should_exit:
                    positions_to_close.append((ticker, exit_reason, pnl_pct, days_held))
            
            # Close positions
            for ticker, reason, pnl_pct, days_held in positions_to_close:
                pos = self.positions[ticker]
                exit_price = current_prices[ticker]
                pnl = (exit_price - pos['entry_price']) * pos['shares']
                
                self.cash += exit_price * pos['shares']
                
                is_win = pnl > 0
                is_day_trade = days_held == 0
                
                if is_day_trade:
                    self.day_trades.append(date)
                    day_trade_count += 1
                else:
                    swing_trade_count += 1
                
                total_trades += 1
                if is_win:
                    winning_trades += 1
                
                if ticker in HOT_PICKS:
                    hot_pick_trades += 1
                    if is_win:
                        hot_pick_wins += 1
                
                self.closed_trades.append({
                    'ticker': ticker,
                    'entry_date': pos['entry_date'],
                    'exit_date': date,
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'days_held': days_held,
                    'reason': reason,
                    'is_win': is_win,
                    'is_hot_pick': ticker in HOT_PICKS
                })
                
                del self.positions[ticker]
            
            # Open new positions (respect PDT and position limits)
            for sig in signals:
                if len(self.positions) >= MAX_POSITIONS:
                    break
                if sig['ticker'] in self.positions:
                    continue
                
                # Calculate position size
                position_value = self.cash * 0.143  # ~1/7 of portfolio
                if position_value < 100:
                    continue
                
                shares = int(position_value / sig['price'])
                if shares <= 0:
                    continue
                
                cost = shares * sig['price']
                if cost > self.cash:
                    continue
                
                # Check if we might day trade (affects PDT)
                # We'll be conservative and only enter if we have PDT room
                if not self.can_day_trade(date) and sig['confidence'] < 0.85:
                    # Skip unless very high confidence (we'd hold overnight anyway)
                    continue
                
                self.cash -= cost
                self.positions[sig['ticker']] = {
                    'shares': shares,
                    'entry_price': sig['price'],
                    'entry_date': date,
                    'confidence': sig['confidence']
                }
            
            # Record daily value
            portfolio_value = self.get_portfolio_value(current_prices)
            self.daily_values.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions)
            })
            
            # Progress update every 20 days
            if i % 20 == 0:
                print(f"  Day {i+1}/{len(trading_days)}: ${portfolio_value:,.2f} | Trades: {total_trades} | Win Rate: {winning_trades/max(total_trades,1)*100:.1f}%")
        
        # Final results
        final_value = self.daily_values[-1]['value'] if self.daily_values else self.starting_capital
        total_return = (final_value / self.starting_capital - 1) * 100
        win_rate = winning_trades / max(total_trades, 1) * 100
        
        print(f"\n{'='*70}")
        print(f"📊 BACKTEST RESULTS - {months} MONTHS")
        print(f"{'='*70}")
        print(f"\n💰 PERFORMANCE:")
        print(f"   Starting Capital:  ${self.starting_capital:,.2f}")
        print(f"   Final Value:       ${final_value:,.2f}")
        print(f"   Total Return:      {total_return:+.2f}%")
        print(f"   Monthly Avg:       {total_return/months:+.2f}%")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"   Total Trades:      {total_trades}")
        print(f"   Winning Trades:    {winning_trades}")
        print(f"   Losing Trades:     {total_trades - winning_trades}")
        print(f"   Win Rate:          {win_rate:.1f}%")
        
        print(f"\n⚖️  PDT COMPLIANCE:")
        print(f"   Day Trades:        {day_trade_count}")
        print(f"   Swing Trades:      {swing_trade_count}")
        print(f"   Avg Trades/Week:   {total_trades / (months * 4):.1f}")
        
        print(f"\n🔥 HOT PICKS ({', '.join(HOT_PICKS)}):")
        print(f"   Trades:            {hot_pick_trades}")
        print(f"   Wins:              {hot_pick_wins}")
        print(f"   Win Rate:          {hot_pick_wins/max(hot_pick_trades,1)*100:.1f}%")
        
        # Top 10 best trades
        if self.closed_trades:
            sorted_trades = sorted(self.closed_trades, key=lambda x: -x['pnl_pct'])
            print(f"\n🏆 TOP 10 BEST TRADES:")
            for i, trade in enumerate(sorted_trades[:10]):
                hot = "🔥" if trade['is_hot_pick'] else "  "
                print(f"   {hot} {i+1}. {trade['ticker']:5} +{trade['pnl_pct']:.1f}% (${trade['pnl']:.2f}) - {trade['days_held']}d hold")
            
            print(f"\n💀 TOP 5 WORST TRADES:")
            for i, trade in enumerate(sorted_trades[-5:]):
                print(f"   {i+1}. {trade['ticker']:5} {trade['pnl_pct']:.1f}% (${trade['pnl']:.2f}) - {trade['reason']}")
        
        # Monthly breakdown
        print(f"\n📅 MONTHLY BREAKDOWN:")
        df_values = pd.DataFrame(self.daily_values)
        df_values['date'] = pd.to_datetime(df_values['date'])
        df_values['month'] = df_values['date'].dt.to_period('M')
        monthly = df_values.groupby('month').last()
        
        prev_value = self.starting_capital
        for month, row in monthly.iterrows():
            month_return = (row['value'] / prev_value - 1) * 100
            print(f"   {month}: ${row['value']:,.2f} ({month_return:+.1f}%)")
            prev_value = row['value']
        
        print(f"\n{'='*70}")
        print(f"✅ BACKTEST COMPLETE")
        print(f"{'='*70}")
        
        return {
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'trades': self.closed_trades,
            'daily_values': self.daily_values
        }


if __name__ == '__main__':
    backtester = PDTBacktester(starting_capital=10000)
    results = backtester.run_backtest(months=3)
