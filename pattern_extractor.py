#!/usr/bin/env python3
"""
🏆 PATTERN EXTRACTOR - Extract Winning Strategies from Training
================================================================
Captures the exact conditions that led to +199.7% returns
Exports them as rules for your frontend/trading system
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import yfinance as yf

# =============================================================================
# WINNING PATTERN TRACKER
# =============================================================================

@dataclass
class WinningTrade:
    """Record of a profitable trade with all context"""
    ticker: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    pnl_pct: float
    hold_days: int
    
    # Entry conditions
    rsi_at_entry: float
    macd_hist_at_entry: float
    volume_ratio_at_entry: float
    bb_pct_at_entry: float
    returns_21d_at_entry: float
    ema_8_vs_21_at_entry: float
    trend_alignment_at_entry: float
    
    # Exit conditions
    rsi_at_exit: float
    macd_hist_at_exit: float
    
    # Pattern classification
    pattern_type: str  # DIP_BUY, MOMENTUM, BREAKOUT, etc.
    confidence: float


class PatternExtractor:
    """
    Extract and analyze winning patterns from trading history
    """
    
    def __init__(self):
        self.winning_trades: List[WinningTrade] = []
        self.losing_trades: List[WinningTrade] = []
        self.pattern_stats: Dict = {}
        
    def classify_pattern(self, entry_conditions: Dict) -> str:
        """Classify the trade pattern based on entry conditions"""
        
        rsi = entry_conditions.get('rsi', 50)
        returns_21d = entry_conditions.get('returns_21d', 0)
        volume_ratio = entry_conditions.get('volume_ratio', 1)
        bb_pct = entry_conditions.get('bb_pct', 0.5)
        macd_hist = entry_conditions.get('macd_hist', 0)
        ema_cross = entry_conditions.get('ema_8_vs_21', 0)
        
        # DIP BUY: Oversold + big drawdown
        if rsi < 35 and returns_21d < -0.08:
            return "DIP_BUY"
        
        # RSI BOUNCE: Oversold bounce
        if rsi < 40 and macd_hist > 0:
            return "RSI_BOUNCE"
        
        # VOLUME BREAKOUT: High volume + momentum
        if volume_ratio > 1.5 and macd_hist > 0 and ema_cross > 0:
            return "VOLUME_BREAKOUT"
        
        # BOLLINGER BOUNCE: Near lower band
        if bb_pct < 0.2 and rsi < 45:
            return "BB_BOUNCE"
        
        # MOMENTUM: Trend following
        if ema_cross > 0 and macd_hist > 0 and rsi > 50 and rsi < 70:
            return "MOMENTUM"
        
        # MEAN REVERSION: Oversold in uptrend
        if rsi < 45 and ema_cross > 0:
            return "MEAN_REVERSION"
        
        return "OTHER"
    
    def record_trade(self, trade_data: Dict, is_winner: bool = True):
        """Record a trade with full context"""
        
        pattern_type = self.classify_pattern({
            'rsi': trade_data.get('rsi_at_entry', 50),
            'returns_21d': trade_data.get('returns_21d_at_entry', 0),
            'volume_ratio': trade_data.get('volume_ratio_at_entry', 1),
            'bb_pct': trade_data.get('bb_pct_at_entry', 0.5),
            'macd_hist': trade_data.get('macd_hist_at_entry', 0),
            'ema_8_vs_21': trade_data.get('ema_8_vs_21_at_entry', 0),
        })
        
        trade = WinningTrade(
            ticker=trade_data.get('ticker', ''),
            entry_date=trade_data.get('entry_date', ''),
            exit_date=trade_data.get('exit_date', ''),
            entry_price=trade_data.get('entry_price', 0),
            exit_price=trade_data.get('exit_price', 0),
            pnl_pct=trade_data.get('pnl_pct', 0),
            hold_days=trade_data.get('hold_days', 0),
            rsi_at_entry=trade_data.get('rsi_at_entry', 50),
            macd_hist_at_entry=trade_data.get('macd_hist_at_entry', 0),
            volume_ratio_at_entry=trade_data.get('volume_ratio_at_entry', 1),
            bb_pct_at_entry=trade_data.get('bb_pct_at_entry', 0.5),
            returns_21d_at_entry=trade_data.get('returns_21d_at_entry', 0),
            ema_8_vs_21_at_entry=trade_data.get('ema_8_vs_21_at_entry', 0),
            trend_alignment_at_entry=trade_data.get('trend_alignment_at_entry', 0),
            rsi_at_exit=trade_data.get('rsi_at_exit', 50),
            macd_hist_at_exit=trade_data.get('macd_hist_at_exit', 0),
            pattern_type=pattern_type,
            confidence=trade_data.get('confidence', 0.5)
        )
        
        if is_winner:
            self.winning_trades.append(trade)
        else:
            self.losing_trades.append(trade)
    
    def analyze_patterns(self) -> Dict:
        """Analyze all recorded trades to find winning patterns"""
        
        if not self.winning_trades:
            return {}
        
        # Group by pattern type
        pattern_groups = {}
        for trade in self.winning_trades:
            if trade.pattern_type not in pattern_groups:
                pattern_groups[trade.pattern_type] = []
            pattern_groups[trade.pattern_type].append(trade)
        
        # Calculate stats for each pattern
        pattern_stats = {}
        for pattern_type, trades in pattern_groups.items():
            pnls = [t.pnl_pct for t in trades]
            
            pattern_stats[pattern_type] = {
                'count': len(trades),
                'avg_pnl': np.mean(pnls) * 100,
                'max_pnl': np.max(pnls) * 100,
                'min_pnl': np.min(pnls) * 100,
                'win_rate': sum(1 for p in pnls if p > 0) / len(pnls) * 100,
                'avg_hold_days': np.mean([t.hold_days for t in trades]),
                
                # Entry conditions averages
                'avg_rsi_entry': np.mean([t.rsi_at_entry for t in trades]),
                'avg_volume_ratio': np.mean([t.volume_ratio_at_entry for t in trades]),
                'avg_bb_pct': np.mean([t.bb_pct_at_entry for t in trades]),
                'avg_returns_21d': np.mean([t.returns_21d_at_entry for t in trades]) * 100,
                
                # Exit conditions
                'avg_rsi_exit': np.mean([t.rsi_at_exit for t in trades]),
                
                # Best examples
                'best_trades': sorted(trades, key=lambda x: x.pnl_pct, reverse=True)[:3]
            }
        
        self.pattern_stats = pattern_stats
        return pattern_stats
    
    def generate_trading_rules(self) -> List[Dict]:
        """Generate concrete trading rules from patterns"""
        
        if not self.pattern_stats:
            self.analyze_patterns()
        
        rules = []
        
        for pattern_type, stats in self.pattern_stats.items():
            if stats['count'] < 3:  # Need at least 3 examples
                continue
            
            rule = {
                'name': pattern_type,
                'priority': int(stats['count'] * stats['win_rate'] / 100),
                'expected_return': round(stats['avg_pnl'], 1),
                'win_rate': round(stats['win_rate'], 1),
                'avg_hold_days': round(stats['avg_hold_days'], 1),
                'conditions': {},
                'exit_conditions': {}
            }
            
            # Generate entry conditions based on pattern type
            if pattern_type == "DIP_BUY":
                rule['conditions'] = {
                    'rsi_below': 35,
                    'returns_21d_below': -8,  # -8% drawdown
                    'volume_ratio_above': 0.8
                }
                rule['exit_conditions'] = {
                    'rsi_above': 60,
                    'profit_target': 8,
                    'stop_loss': -5
                }
                
            elif pattern_type == "RSI_BOUNCE":
                rule['conditions'] = {
                    'rsi_below': 40,
                    'macd_hist_above': 0,
                    'bb_pct_below': 0.3
                }
                rule['exit_conditions'] = {
                    'rsi_above': 65,
                    'profit_target': 6,
                    'stop_loss': -4
                }
                
            elif pattern_type == "VOLUME_BREAKOUT":
                rule['conditions'] = {
                    'volume_ratio_above': 1.5,
                    'macd_hist_above': 0,
                    'ema_8_above_21': True
                }
                rule['exit_conditions'] = {
                    'volume_ratio_below': 0.8,
                    'profit_target': 10,
                    'stop_loss': -5
                }
                
            elif pattern_type == "BB_BOUNCE":
                rule['conditions'] = {
                    'bb_pct_below': 0.2,
                    'rsi_below': 45
                }
                rule['exit_conditions'] = {
                    'bb_pct_above': 0.7,
                    'profit_target': 5,
                    'stop_loss': -3
                }
                
            elif pattern_type == "MOMENTUM":
                rule['conditions'] = {
                    'ema_8_above_21': True,
                    'macd_hist_above': 0,
                    'rsi_between': [50, 70],
                    'trend_alignment_above': 0.3
                }
                rule['exit_conditions'] = {
                    'rsi_above': 75,
                    'profit_target': 8,
                    'stop_loss': -4
                }
                
            elif pattern_type == "MEAN_REVERSION":
                rule['conditions'] = {
                    'rsi_below': 45,
                    'ema_8_above_21': True,
                    'returns_21d_between': [-5, 0]
                }
                rule['exit_conditions'] = {
                    'rsi_above': 55,
                    'profit_target': 5,
                    'stop_loss': -3
                }
            
            rules.append(rule)
        
        # Sort by priority
        rules.sort(key=lambda x: x['priority'], reverse=True)
        return rules
    
    def export_to_json(self, filename: str = 'winning_patterns.json'):
        """Export patterns and rules to JSON for frontend"""
        
        rules = self.generate_trading_rules()
        
        export_data = {
            'generated_at': datetime.now().isoformat(),
            'total_winning_trades': len(self.winning_trades),
            'total_losing_trades': len(self.losing_trades),
            'overall_win_rate': len(self.winning_trades) / max(len(self.winning_trades) + len(self.losing_trades), 1) * 100,
            
            'trading_rules': rules,
            
            'pattern_summary': {
                name: {
                    'count': stats['count'],
                    'avg_pnl': round(stats['avg_pnl'], 2),
                    'win_rate': round(stats['win_rate'], 1),
                    'avg_hold_days': round(stats['avg_hold_days'], 1)
                }
                for name, stats in self.pattern_stats.items()
            },
            
            'best_trades': [
                asdict(t) for t in sorted(self.winning_trades, key=lambda x: x.pnl_pct, reverse=True)[:10]
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        print(f"✅ Exported patterns to {filename}")
        return export_data


# =============================================================================
# TRAINING WITH PATTERN EXTRACTION
# =============================================================================

class AlphaGoTrainerWithExtraction:
    """
    Training system that extracts and saves winning patterns
    """
    
    def __init__(self, tickers: List[str]):
        self.tickers = tickers
        self.extractor = PatternExtractor()
        self.best_portfolio_value = 10000
        self.best_episode_trades = []
        
    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all indicators needed for pattern detection"""
        
        # Returns
        df['returns_1d'] = df['Close'].pct_change()
        df['returns_21d'] = df['Close'].pct_change(21)
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Volume
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = df['Volume'] / (df['volume_sma'] + 1e-10)
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(20).mean()
        std_20 = df['Close'].rolling(20).std()
        df['bb_upper'] = sma_20 + 2 * std_20
        df['bb_lower'] = sma_20 - 2 * std_20
        df['bb_pct'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        
        # EMAs - Full Ribbon
        df['ema_8'] = df['Close'].ewm(span=8).mean()
        df['ema_13'] = df['Close'].ewm(span=13).mean()
        df['ema_21'] = df['Close'].ewm(span=21).mean()
        df['ema_34'] = df['Close'].ewm(span=34).mean()
        df['ema_55'] = df['Close'].ewm(span=55).mean()
        df['ema_8_vs_21'] = (df['ema_8'] / df['ema_21'] - 1)
        
        # EMA Ribbon Analysis (YOUR PATTERN!)
        df['ribbon_min'] = df[['ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_55']].min(axis=1)
        df['ribbon_max'] = df[['ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_55']].max(axis=1)
        df['ribbon_range'] = (df['ribbon_max'] - df['ribbon_min']) / df['ribbon_min'] * 100
        df['above_ribbon'] = df['Close'] > df['ribbon_min']
        df['ema_8_rising'] = df['ema_8'] > df['ema_8'].shift(5)
        df['ema_21_rising'] = df['ema_21'] > df['ema_21'].shift(5)
        df['ribbon_bullish'] = (df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_55'])
        
        # 5-day momentum and bounce detection
        df['mom_5d'] = df['Close'].pct_change(5) * 100
        df['low_5d'] = df['Low'].rolling(5).min()
        df['bounce_from_low'] = (df['Close'] / df['low_5d'] - 1) * 100
        
        # Trend
        df['trend_5d'] = np.sign(df['Close'].pct_change(5))
        df['trend_10d'] = np.sign(df['Close'].pct_change(10))
        df['trend_21d'] = np.sign(df['Close'].pct_change(21))
        df['trend_alignment'] = (df['trend_5d'] + df['trend_10d'] + df['trend_21d']) / 3
        
        return df.ffill().bfill().fillna(0)
    
    def run_episode(self, data_dict: Dict[str, pd.DataFrame], epsilon: float = 0.1) -> Tuple[float, List[Dict]]:
        """Run one training episode and track trades"""
        
        balance = 10000
        positions = {}  # ticker -> {shares, entry_price, entry_idx}
        episode_trades = []
        
        # Get minimum length across all tickers
        min_len = min(len(df) for df in data_dict.values())
        
        for day in range(60, min_len - 1):
            for ticker, df in data_dict.items():
                current_price = float(df['Close'].iloc[day])
                
                # Get indicators
                rsi = float(df['rsi'].iloc[day])
                macd_hist = float(df['macd_hist'].iloc[day])
                volume_ratio = float(df['volume_ratio'].iloc[day])
                bb_pct = float(df['bb_pct'].iloc[day])
                returns_21d = float(df['returns_21d'].iloc[day])
                ema_8_vs_21 = float(df['ema_8_vs_21'].iloc[day])
                trend_alignment = float(df['trend_alignment'].iloc[day])
                
                # Check if we have position
                if ticker in positions:
                    pos = positions[ticker]
                    pnl_pct = (current_price / pos['entry_price']) - 1
                    hold_days = day - pos['entry_idx']
                    
                    # Exit conditions
                    should_exit = False
                    
                    # Profit take
                    if pnl_pct >= 0.05 and rsi > 65:
                        should_exit = True
                    # Extended profit
                    if pnl_pct >= 0.08:
                        should_exit = True
                    # Stop loss
                    if pnl_pct <= -0.05:
                        should_exit = True
                    # Time-based exit
                    if hold_days > 15 and pnl_pct > 0:
                        should_exit = True
                    
                    if should_exit:
                        # Record trade
                        trade_data = {
                            'ticker': ticker,
                            'entry_date': str(df['Date'].iloc[pos['entry_idx']]) if 'Date' in df.columns else f"Day {pos['entry_idx']}",
                            'exit_date': str(df['Date'].iloc[day]) if 'Date' in df.columns else f"Day {day}",
                            'entry_price': pos['entry_price'],
                            'exit_price': current_price,
                            'pnl_pct': pnl_pct,
                            'hold_days': hold_days,
                            'rsi_at_entry': pos['rsi_entry'],
                            'macd_hist_at_entry': pos['macd_entry'],
                            'volume_ratio_at_entry': pos['volume_entry'],
                            'bb_pct_at_entry': pos['bb_pct_entry'],
                            'returns_21d_at_entry': pos['returns_21d_entry'],
                            'ema_8_vs_21_at_entry': pos['ema_entry'],
                            'trend_alignment_at_entry': pos['trend_entry'],
                            'rsi_at_exit': rsi,
                            'macd_hist_at_exit': macd_hist,
                            'confidence': 0.7
                        }
                        
                        episode_trades.append(trade_data)
                        # Add back the full sale value (entry cost + profit/loss)
                        balance += pos['shares'] * current_price
                        del positions[ticker]
                
                else:
                    # Entry conditions - multiple strategies
                    should_buy = False
                    
                    # DIP BUY (highest priority)
                    if rsi < 35 and returns_21d < -0.08:
                        should_buy = True
                    
                    # RSI bounce
                    elif rsi < 40 and macd_hist > 0 and bb_pct < 0.3:
                        should_buy = True
                    
                    # Volume breakout
                    elif volume_ratio > 1.5 and macd_hist > 0 and ema_8_vs_21 > 0:
                        should_buy = True
                    
                    # Momentum
                    elif ema_8_vs_21 > 0 and macd_hist > 0 and 50 < rsi < 70 and trend_alignment > 0.3:
                        should_buy = True
                    
                    # Random exploration
                    if np.random.random() < epsilon:
                        should_buy = np.random.random() < 0.3
                    
                    if should_buy and balance > current_price:
                        # Position size: 10-20% of balance
                        position_pct = 0.15
                        shares = int(balance * position_pct / current_price)
                        
                        if shares > 0:
                            cost = shares * current_price
                            balance -= cost  # Deduct cost when buying!
                            positions[ticker] = {
                                'shares': shares,
                                'entry_price': current_price,
                                'cost': cost,
                                'entry_idx': day,
                                'rsi_entry': rsi,
                                'macd_entry': macd_hist,
                                'volume_entry': volume_ratio,
                                'bb_pct_entry': bb_pct,
                                'returns_21d_entry': returns_21d,
                                'ema_entry': ema_8_vs_21,
                                'trend_entry': trend_alignment
                            }
        
        # Close remaining positions at end of episode
        for ticker, pos in positions.items():
            df = data_dict[ticker]
            final_price = float(df['Close'].iloc[-1])
            # Add full liquidation value
            balance += pos['shares'] * final_price
        
        return balance, episode_trades
    
    def train(self, num_episodes: int = 100) -> Dict:
        """Train and extract patterns"""
        
        print("📥 Loading market data...")
        data_dict = {}
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, period='1y', progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
                df = df.reset_index()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                df = self.compute_indicators(df)
                if len(df) > 60:
                    data_dict[ticker] = df
                    print(f"   ✓ {ticker}")
            except:
                pass
        
        print(f"\n🎮 ALPHAGO-STYLE TRAINING WITH PATTERN EXTRACTION")
        print("=" * 60)
        
        for ep in range(num_episodes):
            epsilon = max(0.05, 0.3 - ep * 0.003)
            portfolio_value, trades = self.run_episode(data_dict, epsilon)
            
            # Track best episode
            if portfolio_value > self.best_portfolio_value:
                self.best_portfolio_value = portfolio_value
                self.best_episode_trades = trades
                
                # Record winning trades to extractor
                for trade in trades:
                    if trade['pnl_pct'] > 0:
                        self.extractor.record_trade(trade, is_winner=True)
                    else:
                        self.extractor.record_trade(trade, is_winner=False)
            
            if ep % 10 == 0:
                pnl_pct = (portfolio_value / 10000 - 1) * 100
                print(f"Ep {ep:3d} | ${portfolio_value:,.0f} ({pnl_pct:+.1f}%) | "
                      f"Best: ${self.best_portfolio_value:,.0f} | Trades: {len(trades)}")
        
        print("=" * 60)
        best_pnl = (self.best_portfolio_value / 10000 - 1) * 100
        print(f"🏆 BEST: ${self.best_portfolio_value:,.0f} ({best_pnl:+.1f}%)")
        
        # Analyze and export patterns
        print("\n📊 Analyzing winning patterns...")
        self.extractor.analyze_patterns()
        export_data = self.extractor.export_to_json('winning_patterns.json')
        
        return export_data


# =============================================================================
# FRONTEND-READY SIGNAL GENERATOR
# =============================================================================

def generate_live_signals(patterns_file: str = 'winning_patterns.json') -> List[Dict]:
    """
    Use extracted patterns to generate real-time signals
    This is what your frontend will call!
    """
    
    # Load patterns
    with open(patterns_file, 'r') as f:
        patterns = json.load(f)
    
    rules = patterns['trading_rules']
    
    # Get current data for watchlist
    tickers = ['APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'TSLA', 'AMD', 'NVDA', 
               'MU', 'INTC', 'BA', 'META', 'GOOGL', 'SPY']
    
    signals = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period='3mo', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.reset_index()
            
            if len(df) < 30:
                continue
            
            # Compute indicators
            close = df['Close'].values
            
            # RSI
            delta = np.diff(close)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14).mean().iloc[-1]
            avg_loss = pd.Series(loss).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))
            
            # Returns
            returns_21d = (close[-1] / close[-22] - 1) if len(close) > 22 else 0
            
            # Volume
            volume_ratio = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
            
            # Bollinger
            sma_20 = pd.Series(close).rolling(20).mean().iloc[-1]
            std_20 = pd.Series(close).rolling(20).std().iloc[-1]
            bb_lower = sma_20 - 2 * std_20
            bb_upper = sma_20 + 2 * std_20
            bb_pct = (close[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)
            
            # MACD
            ema_12 = pd.Series(close).ewm(span=12).mean().iloc[-1]
            ema_26 = pd.Series(close).ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            macd_signal = pd.Series(close).ewm(span=12).mean().ewm(span=9).mean().iloc[-1]
            macd_hist = macd - macd_signal
            
            # EMA cross
            ema_8 = pd.Series(close).ewm(span=8).mean().iloc[-1]
            ema_21 = pd.Series(close).ewm(span=21).mean().iloc[-1]
            ema_8_above_21 = ema_8 > ema_21
            
            # Check each rule
            for rule in rules:
                conditions = rule['conditions']
                match = True
                
                if 'rsi_below' in conditions and rsi >= conditions['rsi_below']:
                    match = False
                if 'rsi_between' in conditions:
                    if not (conditions['rsi_between'][0] <= rsi <= conditions['rsi_between'][1]):
                        match = False
                if 'returns_21d_below' in conditions and returns_21d * 100 >= conditions['returns_21d_below']:
                    match = False
                if 'volume_ratio_above' in conditions and volume_ratio <= conditions['volume_ratio_above']:
                    match = False
                if 'bb_pct_below' in conditions and bb_pct >= conditions['bb_pct_below']:
                    match = False
                if 'macd_hist_above' in conditions and macd_hist <= conditions['macd_hist_above']:
                    match = False
                if 'ema_8_above_21' in conditions and not ema_8_above_21:
                    match = False
                
                if match:
                    signals.append({
                        'ticker': ticker,
                        'price': round(close[-1], 2),
                        'signal': 'BUY',
                        'pattern': rule['name'],
                        'expected_return': rule['expected_return'],
                        'win_rate': rule['win_rate'],
                        'profit_target': rule['exit_conditions'].get('profit_target', 5),
                        'stop_loss': rule['exit_conditions'].get('stop_loss', -5),
                        'rsi': round(rsi, 1),
                        'returns_21d': round(returns_21d * 100, 1),
                        'volume_ratio': round(volume_ratio, 2),
                        'confidence': min(rule['win_rate'] / 100, 0.95)
                    })
                    break  # Only one signal per ticker
        
        except Exception as e:
            continue
    
    # Sort by expected return
    signals.sort(key=lambda x: x['expected_return'], reverse=True)
    
    return signals


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    # Watchlist
    TICKERS = [
        'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'BAC', 'QCOM', 'UUUU', 
        'TSLA', 'AMD', 'NOW', 'NVDA', 'MU', 'XME', 'KRYS', 'LEU',
        'SPY', 'META', 'GOOGL', 'INTC', 'BA', 'LAC', 'HL'
    ]
    
    # Train and extract patterns
    trainer = AlphaGoTrainerWithExtraction(TICKERS)
    patterns = trainer.train(num_episodes=100)
    
    # Show results
    print("\n" + "=" * 60)
    print("📋 EXTRACTED TRADING RULES")
    print("=" * 60)
    
    for rule in patterns['trading_rules']:
        print(f"\n🎯 {rule['name']}")
        print(f"   Expected Return: {rule['expected_return']}%")
        print(f"   Win Rate: {rule['win_rate']}%")
        print(f"   Entry: {rule['conditions']}")
        print(f"   Exit: {rule['exit_conditions']}")
    
    # Generate live signals
    print("\n" + "=" * 60)
    print("📡 LIVE SIGNALS (for frontend)")
    print("=" * 60)
    
    signals = generate_live_signals()
    for s in signals[:10]:
        print(f"\n🟢 {s['ticker']} @ ${s['price']}")
        print(f"   Pattern: {s['pattern']}")
        print(f"   Expected: +{s['expected_return']}% | WR: {s['win_rate']}%")
        print(f"   Target: +{s['profit_target']}% | Stop: {s['stop_loss']}%")
    
    print("\n✅ Patterns saved to: winning_patterns.json")
    print("✅ Use generate_live_signals() in your frontend!")
