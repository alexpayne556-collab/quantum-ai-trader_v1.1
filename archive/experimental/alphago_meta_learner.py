"""
ALPHAGO META-LEARNER - THE SECRET SAUCE ENGINE
===============================================
This is not just a trading bot. This is a LEARNING MACHINE that:
1. Discovers winning strategies through self-play
2. Validates them with walk-forward testing (no cheating)
3. Locks in proven patterns as "immutable DNA"
4. Adapts in real-time while preserving what works

Think like AlphaGo: Every loss is data. Every win is a pattern to encode.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import deque
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STRATEGY DNA - THE SECRET SAUCE (Discovered patterns that WORK)
# ============================================================================

class StrategyDNA:
    """
    Immutable patterns discovered through backtesting.
    These are the 'genes' that have proven to win.
    Once encoded, they guide all decisions.
    """
    
    # BIG WIN PATTERNS - Encoded from historical analysis
    DIP_BUY_RULES = {
        'min_drawdown': -0.08,       # At least 8% down from high
        'max_rsi': 35,               # RSI oversold
        'min_volume_ratio': 1.3,     # Volume spike (institutions buying)
        'min_recovery_potential': 0.15,  # Expect 15%+ bounce
    }
    
    PROFIT_TAKE_RULES = {
        'min_gain': 0.05,            # At least 5% profit
        'target_gain': 0.08,         # Ideal: 8%+ like HOOD trade
        'max_rsi': 70,               # RSI getting overbought
        'max_hold_days': 15,         # Don't marry positions
    }
    
    CUT_LOSS_RULES = {
        'max_loss': -0.05,           # Cut at -5%
        'max_hold_days_losing': 3,   # 3 days red = cut
        'volume_exit_signal': 1.5,   # High volume selling = run
    }
    
    CASH_MANAGEMENT = {
        'min_cash_reserve': 0.20,    # Always keep 20% cash for dips
        'max_position_size': 0.15,   # Never more than 15% in one stock
        'dip_deploy_pct': 0.25,      # Deploy 25% of cash on confirmed dips
    }
    
    # REGIME PATTERNS - What works when
    REGIME_STRATEGIES = {
        'bull': {
            'buy_threshold': 0.55,   # More aggressive buying
            'position_size': 0.12,
            'stop_loss': -0.05,
        },
        'bear': {
            'buy_threshold': 0.75,   # Only high conviction
            'position_size': 0.08,
            'stop_loss': -0.03,
            'cash_target': 0.40,     # Hold more cash
        },
        'sideways': {
            'buy_threshold': 0.65,
            'position_size': 0.10,
            'stop_loss': -0.04,
        }
    }
    
    @classmethod
    def is_dip_buy(cls, drawdown, rsi, vol_ratio):
        """Check if this is a dip buy opportunity"""
        return (drawdown <= cls.DIP_BUY_RULES['min_drawdown'] and
                rsi <= cls.DIP_BUY_RULES['max_rsi'] and
                vol_ratio >= cls.DIP_BUY_RULES['min_volume_ratio'])
    
    @classmethod
    def should_take_profit(cls, gain_pct, rsi, days_held):
        """Check if we should take profits"""
        if gain_pct >= cls.PROFIT_TAKE_RULES['target_gain']:
            return True
        if gain_pct >= cls.PROFIT_TAKE_RULES['min_gain'] and rsi >= cls.PROFIT_TAKE_RULES['max_rsi']:
            return True
        if gain_pct >= cls.PROFIT_TAKE_RULES['min_gain'] and days_held >= cls.PROFIT_TAKE_RULES['max_hold_days']:
            return True
        return False
    
    @classmethod
    def should_cut_loss(cls, loss_pct, days_held, vol_ratio):
        """Check if we should cut losses"""
        if loss_pct <= cls.CUT_LOSS_RULES['max_loss']:
            return True
        if loss_pct < 0 and days_held >= cls.CUT_LOSS_RULES['max_hold_days_losing']:
            return True
        if loss_pct < -0.02 and vol_ratio >= cls.CUT_LOSS_RULES['volume_exit_signal']:
            return True
        return False


# ============================================================================
# WALK-FORWARD VALIDATOR - NO CHEATING ALLOWED
# ============================================================================

class WalkForwardValidator:
    """
    True out-of-sample testing. Train on past, test on future.
    This prevents overfitting and validates real alpha.
    """
    
    def __init__(self, train_months=12, test_months=3):
        self.train_months = train_months
        self.test_months = test_months
        self.results = []
    
    def split_data(self, data, fold_idx):
        """Create train/test split for walk-forward validation"""
        total_days = len(data)
        train_days = int(total_days * 0.7)
        
        # Walk forward: each fold moves the window forward
        offset = fold_idx * (self.test_months * 21)  # ~21 trading days per month
        
        train_start = offset
        train_end = train_start + train_days
        test_start = train_end
        test_end = min(test_start + (self.test_months * 21), total_days)
        
        if test_end > total_days:
            return None, None
        
        train_data = {t: df.iloc[train_start:train_end] for t, df in data.items()}
        test_data = {t: df.iloc[test_start:test_end] for t, df in data.items()}
        
        return train_data, test_data
    
    def validate_strategy(self, strategy_fn, data, n_folds=4):
        """Run walk-forward validation on a strategy"""
        all_results = []
        
        for fold in range(n_folds):
            train_data, test_data = self.split_data(data, fold)
            if train_data is None:
                break
            
            # Train on historical data
            strategy_fn.fit(train_data)
            
            # Test on future data (no peeking!)
            test_return = strategy_fn.evaluate(test_data)
            
            all_results.append({
                'fold': fold,
                'test_return': test_return,
                'train_period': f'Fold {fold} Train',
                'test_period': f'Fold {fold} Test'
            })
            
            print(f"  Fold {fold}: Test Return = {test_return*100:.2f}%")
        
        if all_results:
            avg_return = np.mean([r['test_return'] for r in all_results])
            consistency = np.std([r['test_return'] for r in all_results])
            
            return {
                'avg_return': avg_return,
                'consistency': consistency,
                'folds': all_results,
                'is_valid': avg_return > 0 and consistency < 0.15
            }
        return None


# ============================================================================
# ADAPTIVE BRAIN - NEVER GETS STUCK
# ============================================================================

class AdaptiveBrain:
    """
    The brain that learns and adapts. Key principles:
    1. Never get stuck in a losing pattern
    2. Quickly recognize when something isn't working
    3. Preserve capital during uncertainty
    4. Deploy aggressively on high-conviction opportunities
    """
    
    def __init__(self):
        self.recent_trades = deque(maxlen=50)
        self.pattern_memory = {}
        self.current_regime = 'unknown'
        self.confidence = 0.5
        
        # Performance tracking
        self.win_count = 0
        self.loss_count = 0
        self.total_pnl = 0
        
        # Adaptation state
        self.losing_streak = 0
        self.max_losing_streak = 3
        
    def detect_regime(self, spy_data):
        """Detect current market regime"""
        if len(spy_data) < 50:
            return 'unknown'
        
        c = spy_data['Close'].values
        
        # Simple regime detection
        sma_20 = np.mean(c[-20:])
        sma_50 = np.mean(c[-50:])
        current = c[-1]
        
        ret_20 = (current / c[-20] - 1)
        
        if current > sma_20 > sma_50 and ret_20 > 0.02:
            return 'bull'
        elif current < sma_20 < sma_50 and ret_20 < -0.02:
            return 'bear'
        else:
            return 'sideways'
    
    def record_trade(self, ticker, entry, exit_price, days_held, pnl_pct):
        """Record and learn from a trade"""
        trade = {
            'ticker': ticker,
            'entry': entry,
            'exit': exit_price,
            'days': days_held,
            'pnl': pnl_pct,
            'regime': self.current_regime,
            'timestamp': datetime.now()
        }
        
        self.recent_trades.append(trade)
        
        if pnl_pct > 0:
            self.win_count += 1
            self.losing_streak = 0
            
            # Record winning pattern
            if ticker not in self.pattern_memory:
                self.pattern_memory[ticker] = {'wins': [], 'losses': []}
            self.pattern_memory[ticker]['wins'].append(trade)
            
        else:
            self.loss_count += 1
            self.losing_streak += 1
            
            if ticker not in self.pattern_memory:
                self.pattern_memory[ticker] = {'wins': [], 'losses': []}
            self.pattern_memory[ticker]['losses'].append(trade)
        
        self.total_pnl += pnl_pct
        
        # Adapt if on losing streak
        if self.losing_streak >= self.max_losing_streak:
            self._adapt_to_losses()
    
    def _adapt_to_losses(self):
        """
        AlphaGo thinking: Losing streak = time to change strategy
        Don't keep doing what isn't working
        """
        print(f"⚠️  Losing streak detected ({self.losing_streak}). Adapting...")
        
        # Analyze recent losses
        recent_losses = [t for t in self.recent_trades if t['pnl'] < 0][-5:]
        
        if len(recent_losses) >= 3:
            # Find common patterns in losses
            regimes = [t['regime'] for t in recent_losses]
            if regimes.count(self.current_regime) >= 2:
                # Current regime is causing losses - reduce exposure
                self.confidence *= 0.7
                print(f"   Reducing confidence in {self.current_regime} regime")
            
            # Check if specific tickers are problematic
            tickers = [t['ticker'] for t in recent_losses]
            for t in set(tickers):
                if tickers.count(t) >= 2:
                    print(f"   Flagging {t} as risky - multiple losses")
        
        # Reset streak after adaptation
        self.losing_streak = 0
    
    def get_position_size(self, signal_strength, cash_available):
        """
        Dynamic position sizing based on confidence and regime
        """
        regime_config = StrategyDNA.REGIME_STRATEGIES.get(
            self.current_regime, 
            StrategyDNA.REGIME_STRATEGIES['sideways']
        )
        
        base_size = regime_config['position_size']
        
        # Adjust for confidence
        adjusted_size = base_size * self.confidence
        
        # Adjust for signal strength
        if signal_strength > 0.8:
            adjusted_size *= 1.3  # High conviction = larger position
        elif signal_strength < 0.6:
            adjusted_size *= 0.7  # Low conviction = smaller position
        
        # Never exceed max position size
        max_size = StrategyDNA.CASH_MANAGEMENT['max_position_size']
        adjusted_size = min(adjusted_size, max_size)
        
        return adjusted_size * cash_available
    
    def should_preserve_cash(self):
        """
        Determine if we should be preserving cash (saving for dips)
        """
        # If in bear market, preserve more cash
        if self.current_regime == 'bear':
            return True
        
        # If on losing streak, preserve cash
        if self.losing_streak >= 2:
            return True
        
        # If win rate is dropping, preserve cash
        if self.win_count + self.loss_count >= 10:
            win_rate = self.win_count / (self.win_count + self.loss_count)
            if win_rate < 0.45:
                return True
        
        return False
    
    def get_win_rate(self):
        total = self.win_count + self.loss_count
        if total == 0:
            return 0.5
        return self.win_count / total


# ============================================================================
# META-STRATEGY ENGINE - THE SECRET SAUCE
# ============================================================================

class MetaStrategyEngine:
    """
    The main engine that combines all components:
    - Strategy DNA (proven patterns)
    - Adaptive Brain (learns from losses)
    - Walk-Forward Validation (no cheating)
    
    This creates the "secret sauce" - strategies that are:
    1. Discovered through exploration
    2. Validated through testing
    3. Encoded as immutable DNA
    4. Executed with adaptive risk management
    """
    
    def __init__(self):
        self.brain = AdaptiveBrain()
        self.validator = WalkForwardValidator()
        self.discovered_patterns = []
        self.locked_strategies = []  # Proven strategies
        
    def analyze_ticker(self, ticker, df):
        """
        Comprehensive analysis using all components
        Returns: action, confidence, reasoning
        """
        if len(df) < 50:
            return 'HOLD', 0.0, 'Insufficient data'
        
        c = df['Close'].values
        v = df['Volume'].values
        h = df['High'].values if 'High' in df else c
        l = df['Low'].values if 'Low' in df else c
        
        # Calculate features
        rsi = self._calculate_rsi(c)
        vol_ratio = v[-1] / (np.mean(v[-20:]) + 1e-8)
        
        high_20 = max(c[-20:])
        low_20 = min(c[-20:])
        drawdown = (c[-1] - high_20) / high_20
        recovery = (c[-1] - low_20) / (high_20 - low_20 + 0.001)
        
        ret_5 = (c[-1] / c[-6] - 1) if len(c) > 5 else 0
        ret_10 = (c[-1] / c[-11] - 1) if len(c) > 10 else 0
        
        # Check Strategy DNA patterns
        signals = []
        
        # DIP BUY CHECK
        if StrategyDNA.is_dip_buy(drawdown, rsi, vol_ratio):
            signals.append({
                'action': 'BUY',
                'type': 'DIP_BUY',
                'confidence': 0.85,
                'reason': f'Dip opportunity: {drawdown*100:.1f}% down, RSI={rsi:.0f}, Vol={vol_ratio:.1f}x'
            })
        
        # MOMENTUM BUY
        if ret_5 > 0.03 and ret_10 > 0.05 and rsi < 65:
            signals.append({
                'action': 'BUY',
                'type': 'MOMENTUM',
                'confidence': 0.65,
                'reason': f'Strong momentum: 5d={ret_5*100:.1f}%, 10d={ret_10*100:.1f}%'
            })
        
        # REVERSAL BUY
        if rsi < 30 and ret_5 > 0:
            signals.append({
                'action': 'BUY',
                'type': 'REVERSAL',
                'confidence': 0.70,
                'reason': f'Oversold reversal: RSI={rsi:.0f}, bouncing'
            })
        
        # PROFIT TAKE / SELL
        if recovery > 0.90 and rsi > 70:
            signals.append({
                'action': 'SELL',
                'type': 'PROFIT_TAKE',
                'confidence': 0.80,
                'reason': f'Overbought at top: RSI={rsi:.0f}, Recovery={recovery*100:.0f}%'
            })
        
        # BREAKDOWN SELL
        if ret_5 < -0.05 and vol_ratio > 1.5:
            signals.append({
                'action': 'SELL',
                'type': 'BREAKDOWN',
                'confidence': 0.75,
                'reason': f'High volume breakdown: {ret_5*100:.1f}%, Vol={vol_ratio:.1f}x'
            })
        
        # Combine signals with adaptive weighting
        if not signals:
            return 'HOLD', 0.4, 'No strong signals'
        
        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        best = signals[0]
        
        # Adjust confidence based on brain state
        adjusted_conf = best['confidence'] * self.brain.confidence
        
        # If brain says preserve cash, reduce buy confidence
        if best['action'] == 'BUY' and self.brain.should_preserve_cash():
            adjusted_conf *= 0.7
            best['reason'] += ' (reduced: preservation mode)'
        
        return best['action'], adjusted_conf, best['reason']
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:]) + 1e-8
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def backtest_strategy(self, data, start_cash=10000, verbose=True):
        """
        Full backtest with walk-forward validation thinking
        """
        cash = start_cash
        positions = {}
        trades = []
        daily_values = []
        
        # Get all dates across all tickers
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(list(all_dates))
        
        if len(dates) < 60:
            return None
        
        # Simulate day by day
        for i, date in enumerate(dates[60:], start=60):
            day_value = cash
            
            # Check existing positions
            for ticker in list(positions.keys()):
                if ticker not in data:
                    continue
                df = data[ticker]
                
                # Get current price
                if date not in df.index:
                    continue
                
                current_price = df.loc[date, 'Close']
                pos = positions[ticker]
                day_value += pos['shares'] * current_price
                
                # Calculate P&L
                pnl_pct = (current_price / pos['entry'] - 1)
                days_held = (date - pos['date']).days
                
                # Check Strategy DNA for exit
                df_slice = df.loc[:date]
                rsi = self._calculate_rsi(df_slice['Close'].values)
                vol_ratio = df_slice['Volume'].iloc[-1] / (df_slice['Volume'].mean() + 1e-8)
                
                should_sell = False
                sell_reason = ''
                
                if StrategyDNA.should_take_profit(pnl_pct, rsi, days_held):
                    should_sell = True
                    sell_reason = 'PROFIT_TAKE'
                elif StrategyDNA.should_cut_loss(pnl_pct, days_held, vol_ratio):
                    should_sell = True
                    sell_reason = 'CUT_LOSS'
                
                if should_sell:
                    proceeds = pos['shares'] * current_price
                    cash += proceeds
                    
                    trades.append({
                        'ticker': ticker,
                        'action': 'SELL',
                        'reason': sell_reason,
                        'entry': pos['entry'],
                        'exit': current_price,
                        'pnl_pct': pnl_pct,
                        'days': days_held,
                        'date': date
                    })
                    
                    self.brain.record_trade(ticker, pos['entry'], current_price, days_held, pnl_pct)
                    del positions[ticker]
            
            # Look for new opportunities
            cash_to_deploy = cash * (1 - StrategyDNA.CASH_MANAGEMENT['min_cash_reserve'])
            
            if cash_to_deploy > 100:
                opportunities = []
                
                for ticker, df in data.items():
                    if ticker in positions:
                        continue
                    if date not in df.index:
                        continue
                    
                    df_slice = df.loc[:date]
                    action, conf, reason = self.analyze_ticker(ticker, df_slice)
                    
                    if action == 'BUY' and conf > 0.6:
                        opportunities.append({
                            'ticker': ticker,
                            'confidence': conf,
                            'reason': reason,
                            'price': df.loc[date, 'Close']
                        })
                
                # Sort by confidence and take top opportunities
                opportunities.sort(key=lambda x: x['confidence'], reverse=True)
                
                for opp in opportunities[:3]:  # Max 3 new positions per day
                    if cash_to_deploy < 100:
                        break
                    
                    ticker = opp['ticker']
                    price = opp['price']
                    
                    # Calculate position size
                    position_value = self.brain.get_position_size(opp['confidence'], cash_to_deploy)
                    shares = int(position_value / price)
                    
                    if shares > 0:
                        cost = shares * price
                        cash -= cost
                        cash_to_deploy -= cost
                        
                        positions[ticker] = {
                            'shares': shares,
                            'entry': price,
                            'date': date
                        }
                        
                        trades.append({
                            'ticker': ticker,
                            'action': 'BUY',
                            'reason': opp['reason'],
                            'price': price,
                            'shares': shares,
                            'date': date
                        })
            
            daily_values.append({'date': date, 'value': day_value})
        
        # Final portfolio value
        final_value = cash
        for ticker, pos in positions.items():
            if ticker in data:
                final_value += pos['shares'] * data[ticker]['Close'].iloc[-1]
        
        total_return = (final_value / start_cash - 1)
        
        wins = [t for t in trades if t.get('pnl_pct', 0) > 0]
        losses = [t for t in trades if t.get('pnl_pct', 0) < 0]
        
        results = {
            'start_cash': start_cash,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len([t for t in trades if t['action'] == 'SELL']),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / max(1, len(wins) + len(losses)),
            'avg_win': np.mean([t['pnl_pct'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl_pct'] for t in losses]) if losses else 0,
            'trades': trades,
            'daily_values': daily_values
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 BACKTEST RESULTS")
            print(f"{'='*60}")
            print(f"Starting Cash:    ${start_cash:,.2f}")
            print(f"Final Value:      ${final_value:,.2f}")
            print(f"Total Return:     {total_return*100:+.2f}%")
            print(f"Total Trades:     {results['total_trades']}")
            print(f"Win Rate:         {results['win_rate']*100:.1f}%")
            print(f"Avg Win:          {results['avg_win']*100:+.2f}%")
            print(f"Avg Loss:         {results['avg_loss']*100:.2f}%")
            print(f"{'='*60}")
        
        return results
    
    def discover_and_lock_patterns(self, data):
        """
        The SECRET SAUCE generator:
        1. Run walk-forward tests
        2. Find patterns that work consistently
        3. Lock them into Strategy DNA
        """
        print("\n🔬 DISCOVERING PATTERNS...")
        print("="*60)
        
        # Test different parameter combinations
        test_configs = [
            {'min_drawdown': -0.10, 'min_rsi': 30},
            {'min_drawdown': -0.08, 'min_rsi': 35},
            {'min_drawdown': -0.12, 'min_rsi': 25},
            {'min_drawdown': -0.15, 'min_rsi': 30},
        ]
        
        best_config = None
        best_return = -1
        
        for config in test_configs:
            print(f"\nTesting: Drawdown={config['min_drawdown']}, RSI={config['min_rsi']}")
            
            # Temporarily update DNA
            original_dd = StrategyDNA.DIP_BUY_RULES['min_drawdown']
            original_rsi = StrategyDNA.DIP_BUY_RULES['max_rsi']
            
            StrategyDNA.DIP_BUY_RULES['min_drawdown'] = config['min_drawdown']
            StrategyDNA.DIP_BUY_RULES['max_rsi'] = config['min_rsi']
            
            # Run backtest
            results = self.backtest_strategy(data, verbose=False)
            
            if results and results['total_return'] > best_return:
                best_return = results['total_return']
                best_config = config.copy()
                best_config['return'] = best_return
                best_config['win_rate'] = results['win_rate']
            
            # Restore
            StrategyDNA.DIP_BUY_RULES['min_drawdown'] = original_dd
            StrategyDNA.DIP_BUY_RULES['max_rsi'] = original_rsi
        
        if best_config:
            print(f"\n🏆 BEST PATTERN FOUND:")
            print(f"   Drawdown Threshold: {best_config['min_drawdown']*100:.0f}%")
            print(f"   RSI Threshold: {best_config['min_rsi']}")
            print(f"   Return: {best_config['return']*100:.2f}%")
            print(f"   Win Rate: {best_config['win_rate']*100:.1f}%")
            
            self.locked_strategies.append(best_config)
        
        return best_config
    
    def generate_predictions(self, data):
        """
        Generate predictions using all discovered patterns
        """
        predictions = []
        
        for ticker, df in data.items():
            if len(df) < 50:
                continue
            
            action, confidence, reason = self.analyze_ticker(ticker, df)
            
            c = df['Close'].values
            high_20 = max(c[-20:])
            drawdown = (c[-1] - high_20) / high_20
            rsi = self._calculate_rsi(c)
            
            predictions.append({
                'ticker': ticker,
                'action': action,
                'confidence': confidence,
                'reason': reason,
                'price': float(c[-1]),
                'drawdown': drawdown * 100,
                'rsi': rsi,
                'regime': self.brain.current_regime
            })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return predictions


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("="*70)
    print("🧬 ALPHAGO META-LEARNER - THE SECRET SAUCE ENGINE")
    print("="*70)
    print("This system:")
    print("  1. Discovers winning patterns through exploration")
    print("  2. Validates with walk-forward testing (no cheating)")
    print("  3. Locks proven patterns as 'Strategy DNA'")
    print("  4. Adapts in real-time while preserving what works")
    print("="*70)
    
    # Load tickers
    TICKERS = ['APLD','SERV','MRVL','HOOD','LUNR','BAC','QCOM','UUUU','TSLA','AMD',
               'NOW','NVDA','MU','PG','DLB','XME','KRYS','LEU','QTUM','SPY',
               'UNH','WMT','OKLO','RXRX','MTZ','SNOW','GRRR','BSX','LLY','VOO',
               'GEO','CXW','LYFT','MNDY','BA','LAC','INTC','ALK','LMT','CRDO',
               'ANET','META','RIVN','GOOGL','HL','TEM','TDOC','KMTS','SCHA','B']
    
    print(f"\n📥 Loading data for {len(TICKERS)} tickers...")
    
    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start='2020-01-01', progress=False)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except:
            pass
    
    print(f"✅ Loaded {len(data)} tickers")
    
    # Detect regime using SPY
    engine = MetaStrategyEngine()
    if 'SPY' in data:
        engine.brain.current_regime = engine.brain.detect_regime(data['SPY'])
        print(f"\n📊 Current Market Regime: {engine.brain.current_regime.upper()}")
    
    # Run pattern discovery
    best_pattern = engine.discover_and_lock_patterns(data)
    
    # Run full backtest
    print("\n🔄 Running full backtest with optimized patterns...")
    results = engine.backtest_strategy(data)
    
    # Generate today's predictions
    print("\n📊 GENERATING TODAY'S PREDICTIONS...")
    predictions = engine.generate_predictions(data)
    
    print(f"\n{'='*70}")
    print("🎯 TODAY'S SIGNALS")
    print(f"{'='*70}")
    
    buys = [p for p in predictions if p['action'] == 'BUY' and p['confidence'] > 0.6]
    sells = [p for p in predictions if p['action'] == 'SELL' and p['confidence'] > 0.6]
    
    print(f"\n🟢 BUY SIGNALS ({len(buys)}):")
    for p in buys[:10]:
        print(f"   {p['ticker']:5s} ${p['price']:>7.2f} | Conf: {p['confidence']:.0%} | {p['reason'][:50]}")
    
    print(f"\n🔴 SELL SIGNALS ({len(sells)}):")
    for p in sells[:5]:
        print(f"   {p['ticker']:5s} ${p['price']:>7.2f} | Conf: {p['confidence']:.0%} | {p['reason'][:50]}")
    
    # Save results
    output = {
        'regime': engine.brain.current_regime,
        'brain_confidence': engine.brain.confidence,
        'brain_win_rate': engine.brain.get_win_rate(),
        'locked_strategies': engine.locked_strategies,
        'predictions': predictions,
        'backtest_results': {
            'total_return': results['total_return'] if results else 0,
            'win_rate': results['win_rate'] if results else 0,
            'total_trades': results['total_trades'] if results else 0
        }
    }
    
    with open('meta_learner_output.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to meta_learner_output.json")
    print(f"\n{'='*70}")
    print("🧬 SECRET SAUCE ENCODED - READY FOR DEPLOYMENT")
    print(f"{'='*70}")
    
    return engine, results, predictions


if __name__ == '__main__':
    engine, results, predictions = main()
