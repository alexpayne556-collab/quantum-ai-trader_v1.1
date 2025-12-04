"""
🏆 CHAMPIONSHIP ARENA - AI vs CHAMPION STRATEGIES
==================================================
This is the DRILL SERGEANT. The AI must beat these champions to graduate.

OPPONENTS:
1. BUY_AND_HOLD - The lazy benchmark (hard to beat long-term)
2. MOMENTUM_MASTER - Rides trends like a pro
3. DIP_HUNTER - Buys blood in the streets
4. MEAN_REVERSION - Fades extremes
5. YOUR_TRADES - Your actual Robinhood history (the human benchmark)

The AI must beat ALL of them to graduate to paper trading.
If it loses, it LEARNS from the winner and tries again.

"I fear not the man who has practiced 10,000 kicks once, 
 but I fear the man who has practiced one kick 10,000 times." - Bruce Lee
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from collections import deque
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CHAMPION STRATEGIES - THE OPPONENTS TO BEAT
# ============================================================================

class Champion:
    """Base class for champion strategies"""
    name = "Base Champion"
    description = "Override this"
    
    def decide(self, ticker, data, day, positions, cash):
        """Return 'buy', 'sell', or 'hold'"""
        raise NotImplementedError
    
    def backtest(self, all_data, start_cash=10000):
        """Run full backtest, return results"""
        cash = start_cash
        positions = {}
        trades = []
        daily_values = []
        
        # Get minimum length across all tickers
        min_len = min(len(df) for df in all_data.values())
        
        for day in range(60, min_len - 1):
            day_value = cash
            
            # Check existing positions
            for ticker in list(positions.keys()):
                if ticker not in all_data:
                    continue
                df = all_data[ticker]
                if day >= len(df):
                    continue
                
                price = df['Close'].iloc[day]
                pos = positions[ticker]
                day_value += pos['shares'] * price
                
                # Decide
                action = self.decide(ticker, df, day, positions, cash)
                
                if action == 'sell':
                    proceeds = pos['shares'] * price
                    pnl = (price / pos['entry'] - 1) * 100
                    cash += proceeds
                    trades.append({
                        'ticker': ticker, 'action': 'sell',
                        'price': price, 'pnl': pnl, 'day': day
                    })
                    del positions[ticker]
            
            # Look for buys
            for ticker, df in all_data.items():
                if ticker in positions:
                    continue
                if day >= len(df):
                    continue
                
                action = self.decide(ticker, df, day, positions, cash)
                
                if action == 'buy' and cash > 100:
                    price = df['Close'].iloc[day]
                    # Position sizing: 10% of portfolio per trade
                    position_size = min(cash * 0.10, cash * 0.95)
                    shares = int(position_size / price)
                    
                    if shares > 0:
                        cost = shares * price
                        cash -= cost
                        positions[ticker] = {
                            'shares': shares,
                            'entry': price,
                            'day': day
                        }
                        trades.append({
                            'ticker': ticker, 'action': 'buy',
                            'price': price, 'day': day
                        })
            
            # Recalculate day value
            day_value = cash
            for ticker, pos in positions.items():
                if ticker in all_data and day < len(all_data[ticker]):
                    day_value += pos['shares'] * all_data[ticker]['Close'].iloc[day]
            
            daily_values.append({'day': day, 'value': day_value})
        
        # Final value
        final_value = cash
        for ticker, pos in positions.items():
            if ticker in all_data:
                final_value += pos['shares'] * all_data[ticker]['Close'].iloc[-1]
        
        # Calculate metrics
        wins = [t for t in trades if t.get('pnl', 0) > 0]
        losses = [t for t in trades if t.get('pnl', 0) < 0]
        
        return {
            'champion': self.name,
            'start_cash': start_cash,
            'final_value': final_value,
            'total_return': (final_value / start_cash - 1) * 100,
            'trades': len([t for t in trades if t['action'] == 'sell']),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / max(1, len(wins) + len(losses)),
            'avg_win': np.mean([t['pnl'] for t in wins]) if wins else 0,
            'avg_loss': np.mean([t['pnl'] for t in losses]) if losses else 0,
            'daily_values': daily_values,
            'trade_log': trades
        }


class BuyAndHold(Champion):
    """The lazy benchmark - just buy SPY and hold"""
    name = "BUY & HOLD"
    description = "Buy SPY and hold forever"
    
    def decide(self, ticker, data, day, positions, cash):
        if ticker == 'SPY' and ticker not in positions and day == 60:
            return 'buy'
        return 'hold'


class MomentumMaster(Champion):
    """Rides strong trends"""
    name = "MOMENTUM MASTER"
    description = "Buy strength, sell weakness"
    
    def decide(self, ticker, data, day, positions, cash):
        if day < 20:
            return 'hold'
        
        c = data['Close'].iloc[day-20:day+1].values
        ret_5 = (c[-1] / c[-6] - 1) if len(c) > 5 else 0
        ret_10 = (c[-1] / c[-11] - 1) if len(c) > 10 else 0
        ret_20 = (c[-1] / c[0] - 1) if len(c) > 0 else 0
        
        # RSI
        deltas = np.diff(c)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gain = np.mean(gains[-14:])
        avg_loss = abs(np.mean(losses[-14:])) + 1e-8
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
        
        if ticker in positions:
            pos = positions[ticker]
            pnl = (c[-1] / pos['entry'] - 1)
            
            # Sell on weakness or profit target
            if ret_5 < -0.05 or pnl > 0.10 or pnl < -0.05:
                return 'sell'
            return 'hold'
        else:
            # Buy on strong momentum
            if ret_5 > 0.03 and ret_10 > 0.05 and ret_20 > 0.08 and rsi < 70:
                return 'buy'
            return 'hold'


class DipHunter(Champion):
    """Buys blood in the streets"""
    name = "DIP HUNTER"
    description = "Buy the dip, sell the rip"
    
    def decide(self, ticker, data, day, positions, cash):
        if day < 20:
            return 'hold'
        
        c = data['Close'].iloc[day-20:day+1].values
        v = data['Volume'].iloc[day-20:day+1].values
        
        high_20 = max(c)
        drawdown = (c[-1] - high_20) / high_20
        
        # RSI
        deltas = np.diff(c)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gain = np.mean(gains[-14:])
        avg_loss = abs(np.mean(losses[-14:])) + 1e-8
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
        
        vol_ratio = v[-1] / (np.mean(v) + 1)
        
        if ticker in positions:
            pos = positions[ticker]
            pnl = (c[-1] / pos['entry'] - 1)
            days_held = day - pos['day']
            
            # Take profit at 8%+ or cut loss at -5%
            if pnl > 0.08 or pnl < -0.05 or (days_held > 10 and pnl > 0.03):
                return 'sell'
            return 'hold'
        else:
            # Buy the dip: 10%+ down, oversold, volume spike
            if drawdown < -0.08 and rsi < 35 and vol_ratio > 1.3:
                return 'buy'
            return 'hold'


class MeanReversion(Champion):
    """Fades extremes"""
    name = "MEAN REVERSION"
    description = "Buy oversold, sell overbought"
    
    def decide(self, ticker, data, day, positions, cash):
        if day < 20:
            return 'hold'
        
        c = data['Close'].iloc[day-20:day+1].values
        sma_20 = np.mean(c)
        std_20 = np.std(c)
        
        z_score = (c[-1] - sma_20) / (std_20 + 1e-8)
        
        # RSI
        deltas = np.diff(c)
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        avg_gain = np.mean(gains[-14:])
        avg_loss = abs(np.mean(losses[-14:])) + 1e-8
        rsi = 100 - 100 / (1 + avg_gain / avg_loss)
        
        if ticker in positions:
            pos = positions[ticker]
            pnl = (c[-1] / pos['entry'] - 1)
            
            # Sell when mean reverted or stop loss
            if z_score > 1.5 or pnl > 0.06 or pnl < -0.04:
                return 'sell'
            return 'hold'
        else:
            # Buy when extremely oversold
            if z_score < -2.0 and rsi < 30:
                return 'buy'
            return 'hold'


class HumanTrader(Champion):
    """Your actual Robinhood trades - the human benchmark"""
    name = "HUMAN (YOU)"
    description = "Your actual trading history"
    
    def __init__(self, trade_history=None):
        self.trade_history = trade_history or []
        self.trade_index = 0
    
    def load_trades(self, filepath):
        """Load trades from JSON file"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.trade_history = json.load(f)
    
    def decide(self, ticker, data, day, positions, cash):
        # Replay human trades
        # This would need actual trade history from Robinhood
        return 'hold'


# ============================================================================
# ARENA - WHERE CHAMPIONS COMPETE
# ============================================================================

class ChampionshipArena:
    """
    The training arena where AI fights champions.
    
    Process:
    1. All champions trade the same data
    2. AI trades the same data
    3. Compare results
    4. AI learns from winners
    5. Repeat until AI beats all champions
    """
    
    def __init__(self, data):
        self.data = data
        self.champions = [
            BuyAndHold(),
            MomentumMaster(),
            DipHunter(),
            MeanReversion(),
        ]
        self.leaderboard = []
        self.ai_history = []
        self.graduation_threshold = 3  # Must beat 3 out of 4 champions
        
    def run_tournament(self, ai_strategy=None):
        """
        Run all champions and AI on the same data.
        Returns leaderboard.
        """
        print("\n" + "="*70)
        print("🏆 CHAMPIONSHIP TOURNAMENT")
        print("="*70)
        
        results = []
        
        # Run each champion
        for champion in self.champions:
            print(f"\n⚔️  Running {champion.name}...")
            result = champion.backtest(self.data)
            results.append(result)
            print(f"   Return: {result['total_return']:+.2f}% | "
                  f"Win Rate: {result['win_rate']:.0%} | "
                  f"Trades: {result['trades']}")
        
        # Run AI if provided
        if ai_strategy:
            print(f"\n🤖 Running AI CHALLENGER...")
            ai_result = ai_strategy.backtest(self.data)
            ai_result['champion'] = 'AI CHALLENGER'
            results.append(ai_result)
            print(f"   Return: {ai_result['total_return']:+.2f}% | "
                  f"Win Rate: {ai_result['win_rate']:.0%} | "
                  f"Trades: {ai_result['trades']}")
        
        # Sort by return
        results.sort(key=lambda x: x['total_return'], reverse=True)
        self.leaderboard = results
        
        return results
    
    def display_leaderboard(self):
        """Display tournament results"""
        print("\n" + "="*70)
        print("🏆 LEADERBOARD")
        print("="*70)
        print(f"{'Rank':<6} {'Champion':<20} {'Return':>10} {'Win Rate':>10} {'Trades':>8}")
        print("-"*70)
        
        for i, result in enumerate(self.leaderboard):
            medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
            is_ai = result['champion'] == 'AI CHALLENGER'
            name = f"{'→ ' if is_ai else ''}{result['champion']}"
            
            print(f"{medal} {i+1:<3} {name:<20} {result['total_return']:>+9.2f}% "
                  f"{result['win_rate']:>9.0%} {result['trades']:>8}")
        
        print("="*70)
    
    def ai_vs_champions(self, ai_result):
        """
        Compare AI to champions.
        Returns: (wins, losses, lessons_learned)
        """
        wins = 0
        losses = 0
        lessons = []
        
        for result in self.leaderboard:
            if result['champion'] == 'AI CHALLENGER':
                continue
            
            if ai_result['total_return'] > result['total_return']:
                wins += 1
                print(f"✅ AI BEATS {result['champion']}: "
                      f"{ai_result['total_return']:.2f}% vs {result['total_return']:.2f}%")
            else:
                losses += 1
                print(f"❌ AI LOSES to {result['champion']}: "
                      f"{ai_result['total_return']:.2f}% vs {result['total_return']:.2f}%")
                
                # Learn from the champion
                if result['win_rate'] > ai_result['win_rate']:
                    lessons.append(f"Learn from {result['champion']}: Better win rate ({result['win_rate']:.0%})")
                if result['avg_win'] > ai_result.get('avg_win', 0):
                    lessons.append(f"Learn from {result['champion']}: Bigger wins ({result['avg_win']:.1f}%)")
                if abs(result.get('avg_loss', -999)) < abs(ai_result.get('avg_loss', -999)):
                    lessons.append(f"Learn from {result['champion']}: Smaller losses ({result['avg_loss']:.1f}%)")
        
        return wins, losses, lessons
    
    def graduation_check(self, ai_result):
        """
        Check if AI has graduated (beaten enough champions)
        """
        wins, losses, lessons = self.ai_vs_champions(ai_result)
        
        print(f"\n{'='*70}")
        print(f"📊 AI RECORD: {wins} WINS - {losses} LOSSES")
        
        if wins >= self.graduation_threshold:
            print(f"🎓 GRADUATION ACHIEVED! AI beats {wins}/{len(self.champions)} champions!")
            print(f"   AI is READY for paper trading!")
            return True, lessons
        else:
            print(f"📚 NOT YET READY. Need {self.graduation_threshold - wins} more wins.")
            print(f"\n📝 LESSONS TO LEARN:")
            for lesson in lessons:
                print(f"   • {lesson}")
            return False, lessons


# ============================================================================
# TRAINING CAMP - WHERE AI LEARNS FROM CHAMPIONS
# ============================================================================

class TrainingCamp:
    """
    Training camp where AI learns from champion strategies.
    
    Process:
    1. Analyze what makes each champion successful
    2. Extract their patterns
    3. Feed to AI as training signal
    4. Test against champions
    5. Repeat until graduation
    """
    
    def __init__(self, arena):
        self.arena = arena
        self.training_sessions = []
        self.best_ai_performance = {'total_return': -999}
        
    def extract_champion_patterns(self, champion_result):
        """
        Extract winning patterns from a champion's trades.
        """
        patterns = {
            'avg_hold_time': [],
            'entry_conditions': [],
            'exit_conditions': [],
            'win_setups': [],
            'loss_setups': [],
        }
        
        trades = champion_result.get('trade_log', [])
        
        # Analyze trades
        buys = [t for t in trades if t['action'] == 'buy']
        sells = [t for t in trades if t['action'] == 'sell']
        
        for sell in sells:
            # Find matching buy
            matching_buys = [b for b in buys 
                           if b['ticker'] == sell['ticker'] and b['day'] < sell['day']]
            if matching_buys:
                buy = matching_buys[-1]
                hold_time = sell['day'] - buy['day']
                patterns['avg_hold_time'].append(hold_time)
                
                if sell.get('pnl', 0) > 0:
                    patterns['win_setups'].append({
                        'ticker': sell['ticker'],
                        'hold_time': hold_time,
                        'pnl': sell['pnl']
                    })
                else:
                    patterns['loss_setups'].append({
                        'ticker': sell['ticker'],
                        'hold_time': hold_time,
                        'pnl': sell['pnl']
                    })
        
        return patterns
    
    def train_ai_from_champions(self, ai_brain, n_sessions=10):
        """
        Train AI by learning from champion patterns.
        """
        print("\n" + "="*70)
        print("🏋️  TRAINING CAMP - LEARNING FROM CHAMPIONS")
        print("="*70)
        
        for session in range(n_sessions):
            print(f"\n📚 Training Session {session + 1}/{n_sessions}")
            
            # Run tournament to get champion results
            results = self.arena.run_tournament()
            
            # Extract patterns from each champion
            for result in results:
                patterns = self.extract_champion_patterns(result)
                
                if patterns['win_setups']:
                    avg_win_hold = np.mean([s['hold_time'] for s in patterns['win_setups']])
                    avg_win_pnl = np.mean([s['pnl'] for s in patterns['win_setups']])
                    print(f"   {result['champion']}: Avg win hold {avg_win_hold:.0f} days, "
                          f"Avg win {avg_win_pnl:.1f}%")
            
            self.training_sessions.append({
                'session': session,
                'results': results,
                'timestamp': datetime.now().isoformat()
            })
        
        return self.training_sessions


# ============================================================================
# PAPER TRADING SIMULATOR - WHERE GRADUATES COMPETE LIVE
# ============================================================================

class PaperTradingSimulator:
    """
    Paper trading simulator for graduated AI.
    
    Features:
    - Real-time data simulation
    - Side-by-side with human trades
    - Performance tracking
    - Learning from live results
    """
    
    def __init__(self, starting_cash=10000):
        self.starting_cash = starting_cash
        self.ai_cash = starting_cash
        self.human_cash = starting_cash
        self.ai_positions = {}
        self.human_positions = {}
        self.ai_trades = []
        self.human_trades = []
        self.daily_log = []
        
    def log_ai_trade(self, ticker, action, price, shares, reason=""):
        """Log an AI trade"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'trader': 'AI',
            'ticker': ticker,
            'action': action,
            'price': price,
            'shares': shares,
            'reason': reason
        }
        self.ai_trades.append(trade)
        print(f"🤖 AI {action.upper()}: {shares} {ticker} @ ${price:.2f} - {reason}")
        return trade
    
    def log_human_trade(self, ticker, action, price, shares, reason=""):
        """Log a human trade"""
        trade = {
            'timestamp': datetime.now().isoformat(),
            'trader': 'HUMAN',
            'ticker': ticker,
            'action': action,
            'price': price,
            'shares': shares,
            'reason': reason
        }
        self.human_trades.append(trade)
        print(f"👤 HUMAN {action.upper()}: {shares} {ticker} @ ${price:.2f} - {reason}")
        return trade
    
    def daily_update(self, prices):
        """
        Update daily values for both AI and human.
        prices: dict of {ticker: current_price}
        """
        ai_value = self.ai_cash
        human_value = self.human_cash
        
        for ticker, pos in self.ai_positions.items():
            if ticker in prices:
                ai_value += pos['shares'] * prices[ticker]
        
        for ticker, pos in self.human_positions.items():
            if ticker in prices:
                human_value += pos['shares'] * prices[ticker]
        
        log_entry = {
            'date': datetime.now().isoformat(),
            'ai_value': ai_value,
            'human_value': human_value,
            'ai_return': (ai_value / self.starting_cash - 1) * 100,
            'human_return': (human_value / self.starting_cash - 1) * 100,
            'ai_trades_today': len([t for t in self.ai_trades 
                                   if t['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))]),
            'human_trades_today': len([t for t in self.human_trades 
                                      if t['timestamp'].startswith(datetime.now().strftime('%Y-%m-%d'))]),
        }
        
        self.daily_log.append(log_entry)
        
        # Display comparison
        print(f"\n📊 DAILY UPDATE - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"   🤖 AI:    ${ai_value:,.2f} ({log_entry['ai_return']:+.2f}%)")
        print(f"   👤 HUMAN: ${human_value:,.2f} ({log_entry['human_return']:+.2f}%)")
        
        if log_entry['ai_return'] > log_entry['human_return']:
            print(f"   🏆 AI is WINNING by {log_entry['ai_return'] - log_entry['human_return']:.2f}%")
        else:
            print(f"   🏆 HUMAN is WINNING by {log_entry['human_return'] - log_entry['ai_return']:.2f}%")
        
        return log_entry
    
    def get_scoreboard(self):
        """Get current scoreboard"""
        if not self.daily_log:
            return None
        
        latest = self.daily_log[-1]
        
        scoreboard = {
            'ai': {
                'current_value': latest['ai_value'],
                'total_return': latest['ai_return'],
                'total_trades': len(self.ai_trades),
                'positions': len(self.ai_positions),
            },
            'human': {
                'current_value': latest['human_value'],
                'total_return': latest['human_return'],
                'total_trades': len(self.human_trades),
                'positions': len(self.human_positions),
            },
            'leader': 'AI' if latest['ai_return'] > latest['human_return'] else 'HUMAN',
            'lead_by': abs(latest['ai_return'] - latest['human_return']),
        }
        
        return scoreboard
    
    def save_state(self, filepath='paper_trading_state.json'):
        """Save current state"""
        state = {
            'ai_cash': self.ai_cash,
            'human_cash': self.human_cash,
            'ai_positions': self.ai_positions,
            'human_positions': self.human_positions,
            'ai_trades': self.ai_trades,
            'human_trades': self.human_trades,
            'daily_log': self.daily_log,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print(f"✅ State saved to {filepath}")
        return filepath
    
    def load_state(self, filepath='paper_trading_state.json'):
        """Load saved state"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.ai_cash = state['ai_cash']
            self.human_cash = state['human_cash']
            self.ai_positions = state['ai_positions']
            self.human_positions = state['human_positions']
            self.ai_trades = state['ai_trades']
            self.human_trades = state['human_trades']
            self.daily_log = state['daily_log']
            
            print(f"✅ State loaded from {filepath}")
            return True
        return False


# ============================================================================
# MAIN - RUN THE CHAMPIONSHIP
# ============================================================================

def main():
    print("="*70)
    print("🏆 CHAMPIONSHIP ARENA - AI vs CHAMPION STRATEGIES")
    print("="*70)
    print("\nThe AI must beat the champions to graduate to paper trading.")
    print("This is where it learns to be a CONFIDENT PRO.\n")
    
    # Load tickers
    TICKERS = ['APLD','SERV','MRVL','HOOD','LUNR','BAC','QCOM','UUUU','TSLA','AMD',
               'NOW','NVDA','MU','PG','DLB','XME','KRYS','LEU','QTUM','SPY',
               'UNH','WMT','OKLO','RXRX','MTZ','SNOW','GRRR','BSX','LLY','VOO',
               'GEO','CXW','LYFT','MNDY','BA','LAC','INTC','ALK','LMT','CRDO',
               'ANET','META','RIVN','GOOGL','HL','TEM','TDOC','KMTS','SCHA','B']
    
    print(f"📥 Loading data for {len(TICKERS)} tickers...")
    
    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start='2022-01-01', progress=False)
            if len(df) > 100:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                data[ticker] = df
        except:
            pass
    
    print(f"✅ Loaded {len(data)} tickers\n")
    
    # Create arena
    arena = ChampionshipArena(data)
    
    # Run tournament
    results = arena.run_tournament()
    arena.display_leaderboard()
    
    # Save results
    output = {
        'tournament_date': datetime.now().isoformat(),
        'leaderboard': [
            {
                'rank': i + 1,
                'champion': r['champion'],
                'total_return': r['total_return'],
                'win_rate': r['win_rate'],
                'trades': r['trades'],
            }
            for i, r in enumerate(results)
        ]
    }
    
    with open('championship_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to championship_results.json")
    
    # Next steps
    print("\n" + "="*70)
    print("📋 NEXT STEPS TO TRAIN AI:")
    print("="*70)
    print("1. Run ALPHAGO_TRADER.ipynb in Colab to train the AI brain")
    print("2. Download trained model files")
    print("3. Run this arena with AI to see if it can beat the champions")
    print("4. If AI wins 3/4 champions → GRADUATE to paper trading")
    print("5. Paper trade AI vs YOUR trades for 1 week")
    print("6. Learn from results and improve")
    print("="*70)
    
    return arena, results


if __name__ == '__main__':
    arena, results = main()
