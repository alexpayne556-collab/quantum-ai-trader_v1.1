#!/usr/bin/env python3
"""
=============================================================================
SKILL CAPTURE ENGINE - Learn from YOUR 20%+ Monthly Returns
=============================================================================

This module captures and analyzes YOUR trading patterns to:
1. Identify what makes YOUR trades successful
2. Find patterns in your best vs worst trades
3. Build a "trading fingerprint" of your edge
4. Combine your intuition with systematic signals

Your 20%+ monthly return shows REAL EDGE. Let's quantify it.

Usage:
    python SKILL_CAPTURE_ENGINE.py

Commands:
    manual    - Log a manual trade with full context
    analyze   - Deep analysis of your trading patterns
    fingerprint - Build your trading edge profile
    coach     - Get AI coaching based on your patterns
"""

import os
import sys
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import warnings

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings('ignore')


# Data path
DATA_PATH = Path('./companion_data/')
DATA_PATH.mkdir(exist_ok=True)


# ============================================================================
# MARKET CONTEXT CAPTURER
# ============================================================================

class MarketContextCapturer:
    """Captures market state at the moment of each trade."""
    
    @staticmethod
    def get_market_context(symbol: str, trade_date: datetime = None) -> Dict:
        """Get comprehensive market context for a trade."""
        if trade_date is None:
            trade_date = datetime.now()
        
        context = {
            'timestamp': trade_date.isoformat(),
            'symbol': symbol,
        }
        
        # Fetch recent data
        try:
            # Asset data
            df = yf.download(symbol, period='60d', progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [c.lower().replace(' ', '_') for c in df.columns]
            
            if not df.empty:
                price = df['close'].iloc[-1]
                context['price'] = price
                
                # Momentum
                context['return_1d'] = df['close'].pct_change(1).iloc[-1]
                context['return_5d'] = df['close'].pct_change(5).iloc[-1]
                context['return_21d'] = df['close'].pct_change(21).iloc[-1]
                
                # Relative to MAs
                ma20 = df['close'].rolling(20).mean().iloc[-1]
                ma50 = df['close'].rolling(50).mean().iloc[-1]
                context['vs_ma20'] = (price / ma20 - 1) * 100
                context['vs_ma50'] = (price / ma50 - 1) * 100
                
                # Volatility
                context['volatility_20d'] = df['close'].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
                
                # Volume
                avg_vol = df['volume'].rolling(20).mean().iloc[-1]
                context['vol_ratio'] = df['volume'].iloc[-1] / avg_vol if avg_vol > 0 else 1
                
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                context['rsi'] = rsi.iloc[-1]
                
                # 52-week position
                high_52w = df['close'].rolling(252).max().iloc[-1] if len(df) >= 252 else df['close'].max()
                low_52w = df['close'].rolling(252).min().iloc[-1] if len(df) >= 252 else df['close'].min()
                context['pct_from_52w_high'] = (price / high_52w - 1) * 100
                context['pct_from_52w_low'] = (price / low_52w - 1) * 100
            
            # VIX
            vix = yf.download('^VIX', period='5d', progress=False)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix.columns = [c.lower().replace(' ', '_') for c in vix.columns]
            if not vix.empty:
                context['vix'] = vix['close'].iloc[-1]
            
            # SPY (market)
            spy = yf.download('SPY', period='22d', progress=False)
            if isinstance(spy.columns, pd.MultiIndex):
                spy.columns = spy.columns.get_level_values(0)
            spy.columns = [c.lower().replace(' ', '_') for c in spy.columns]
            if not spy.empty:
                context['spy_return_5d'] = spy['close'].pct_change(5).iloc[-1]
                context['spy_return_21d'] = spy['close'].pct_change(21).iloc[-1]
        
        except Exception as e:
            context['error'] = str(e)
        
        return context


# ============================================================================
# ENHANCED TRADE JOURNAL
# ============================================================================

class EnhancedTradeJournal:
    """
    Enhanced trade journal that captures YOUR trading patterns.
    """
    
    def __init__(self):
        self.journal_file = DATA_PATH / 'enhanced_trade_journal.json'
        self.trades = []
        self.load_journal()
    
    def load_journal(self):
        """Load journal from file."""
        if self.journal_file.exists():
            with open(self.journal_file, 'r') as f:
                self.trades = json.load(f)
    
    def save_journal(self):
        """Save journal to file."""
        with open(self.journal_file, 'w') as f:
            json.dump(self.trades, f, indent=2, default=str)
    
    def log_manual_trade(self, 
                         symbol: str,
                         side: str,
                         entry_price: float,
                         position_size: float,
                         reasoning: str,
                         confidence: int = 5,
                         setup_type: str = 'unknown',
                         catalyst: str = None,
                         timeframe: str = 'swing',  # scalp, day, swing, position
                         stop_loss: float = None,
                         target: float = None) -> int:
        """
        Log a manual trade with full context capture.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            entry_price: Entry price
            position_size: Dollar amount or shares
            reasoning: Your reasoning in your own words
            confidence: 1-10 confidence level
            setup_type: Type of setup (breakout, pullback, reversal, momentum, etc.)
            catalyst: News/event catalyst if any
            timeframe: Expected holding period
            stop_loss: Stop loss price if set
            target: Target price if set
        """
        # Capture market context automatically
        context = MarketContextCapturer.get_market_context(symbol)
        
        trade = {
            'id': len(self.trades) + 1,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol.upper(),
            'side': side.lower(),
            'entry_price': entry_price,
            'position_size': position_size,
            'reasoning': reasoning,
            'confidence': confidence,
            'setup_type': setup_type.lower(),
            'catalyst': catalyst,
            'timeframe': timeframe.lower(),
            'stop_loss': stop_loss,
            'target': target,
            'risk_reward': (target - entry_price) / (entry_price - stop_loss) if stop_loss and target else None,
            'market_context': context,
            'status': 'open',
            'exit_price': None,
            'exit_timestamp': None,
            'exit_reasoning': None,
            'partial_exits': [],
            'pnl': None,
            'pnl_pct': None,
            'lessons': None,
            'rating': None,  # Self-rate the trade 1-10
        }
        
        self.trades.append(trade)
        self.save_journal()
        
        print(f"\n✅ Trade #{trade['id']} logged: {side.upper()} {symbol}")
        print(f"   Entry: ${entry_price:.2f}")
        print(f"   Size: ${position_size:,.0f}")
        print(f"   Setup: {setup_type}")
        print(f"   Confidence: {confidence}/10")
        if stop_loss:
            risk = abs(entry_price - stop_loss) / entry_price * 100
            print(f"   Stop Loss: ${stop_loss:.2f} ({risk:.1f}% risk)")
        if target:
            reward = abs(target - entry_price) / entry_price * 100
            print(f"   Target: ${target:.2f} ({reward:.1f}% reward)")
        if trade['risk_reward']:
            print(f"   R:R Ratio: {trade['risk_reward']:.1f}:1")
        
        # Market context snapshot
        print(f"\n   📊 Market Context Captured:")
        print(f"      VIX: {context.get('vix', 'N/A'):.1f}" if context.get('vix') else "      VIX: N/A")
        print(f"      RSI: {context.get('rsi', 'N/A'):.1f}" if context.get('rsi') else "      RSI: N/A")
        print(f"      5d Return: {context.get('return_5d', 0)*100:.1f}%")
        print(f"      vs MA20: {context.get('vs_ma20', 0):.1f}%")
        
        return trade['id']
    
    def add_partial_exit(self, trade_id: int, shares_or_pct: float, 
                         exit_price: float, reasoning: str = None):
        """Add a partial exit to a trade."""
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['partial_exits'].append({
                    'timestamp': datetime.now().isoformat(),
                    'amount': shares_or_pct,
                    'price': exit_price,
                    'reasoning': reasoning,
                })
                self.save_journal()
                print(f"✅ Partial exit added to trade #{trade_id}")
                return True
        return False
    
    def close_trade(self, trade_id: int, exit_price: float,
                    exit_reasoning: str = None, 
                    lessons: str = None,
                    rating: int = None):
        """Close a trade with full analysis."""
        for trade in self.trades:
            if trade['id'] == trade_id and trade['status'] == 'open':
                trade['exit_price'] = exit_price
                trade['exit_timestamp'] = datetime.now().isoformat()
                trade['exit_reasoning'] = exit_reasoning
                trade['lessons'] = lessons
                trade['rating'] = rating
                trade['status'] = 'closed'
                
                # Calculate P&L
                if trade['side'] == 'buy':
                    trade['pnl_pct'] = (exit_price - trade['entry_price']) / trade['entry_price']
                else:
                    trade['pnl_pct'] = (trade['entry_price'] - exit_price) / trade['entry_price']
                
                trade['pnl'] = trade['position_size'] * trade['pnl_pct']
                
                # Calculate hold duration
                entry_time = datetime.fromisoformat(trade['timestamp'])
                exit_time = datetime.fromisoformat(trade['exit_timestamp'])
                trade['hold_days'] = (exit_time - entry_time).days
                
                # Did it hit stop or target?
                if trade['stop_loss'] and trade['side'] == 'buy':
                    if exit_price <= trade['stop_loss']:
                        trade['exit_type'] = 'stopped_out'
                    elif trade['target'] and exit_price >= trade['target']:
                        trade['exit_type'] = 'target_hit'
                    else:
                        trade['exit_type'] = 'manual'
                
                self.save_journal()
                
                print(f"\n✅ Trade #{trade_id} closed")
                print(f"   Entry: ${trade['entry_price']:.2f}")
                print(f"   Exit: ${exit_price:.2f}")
                print(f"   P&L: ${trade['pnl']:,.0f} ({trade['pnl_pct']*100:+.1f}%)")
                print(f"   Duration: {trade['hold_days']} days")
                
                return True
        
        print(f"❌ Trade {trade_id} not found or already closed")
        return False


# ============================================================================
# TRADING FINGERPRINT ANALYZER
# ============================================================================

class TradingFingerprintAnalyzer:
    """
    Analyzes YOUR trades to build a "trading fingerprint".
    Identifies:
    - What setups you trade best
    - What market conditions favor your style
    - Your edge patterns
    - Your weaknesses
    """
    
    def __init__(self, journal: EnhancedTradeJournal):
        self.journal = journal
        self.fingerprint = {}
    
    def analyze(self) -> Dict:
        """Build comprehensive trading fingerprint."""
        trades = [t for t in self.journal.trades if t['status'] == 'closed']
        
        if len(trades) < 5:
            print("Need at least 5 closed trades for analysis")
            return {}
        
        print("\n" + "="*70)
        print("🧬 YOUR TRADING FINGERPRINT")
        print("="*70)
        
        # Basic stats
        pnls = [t['pnl_pct'] for t in trades if t.get('pnl_pct')]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        print(f"\n📊 OVERALL PERFORMANCE")
        print(f"   Total trades: {len(trades)}")
        print(f"   Win rate: {len(wins)/len(pnls)*100:.1f}%")
        print(f"   Avg win: {np.mean(wins)*100:+.1f}%" if wins else "   Avg win: N/A")
        print(f"   Avg loss: {np.mean(losses)*100:.1f}%" if losses else "   Avg loss: N/A")
        print(f"   Total return: {sum(pnls)*100:+.1f}%")
        
        if wins and losses:
            profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
            print(f"   Profit factor: {profit_factor:.2f}")
        
        self.fingerprint['overall'] = {
            'total_trades': len(trades),
            'win_rate': len(wins)/len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'total_return': sum(pnls),
        }
        
        # By setup type
        print(f"\n🎯 PERFORMANCE BY SETUP TYPE")
        setups = {}
        for t in trades:
            setup = t.get('setup_type', 'unknown')
            if setup not in setups:
                setups[setup] = []
            if t.get('pnl_pct'):
                setups[setup].append(t['pnl_pct'])
        
        setup_stats = []
        for setup, pnls_list in setups.items():
            if len(pnls_list) >= 2:
                wins_list = [p for p in pnls_list if p > 0]
                stat = {
                    'setup': setup,
                    'trades': len(pnls_list),
                    'win_rate': len(wins_list)/len(pnls_list),
                    'avg_return': np.mean(pnls_list),
                    'total_return': sum(pnls_list),
                }
                setup_stats.append(stat)
        
        setup_stats.sort(key=lambda x: -x['total_return'])
        
        for stat in setup_stats:
            emoji = "🏆" if stat['avg_return'] > 0 else "❌"
            print(f"   {emoji} {stat['setup']}: {stat['win_rate']*100:.0f}% WR, "
                  f"{stat['avg_return']*100:+.1f}% avg ({stat['trades']} trades)")
        
        if setup_stats:
            best_setup = setup_stats[0]
            self.fingerprint['best_setup'] = best_setup
            print(f"\n   ⭐ YOUR EDGE: {best_setup['setup'].upper()} setups")
        
        # By confidence level
        print(f"\n💪 PERFORMANCE BY CONFIDENCE")
        high_conf = [t for t in trades if t.get('confidence', 5) >= 7 and t.get('pnl_pct')]
        med_conf = [t for t in trades if 4 <= t.get('confidence', 5) < 7 and t.get('pnl_pct')]
        low_conf = [t for t in trades if t.get('confidence', 5) < 4 and t.get('pnl_pct')]
        
        for label, group in [("High (7-10)", high_conf), ("Medium (4-6)", med_conf), ("Low (1-3)", low_conf)]:
            if group:
                pnls_g = [t['pnl_pct'] for t in group]
                wr = len([p for p in pnls_g if p > 0]) / len(pnls_g)
                print(f"   {label}: {wr*100:.0f}% WR, {np.mean(pnls_g)*100:+.1f}% avg ({len(group)} trades)")
        
        # By market context
        print(f"\n📈 PERFORMANCE BY MARKET CONDITIONS")
        
        # VIX levels
        high_vix_trades = [t for t in trades if t.get('market_context', {}).get('vix', 0) > 20 and t.get('pnl_pct')]
        low_vix_trades = [t for t in trades if t.get('market_context', {}).get('vix', 100) <= 20 and t.get('pnl_pct')]
        
        for label, group in [("High VIX (>20)", high_vix_trades), ("Low VIX (≤20)", low_vix_trades)]:
            if len(group) >= 2:
                pnls_g = [t['pnl_pct'] for t in group]
                wr = len([p for p in pnls_g if p > 0]) / len(pnls_g)
                print(f"   {label}: {wr*100:.0f}% WR, {np.mean(pnls_g)*100:+.1f}% avg ({len(group)} trades)")
        
        # RSI levels at entry
        oversold_trades = [t for t in trades if t.get('market_context', {}).get('rsi', 50) < 30 and t.get('pnl_pct')]
        overbought_trades = [t for t in trades if t.get('market_context', {}).get('rsi', 50) > 70 and t.get('pnl_pct')]
        neutral_trades = [t for t in trades if 30 <= t.get('market_context', {}).get('rsi', 50) <= 70 and t.get('pnl_pct')]
        
        for label, group in [("Oversold (<30 RSI)", oversold_trades), ("Overbought (>70 RSI)", overbought_trades), ("Neutral RSI", neutral_trades)]:
            if len(group) >= 2:
                pnls_g = [t['pnl_pct'] for t in group]
                wr = len([p for p in pnls_g if p > 0]) / len(pnls_g)
                print(f"   {label}: {wr*100:.0f}% WR, {np.mean(pnls_g)*100:+.1f}% avg ({len(group)} trades)")
        
        # By timeframe
        print(f"\n⏱️ PERFORMANCE BY HOLDING PERIOD")
        quick_trades = [t for t in trades if t.get('hold_days', 0) <= 2 and t.get('pnl_pct')]
        swing_trades = [t for t in trades if 2 < t.get('hold_days', 0) <= 10 and t.get('pnl_pct')]
        position_trades = [t for t in trades if t.get('hold_days', 0) > 10 and t.get('pnl_pct')]
        
        for label, group in [("Quick (≤2 days)", quick_trades), ("Swing (3-10 days)", swing_trades), ("Position (>10 days)", position_trades)]:
            if len(group) >= 2:
                pnls_g = [t['pnl_pct'] for t in group]
                wr = len([p for p in pnls_g if p > 0]) / len(pnls_g)
                print(f"   {label}: {wr*100:.0f}% WR, {np.mean(pnls_g)*100:+.1f}% avg ({len(group)} trades)")
        
        # Best and worst trades
        print(f"\n🏆 TOP 3 WINNERS")
        sorted_trades = sorted([t for t in trades if t.get('pnl_pct')], key=lambda x: -x['pnl_pct'])
        for t in sorted_trades[:3]:
            print(f"   {t['symbol']}: {t['pnl_pct']*100:+.1f}% ({t.get('setup_type', 'unknown')}) - {t.get('reasoning', '')[:50]}")
        
        print(f"\n💀 TOP 3 LOSERS")
        for t in sorted_trades[-3:]:
            print(f"   {t['symbol']}: {t['pnl_pct']*100:+.1f}% ({t.get('setup_type', 'unknown')}) - {t.get('lessons', t.get('reasoning', ''))[:50]}")
        
        # Patterns in winners vs losers
        print(f"\n🔍 WINNER VS LOSER PATTERNS")
        winners_full = [t for t in trades if t.get('pnl_pct', 0) > 0]
        losers_full = [t for t in trades if t.get('pnl_pct', 0) <= 0]
        
        if winners_full and losers_full:
            # Confidence difference
            win_conf = np.mean([t.get('confidence', 5) for t in winners_full])
            lose_conf = np.mean([t.get('confidence', 5) for t in losers_full])
            print(f"   Avg confidence - Winners: {win_conf:.1f}, Losers: {lose_conf:.1f}")
            
            # RSI at entry
            win_rsi = np.mean([t.get('market_context', {}).get('rsi', 50) for t in winners_full if t.get('market_context', {}).get('rsi')])
            lose_rsi = np.mean([t.get('market_context', {}).get('rsi', 50) for t in losers_full if t.get('market_context', {}).get('rsi')])
            if win_rsi and lose_rsi:
                print(f"   Avg RSI at entry - Winners: {win_rsi:.1f}, Losers: {lose_rsi:.1f}")
        
        # Save fingerprint
        fingerprint_file = DATA_PATH / 'trading_fingerprint.json'
        with open(fingerprint_file, 'w') as f:
            json.dump(self.fingerprint, f, indent=2, default=str)
        
        print(f"\n✅ Fingerprint saved to {fingerprint_file}")
        
        return self.fingerprint


# ============================================================================
# AI COACH
# ============================================================================

class AICoach:
    """
    Provides coaching based on your trading patterns.
    """
    
    def __init__(self, journal: EnhancedTradeJournal):
        self.journal = journal
        self.fingerprint_file = DATA_PATH / 'trading_fingerprint.json'
    
    def get_pre_trade_advice(self, symbol: str, setup_type: str, confidence: int) -> List[str]:
        """Get advice before entering a trade."""
        advice = []
        
        # Load fingerprint if exists
        fingerprint = {}
        if self.fingerprint_file.exists():
            with open(self.fingerprint_file, 'r') as f:
                fingerprint = json.load(f)
        
        # Get market context
        context = MarketContextCapturer.get_market_context(symbol)
        
        # Check against your patterns
        if fingerprint.get('best_setup'):
            best = fingerprint['best_setup']
            if setup_type.lower() != best['setup'].lower():
                advice.append(f"📊 Your best setup is '{best['setup']}' ({best['win_rate']*100:.0f}% WR). "
                            f"Consider if '{setup_type}' is as strong for you.")
        
        # Confidence check
        if confidence < 7:
            advice.append(f"⚠️ Confidence is {confidence}/10. Your high-confidence trades tend to perform better. "
                         "Consider waiting for a higher conviction setup.")
        
        # VIX check
        vix = context.get('vix', 15)
        if vix > 25:
            advice.append(f"🔥 VIX is {vix:.1f}. High volatility - consider smaller position size "
                         "or wider stops.")
        
        # RSI check
        rsi = context.get('rsi', 50)
        if rsi < 30:
            advice.append(f"📉 RSI is {rsi:.0f} (oversold). This is a mean reversion setup - "
                         "historically your edge area based on validated signals.")
        elif rsi > 70:
            advice.append(f"📈 RSI is {rsi:.0f} (overbought). Be cautious of chasing. "
                         "Consider waiting for a pullback.")
        
        # Recent momentum
        ret_5d = context.get('return_5d', 0)
        if ret_5d < -0.05:
            advice.append(f"📊 {symbol} is down {abs(ret_5d)*100:.1f}% in 5 days. "
                         "If buying, you're catching a falling knife - have a clear plan.")
        elif ret_5d > 0.10:
            advice.append(f"🚀 {symbol} is up {ret_5d*100:.1f}% in 5 days. "
                         "Strong momentum but extended - consider scaling in vs all at once.")
        
        return advice
    
    def get_coaching_summary(self):
        """Get overall coaching based on recent trades."""
        trades = self.journal.trades
        closed = [t for t in trades if t['status'] == 'closed']
        
        if len(closed) < 3:
            print("\n🎓 Not enough closed trades for coaching yet. Keep logging!")
            return
        
        print("\n" + "="*70)
        print("🎓 AI COACHING SESSION")
        print("="*70)
        
        # Recent performance
        recent = closed[-10:]  # Last 10 trades
        recent_pnls = [t['pnl_pct'] for t in recent if t.get('pnl_pct')]
        
        if recent_pnls:
            recent_wr = len([p for p in recent_pnls if p > 0]) / len(recent_pnls)
            recent_avg = np.mean(recent_pnls)
            
            print(f"\n📈 Last 10 Trades:")
            print(f"   Win Rate: {recent_wr*100:.0f}%")
            print(f"   Average: {recent_avg*100:+.1f}%")
            
            if recent_wr >= 0.6 and recent_avg > 0.02:
                print(f"\n   🔥 You're on a HOT STREAK! Confidence justified.")
            elif recent_wr < 0.4:
                print(f"\n   ⚠️ Recent slump detected. Consider:")
                print(f"      - Taking smaller position sizes")
                print(f"      - Waiting for A+ setups only")
                print(f"      - Reviewing what changed in the market")
        
        # Analyze lessons learned
        all_lessons = [t['lessons'] for t in closed if t.get('lessons')]
        if all_lessons:
            print(f"\n📝 Recurring Themes in Your Lessons ({len(all_lessons)} entries):")
            # Simple keyword analysis
            keywords = {
                'patience': ['patience', 'wait', 'too early', 'fomo'],
                'position_size': ['size', 'too big', 'too small', 'scale'],
                'stops': ['stop', 'stopped', 'stop loss'],
                'targets': ['target', 'took profit', 'let run', 'sold too early'],
                'discipline': ['discipline', 'plan', 'rules'],
            }
            
            for theme, words in keywords.items():
                count = sum(1 for lesson in all_lessons 
                           if any(w in lesson.lower() for w in words))
                if count >= 2:
                    print(f"      • {theme.replace('_', ' ').title()}: mentioned {count}x")
        
        # Specific recommendations
        print(f"\n💡 PERSONALIZED RECOMMENDATIONS:")
        
        # Check for overtrading
        if len(closed) >= 20:
            monthly_count = len([t for t in closed 
                               if datetime.fromisoformat(t['timestamp']) > datetime.now() - timedelta(days=30)])
            if monthly_count > 30:
                print(f"   ⚠️ {monthly_count} trades in 30 days. Consider reducing frequency to focus on A+ setups.")
        
        # Check for position sizing discipline
        sizes = [t['position_size'] for t in closed if t.get('position_size')]
        if sizes and max(sizes) > 3 * np.median(sizes):
            print(f"   ⚠️ Position sizes vary a lot. Consider more consistent sizing for risk management.")
        
        # Best time/setup combination
        fingerprint_file = DATA_PATH / 'trading_fingerprint.json'
        if fingerprint_file.exists():
            with open(fingerprint_file, 'r') as f:
                fp = json.load(f)
            
            if fp.get('best_setup'):
                print(f"\n   ⭐ FOCUS ON: {fp['best_setup']['setup'].upper()} setups")
                print(f"      This is where you have the most edge!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    journal = EnhancedTradeJournal()
    analyzer = TradingFingerprintAnalyzer(journal)
    coach = AICoach(journal)
    
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("SKILL CAPTURE ENGINE")
        print("="*70)
        print("\nCapture and analyze YOUR trading edge to combine with systematic signals.")
        print("\nCommands:")
        print("  manual      - Log a manual trade with full context")
        print("  close       - Close an open trade")
        print("  open        - Show open trades")
        print("  analyze     - Deep analysis of your patterns")
        print("  fingerprint - Build your trading edge profile")
        print("  coach       - Get AI coaching")
        print("  advice      - Get pre-trade advice")
        
        # Show quick stats
        closed = [t for t in journal.trades if t['status'] == 'closed']
        open_trades = [t for t in journal.trades if t['status'] == 'open']
        
        print(f"\n📊 Journal Stats:")
        print(f"   Open trades: {len(open_trades)}")
        print(f"   Closed trades: {len(closed)}")
        
        if closed:
            pnls = [t['pnl_pct'] for t in closed if t.get('pnl_pct')]
            if pnls:
                print(f"   Total P&L: {sum(pnls)*100:+.1f}%")
        
        return
    
    command = sys.argv[1].lower()
    
    if command == 'manual':
        print("\n📝 LOG MANUAL TRADE")
        print("-"*40)
        
        symbol = input("Symbol: ").upper()
        side = input("Side (buy/sell): ").lower()
        entry_price = float(input("Entry price: "))
        position_size = float(input("Position size ($): "))
        reasoning = input("Your reasoning: ")
        setup_type = input("Setup type (breakout/pullback/reversal/momentum/earnings/news/other): ")
        confidence = int(input("Confidence (1-10): "))
        
        has_stops = input("Have stop/target? (y/n): ").lower() == 'y'
        stop_loss = float(input("Stop loss price: ")) if has_stops else None
        target = float(input("Target price: ")) if has_stops else None
        
        catalyst = input("Catalyst (optional, press enter to skip): ") or None
        timeframe = input("Expected timeframe (scalp/day/swing/position) [swing]: ") or 'swing'
        
        journal.log_manual_trade(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            position_size=position_size,
            reasoning=reasoning,
            confidence=confidence,
            setup_type=setup_type,
            catalyst=catalyst,
            timeframe=timeframe,
            stop_loss=stop_loss,
            target=target
        )
    
    elif command == 'close':
        # Show open trades first
        open_trades = [t for t in journal.trades if t['status'] == 'open']
        if not open_trades:
            print("\nNo open trades to close")
            return
        
        print("\n📋 OPEN TRADES")
        for t in open_trades:
            print(f"   #{t['id']}: {t['side'].upper()} {t['symbol']} @ ${t['entry_price']:.2f}")
        
        trade_id = int(input("\nTrade ID to close: "))
        exit_price = float(input("Exit price: "))
        exit_reasoning = input("Exit reasoning (optional): ") or None
        lessons = input("Lessons learned (optional): ") or None
        rating = input("Rate this trade 1-10 (optional): ")
        rating = int(rating) if rating else None
        
        journal.close_trade(trade_id, exit_price, exit_reasoning, lessons, rating)
    
    elif command == 'open':
        open_trades = [t for t in journal.trades if t['status'] == 'open']
        if not open_trades:
            print("\nNo open trades")
        else:
            print("\n📋 OPEN TRADES")
            for t in open_trades:
                print(f"\n   #{t['id']}: {t['side'].upper()} {t['symbol']}")
                print(f"      Entry: ${t['entry_price']:.2f}")
                print(f"      Size: ${t['position_size']:,.0f}")
                print(f"      Setup: {t.get('setup_type', 'unknown')}")
                print(f"      Reasoning: {t.get('reasoning', '')[:60]}")
    
    elif command == 'analyze' or command == 'fingerprint':
        analyzer.analyze()
    
    elif command == 'coach':
        coach.get_coaching_summary()
    
    elif command == 'advice':
        print("\n🎯 PRE-TRADE ADVICE")
        symbol = input("Symbol: ").upper()
        setup = input("Setup type: ")
        confidence = int(input("Confidence (1-10): "))
        
        advice = coach.get_pre_trade_advice(symbol, setup, confidence)
        
        print("\n" + "-"*40)
        for a in advice:
            print(f"\n   {a}")
        
        if not advice:
            print("   ✅ No red flags detected. Trade looks aligned with your edge!")
    
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
