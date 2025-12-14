"""
HISTORY LEARNING ENGINE
=======================
Learn from past trades to improve future decisions.

Features:
- Analyze last 3 days of trades
- Identify winning patterns (what worked)
- Identify losing patterns (what didn't work)
- Track model accuracy by cluster
- Learn optimal hold times
- Detect repeated mistakes

This makes the AI smarter every day - it learns from YOUR trading history.

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HistoryAnalyzer:
    """
    Analyze trading history to learn patterns.
    """
    
    def __init__(
        self,
        lookback_days: int = 3,
        min_trades_for_pattern: int = 3
    ):
        """
        Args:
            lookback_days: Days of history to analyze
            min_trades_for_pattern: Minimum trades to identify pattern
        """
        self.lookback_days = lookback_days
        self.min_trades_for_pattern = min_trades_for_pattern
    
    def analyze_recent_trades(self, trade_history: List[Dict]) -> Dict:
        """
        Analyze recent trades for patterns.
        
        Args:
            trade_history: List of trade dicts from portfolio tracker
            
        Returns:
            Analysis results
        """
        logger.info("\n" + "="*60)
        logger.info(f"ANALYZING LAST {self.lookback_days} DAYS OF TRADES")
        logger.info("="*60)
        
        if not trade_history:
            logger.warning("⚠️ No trade history available")
            return {'message': 'No trades to analyze'}
        
        # Filter to recent trades
        cutoff = datetime.now() - timedelta(days=self.lookback_days)
        recent_trades = [
            t for t in trade_history
            if datetime.fromisoformat(t['exit_date']) > cutoff
        ]
        
        if not recent_trades:
            logger.warning(f"⚠️ No trades in last {self.lookback_days} days")
            return {'message': f'No trades in last {self.lookback_days} days'}
        
        logger.info(f"Found {len(recent_trades)} trades in last {self.lookback_days} days")
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(recent_trades)
        
        # Analyze patterns
        results = {
            'period': f'Last {self.lookback_days} days',
            'total_trades': len(df),
            'wins': len(df[df['realized_return'] > 0]),
            'losses': len(df[df['realized_return'] <= 0]),
            'win_rate': len(df[df['realized_return'] > 0]) / len(df) if len(df) > 0 else 0,
            'total_pnl': df['realized_pnl'].sum(),
            'avg_win': df[df['realized_pnl'] > 0]['realized_pnl'].mean() if len(df[df['realized_pnl'] > 0]) > 0 else 0,
            'avg_loss': df[df['realized_pnl'] < 0]['realized_pnl'].mean() if len(df[df['realized_pnl'] < 0]) > 0 else 0,
            'avg_hold_time': df['days_held'].mean(),
            'day_trades': len(df[df['is_day_trade']]),
        }
        
        # Find winning patterns
        results['winning_patterns'] = self._find_winning_patterns(df)
        
        # Find losing patterns
        results['losing_patterns'] = self._find_losing_patterns(df)
        
        # Model accuracy by cluster
        results['cluster_performance'] = self._analyze_cluster_performance(df)
        
        # Optimal hold times
        results['optimal_hold_times'] = self._analyze_hold_times(df)
        
        # Repeated mistakes
        results['repeated_mistakes'] = self._find_repeated_mistakes(df)
        
        # Log summary
        self._log_analysis(results)
        
        return results
    
    def _find_winning_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Find patterns that led to wins."""
        wins = df[df['realized_return'] > 0]
        
        if len(wins) < self.min_trades_for_pattern:
            return []
        
        patterns = []
        
        # Pattern 1: High entry confidence wins
        high_conf_wins = wins[wins['entry_confidence'] > 80]
        if len(high_conf_wins) >= self.min_trades_for_pattern:
            patterns.append({
                'pattern': 'High confidence entries (>80%)',
                'trades': len(high_conf_wins),
                'win_rate': len(high_conf_wins[high_conf_wins['realized_return'] > 0]) / len(high_conf_wins),
                'avg_return': high_conf_wins['realized_return'].mean(),
                'recommendation': 'CONTINUE - High confidence entries work well'
            })
        
        # Pattern 2: Quick wins (1-3 days)
        quick_wins = wins[wins['days_held'] <= 3]
        if len(quick_wins) >= self.min_trades_for_pattern:
            patterns.append({
                'pattern': 'Quick exits (1-3 days)',
                'trades': len(quick_wins),
                'win_rate': len(quick_wins[quick_wins['realized_return'] > 0]) / len(quick_wins),
                'avg_return': quick_wins['realized_return'].mean(),
                'recommendation': 'CONTINUE - Quick exits capture profits well'
            })
        
        # Pattern 3: Specific tickers that work
        ticker_wins = wins.groupby('ticker').agg({
            'realized_return': ['count', 'mean'],
            'realized_pnl': 'sum'
        }).reset_index()
        ticker_wins.columns = ['ticker', 'trades', 'avg_return', 'total_pnl']
        best_tickers = ticker_wins[ticker_wins['trades'] >= 2].nlargest(3, 'avg_return')
        
        for _, row in best_tickers.iterrows():
            patterns.append({
                'pattern': f'{row["ticker"]} - proven winner',
                'trades': row['trades'],
                'avg_return': row['avg_return'],
                'total_pnl': row['total_pnl'],
                'recommendation': f'CONTINUE - {row["ticker"]} has edge for you'
            })
        
        return patterns
    
    def _find_losing_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Find patterns that led to losses."""
        losses = df[df['realized_return'] < 0]
        
        if len(losses) < self.min_trades_for_pattern:
            return []
        
        patterns = []
        
        # Pattern 1: Holding too long
        long_holds = losses[losses['days_held'] > 10]
        if len(long_holds) >= 2:
            patterns.append({
                'pattern': 'Holding losers too long (>10 days)',
                'trades': len(long_holds),
                'avg_loss': long_holds['realized_return'].mean(),
                'recommendation': 'AVOID - Cut losses faster, max 32 days'
            })
        
        # Pattern 2: Low confidence entries that lost
        low_conf_losses = losses[losses['entry_confidence'] < 70]
        if len(low_conf_losses) >= 2:
            patterns.append({
                'pattern': 'Low confidence entries (<70%)',
                'trades': len(low_conf_losses),
                'avg_loss': low_conf_losses['realized_return'].mean(),
                'recommendation': 'AVOID - Stick to 70%+ confidence'
            })
        
        # Pattern 3: Specific tickers that don't work
        ticker_losses = losses.groupby('ticker').agg({
            'realized_return': ['count', 'mean'],
            'realized_pnl': 'sum'
        }).reset_index()
        ticker_losses.columns = ['ticker', 'trades', 'avg_return', 'total_pnl']
        worst_tickers = ticker_losses[ticker_losses['trades'] >= 2].nsmallest(3, 'avg_return')
        
        for _, row in worst_tickers.iterrows():
            patterns.append({
                'pattern': f'{row["ticker"]} - consistent loser',
                'trades': row['trades'],
                'avg_loss': row['avg_return'],
                'total_loss': row['total_pnl'],
                'recommendation': f'AVOID - {row["ticker"]} doesn\'t work for your strategy'
            })
        
        return patterns
    
    def _analyze_cluster_performance(self, df: pd.DataFrame) -> Dict:
        """Analyze model performance by cluster."""
        if 'cluster_id' not in df.columns:
            return {}
        
        cluster_perf = df.groupby('cluster_id').agg({
            'realized_return': ['count', 'mean', lambda x: (x > 0).sum() / len(x)],
            'realized_pnl': 'sum'
        }).reset_index()
        
        cluster_perf.columns = ['cluster_id', 'trades', 'avg_return', 'win_rate', 'total_pnl']
        
        return cluster_perf.to_dict('records')
    
    def _analyze_hold_times(self, df: pd.DataFrame) -> Dict:
        """Analyze optimal hold times."""
        wins = df[df['realized_return'] > 0]
        losses = df[df['realized_return'] < 0]
        
        return {
            'avg_winning_hold': wins['days_held'].mean() if len(wins) > 0 else 0,
            'avg_losing_hold': losses['days_held'].mean() if len(losses) > 0 else 0,
            'best_hold_range': self._find_best_hold_range(df)
        }
    
    def _find_best_hold_range(self, df: pd.DataFrame) -> str:
        """Find hold time range with best results."""
        # Bin hold times
        df['hold_bin'] = pd.cut(df['days_held'], bins=[0, 1, 3, 7, 14, 32, 100], 
                                labels=['0-1d', '1-3d', '3-7d', '7-14d', '14-32d', '32+d'])
        
        bin_perf = df.groupby('hold_bin').agg({
            'realized_return': ['count', 'mean', lambda x: (x > 0).sum() / len(x) if len(x) > 0 else 0]
        }).reset_index()
        
        bin_perf.columns = ['hold_range', 'trades', 'avg_return', 'win_rate']
        
        # Find best bin with at least 2 trades
        best_bin = bin_perf[bin_perf['trades'] >= 2].nlargest(1, 'win_rate')
        
        if len(best_bin) > 0:
            return f"{best_bin.iloc[0]['hold_range']} ({best_bin.iloc[0]['win_rate']:.1%} WR)"
        else:
            return "Insufficient data"
    
    def _find_repeated_mistakes(self, df: pd.DataFrame) -> List[str]:
        """Identify repeated mistakes."""
        mistakes = []
        
        # Mistake 1: Revenge trading after loss
        for i in range(1, len(df)):
            prev_trade = df.iloc[i-1]
            curr_trade = df.iloc[i]
            
            # Check if previous was big loss and current was quick entry
            if prev_trade['realized_return'] < -0.10:
                prev_exit = datetime.fromisoformat(prev_trade['exit_date'])
                curr_entry = datetime.fromisoformat(curr_trade['entry_date'])
                hours_between = (curr_entry - prev_exit).total_seconds() / 3600
                
                if hours_between < 24:
                    mistakes.append(f"Revenge trading: Entered {curr_trade['ticker']} {hours_between:.1f}h after {prev_trade['realized_return']:.1%} loss on {prev_trade['ticker']}")
        
        # Mistake 2: Same ticker multiple losses
        ticker_losses = df[df['realized_return'] < 0].groupby('ticker').size()
        repeat_losers = ticker_losses[ticker_losses >= 2]
        
        for ticker, count in repeat_losers.items():
            mistakes.append(f"Repeated losses on {ticker}: {count} losses")
        
        # Mistake 3: Day trading when should avoid
        if 'is_day_trade' in df.columns:
            day_trade_losses = df[(df['is_day_trade']) & (df['realized_return'] < 0)]
            if len(day_trade_losses) >= 2:
                mistakes.append(f"Day trade losses: {len(day_trade_losses)} day trades resulted in losses")
        
        return mistakes
    
    def _log_analysis(self, results: Dict):
        """Log analysis results."""
        logger.info(f"\n📊 SUMMARY")
        logger.info(f"   Period: {results['period']}")
        logger.info(f"   Total trades: {results['total_trades']}")
        logger.info(f"   Win rate: {results['win_rate']:.1%}")
        logger.info(f"   Total P&L: ${results['total_pnl']:+.2f}")
        logger.info(f"   Avg win: ${results['avg_win']:+.2f}")
        logger.info(f"   Avg loss: ${results['avg_loss']:+.2f}")
        logger.info(f"   Avg hold: {results['avg_hold_time']:.1f} days")
        logger.info(f"   Day trades: {results['day_trades']}")
        
        if results['winning_patterns']:
            logger.info(f"\n✅ WINNING PATTERNS ({len(results['winning_patterns'])})")
            for pattern in results['winning_patterns']:
                logger.info(f"   • {pattern['pattern']}")
                logger.info(f"     Trades: {pattern['trades']}, Recommendation: {pattern['recommendation']}")
        
        if results['losing_patterns']:
            logger.info(f"\n❌ LOSING PATTERNS ({len(results['losing_patterns'])})")
            for pattern in results['losing_patterns']:
                logger.info(f"   • {pattern['pattern']}")
                logger.info(f"     Trades: {pattern['trades']}, Recommendation: {pattern['recommendation']}")
        
        if results['repeated_mistakes']:
            logger.info(f"\n⚠️ REPEATED MISTAKES ({len(results['repeated_mistakes'])})")
            for mistake in results['repeated_mistakes']:
                logger.info(f"   • {mistake}")
    
    def get_recommendations(self, analysis: Dict) -> List[str]:
        """Get actionable recommendations from analysis."""
        recommendations = []
        
        # From winning patterns
        for pattern in analysis.get('winning_patterns', []):
            recommendations.append(pattern['recommendation'])
        
        # From losing patterns
        for pattern in analysis.get('losing_patterns', []):
            recommendations.append(pattern['recommendation'])
        
        # From repeated mistakes
        if analysis.get('repeated_mistakes'):
            recommendations.append("AVOID: Revenge trading and repeated ticker losses")
        
        # From hold times
        hold_info = analysis.get('optimal_hold_times', {})
        if hold_info.get('best_hold_range'):
            recommendations.append(f"OPTIMAL HOLD TIME: {hold_info['best_hold_range']}")
        
        return recommendations


def example_usage():
    """Example: Analyze trading history."""
    logger.info("\n" + "="*60)
    logger.info("HISTORY ANALYZER - Learn from Past Trades")
    logger.info("="*60 + "\n")
    
    # Mock trade history (replace with real data from portfolio tracker)
    trade_history = [
        {
            'ticker': 'PALI',
            'entry_date': '2025-12-08T10:30:00',
            'exit_date': '2025-12-10T14:00:00',
            'entry_price': 2.10,
            'exit_price': 2.38,
            'realized_pnl': 14.00,
            'realized_return': 0.1333,
            'days_held': 2,
            'entry_confidence': 85,
            'is_day_trade': False,
            'cluster_id': 2
        },
        {
            'ticker': 'HOOD',
            'entry_date': '2025-12-07T09:00:00',
            'exit_date': '2025-12-09T15:30:00',
            'entry_price': 18.50,
            'exit_price': 34.20,
            'realized_pnl': 271.50,
            'realized_return': 0.8486,
            'days_held': 2,
            'entry_confidence': 92,
            'is_day_trade': False,
            'cluster_id': 1
        },
        {
            'ticker': 'ASTS',
            'entry_date': '2025-12-09T11:00:00',
            'exit_date': '2025-12-10T11:30:00',
            'entry_price': 4.86,
            'exit_price': 4.20,
            'realized_pnl': -76.53,
            'realized_return': -0.1358,
            'days_held': 1,
            'entry_confidence': 68,
            'is_day_trade': True,
            'cluster_id': 3
        }
    ]
    
    # Analyze
    analyzer = HistoryAnalyzer(lookback_days=3)
    analysis = analyzer.analyze_recent_trades(trade_history)
    
    # Get recommendations
    logger.info("\n🎯 RECOMMENDATIONS")
    recommendations = analyzer.get_recommendations(analysis)
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"   {i}. {rec}")


if __name__ == '__main__':
    example_usage()
