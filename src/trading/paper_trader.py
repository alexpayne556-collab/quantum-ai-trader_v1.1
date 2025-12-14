"""
📄 PAPER TRADING INTEGRATION

Philosophy: "The only way to teach a baby to swim is to throw it in the water"

Features:
- Alpaca paper trading API
- 20% allocation initially
- Complete logging (every signal, trade, outcome)
- Real-time monitoring
- Daily retraining
- Weekly analysis
- Companion AI integration
"""

import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json
import os
from dataclasses import dataclass, asdict

from src.trading.companion_ai import CompanionAI, Position, Warning

logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Trading signal from model"""
    ticker: str
    signal: str  # 'BUY' or 'SELL'
    confidence: float
    probability: float
    cluster_id: int
    model_votes: Dict[str, int]
    features: Dict[str, float]
    timestamp: datetime
    
    # Price levels
    entry_price: float
    target_profit: float
    stop_loss: float


@dataclass
class TradeLog:
    """Complete trade record"""
    trade_id: str
    ticker: str
    signal: str
    
    # Entry
    entry_time: datetime
    entry_price: float
    shares: int
    entry_value: float
    
    # Signal details
    confidence: float
    probability: float
    cluster_id: int
    model_votes: Dict[str, int]
    
    # Targets
    target_profit: float
    stop_loss: float
    
    # Exit (filled when trade closes)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    
    # Outcome
    profit_loss: Optional[float] = None
    profit_loss_pct: Optional[float] = None
    holding_period_hours: Optional[float] = None
    
    # Companion AI
    warnings_received: List[str] = None
    
    def __post_init__(self):
        if self.warnings_received is None:
            self.warnings_received = []


class PaperTrader:
    """
    Paper trading system with Alpaca
    
    Features:
    - Automated signal execution
    - Position monitoring
    - Risk management
    - Complete logging
    - Companion AI integration
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = 'https://paper-api.alpaca.markets',
        allocation_pct: float = 0.20,  # Start with 20%
        max_position_size_pct: float = 0.10,  # Max 10% per position
        min_confidence: float = 0.60,  # Min 60% confidence
        log_dir: str = 'data/paper_trading'
    ):
        # Initialize Alpaca API
        self.api = tradeapi.REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url
        )
        
        self.allocation_pct = allocation_pct
        self.max_position_size = max_position_size_pct
        self.min_confidence = min_confidence
        self.log_dir = log_dir
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize Companion AI
        self.companion = CompanionAI()
        
        # Active trades
        self.active_trades: Dict[str, TradeLog] = {}
        
        # Trade history
        self.trade_history: List[TradeLog] = []
        
        # Load existing logs
        self._load_trade_history()
        
        # Get account info
        account = self.api.get_account()
        logger.info("📄 Paper Trader initialized")
        logger.info(f"   Account: {account.account_number}")
        logger.info(f"   Cash: ${float(account.cash):,.2f}")
        logger.info(f"   Portfolio: ${float(account.portfolio_value):,.2f}")
        logger.info(f"   Allocation: {self.allocation_pct:.0%}")
        logger.info(f"   Min confidence: {self.min_confidence:.0%}")
    
    def process_signals(
        self,
        signals: List[TradeSignal],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Process trading signals from model
        
        Steps:
        1. Filter by confidence
        2. Check Companion AI
        3. Size positions
        4. Execute trades
        5. Log everything
        
        Returns:
            {
                'signals_received': int,
                'signals_filtered': int,
                'trades_executed': int,
                'trades_rejected': int,
                'reasons': Dict
            }
        """
        
        stats = {
            'signals_received': len(signals),
            'signals_filtered': 0,
            'trades_executed': 0,
            'trades_rejected': 0,
            'reasons': {}
        }
        
        # Filter by confidence
        high_confidence_signals = [
            s for s in signals if s.confidence >= self.min_confidence
        ]
        stats['signals_filtered'] = len(high_confidence_signals)
        
        # Sort by confidence (trade best first)
        high_confidence_signals.sort(key=lambda x: x.confidence, reverse=True)
        
        # Execute trades
        for signal in high_confidence_signals:
            result = self._execute_signal(signal, market_data)
            
            if result['success']:
                stats['trades_executed'] += 1
            else:
                stats['trades_rejected'] += 1
                reason = result['reason']
                stats['reasons'][reason] = stats['reasons'].get(reason, 0) + 1
        
        logger.info(f"📊 Signal processing complete:")
        logger.info(f"   Received: {stats['signals_received']}")
        logger.info(f"   High confidence: {stats['signals_filtered']}")
        logger.info(f"   Executed: {stats['trades_executed']}")
        logger.info(f"   Rejected: {stats['trades_rejected']}")
        
        return stats
    
    def _execute_signal(
        self,
        signal: TradeSignal,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """Execute a single trading signal"""
        
        try:
            # Check if already in position
            if signal.ticker in self.active_trades:
                return {
                    'success': False,
                    'reason': 'ALREADY_IN_POSITION'
                }
            
            # Get account info
            account = self.api.get_account()
            cash_available = float(account.cash) * self.allocation_pct
            
            # Calculate position size
            position_value = min(
                cash_available * self.max_position_size,
                cash_available  # Don't exceed available cash
            )
            
            shares = int(position_value / signal.entry_price)
            
            if shares < 1:
                return {
                    'success': False,
                    'reason': 'INSUFFICIENT_FUNDS'
                }
            
            # Check Companion AI (for existing holdings - skip for new entry)
            # Just log the entry
            
            # Execute trade
            if signal.signal == 'BUY':
                order = self.api.submit_order(
                    symbol=signal.ticker,
                    qty=shares,
                    side='buy',
                    type='limit',
                    time_in_force='day',
                    limit_price=signal.entry_price
                )
            else:
                # For short selling (if enabled)
                return {
                    'success': False,
                    'reason': 'SHORT_SELLING_DISABLED'
                }
            
            # Create trade log
            trade_log = TradeLog(
                trade_id=order.id,
                ticker=signal.ticker,
                signal=signal.signal,
                entry_time=datetime.now(),
                entry_price=signal.entry_price,
                shares=shares,
                entry_value=shares * signal.entry_price,
                confidence=signal.confidence,
                probability=signal.probability,
                cluster_id=signal.cluster_id,
                model_votes=signal.model_votes,
                target_profit=signal.target_profit,
                stop_loss=signal.stop_loss
            )
            
            # Add to active trades
            self.active_trades[signal.ticker] = trade_log
            
            # Save log
            self._save_trade_log(trade_log)
            
            logger.info(f"✅ Executed {signal.signal} {shares} {signal.ticker} @ ${signal.entry_price:.2f}")
            logger.info(f"   Confidence: {signal.confidence:.1%}")
            logger.info(f"   Target: ${signal.target_profit:.2f}")
            logger.info(f"   Stop: ${signal.stop_loss:.2f}")
            
            return {
                'success': True,
                'trade_id': order.id,
                'shares': shares
            }
        
        except Exception as e:
            logger.error(f"Failed to execute {signal.ticker}: {e}")
            return {
                'success': False,
                'reason': f'ERROR: {str(e)}'
            }
    
    def monitor_positions(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Monitor all active positions with Companion AI
        
        Returns:
            {
                'positions_monitored': int,
                'warnings_issued': int,
                'exits_recommended': int,
                'exits_executed': int
            }
        """
        
        stats = {
            'positions_monitored': len(self.active_trades),
            'warnings_issued': 0,
            'exits_recommended': 0,
            'exits_executed': 0
        }
        
        # Convert active trades to Position objects
        positions = []
        for ticker, trade_log in self.active_trades.items():
            # Get current price
            if ticker not in market_data:
                logger.warning(f"No market data for {ticker}")
                continue
            
            current_price = market_data[ticker]['close'].iloc[-1]
            
            position = Position(
                ticker=ticker,
                entry_price=trade_log.entry_price,
                entry_time=trade_log.entry_time,
                current_price=current_price,
                shares=trade_log.shares,
                signal_confidence=trade_log.confidence,
                target_profit=trade_log.target_profit,
                stop_loss=trade_log.stop_loss,
                cluster_id=trade_log.cluster_id
            )
            
            positions.append(position)
        
        # Monitor with Companion AI
        monitoring_results = self.companion.monitor_all_positions(
            positions,
            market_data
        )
        
        # Process recommendations
        for ticker, result in monitoring_results.items():
            warnings = result['warnings']
            recommendation = result['recommendation']
            
            if len(warnings) > 0:
                stats['warnings_issued'] += len(warnings)
                
                # Log warnings
                for warning in warnings:
                    self.active_trades[ticker].warnings_received.append(
                        f"[{warning.severity}] {warning.message}"
                    )
            
            # Execute exits if recommended
            if recommendation['action'] in ['EMERGENCY_EXIT', 'FULL_EXIT']:
                stats['exits_recommended'] += 1
                
                exit_result = self._execute_exit(
                    ticker,
                    recommendation['message'],
                    market_data[ticker]['close'].iloc[-1]
                )
                
                if exit_result['success']:
                    stats['exits_executed'] += 1
        
        logger.info(f"👀 Position monitoring complete:")
        logger.info(f"   Positions: {stats['positions_monitored']}")
        logger.info(f"   Warnings: {stats['warnings_issued']}")
        logger.info(f"   Exits recommended: {stats['exits_recommended']}")
        logger.info(f"   Exits executed: {stats['exits_executed']}")
        
        return stats
    
    def _execute_exit(
        self,
        ticker: str,
        reason: str,
        exit_price: float
    ) -> Dict:
        """Close a position"""
        
        try:
            if ticker not in self.active_trades:
                return {
                    'success': False,
                    'reason': 'NOT_IN_POSITION'
                }
            
            trade_log = self.active_trades[ticker]
            
            # Submit sell order
            order = self.api.submit_order(
                symbol=ticker,
                qty=trade_log.shares,
                side='sell',
                type='market',
                time_in_force='day'
            )
            
            # Update trade log
            trade_log.exit_time = datetime.now()
            trade_log.exit_price = exit_price
            trade_log.exit_reason = reason
            trade_log.profit_loss = (exit_price - trade_log.entry_price) * trade_log.shares
            trade_log.profit_loss_pct = (exit_price - trade_log.entry_price) / trade_log.entry_price
            trade_log.holding_period_hours = (trade_log.exit_time - trade_log.entry_time).total_seconds() / 3600
            
            # Move to history
            self.trade_history.append(trade_log)
            del self.active_trades[ticker]
            
            # Save
            self._save_trade_log(trade_log)
            
            logger.info(f"🔒 Closed {ticker} @ ${exit_price:.2f}")
            logger.info(f"   Reason: {reason}")
            logger.info(f"   P/L: ${trade_log.profit_loss:+.2f} ({trade_log.profit_loss_pct:+.1%})")
            logger.info(f"   Held: {trade_log.holding_period_hours:.1f} hours")
            
            return {
                'success': True,
                'profit_loss': trade_log.profit_loss,
                'profit_loss_pct': trade_log.profit_loss_pct
            }
        
        except Exception as e:
            logger.error(f"Failed to exit {ticker}: {e}")
            return {
                'success': False,
                'reason': f'ERROR: {str(e)}'
            }
    
    def daily_summary(self) -> Dict:
        """Generate daily trading summary"""
        
        # Get today's trades
        today = datetime.now().date()
        today_trades = [
            t for t in self.trade_history
            if t.exit_time and t.exit_time.date() == today
        ]
        
        if len(today_trades) == 0:
            return {
                'date': today,
                'trades': 0,
                'message': 'No trades closed today'
            }
        
        # Calculate metrics
        wins = [t for t in today_trades if t.profit_loss > 0]
        losses = [t for t in today_trades if t.profit_loss <= 0]
        
        total_pnl = sum(t.profit_loss for t in today_trades)
        win_rate = len(wins) / len(today_trades) if today_trades else 0
        
        avg_win = np.mean([t.profit_loss for t in wins]) if wins else 0
        avg_loss = np.mean([t.profit_loss for t in losses]) if losses else 0
        
        avg_holding = np.mean([t.holding_period_hours for t in today_trades])
        
        summary = {
            'date': today,
            'trades': len(today_trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_holding_hours': avg_holding,
            'best_trade': max(today_trades, key=lambda t: t.profit_loss),
            'worst_trade': min(today_trades, key=lambda t: t.profit_loss)
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"📊 DAILY SUMMARY - {today}")
        print(f"{'='*60}")
        print(f"Trades: {summary['trades']} ({summary['wins']}W / {summary['losses']}L)")
        print(f"Win Rate: {summary['win_rate']:.1%}")
        print(f"Total P/L: ${summary['total_pnl']:+,.2f}")
        print(f"Avg Win: ${summary['avg_win']:+.2f}")
        print(f"Avg Loss: ${summary['avg_loss']:+.2f}")
        print(f"Avg Hold: {summary['avg_holding_hours']:.1f}h")
        print(f"\nBest: {summary['best_trade'].ticker} ${summary['best_trade'].profit_loss:+.2f}")
        print(f"Worst: {summary['worst_trade'].ticker} ${summary['worst_trade'].profit_loss:+.2f}")
        print(f"{'='*60}\n")
        
        return summary
    
    def _save_trade_log(self, trade_log: TradeLog):
        """Save trade log to file"""
        
        filename = f"{self.log_dir}/trades_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        with open(filename, 'a') as f:
            f.write(json.dumps(asdict(trade_log), default=str) + '\n')
    
    def _load_trade_history(self):
        """Load existing trade logs"""
        
        # Load all .jsonl files
        for filename in os.listdir(self.log_dir):
            if filename.endswith('.jsonl'):
                filepath = os.path.join(self.log_dir, filename)
                
                with open(filepath, 'r') as f:
                    for line in f:
                        data = json.loads(line)
                        
                        # Convert dates
                        data['entry_time'] = datetime.fromisoformat(data['entry_time'])
                        if data['exit_time']:
                            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
                        
                        trade_log = TradeLog(**data)
                        
                        # Add to history if closed
                        if trade_log.exit_time:
                            self.trade_history.append(trade_log)
                        else:
                            # Add to active if still open
                            self.active_trades[trade_log.ticker] = trade_log
        
        logger.info(f"Loaded {len(self.trade_history)} historical trades")
        logger.info(f"Found {len(self.active_trades)} active positions")


# Example usage
if __name__ == '__main__':
    # Initialize trader
    trader = PaperTrader(
        api_key=os.getenv('ALPACA_API_KEY'),
        api_secret=os.getenv('ALPACA_API_SECRET'),
        allocation_pct=0.20,
        min_confidence=0.60
    )
    
    # Mock signals
    signals = [
        TradeSignal(
            ticker='NVDA',
            signal='BUY',
            confidence=0.85,
            probability=0.92,
            cluster_id=0,
            model_votes={'xgb': 1, 'lgb': 1, 'cat': 1},
            features={},
            timestamp=datetime.now(),
            entry_price=475.00,
            target_profit=522.50,
            stop_loss=451.25
        )
    ]
    
    # Process signals
    # stats = trader.process_signals(signals, market_data)
    
    # Monitor positions
    # monitoring_stats = trader.monitor_positions(market_data)
    
    # Daily summary
    # summary = trader.daily_summary()
    
    print("✅ Paper Trader ready")
