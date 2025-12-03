"""
Trade Executor Module
Research-backed: Kelly criterion, Almgren-Chriss slippage, risk validation
2024-2025 institutional best practices
"""
from typing import Dict, List, Tuple
import numpy as np

class TradeExecutor:
    def validate_order(self, order: Dict, portfolio_state: Dict, market_data: Dict) -> Tuple[bool, List[str]]:
        violations = []
        # Position size check (Kelly criterion)
        max_position_size = self.calculate_kelly_size(
            portfolio_state['equity'],
            market_data[order['ticker']]['volatility'],
            order['entry_price'],
            win_rate=0.55  # Example win rate
        )
        if order['quantity'] > max_position_size:
            violations.append(f"Position size {order['quantity']} > Kelly limit {max_position_size}")
        # Margin requirement check
        required_margin = order['quantity'] * order['entry_price'] * 0.5
        available_margin = portfolio_state['cash'] - portfolio_state['used_margin']
        if required_margin > available_margin:
            violations.append(f"Insufficient margin: need {required_margin}, have {available_margin}")
        # Drawdown stop check
        current_drawdown = self.calculate_drawdown(portfolio_state['equity'])
        if current_drawdown < -0.20 and order['side'] == 'buy':
            violations.append(f"Portfolio drawdown {current_drawdown:.1%} > -20%, stopping new longs")
        return len(violations) == 0, violations
    def calculate_kelly_size(self, equity: float, volatility: float, entry_price: float, win_rate: float = 0.55, risk_per_trade: float = 0.02) -> int:
        max_loss_per_trade = equity * risk_per_trade
        risk_per_unit = entry_price * volatility
        reward_per_unit = entry_price * 0.02
        b = reward_per_unit / (risk_per_unit + 1e-8)
        p = win_rate
        q = 1 - win_rate
        kelly_fraction = (b * p - q) / (b + 1e-8)
        kelly_fraction_safe = max(0.0, kelly_fraction / 2)
        kelly_size = int((equity * kelly_fraction_safe) / (risk_per_unit + 1e-8))
        return max(1, kelly_size)
    def calculate_drawdown(self, equity: float) -> float:
        # Simulate drawdown for demo
        # In production, use historical equity curve
        return np.random.uniform(-0.25, 0.05)
    def estimate_slippage(self, ticker: str, quantity: int, current_price: float, market_data: Dict) -> Dict[str, float]:
        commission_rate = 0.0001
        commission = quantity * current_price * commission_rate
        adv = market_data[ticker]['average_daily_volume']
        volume_fraction = quantity / adv
        lambda_param = 0.3
        market_impact_bps = lambda_param * np.sqrt(volume_fraction) * 100
        market_impact = quantity * current_price * (market_impact_bps / 10000)
        bid_ask_spread = market_data[ticker].get('spread', 0.01)
        adverse_selection = quantity * current_price * (bid_ask_spread / 2)
        total_slippage = commission + market_impact + adverse_selection
        total_slippage_bps = (total_slippage / (quantity * current_price)) * 10000
        return {
            'commission': commission,
            'commission_bps': commission_rate * 10000,
            'market_impact': market_impact,
            'market_impact_bps': market_impact_bps,
            'adverse_selection': adverse_selection,
            'adverse_selection_bps': (adverse_selection / (quantity * current_price)) * 10000,
            'total_slippage': total_slippage,
            'total_slippage_bps': total_slippage_bps,
            'estimated_fill_price': current_price + (total_slippage / quantity)
        }

    def calculate_realistic_pnl(self, entry_price: float, exit_price: float, quantity: int,
                               commission_pct: float = 0.001, slippage_pct: float = 0.001) -> Dict[str, float]:
        """
        Calculate realistic P&L including commissions and slippage
        Returns dict with gross_pnl, net_pnl, commissions, slippage, and total_costs
        """
        # Apply slippage to entry and exit prices
        entry_slippage = entry_price * slippage_pct
        exit_slippage = exit_price * slippage_pct

        # Effective entry/exit prices after slippage
        effective_entry = entry_price + entry_slippage  # Buy at higher price
        effective_exit = exit_price - exit_slippage     # Sell at lower price

        # Calculate commissions
        entry_commission = effective_entry * quantity * commission_pct
        exit_commission = effective_exit * quantity * commission_pct
        total_commission = entry_commission + exit_commission

        # Calculate slippage costs
        entry_slippage_cost = entry_slippage * quantity
        exit_slippage_cost = exit_slippage * quantity
        total_slippage_cost = entry_slippage_cost + exit_slippage_cost

        # Calculate P&L
        gross_pnl = (effective_exit - effective_entry) * quantity
        total_costs = total_commission + total_slippage_cost
        net_pnl = gross_pnl - total_costs

        return {
            'gross_pnl': gross_pnl,
            'net_pnl': net_pnl,
            'total_costs': total_costs,
            'total_commission': total_commission,
            'total_slippage': total_slippage_cost,
            'effective_entry': effective_entry,
            'effective_exit': effective_exit,
            'commission_pct': commission_pct,
            'slippage_pct': slippage_pct
        }

    def execute_trade_with_costs(self, ticker: str, quantity: int, entry_price: float,
                                exit_price: float, commission_pct: float = 0.001,
                                slippage_pct: float = 0.001) -> Dict[str, float]:
        """
        Execute a complete trade round-trip with realistic costs
        Simplified version for backtesting and live trading
        """
        return self.calculate_realistic_pnl(entry_price, exit_price, quantity,
                                          commission_pct, slippage_pct)
