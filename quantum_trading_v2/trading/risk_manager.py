"""Simple risk manager for position sizing and stops."""
from __future__ import annotations


class RiskManager:
    def __init__(self, max_risk_pct: float = 0.02):
        self.max_risk_pct = max_risk_pct

    def max_shares(self, account_balance: float, entry_price: float, stop_price: float) -> int:
        risk_per_share = max(0.0, entry_price - stop_price)
        if risk_per_share <= 0:
            return 0
        max_risk = account_balance * self.max_risk_pct
        return int(max_risk / risk_per_share)


__all__ = ["RiskManager"]
