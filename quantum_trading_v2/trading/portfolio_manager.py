"""Portfolio manager stub for examples and unit tests."""
from __future__ import annotations

from typing import List, Dict


class PortfolioManager:
    def __init__(self, cash: float = 10000.0):
        self.cash = cash
        self.positions: List[Dict] = []

    def buy(self, ticker: str, shares: int, price: float) -> None:
        self.positions.append({"ticker": ticker, "shares": shares, "price": price})
        self.cash -= shares * price

    def sell(self, ticker: str) -> None:
        self.positions = [p for p in self.positions if p["ticker"] != ticker]

    def summary(self) -> Dict:
        return {"cash": self.cash, "positions": self.positions}


__all__ = ["PortfolioManager"]
