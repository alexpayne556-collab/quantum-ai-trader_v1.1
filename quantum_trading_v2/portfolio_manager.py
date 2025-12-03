"""
PORTFOLIO MANAGER MODULE
========================
Manages portfolio positions, allocations, and performance tracking.
"""
import pandas as pd

class PortfolioManager:
    def __init__(self, initial_cash=100000):
        self.cash = initial_cash
        self.positions = {}
        self.history = []

    def buy(self, ticker, shares, price):
        cost = shares * price
        if self.cash >= cost:
            self.cash -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + shares
            self.history.append({'action': 'buy', 'ticker': ticker, 'shares': shares, 'price': price})
            return True
        return False

    def sell(self, ticker, shares, price):
        if self.positions.get(ticker, 0) >= shares:
            self.positions[ticker] -= shares
            self.cash += shares * price
            self.history.append({'action': 'sell', 'ticker': ticker, 'shares': shares, 'price': price})
            return True
        return False

    def get_portfolio_value(self, prices):
        value = self.cash
        for ticker, shares in self.positions.items():
            value += shares * prices.get(ticker, 0)
        return value

    def get_positions(self):
        return self.positions.copy()

    def get_history(self):
        return pd.DataFrame(self.history)
