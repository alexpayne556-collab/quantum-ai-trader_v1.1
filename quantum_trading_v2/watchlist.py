"""
WATCHLIST MODULE
================
Tracks tickers for monitoring and signal generation.
"""
class Watchlist:
    def __init__(self):
        self.tickers = set()

    def add(self, ticker):
        self.tickers.add(ticker.upper())

    def remove(self, ticker):
        self.tickers.discard(ticker.upper())

    def get_all(self):
        return list(self.tickers)

    def clear(self):
        self.tickers.clear()
