"""
PORTFOLIO ANALYZER - Track Your Current Holdings
Fetches real-time prices, calculates P&L, suggests actions based on edges
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import json
import os

class PortfolioAnalyzer:
    """
    Analyze your portfolio with real prices and edge signals.
    """
    
    def __init__(self, portfolio_path: str = None):
        self.portfolio_path = portfolio_path or "/workspaces/quantum-ai-trader_v1.1/MY_PORTFOLIO.json"
        self.portfolio = self._load_portfolio()
        self.price_cache = {}
        
    def _load_portfolio(self) -> Dict:
        """Load portfolio from JSON file"""
        if os.path.exists(self.portfolio_path):
            with open(self.portfolio_path, 'r') as f:
                return json.load(f)
        return {"cash": 0, "positions": []}
    
    def save_portfolio(self, portfolio: Dict = None):
        """Save portfolio to JSON file"""
        if portfolio:
            self.portfolio = portfolio
        with open(self.portfolio_path, 'w') as f:
            json.dump(self.portfolio, f, indent=2, default=str)
    
    def get_current_price(self, ticker: str) -> float:
        """Get current price for a ticker"""
        if ticker in self.price_cache:
            return self.price_cache[ticker]
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='1d')
            if len(hist) > 0:
                price = hist['Close'].iloc[-1]
                self.price_cache[ticker] = price
                return price
        except:
            pass
        return 0.0
    
    def get_rsi(self, ticker: str, period: int = 14) -> float:
        """Calculate current RSI for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='3mo')
            if len(hist) < period + 1:
                return 50.0
            
            close = hist['Close']
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(period).mean()
            avg_loss = loss.rolling(period).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1]
        except:
            return 50.0
    
    def analyze_position(self, position: Dict) -> Dict:
        """Analyze a single position"""
        ticker = position['ticker']
        entry_price = position.get('entry_price', 0)
        shares = position.get('shares', 0)
        
        current_price = self.get_current_price(ticker)
        rsi = self.get_rsi(ticker)
        
        # Calculate P&L
        if entry_price > 0 and shares > 0:
            pnl_pct = (current_price / entry_price - 1) * 100
            pnl_dollar = (current_price - entry_price) * shares
        else:
            pnl_pct = 0
            pnl_dollar = 0
        
        # Determine action based on our validated edges
        action = "HOLD"
        reason = ""
        
        if rsi < 10:
            action = "STRONG BUY"
            reason = f"RSI={rsi:.1f} < 10 (80.8% WR edge)"
        elif rsi < 15:
            action = "BUY"
            reason = f"RSI={rsi:.1f} < 15 (72.2% WR edge)"
        elif rsi < 20:
            action = "CONSIDER BUY"
            reason = f"RSI={rsi:.1f} < 20 (67.5% WR edge)"
        elif pnl_pct >= 15:
            action = "TAKE PROFIT"
            reason = f"Up {pnl_pct:.1f}%, lock in gains"
        elif pnl_pct <= -10:
            action = "CUT LOSS"
            reason = f"Down {pnl_pct:.1f}%, protect capital"
        elif rsi > 80:
            action = "CONSIDER SELL"
            reason = f"RSI={rsi:.1f} > 80 (overbought)"
        
        return {
            'ticker': ticker,
            'shares': shares,
            'entry_price': entry_price,
            'current_price': round(current_price, 2),
            'rsi': round(rsi, 1),
            'pnl_pct': round(pnl_pct, 2),
            'pnl_dollar': round(pnl_dollar, 2),
            'market_value': round(current_price * shares, 2),
            'action': action,
            'reason': reason,
            'sector': position.get('sector', 'UNKNOWN')
        }
    
    def analyze_portfolio(self) -> pd.DataFrame:
        """Analyze entire portfolio"""
        if not self.portfolio.get('positions'):
            print("No positions in portfolio")
            return pd.DataFrame()
        
        analyses = []
        for pos in self.portfolio['positions']:
            if pos.get('shares', 0) > 0:  # Only analyze positions with shares
                analysis = self.analyze_position(pos)
                analyses.append(analysis)
        
        if not analyses:
            print("No active positions found")
            return pd.DataFrame()
        
        df = pd.DataFrame(analyses)
        return df.sort_values('action', ascending=False)
    
    def get_portfolio_summary(self) -> Dict:
        """Get summary statistics for portfolio"""
        df = self.analyze_portfolio()
        
        if len(df) == 0:
            return {
                'total_value': self.portfolio.get('cash', 0),
                'cash': self.portfolio.get('cash', 0),
                'positions_value': 0,
                'total_pnl': 0,
                'total_pnl_pct': 0,
                'positions': []
            }
        
        positions_value = df['market_value'].sum()
        cash = self.portfolio.get('cash', 0)
        total_value = positions_value + cash
        total_pnl = df['pnl_dollar'].sum()
        
        return {
            'total_value': round(total_value, 2),
            'cash': round(cash, 2),
            'positions_value': round(positions_value, 2),
            'total_pnl': round(total_pnl, 2),
            'num_positions': len(df),
            'buy_signals': len(df[df['action'].str.contains('BUY')]),
            'sell_signals': len(df[df['action'].str.contains('SELL|PROFIT|LOSS')]),
            'positions': df.to_dict('records')
        }
    
    def print_portfolio_report(self):
        """Print formatted portfolio report"""
        summary = self.get_portfolio_summary()
        
        print("\n" + "="*70)
        print("📊 PORTFOLIO ANALYSIS REPORT")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("-"*70)
        
        print(f"\n💰 PORTFOLIO VALUE:")
        print(f"   Total Value:      ${summary['total_value']:,.2f}")
        print(f"   Cash:             ${summary['cash']:,.2f}")
        print(f"   Positions Value:  ${summary['positions_value']:,.2f}")
        print(f"   Total P&L:        ${summary['total_pnl']:,.2f}")
        
        print(f"\n📈 POSITIONS ({summary['num_positions']}):")
        print("-"*70)
        print(f"{'Ticker':<8} {'Price':<10} {'RSI':<8} {'P&L%':<10} {'Action':<15} {'Reason'}")
        print("-"*70)
        
        for pos in summary['positions']:
            pnl_str = f"{pos['pnl_pct']:+.1f}%"
            print(f"{pos['ticker']:<8} ${pos['current_price']:<8.2f} {pos['rsi']:<8.1f} {pnl_str:<10} {pos['action']:<15} {pos['reason'][:30]}")
        
        print("\n" + "="*70)
        print(f"🎯 SIGNALS: {summary['buy_signals']} BUY | {summary['sell_signals']} SELL")
        print("="*70)
        
        return summary
    
    def add_position(self, ticker: str, shares: int, entry_price: float, sector: str = "UNKNOWN"):
        """Add a new position to portfolio"""
        position = {
            'ticker': ticker.upper(),
            'entry_price': entry_price,
            'shares': shares,
            'entry_date': datetime.now().isoformat(),
            'sector': sector,
            'current_price': 0,
            'stop_loss': None,
            'target_price': None
        }
        self.portfolio['positions'].append(position)
        self.save_portfolio()
        print(f"✅ Added {shares} shares of {ticker} at ${entry_price}")
    
    def update_position(self, ticker: str, shares: int = None, entry_price: float = None):
        """Update an existing position"""
        for pos in self.portfolio['positions']:
            if pos['ticker'].upper() == ticker.upper():
                if shares is not None:
                    pos['shares'] = shares
                if entry_price is not None:
                    pos['entry_price'] = entry_price
                self.save_portfolio()
                print(f"✅ Updated {ticker}")
                return
        print(f"❌ Position {ticker} not found")
    
    def remove_position(self, ticker: str):
        """Remove a position from portfolio"""
        self.portfolio['positions'] = [
            p for p in self.portfolio['positions'] 
            if p['ticker'].upper() != ticker.upper()
        ]
        self.save_portfolio()
        print(f"✅ Removed {ticker}")


class WatchlistScanner:
    """
    Scan watchlist for edge signals.
    Uses validated edges: RSI<10 (80.8% WR), RSI<15 (72.2% WR), RSI<20 (67.5% WR)
    """
    
    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or []
        self.data_cache = {}
        
    def load_from_file(self, filepath: str):
        """Load tickers from file (one per line)"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.tickers = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return self.tickers
    
    def get_ticker_data(self, ticker: str) -> Dict:
        """Get price and RSI data for a ticker"""
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='3mo')
            if len(hist) < 20:
                return None
            
            close = hist['Close']
            volume = hist['Volume']
            
            # Calculate RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate volume ratio
            vol_ma = volume.rolling(20).mean()
            vol_ratio = volume.iloc[-1] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1
            
            # Calculate recent drop
            high_21d = close.rolling(21).max()
            drop_from_high = (close.iloc[-1] / high_21d.iloc[-1] - 1) * 100
            
            return {
                'ticker': ticker,
                'price': round(close.iloc[-1], 2),
                'rsi': round(rsi.iloc[-1], 1),
                'volume_ratio': round(vol_ratio, 2),
                'drop_from_21d_high': round(drop_from_high, 1),
                'change_1d': round((close.iloc[-1] / close.iloc[-2] - 1) * 100, 2),
                'change_5d': round((close.iloc[-1] / close.iloc[-5] - 1) * 100, 2) if len(close) >= 5 else 0
            }
        except Exception as e:
            return None
    
    def scan_for_signals(self) -> pd.DataFrame:
        """Scan all tickers for edge signals"""
        signals = []
        
        print(f"Scanning {len(self.tickers)} tickers for signals...")
        for i, ticker in enumerate(self.tickers):
            if (i + 1) % 10 == 0:
                print(f"  Progress: {i+1}/{len(self.tickers)}")
            
            data = self.get_ticker_data(ticker)
            if data:
                # Determine signal based on validated edges
                signal = None
                win_rate = 0
                
                if data['rsi'] < 10:
                    signal = "🔥 RSI<10"
                    win_rate = 80.8
                elif data['rsi'] < 15:
                    signal = "✅ RSI<15"
                    win_rate = 72.2
                elif data['rsi'] < 20:
                    signal = "📊 RSI<20"
                    win_rate = 67.5
                elif data['drop_from_21d_high'] <= -20:
                    signal = "📉 20% Drop"
                    win_rate = 61.3
                
                if signal:
                    data['signal'] = signal
                    data['expected_wr'] = win_rate
                    signals.append(data)
        
        if not signals:
            print("No signals found")
            return pd.DataFrame()
        
        df = pd.DataFrame(signals)
        return df.sort_values('rsi', ascending=True)
    
    def print_signal_report(self):
        """Print formatted signal report"""
        df = self.scan_for_signals()
        
        print("\n" + "="*70)
        print("🎯 WATCHLIST SIGNAL SCANNER")
        print("="*70)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print(f"Tickers scanned: {len(self.tickers)}")
        print("-"*70)
        
        if len(df) == 0:
            print("No signals found - market not oversold")
            return df
        
        # RSI < 10 signals (BEST)
        rsi10 = df[df['rsi'] < 10]
        if len(rsi10) > 0:
            print(f"\n🔥 RSI < 10 SIGNALS (80.8% Win Rate) - {len(rsi10)} found:")
            for _, row in rsi10.iterrows():
                print(f"   {row['ticker']:<6} RSI={row['rsi']:<5.1f} Price=${row['price']:<8.2f} Drop={row['drop_from_21d_high']:+.1f}%")
        
        # RSI < 15 signals
        rsi15 = df[(df['rsi'] >= 10) & (df['rsi'] < 15)]
        if len(rsi15) > 0:
            print(f"\n✅ RSI < 15 SIGNALS (72.2% Win Rate) - {len(rsi15)} found:")
            for _, row in rsi15.iterrows():
                print(f"   {row['ticker']:<6} RSI={row['rsi']:<5.1f} Price=${row['price']:<8.2f} Drop={row['drop_from_21d_high']:+.1f}%")
        
        # RSI < 20 signals
        rsi20 = df[(df['rsi'] >= 15) & (df['rsi'] < 20)]
        if len(rsi20) > 0:
            print(f"\n📊 RSI < 20 SIGNALS (67.5% Win Rate) - {len(rsi20)} found:")
            for _, row in rsi20.head(10).iterrows():  # Top 10 only
                print(f"   {row['ticker']:<6} RSI={row['rsi']:<5.1f} Price=${row['price']:<8.2f} Drop={row['drop_from_21d_high']:+.1f}%")
        
        print("\n" + "="*70)
        print(f"SUMMARY: {len(rsi10)} hot | {len(rsi15)} good | {len(rsi20)} okay")
        print("="*70)
        
        return df


def quick_test():
    """Quick test of portfolio analyzer"""
    print("Testing Portfolio Analyzer...")
    
    # Test with sample tickers
    scanner = WatchlistScanner(['NVDA', 'AMD', 'TSLA', 'IONQ', 'RGTI'])
    scanner.print_signal_report()


if __name__ == "__main__":
    quick_test()
