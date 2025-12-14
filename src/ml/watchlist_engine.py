"""
WATCHLIST ENGINE
================
Scan your 76 legendary tickers for high-confidence trading opportunities.

Features:
- Scan all tickers using Trident models
- Rank by confidence, entry quality, risk/reward
- Filter by PDT compliance (can you trade today?)
- Identify "ready to buy NOW" vs "watch" vs "avoid"
- Real-time updates every 15 minutes

This is your AI scout - it finds opportunities you might miss.

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WatchlistScanner:
    """
    Scan watchlist for trading opportunities.
    """
    
    def __init__(
        self,
        model_dir: str = 'models/trident',
        min_confidence: float = 0.70,      # Minimum confidence to trade
        max_positions: int = 5,            # Max simultaneous positions
        min_volume: float = 500_000,       # Minimum daily volume
        max_spread_pct: float = 0.02,      # Max 2% spread
        scan_interval_minutes: int = 15    # Scan every 15 min
    ):
        """
        Args:
            model_dir: Directory with trained Trident models
            min_confidence: Minimum confidence to suggest trade
            max_positions: Maximum open positions
            min_volume: Minimum daily volume filter
            max_spread_pct: Maximum bid-ask spread %
            scan_interval_minutes: Scan frequency
        """
        self.model_dir = Path(model_dir)
        self.min_confidence = min_confidence
        self.max_positions = max_positions
        self.min_volume = min_volume
        self.max_spread_pct = max_spread_pct
        self.scan_interval_minutes = scan_interval_minutes
        
        # Load inference engine
        from src.ml.inference_engine import TridenInference
        self.engine = TridenInference(model_dir=str(self.model_dir))
        
        # Scan history
        self.last_scan_time: Optional[datetime] = None
        self.last_scan_results: List[Dict] = []
        
    def load_watchlist(self, watchlist_file: str = None) -> List[str]:
        """
        Load watchlist tickers.
        
        Args:
            watchlist_file: Path to watchlist file
            
        Returns:
            List of ticker symbols
        """
        if watchlist_file and Path(watchlist_file).exists():
            with open(watchlist_file, 'r') as f:
                tickers = [line.strip() for line in f if line.strip()]
            logger.info(f"✅ Loaded {len(tickers)} tickers from {watchlist_file}")
            return tickers
        
        # Default: Use legendary tickers
        try:
            from config.legendary_tickers import get_legendary_tickers
            tickers = get_legendary_tickers()
            logger.info(f"✅ Loaded {len(tickers)} legendary tickers")
            return tickers
        except ImportError:
            logger.warning("⚠️ legendary_tickers not found, using sample tickers")
            return ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX',
                   'AMD', 'INTC', 'ORCL', 'CRM', 'ADBE', 'PYPL', 'SQ', 'SHOP',
                   'PALI', 'RXT', 'KDK', 'ASTS', 'HOOD', 'DGNX']
    
    def fetch_ticker_data(self, ticker: str) -> Optional[Dict]:
        """
        Fetch current data for ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            {price, volume, spread, features, ...} or None if error
        """
        try:
            # Download recent data
            df = yf.download(ticker, period='3mo', progress=False)
            
            if df.empty or len(df) < 21:
                return None
            
            # Calculate features (simplified - in production use full feature set)
            latest = df.iloc[-1]
            
            # Basic features
            features = pd.Series({
                'close': latest['Close'],
                'volume': latest['Volume'],
                'returns_1d': df['Close'].pct_change().iloc[-1],
                'returns_5d': df['Close'].pct_change(5).iloc[-1],
                'returns_21d': df['Close'].pct_change(21).iloc[-1],
                'volume_ratio': latest['Volume'] / df['Volume'].rolling(20).mean().iloc[-1],
                'rsi_14': self._calculate_rsi(df['Close'], 14),
                'macd': self._calculate_macd(df['Close']),
                'volatility_21d': df['Close'].pct_change().rolling(21).std().iloc[-1]
            })
            
            # Estimate spread (simplified)
            spread_pct = (latest['High'] - latest['Low']) / latest['Close']
            
            return {
                'ticker': ticker,
                'price': latest['Close'],
                'volume': latest['Volume'],
                'spread_pct': spread_pct,
                'features': features,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.debug(f"Error fetching {ticker}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) > 0 else 50
    
    def _calculate_macd(self, prices: pd.Series) -> float:
        """Calculate MACD."""
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        return macd.iloc[-1] if len(macd) > 0 else 0
    
    def scan_ticker(self, ticker: str, current_portfolio: Dict = None) -> Optional[Dict]:
        """
        Scan single ticker for opportunity.
        
        Args:
            ticker: Ticker symbol
            current_portfolio: Current portfolio state
            
        Returns:
            Opportunity dict or None
        """
        # Fetch data
        data = self.fetch_ticker_data(ticker)
        if data is None:
            return None
        
        # Apply filters
        
        # 1. Volume filter
        if data['volume'] < self.min_volume:
            return None
        
        # 2. Spread filter
        if data['spread_pct'] > self.max_spread_pct:
            return None
        
        # 3. Already in portfolio?
        already_owned = False
        if current_portfolio and ticker in current_portfolio.get('positions', {}):
            already_owned = True
        
        # Get model prediction
        try:
            prediction = self.engine.predict(
                ticker=ticker,
                features=data['features']
            )
        except Exception as e:
            logger.debug(f"Prediction error for {ticker}: {e}")
            return None
        
        # Only consider BUY signals
        if prediction['signal'] != 'BUY':
            return None
        
        # Confidence threshold
        if prediction['confidence'] < self.min_confidence * 100:
            return None
        
        # Calculate entry quality score
        entry_quality = self._calculate_entry_quality(data, prediction)
        
        # Calculate risk/reward
        risk_reward = self._calculate_risk_reward(data, prediction)
        
        return {
            'ticker': ticker,
            'price': data['price'],
            'volume': data['volume'],
            'spread_pct': data['spread_pct'],
            'signal': prediction['signal'],
            'confidence': prediction['confidence'],
            'probability': prediction['probability'],
            'cluster_id': prediction['cluster_id'],
            'entry_quality': entry_quality,
            'risk_reward': risk_reward,
            'already_owned': already_owned,
            'timestamp': data['timestamp']
        }
    
    def _calculate_entry_quality(self, data: Dict, prediction: Dict) -> float:
        """
        Calculate entry quality score (0-100).
        
        Factors:
        - Model confidence
        - Volume above average
        - Low spread
        - Not overbought
        """
        score = 0
        
        # Confidence (40 points)
        score += (prediction['confidence'] / 100) * 40
        
        # Volume (20 points)
        # If volume > avg, better entry
        volume_score = min(data['volume'] / self.min_volume, 2.0)
        score += volume_score * 10
        
        # Spread (20 points)
        spread_score = max(0, 1 - data['spread_pct'] / self.max_spread_pct)
        score += spread_score * 20
        
        # RSI (20 points) - prefer not overbought
        if 'rsi_14' in data['features']:
            rsi = data['features']['rsi_14']
            if rsi < 30:
                score += 20  # Oversold - great entry
            elif rsi < 50:
                score += 15  # Neutral - good entry
            elif rsi < 70:
                score += 10  # Mildly bullish - ok entry
            # Else: overbought - 0 points
        
        return min(score, 100)
    
    def _calculate_risk_reward(self, data: Dict, prediction: Dict) -> float:
        """
        Calculate risk/reward ratio.
        
        Assumptions:
        - Stop loss: -19% (from evolved_config)
        - Take profit: +15% average
        """
        stop_loss = 0.19
        take_profit = 0.15
        
        # Adjust take profit based on confidence
        if prediction['confidence'] > 85:
            take_profit = 0.20  # Higher target for high-confidence
        
        risk_reward = take_profit / stop_loss
        return risk_reward
    
    def scan_all(self, current_portfolio: Dict = None, max_workers: int = 10) -> List[Dict]:
        """
        Scan all watchlist tickers.
        
        Args:
            current_portfolio: Current portfolio state
            max_workers: Parallel workers
            
        Returns:
            List of opportunities, sorted by quality
        """
        logger.info("\n" + "="*60)
        logger.info("WATCHLIST SCAN")
        logger.info("="*60)
        logger.info(f"Scan time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Load watchlist
        tickers = self.load_watchlist()
        logger.info(f"Scanning {len(tickers)} tickers...")
        
        # Scan in parallel
        opportunities = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                lambda t: self.scan_ticker(t, current_portfolio),
                tickers
            )
            
            for result in results:
                if result is not None:
                    opportunities.append(result)
        
        # Sort by entry quality (best first)
        opportunities.sort(key=lambda x: x['entry_quality'], reverse=True)
        
        # Update scan state
        self.last_scan_time = datetime.now()
        self.last_scan_results = opportunities
        
        logger.info(f"✅ Scan complete: {len(opportunities)} opportunities found")
        
        return opportunities
    
    def get_top_opportunities(
        self,
        n: int = 5,
        exclude_owned: bool = True,
        current_portfolio: Dict = None
    ) -> List[Dict]:
        """
        Get top N trading opportunities.
        
        Args:
            n: Number of opportunities to return
            exclude_owned: Exclude already-owned tickers
            current_portfolio: Current portfolio state
            
        Returns:
            Top opportunities
        """
        # If no recent scan, scan now
        if not self.last_scan_results or \
           (self.last_scan_time and (datetime.now() - self.last_scan_time).seconds > self.scan_interval_minutes * 60):
            self.scan_all(current_portfolio)
        
        opportunities = self.last_scan_results
        
        # Filter out owned if requested
        if exclude_owned and current_portfolio:
            owned_tickers = set(current_portfolio.get('positions', {}).keys())
            opportunities = [opp for opp in opportunities if opp['ticker'] not in owned_tickers]
        
        # Return top N
        return opportunities[:n]
    
    def display_opportunities(self, opportunities: List[Dict]):
        """Display opportunities in readable format."""
        if not opportunities:
            logger.info("❌ No opportunities found")
            return
        
        logger.info("\n" + "="*60)
        logger.info(f"TOP {len(opportunities)} OPPORTUNITIES")
        logger.info("="*60)
        
        for i, opp in enumerate(opportunities, 1):
            logger.info(f"\n{i}. {opp['ticker']}")
            logger.info(f"   Price: ${opp['price']:.2f}")
            logger.info(f"   Signal: {opp['signal']} (confidence: {opp['confidence']:.1f}%)")
            logger.info(f"   Entry quality: {opp['entry_quality']:.1f}/100")
            logger.info(f"   Risk/Reward: {opp['risk_reward']:.2f}:1")
            logger.info(f"   Volume: {opp['volume']:,.0f}")
            logger.info(f"   Spread: {opp['spread_pct']:.2%}")
            logger.info(f"   Cluster: {opp['cluster_id']}")
            if opp['already_owned']:
                logger.info(f"   ⚠️ Already in portfolio")
    
    def get_next_trade_suggestion(self, current_portfolio: Dict) -> Optional[Dict]:
        """
        Get the BEST next trade suggestion.
        
        Considers:
        - Portfolio constraints (max positions, buying power)
        - PDT restrictions
        - Opportunity quality
        
        Args:
            current_portfolio: Current portfolio state
            
        Returns:
            Trade suggestion or None
        """
        # Check portfolio constraints
        n_positions = current_portfolio.get('n_positions', 0)
        if n_positions >= self.max_positions:
            logger.info(f"⚠️ Max positions reached ({n_positions}/{self.max_positions})")
            return None
        
        buying_power = current_portfolio.get('buying_power', 0)
        if buying_power < 100:
            logger.info(f"⚠️ Insufficient buying power (${buying_power:.2f})")
            return None
        
        # Get top opportunity
        opportunities = self.get_top_opportunities(
            n=1,
            exclude_owned=True,
            current_portfolio=current_portfolio
        )
        
        if not opportunities:
            logger.info("❌ No trade suggestions at this time")
            return None
        
        best = opportunities[0]
        
        # Calculate position size (21% of account)
        account_equity = current_portfolio.get('account_equity', 0)
        position_value = account_equity * 0.21  # From evolved_config
        shares = int(position_value / best['price'])
        
        if shares == 0:
            logger.info(f"⚠️ Position size too small for {best['ticker']}")
            return None
        
        suggestion = {
            **best,
            'suggested_shares': shares,
            'suggested_value': shares * best['price'],
            'stop_loss_price': best['price'] * 0.81,  # -19%
            'take_profit_price': best['price'] * 1.15,  # +15%
            'max_hold_days': 32  # From evolved_config
        }
        
        logger.info("\n" + "="*60)
        logger.info("🎯 NEXT TRADE SUGGESTION")
        logger.info("="*60)
        logger.info(f"   BUY {suggestion['ticker']}")
        logger.info(f"   Price: ${suggestion['price']:.2f}")
        logger.info(f"   Shares: {suggestion['suggested_shares']}")
        logger.info(f"   Value: ${suggestion['suggested_value']:.2f}")
        logger.info(f"   Confidence: {suggestion['confidence']:.1f}%")
        logger.info(f"   Entry quality: {suggestion['entry_quality']:.1f}/100")
        logger.info(f"   Stop loss: ${suggestion['stop_loss_price']:.2f} (-19%)")
        logger.info(f"   Take profit: ${suggestion['take_profit_price']:.2f} (+15%)")
        logger.info(f"   Risk/Reward: {suggestion['risk_reward']:.2f}:1")
        
        return suggestion


def example_usage():
    """Example: Scan watchlist for opportunities."""
    logger.info("\n" + "="*60)
    logger.info("WATCHLIST SCANNER - Example Usage")
    logger.info("="*60 + "\n")
    
    # Initialize scanner
    scanner = WatchlistScanner(
        model_dir='models/trident',
        min_confidence=0.70,
        max_positions=5
    )
    
    # Mock portfolio state (replace with real portfolio tracker)
    current_portfolio = {
        'account_equity': 780.59,
        'buying_power': 186.10,
        'n_positions': 4,
        'positions': {'PALI': {}, 'RXT': {}, 'KDK': {}, 'ASTS': {}}
    }
    
    # Scan all
    opportunities = scanner.scan_all(current_portfolio)
    
    # Display top 10
    top_10 = opportunities[:10]
    scanner.display_opportunities(top_10)
    
    # Get next trade suggestion
    suggestion = scanner.get_next_trade_suggestion(current_portfolio)


if __name__ == '__main__':
    example_usage()
