"""
Simple ML-Powered Scanners - Work in Colab
All scanners in one file for easy import
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import ta
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PRE-GAINER SCANNER
# ============================================================================

class PreGainerScanner:
    """Scan for stocks likely to gap up at open"""
    
    def __init__(self):
        self.name = "Pre-Gainer Scanner"
    
    def scan(self, symbols, min_score=0.6):
        """Scan for pre-market gainers"""
        results = []
        
        for symbol in symbols:
            try:
                # Download recent data
                df = yf.download(symbol, period='5d', progress=False)
                if len(df) < 3:
                    continue
                
                # Calculate momentum
                returns = df['Close'].pct_change()
                momentum = returns.iloc[-1]
                
                # Volume surge
                vol_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
                
                # Score
                score = (momentum * 10 + vol_ratio) / 2
                score = min(max(score, 0), 1)
                
                if score >= min_score:
                    results.append({
                        'symbol': symbol,
                        'score': float(score),
                        'momentum': float(momentum),
                        'volume_ratio': float(vol_ratio),
                        'current_price': float(df['Close'].iloc[-1])
                    })
            except:
                continue
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# ============================================================================
# DAY TRADING SCANNER
# ============================================================================

class DayTradingScanner:
    """Scan for intraday trading opportunities"""
    
    def __init__(self):
        self.name = "Day Trading Scanner"
    
    def scan(self, symbols, min_score=0.6):
        """Scan for day trading setups"""
        results = []
        
        for symbol in symbols:
            try:
                df = yf.download(symbol, period='10d', progress=False)
                if len(df) < 5:
                    continue
                
                # RSI
                rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
                rsi_score = 1 - abs(rsi - 50) / 50  # Closer to 50 = neutral = good
                
                # Volatility
                returns = df['Close'].pct_change()
                volatility = returns.std()
                vol_score = min(volatility * 20, 1.0)  # Higher vol = better for day trading
                
                # Volume
                vol_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
                volume_score = min(vol_ratio / 2, 1.0)
                
                # Combined score
                score = (rsi_score + vol_score + volume_score) / 3
                
                if score >= min_score:
                    results.append({
                        'symbol': symbol,
                        'score': float(score),
                        'rsi': float(rsi),
                        'volatility': float(volatility),
                        'volume_ratio': float(vol_ratio),
                        'current_price': float(df['Close'].iloc[-1])
                    })
            except:
                continue
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# ============================================================================
# OPPORTUNITY SCANNER
# ============================================================================

class OpportunityScanner:
    """Scan for swing trade opportunities"""
    
    def __init__(self):
        self.name = "Opportunity Scanner"
    
    def scan(self, symbols, min_score=0.6):
        """Scan for swing trading setups"""
        results = []
        
        for symbol in symbols:
            try:
                df = yf.download(symbol, period='3mo', progress=False)
                if len(df) < 50:
                    continue
                
                # Moving averages
                ma20 = df['Close'].rolling(20).mean().iloc[-1]
                ma50 = df['Close'].rolling(50).mean().iloc[-1]
                current = df['Close'].iloc[-1]
                
                # Trend score
                trend_score = 1 if current > ma20 > ma50 else 0.5
                
                # RSI (looking for oversold)
                rsi = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
                rsi_score = 1 - (rsi / 100) if rsi < 50 else 0.5
                
                # MACD
                macd = ta.trend.MACD(df['Close'])
                macd_diff = macd.macd_diff().iloc[-1]
                macd_score = 1 if macd_diff > 0 else 0.3
                
                # Combined score
                score = (trend_score * 0.4 + rsi_score * 0.3 + macd_score * 0.3)
                
                if score >= min_score:
                    results.append({
                        'symbol': symbol,
                        'score': float(score),
                        'rsi': float(rsi),
                        'macd': float(macd_diff),
                        'trend': 'Bullish' if trend_score > 0.7 else 'Neutral',
                        'current_price': float(current)
                    })
            except:
                continue
        
        return sorted(results, key=lambda x: x['score'], reverse=True)

# ============================================================================
# PENNY PUMP DETECTOR
# ============================================================================

class PennyPumpDetector:
    """Detect potential penny stock pumps"""
    
    def __init__(self):
        self.name = "Penny Pump Detector"
    
    def scan(self, symbols, max_price=10):
        """Scan for penny stock pump patterns"""
        results = []
        
        for symbol in symbols:
            try:
                df = yf.download(symbol, period='1mo', progress=False)
                if len(df) < 10:
                    continue
                
                current_price = df['Close'].iloc[-1]
                
                # Only penny stocks
                if current_price > max_price:
                    continue
                
                # Huge volume spike
                vol_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
                
                # Price acceleration
                returns_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) if len(df) >= 5 else 0
                
                # Risk score (higher = more suspicious)
                risk_score = (vol_ratio / 5 + abs(returns_5d) * 5) / 2
                risk_score = min(risk_score, 1.0)
                
                if risk_score > 0.5:
                    results.append({
                        'symbol': symbol,
                        'risk_score': float(risk_score),
                        'price': float(current_price),
                        'volume_spike': float(vol_ratio),
                        'return_5d': float(returns_5d),
                        'warning': 'HIGH RISK' if risk_score > 0.8 else 'MODERATE RISK'
                    })
            except:
                continue
        
        return sorted(results, key=lambda x: x['risk_score'], reverse=True)

# ============================================================================
# SOCIAL SENTIMENT DETECTOR
# ============================================================================

class SocialSentimentDetector:
    """Detect social media sentiment explosions"""
    
    def __init__(self):
        self.name = "Social Sentiment Detector"
    
    def scan(self, symbols, min_score=0.6):
        """
        Scan for viral stock momentum
        (Note: Real social sentiment requires APIs - this uses volume as proxy)
        """
        results = []
        
        for symbol in symbols:
            try:
                df = yf.download(symbol, period='1mo', progress=False)
                if len(df) < 10:
                    continue
                
                # Volume explosion (proxy for social attention)
                vol_ratio = df['Volume'].iloc[-1] / df['Volume'].mean()
                
                # Price momentum
                returns_1d = df['Close'].pct_change().iloc[-1]
                returns_5d = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) if len(df) >= 5 else 0
                
                # Volatility (viral stocks are volatile)
                volatility = df['Close'].pct_change().std()
                
                # Social score (volume + momentum + volatility)
                social_score = (vol_ratio / 5 + abs(returns_5d) * 5 + volatility * 10) / 3
                social_score = min(social_score, 1.0)
                
                if social_score >= min_score:
                    results.append({
                        'symbol': symbol,
                        'social_score': float(social_score),
                        'volume_ratio': float(vol_ratio),
                        'momentum_5d': float(returns_5d),
                        'volatility': float(volatility),
                        'current_price': float(df['Close'].iloc[-1]),
                        'status': 'VIRAL' if social_score > 0.8 else 'TRENDING'
                    })
            except:
                continue
        
        return sorted(results, key=lambda x: x['social_score'], reverse=True)

# ============================================================================
# MORNING BRIEF GENERATOR
# ============================================================================

class MorningBriefGenerator:
    """Generate morning trading brief"""
    
    def __init__(self):
        self.name = "Morning Brief Generator"
    
    def generate(self, symbols):
        """Generate morning market brief"""
        try:
            # Download SPY for market overview
            spy = yf.download('SPY', period='5d', progress=False)
            spy_return = spy['Close'].pct_change().iloc[-1]
            spy_trend = 'Bullish' if spy_return > 0.01 else ('Bearish' if spy_return < -0.01 else 'Neutral')
            
            # Scan top movers
            movers = []
            for symbol in symbols[:20]:  # Top 20
                try:
                    df = yf.download(symbol, period='2d', progress=False)
                    if len(df) >= 2:
                        ret = df['Close'].pct_change().iloc[-1]
                        movers.append((symbol, ret))
                except:
                    continue
            
            movers.sort(key=lambda x: abs(x[1]), reverse=True)
            
            brief = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'market_trend': spy_trend,
                'spy_return': float(spy_return),
                'top_gainers': [(s, float(r)) for s, r in movers[:5] if r > 0],
                'top_losers': [(s, float(r)) for s, r in movers[:5] if r < 0],
                'key_levels': {
                    'SPY': float(spy['Close'].iloc[-1])
                }
            }
            
            return brief
        except Exception as e:
            return {'error': str(e)}

