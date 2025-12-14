"""
CROSS-ASSET INTELLIGENCE BRAIN
===============================
Learns: "When X moves, Y follows"
- Discovers lead-lag relationships between your 50 tickers
- Updates daily with new correlations
- Generates signals based on what happened to LEADERS today

FREE DATA ONLY - Uses yfinance
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Your tickers
TICKERS = ['APLD','SERV','MRVL','HOOD','LUNR','BAC','QCOM','UUUU','TSLA','AMD',
           'NOW','NVDA','MU','PG','DLB','XME','KRYS','LEU','QTUM','SPY',
           'UNH','WMT','OKLO','RXRX','MTZ','SNOW','GRRR','BSX','LLY','VOO',
           'GEO','CXW','LYFT','MNDY','BA','LAC','INTC','ALK','LMT','CRDO',
           'ANET','META','RIVN','GOOGL','HL','TEM','TDOC','KMTS']

# Known sector relationships (domain knowledge boost)
SECTOR_MAP = {
    'SEMI': ['NVDA', 'AMD', 'MRVL', 'MU', 'INTC', 'QCOM', 'CRDO', 'ANET'],
    'TECH': ['META', 'GOOGL', 'NOW', 'SNOW', 'MNDY'],
    'ENERGY': ['UUUU', 'LEU', 'OKLO', 'XME', 'HL', 'LAC'],
    'FINTECH': ['HOOD', 'BAC'],
    'HEALTHCARE': ['LLY', 'BSX', 'UNH', 'KRYS', 'RXRX', 'TDOC', 'TEM'],
    'EV_AUTO': ['TSLA', 'RIVN'],
    'SPACE': ['LUNR'],
    'DEFENSE': ['LMT', 'BA'],
    'RETAIL': ['WMT', 'PG'],
    'SPEC': ['APLD', 'SERV', 'GRRR', 'GEO', 'CXW', 'LYFT', 'ALK', 'DLB', 'MTZ', 'KMTS'],
    'INDEX': ['SPY', 'VOO', 'QTUM', 'SCHA']
}

class CrossAssetBrain:
    """
    Learns which assets predict which other assets.
    Updates relationships daily.
    """
    
    def __init__(self, lookback_days=252, min_correlation=0.08):  # Lower threshold
        self.lookback = lookback_days
        self.min_corr = min_correlation
        self.relationships = {}  # {follower: [{leader, lag, corr, direction}]}
        self.data = {}
        self.last_update = None
        self.cache_file = 'cross_asset_relationships.json'
        
        # Load cached relationships if exists
        self._load_cache()
    
    def _load_cache(self):
        """Load previously discovered relationships"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    cached = json.load(f)
                    self.relationships = cached.get('relationships', {})
                    self.last_update = cached.get('last_update')
                    print(f"📂 Loaded {len(self.relationships)} cached relationships")
            except:
                pass
    
    def _save_cache(self):
        """Save discovered relationships"""
        with open(self.cache_file, 'w') as f:
            json.dump({
                'relationships': self.relationships,
                'last_update': datetime.now().isoformat(),
                'n_tickers': len(self.data)
            }, f, indent=2)
    
    def download_data(self, tickers=None):
        """Download price data for all tickers"""
        tickers = tickers or TICKERS
        print(f"📥 Downloading {len(tickers)} tickers...")
        
        for t in tickers:
            try:
                df = yf.download(t, period='2y', progress=False)
                if len(df) > 100:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    self.data[t] = df
            except Exception as e:
                pass
        
        print(f"✅ {len(self.data)} tickers loaded")
        return self.data
    
    def discover_relationships(self, max_lag=5):
        """
        Find all lead-lag relationships in the data.
        A LEADS B if: A's move today predicts B's move in 1-5 days
        """
        print("\n🔍 DISCOVERING CROSS-ASSET RELATIONSHIPS")
        print("="*60)
        
        if not self.data:
            self.download_data()
        
        # Calculate returns for all tickers
        returns = {}
        for t, df in self.data.items():
            returns[t] = df['Close'].pct_change()
        
        # Find relationships
        all_relationships = {}
        
        for follower in self.data.keys():
            follower_rels = []
            follower_ret = returns[follower]
            
            for leader in self.data.keys():
                if leader == follower:
                    continue
                
                leader_ret = returns[leader]
                
                # Align data
                combined = pd.DataFrame({
                    'follower': follower_ret,
                    'leader': leader_ret
                }).dropna()
                
                if len(combined) < 100:
                    continue
                
                # Test different lags (leader moves BEFORE follower)
                for lag in range(1, max_lag + 1):
                    # Does leader[t-lag] predict follower[t]?
                    corr = combined['follower'].corr(combined['leader'].shift(lag))
                    
                    if abs(corr) >= self.min_corr:
                        # Calculate predictive accuracy
                        signals = combined['leader'].shift(lag)
                        correct = ((signals > 0) & (combined['follower'] > 0)) | \
                                  ((signals < 0) & (combined['follower'] < 0))
                        accuracy = correct.mean()
                        
                        if accuracy > 0.52:  # Better than random
                            follower_rels.append({
                                'leader': leader,
                                'lag': lag,
                                'correlation': round(corr, 3),
                                'direction': 'SAME' if corr > 0 else 'OPPOSITE',
                                'accuracy': round(accuracy, 3),
                                'strength': round(abs(corr) * accuracy, 3)
                            })
            
            # Keep top relationships for each follower
            if follower_rels:
                follower_rels.sort(key=lambda x: -x['strength'])
                all_relationships[follower] = follower_rels[:5]  # Top 5 leaders
        
        self.relationships = all_relationships
        self._save_cache()
        
        # Print discoveries
        print(f"\n📊 Found relationships for {len(all_relationships)} tickers")
        return all_relationships
    
    def print_relationships(self, top_n=20):
        """Print the strongest relationships discovered"""
        print("\n🔗 TOP CROSS-ASSET RELATIONSHIPS")
        print("="*70)
        print(f"{'FOLLOWER':<8} {'LEADER':<8} {'LAG':<4} {'CORR':<8} {'DIR':<8} {'ACC':<6}")
        print("-"*70)
        
        # Flatten and sort by strength
        all_rels = []
        for follower, rels in self.relationships.items():
            for r in rels:
                all_rels.append({**r, 'follower': follower})
        
        all_rels.sort(key=lambda x: -x['strength'])
        
        for r in all_rels[:top_n]:
            print(f"{r['follower']:<8} {r['leader']:<8} {r['lag']:<4} "
                  f"{r['correlation']:>+.3f}   {r['direction']:<8} {r['accuracy']:.1%}")
        
        return all_rels[:top_n]
    
    def get_signals_from_leaders(self):
        """
        Generate signals for TODAY based on what leaders did YESTERDAY.
        This is the money maker.
        """
        print("\n🎯 GENERATING SIGNALS FROM LEADER MOVES")
        print("="*60)
        
        if not self.data:
            self.download_data()
        
        signals = []
        
        for follower, rels in self.relationships.items():
            if follower not in self.data:
                continue
            
            follower_signals = []
            
            for rel in rels:
                leader = rel['leader']
                lag = rel['lag']
                direction = rel['direction']
                accuracy = rel['accuracy']
                
                if leader not in self.data:
                    continue
                
                leader_df = self.data[leader]
                
                # What did the leader do 'lag' days ago?
                if len(leader_df) < lag + 1:
                    continue
                
                # Leader's return 'lag' days ago
                leader_ret = (leader_df['Close'].iloc[-lag] / leader_df['Close'].iloc[-lag-1] - 1)
                
                # Predict follower direction
                if direction == 'SAME':
                    predicted_dir = 'UP' if leader_ret > 0 else 'DOWN'
                else:  # OPPOSITE
                    predicted_dir = 'DOWN' if leader_ret > 0 else 'UP'
                
                follower_signals.append({
                    'leader': leader,
                    'leader_move': leader_ret,
                    'predicted': predicted_dir,
                    'accuracy': accuracy,
                    'lag': lag
                })
            
            if follower_signals:
                # Combine signals (weighted vote)
                up_weight = sum(s['accuracy'] for s in follower_signals if s['predicted'] == 'UP')
                down_weight = sum(s['accuracy'] for s in follower_signals if s['predicted'] == 'DOWN')
                
                total_weight = up_weight + down_weight
                if total_weight > 0:
                    up_prob = up_weight / total_weight
                    consensus = 'BUY' if up_prob > 0.55 else ('SELL' if up_prob < 0.45 else 'HOLD')
                    confidence = abs(up_prob - 0.5) * 2  # 0-1 scale
                    
                    # Get current price info
                    follower_df = self.data[follower]
                    current_price = follower_df['Close'].iloc[-1]
                    
                    signals.append({
                        'ticker': follower,
                        'action': consensus,
                        'up_probability': round(up_prob, 3),
                        'confidence': round(confidence, 3),
                        'n_leaders': len(follower_signals),
                        'leaders': [s['leader'] for s in follower_signals[:3]],
                        'price': round(current_price, 2)
                    })
        
        # Sort by confidence
        signals.sort(key=lambda x: -x['confidence'])
        
        return signals
    
    def print_signals(self, signals=None):
        """Print today's signals"""
        if signals is None:
            signals = self.get_signals_from_leaders()
        
        print("\n" + "="*70)
        print("📊 TODAY'S CROSS-ASSET SIGNALS")
        print(f"   Based on what LEADERS did recently")
        print("="*70)
        
        buys = [s for s in signals if s['action'] == 'BUY']
        sells = [s for s in signals if s['action'] == 'SELL']
        
        if buys:
            print("\n🟢 BUY SIGNALS (Leaders say UP):")
            print("-"*60)
            for s in buys[:10]:
                leaders_str = ', '.join(s['leaders'])
                print(f"   {s['ticker']:6} | Conf: {s['confidence']*100:4.0f}% | "
                      f"P(up): {s['up_probability']:.0%} | Led by: {leaders_str}")
        
        if sells:
            print("\n🔴 SELL/AVOID SIGNALS (Leaders say DOWN):")
            print("-"*60)
            for s in sells[:10]:
                leaders_str = ', '.join(s['leaders'])
                print(f"   {s['ticker']:6} | Conf: {s['confidence']*100:4.0f}% | "
                      f"P(up): {s['up_probability']:.0%} | Led by: {leaders_str}")
        
        return signals
    
    def explain_ticker(self, ticker):
        """Explain what drives a specific ticker"""
        print(f"\n🔍 WHAT DRIVES {ticker}?")
        print("="*50)
        
        if ticker not in self.relationships:
            print(f"   No relationships found for {ticker}")
            return
        
        rels = self.relationships[ticker]
        
        print(f"   {ticker} tends to follow these leaders:\n")
        for r in rels:
            dir_word = "moves with" if r['direction'] == 'SAME' else "moves OPPOSITE to"
            print(f"   • {r['leader']}: {ticker} {dir_word} {r['leader']} "
                  f"({r['lag']} day lag, {r['accuracy']:.0%} accurate)")
        
        # Current signals for this ticker
        signals = self.get_signals_from_leaders()
        ticker_signal = next((s for s in signals if s['ticker'] == ticker), None)
        
        if ticker_signal:
            print(f"\n   📊 TODAY'S SIGNAL: {ticker_signal['action']}")
            print(f"      Confidence: {ticker_signal['confidence']*100:.0f}%")
            print(f"      P(up): {ticker_signal['up_probability']:.0%}")


def run_cross_asset_analysis():
    """Run full analysis and print signals"""
    brain = CrossAssetBrain()
    
    # Download fresh data
    brain.download_data()
    
    # Discover relationships (or use cached)
    today = datetime.now().strftime('%Y-%m-%d')
    if brain.last_update != today:
        print("\n🔄 Updating relationships (daily refresh)...")
        brain.discover_relationships()
    else:
        print("\n📂 Using cached relationships from today")
    
    # Print top relationships
    brain.print_relationships(top_n=15)
    
    # Generate and print signals
    signals = brain.print_signals()
    
    # Save signals
    with open('cross_asset_signals.json', 'w') as f:
        json.dump({
            'date': today,
            'signals': signals
        }, f, indent=2)
    
    print(f"\n✅ Saved {len(signals)} signals to cross_asset_signals.json")
    
    return brain, signals


if __name__ == "__main__":
    brain, signals = run_cross_asset_analysis()
    
    # Explain a few key tickers
    print("\n" + "="*70)
    print("🔍 DEEP DIVE ON YOUR FAVORITES")
    print("="*70)
    
    for ticker in ['APLD', 'SERV', 'NVDA', 'TSLA']:
        brain.explain_ticker(ticker)
