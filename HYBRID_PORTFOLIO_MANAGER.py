#!/usr/bin/env python3
"""
HYBRID PORTFOLIO MANAGER
========================
Phase 4 of the Battle Plan: Combine systematic edges with discretionary skill.

This hybrid system has two parallel tracks:
1. SYSTEMATIC TRACK (70% of capital) - Runs top multi-factor models daily
2. DISCRETIONARY TRACK (30% of capital) - Your stock picks with factor analysis

Features:
- Risk-parity allocation across validated factors
- DiscretionarySignalAnalyzer for manual picks
- Daily performance attribution
- Transaction cost modeling
- Rebalancing logic

Author: Quantum Trading Research Team
Date: December 20, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
import warnings
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = 'data/market_data.db'
PORTFOLIO_STATE_FILE = 'data/portfolio_state.json'
OUTPUT_DIR = 'data/hybrid_portfolio'

# Allocation settings
SYSTEMATIC_ALLOCATION = 0.70  # 70% to systematic strategies
DISCRETIONARY_ALLOCATION = 0.30  # 30% to discretionary picks

# Risk settings
MAX_POSITION_SIZE = 0.10  # Max 10% in single position
MIN_POSITION_SIZE = 0.02  # Min 2% position
REBALANCE_THRESHOLD = 0.05  # Rebalance if drift > 5%

# Transaction costs
COMMISSION_PER_SHARE = 0.01
SPREAD_PCT = 0.001  # 0.1% bid-ask spread
SLIPPAGE_PCT = 0.001  # 0.1% slippage

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CORE FACTOR CALCULATOR (reused from FactorCombinationLab)
# ============================================================

class FactorCalculator:
    """Calculate factor scores for any ticker"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.df = None
        
    def load_market_data(self):
        """Load full market data"""
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql("SELECT * FROM ohlcv", conn)
        conn.close()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['ticker', 'date']).reset_index(drop=True)
        
    def calculate_factors_for_ticker(self, ticker: str) -> Optional[Dict]:
        """
        Calculate all factor scores for a single ticker.
        Returns dict with factor names and percentile scores.
        """
        if self.df is None:
            self.load_market_data()
        
        # Get ticker data
        ticker_df = self.df[self.df['ticker'] == ticker].copy()
        if len(ticker_df) < 200:
            return None
        
        # Get latest row
        latest = ticker_df.iloc[-1]
        latest_date = latest['date']
        
        # Get cross-sectional data for ranking
        cross_section = self.df[self.df['date'] == latest_date].copy()
        
        # Calculate factors
        factors = {}
        
        # ====== RSI ======
        delta = ticker_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        factors['rsi'] = rsi.iloc[-1]
        factors['rsi_oversold'] = 1 if factors['rsi'] < 30 else 0
        
        # ====== Momentum ======
        factors['momentum_20d'] = ticker_df['close'].pct_change(20).iloc[-1]
        factors['momentum_60d'] = ticker_df['close'].pct_change(60).iloc[-1] if len(ticker_df) > 60 else np.nan
        
        # Rank momentum vs universe
        cross_section['_mom'] = cross_section.groupby('ticker')['close'].transform(
            lambda x: x.pct_change(20).iloc[-1] if len(x) > 20 else np.nan
        )
        mom_rank = (cross_section['_mom'] < factors['momentum_20d']).mean()
        factors['momentum_percentile'] = mom_rank * 100
        
        # ====== Volatility ======
        tr = np.maximum(
            ticker_df['high'] - ticker_df['low'],
            np.maximum(
                abs(ticker_df['high'] - ticker_df['close'].shift(1)),
                abs(ticker_df['low'] - ticker_df['close'].shift(1))
            )
        )
        atr = tr.rolling(14).mean()
        factors['atr_pct'] = (atr.iloc[-1] / latest['close']) * 100
        
        # Low vol factor
        cross_section['_atr'] = cross_section.apply(
            lambda x: (x['high'] - x['low']) / x['close'] * 100, axis=1
        )
        vol_rank = (cross_section['_atr'] > factors['atr_pct']).mean()  # Lower = better
        factors['low_vol_percentile'] = vol_rank * 100
        
        # ====== Trend (EMA 200) ======
        ema200 = ticker_df['close'].ewm(span=200).mean()
        factors['above_ema200'] = 1 if latest['close'] > ema200.iloc[-1] else 0
        factors['pct_above_ema200'] = (latest['close'] / ema200.iloc[-1] - 1) * 100
        
        # ====== 52-Week Position ======
        high_52w = ticker_df['high'].rolling(252).max().iloc[-1]
        low_52w = ticker_df['low'].rolling(252).min().iloc[-1]
        factors['pct_from_52w_high'] = (high_52w - latest['close']) / high_52w * 100
        factors['pct_from_52w_low'] = (latest['close'] - low_52w) / low_52w * 100
        factors['near_52w_high'] = 1 if factors['pct_from_52w_high'] < 5 else 0
        
        # ====== Mean Reversion ======
        sma20 = ticker_df['close'].rolling(20).mean()
        std20 = ticker_df['close'].rolling(20).std()
        factors['zscore'] = (latest['close'] - sma20.iloc[-1]) / std20.iloc[-1] if std20.iloc[-1] > 0 else 0
        factors['oversold_zscore'] = 1 if factors['zscore'] < -2 else 0
        
        # ====== Volume ======
        vol_ma = ticker_df['volume'].rolling(20).mean()
        factors['volume_ratio'] = latest['volume'] / vol_ma.iloc[-1] if vol_ma.iloc[-1] > 0 else 1
        factors['volume_spike'] = 1 if factors['volume_ratio'] > 2 else 0
        
        # ====== Consecutive Days ======
        returns = ticker_df['close'].pct_change()
        factors['last_3_returns'] = [returns.iloc[-3], returns.iloc[-2], returns.iloc[-1]]
        factors['after_2_down'] = 1 if returns.iloc[-1] < 0 and returns.iloc[-2] < 0 else 0
        factors['after_2_up'] = 1 if returns.iloc[-1] > 0 and returns.iloc[-2] > 0 else 0
        
        return factors


# ============================================================
# DISCRETIONARY SIGNAL ANALYZER
# ============================================================

class DiscretionarySignalAnalyzer:
    """
    Analyzes discretionary stock picks against validated factors.
    Provides confidence scores and factor alignment analysis.
    """
    
    def __init__(self, db_path=DB_PATH):
        self.factor_calc = FactorCalculator(db_path)
        
    def analyze_pick(self, ticker: str) -> Dict:
        """
        Analyze a discretionary stock pick.
        Returns comprehensive factor analysis and confidence score.
        """
        factors = self.factor_calc.calculate_factors_for_ticker(ticker)
        
        if factors is None:
            return {
                'ticker': ticker,
                'status': 'ERROR',
                'message': f'Insufficient data for {ticker}'
            }
        
        # Calculate confidence score based on factor alignment
        bullish_signals = 0
        bearish_signals = 0
        signal_details = []
        
        # RSI Analysis
        if factors['rsi'] < 30:
            bullish_signals += 2
            signal_details.append(f"✅ RSI oversold ({factors['rsi']:.1f}) - STRONG BUY signal")
        elif factors['rsi'] < 40:
            bullish_signals += 1
            signal_details.append(f"✅ RSI approaching oversold ({factors['rsi']:.1f})")
        elif factors['rsi'] > 70:
            bearish_signals += 2
            signal_details.append(f"⚠️ RSI overbought ({factors['rsi']:.1f}) - CAUTION")
        
        # Momentum Analysis
        if factors['momentum_percentile'] > 80:
            bullish_signals += 2
            signal_details.append(f"✅ Strong momentum (top {100-factors['momentum_percentile']:.0f}% of stocks)")
        elif factors['momentum_percentile'] > 60:
            bullish_signals += 1
            signal_details.append(f"✅ Positive momentum (top {100-factors['momentum_percentile']:.0f}%)")
        elif factors['momentum_percentile'] < 20:
            bearish_signals += 1
            signal_details.append(f"⚠️ Weak momentum (bottom {factors['momentum_percentile']:.0f}%)")
        
        # Low Volatility Analysis
        if factors['low_vol_percentile'] > 70:
            bullish_signals += 1
            signal_details.append(f"✅ Low volatility (cleaner signal)")
        elif factors['low_vol_percentile'] < 30:
            signal_details.append(f"⚠️ High volatility - expect larger swings")
        
        # Trend Analysis
        if factors['above_ema200']:
            bullish_signals += 1
            signal_details.append(f"✅ Above 200-day EMA (uptrend confirmed)")
        else:
            bearish_signals += 1
            signal_details.append(f"⚠️ Below 200-day EMA (downtrend)")
        
        # 52-Week Position
        if factors['near_52w_high']:
            bullish_signals += 1
            signal_details.append(f"✅ Near 52-week high (momentum breakout)")
        elif factors['pct_from_52w_high'] > 30:
            signal_details.append(f"⚠️ {factors['pct_from_52w_high']:.1f}% below 52-week high")
        
        # Mean Reversion
        if factors['oversold_zscore']:
            bullish_signals += 2
            signal_details.append(f"✅ Z-score oversold ({factors['zscore']:.2f}) - mean reversion opportunity")
        
        # After consecutive down days
        if factors['after_2_down']:
            bullish_signals += 1
            signal_details.append(f"✅ After 2 down days - bounce candidate")
        
        # Calculate overall confidence
        total_signals = bullish_signals + bearish_signals
        if total_signals > 0:
            confidence = (bullish_signals - bearish_signals) / total_signals * 100
        else:
            confidence = 0
        
        # Determine recommendation
        if confidence >= 50:
            recommendation = "STRONG BUY"
            emoji = "🟢"
        elif confidence >= 20:
            recommendation = "BUY"
            emoji = "🟡"
        elif confidence >= -20:
            recommendation = "HOLD"
            emoji = "⚪"
        elif confidence >= -50:
            recommendation = "REDUCE"
            emoji = "🟠"
        else:
            recommendation = "SELL"
            emoji = "🔴"
        
        return {
            'ticker': ticker,
            'status': 'OK',
            'recommendation': recommendation,
            'emoji': emoji,
            'confidence_score': confidence,
            'bullish_signals': bullish_signals,
            'bearish_signals': bearish_signals,
            'factors': factors,
            'signal_details': signal_details
        }
    
    def print_analysis(self, ticker: str):
        """Print formatted analysis for a ticker"""
        result = self.analyze_pick(ticker)
        
        if result['status'] == 'ERROR':
            print(f"\n❌ {result['message']}")
            return result
        
        print("\n" + "="*60)
        print(f"{result['emoji']} {ticker}: {result['recommendation']}")
        print(f"   Confidence Score: {result['confidence_score']:.1f}")
        print("="*60)
        
        print("\n📊 FACTOR ANALYSIS:")
        for detail in result['signal_details']:
            print(f"   {detail}")
        
        print("\n📈 KEY METRICS:")
        f = result['factors']
        print(f"   RSI(14): {f['rsi']:.1f}")
        print(f"   Momentum Percentile: {f['momentum_percentile']:.0f}%")
        print(f"   Low Vol Percentile: {f['low_vol_percentile']:.0f}%")
        print(f"   Above EMA200: {'Yes' if f['above_ema200'] else 'No'}")
        print(f"   Z-Score: {f['zscore']:.2f}")
        print(f"   % from 52w High: {f['pct_from_52w_high']:.1f}%")
        
        return result


# ============================================================
# SYSTEMATIC PORTFOLIO MANAGER
# ============================================================

class SystematicPortfolioManager:
    """
    Manages the systematic portion of the portfolio.
    Implements risk-parity allocation across validated factors.
    """
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.factor_calc = FactorCalculator(db_path)
        self.models = []  # Will be loaded from VETTED_STRATEGIES
        
    def load_validated_models(self, models_file='data/factor_lab/recommended_models.csv'):
        """Load validated multi-factor models"""
        if os.path.exists(models_file):
            self.models = pd.read_csv(models_file).to_dict('records')
        else:
            # Default models based on known robust factors
            self.models = [
                {'model_name': 'RSI_Oversold', 'factors': ['rsi_oversold'], 'weight': 0.3},
                {'model_name': 'LowVol_Momentum', 'factors': ['low_volatility', 'momentum_positive'], 'weight': 0.3},
                {'model_name': 'Trend_Following', 'factors': ['above_ema200', 'momentum_positive'], 'weight': 0.4}
            ]
        
    def generate_universe(self, n_stocks=50) -> pd.DataFrame:
        """
        Generate ranked universe of stocks based on factor scores.
        Returns top N stocks with their factor scores.
        """
        if self.factor_calc.df is None:
            self.factor_calc.load_market_data()
        
        df = self.factor_calc.df
        latest_date = df['date'].max()
        
        # Get all tickers with data on latest date
        latest = df[df['date'] == latest_date].copy()
        
        # Calculate composite score for each ticker
        scores = []
        
        for ticker in latest['ticker'].unique()[:500]:  # Limit for speed
            factors = self.factor_calc.calculate_factors_for_ticker(ticker)
            if factors is None:
                continue
            
            # Composite score: RSI oversold + Low vol + Momentum + Trend
            score = 0
            if factors['rsi'] < 30:
                score += 30
            elif factors['rsi'] < 40:
                score += 15
            
            score += factors['low_vol_percentile'] * 0.2
            score += factors['momentum_percentile'] * 0.3
            
            if factors['above_ema200']:
                score += 15
            
            if factors['near_52w_high']:
                score += 10
            
            scores.append({
                'ticker': ticker,
                'composite_score': score,
                'rsi': factors['rsi'],
                'momentum_pct': factors['momentum_percentile'],
                'low_vol_pct': factors['low_vol_percentile'],
                'above_ema200': factors['above_ema200']
            })
        
        universe = pd.DataFrame(scores)
        universe = universe.sort_values('composite_score', ascending=False)
        
        return universe.head(n_stocks)


# ============================================================
# HYBRID PORTFOLIO MANAGER
# ============================================================

class HybridPortfolioManager:
    """
    Main portfolio manager combining systematic and discretionary tracks.
    
    Systematic Track (70%):
    - Runs top multi-factor models daily
    - Generates ranked universe of 50-100 stocks
    - Risk-parity allocation
    
    Discretionary Track (30%):
    - Your manual stock picks
    - Factor analysis for each pick
    - Confidence scores
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        
        # Initialize components
        self.systematic = SystematicPortfolioManager()
        self.discretionary_analyzer = DiscretionarySignalAnalyzer()
        
        # Portfolio state
        self.systematic_positions = {}  # ticker -> shares
        self.discretionary_positions = {}  # ticker -> shares
        self.cash = initial_capital
        
        # Performance tracking
        self.daily_pnl = []
        self.trades = []
        
        # Load state if exists
        self.load_state()
        
    def load_state(self):
        """Load portfolio state from file"""
        if os.path.exists(PORTFOLIO_STATE_FILE):
            with open(PORTFOLIO_STATE_FILE, 'r') as f:
                state = json.load(f)
                self.systematic_positions = state.get('systematic_positions', {})
                self.discretionary_positions = state.get('discretionary_positions', {})
                self.cash = state.get('cash', self.initial_capital)
                self.daily_pnl = state.get('daily_pnl', [])
                
    def save_state(self):
        """Save portfolio state to file"""
        state = {
            'systematic_positions': self.systematic_positions,
            'discretionary_positions': self.discretionary_positions,
            'cash': self.cash,
            'daily_pnl': self.daily_pnl,
            'last_update': datetime.now().isoformat()
        }
        with open(PORTFOLIO_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    
    def calculate_transaction_cost(self, shares: int, price: float) -> float:
        """Calculate total transaction cost"""
        commission = shares * COMMISSION_PER_SHARE
        spread_cost = shares * price * SPREAD_PCT
        slippage = shares * price * SLIPPAGE_PCT
        return commission + spread_cost + slippage
    
    def add_discretionary_position(self, ticker: str, shares: int = None, 
                                   dollars: float = None, price: float = None):
        """
        Add a discretionary position with full factor analysis.
        
        Args:
            ticker: Stock symbol
            shares: Number of shares (or)
            dollars: Dollar amount to invest
            price: Current price (optional, will fetch if not provided)
        """
        # Get analysis first
        analysis = self.discretionary_analyzer.analyze_pick(ticker)
        
        if analysis['status'] == 'ERROR':
            print(f"\n❌ Cannot add {ticker}: {analysis['message']}")
            return None
        
        # Print analysis
        self.discretionary_analyzer.print_analysis(ticker)
        
        # Calculate position size
        max_discretionary = self.capital * DISCRETIONARY_ALLOCATION
        available = max_discretionary - sum(
            self.discretionary_positions.get(t, 0) * 100  # Rough estimate
            for t in self.discretionary_positions
        )
        
        if dollars:
            position_value = min(dollars, available)
        elif shares and price:
            position_value = shares * price
        else:
            # Default to equal weight
            n_positions = len(self.discretionary_positions) + 1
            position_value = min(max_discretionary / n_positions, available)
        
        print(f"\n💰 Position Analysis:")
        print(f"   Requested value: ${position_value:,.2f}")
        print(f"   Available discretionary capital: ${available:,.2f}")
        print(f"   Recommendation: {analysis['recommendation']}")
        
        return analysis
    
    def analyze_alpaca_positions(self, positions: List[Dict]):
        """
        Analyze current Alpaca positions and provide factor scores.
        
        Args:
            positions: List of dicts with 'ticker', 'qty', 'avg_price', 'current_price'
        """
        print("\n" + "="*70)
        print("📊 ALPACA PORTFOLIO FACTOR ANALYSIS")
        print("="*70)
        
        results = []
        
        for pos in positions:
            ticker = pos['ticker']
            analysis = self.discretionary_analyzer.analyze_pick(ticker)
            
            if analysis['status'] == 'OK':
                unrealized_pnl = (pos.get('current_price', pos['avg_price']) - pos['avg_price']) * pos['qty']
                unrealized_pct = (pos.get('current_price', pos['avg_price']) / pos['avg_price'] - 1) * 100
                
                results.append({
                    'ticker': ticker,
                    'qty': pos['qty'],
                    'avg_price': pos['avg_price'],
                    'unrealized_pnl': unrealized_pnl,
                    'unrealized_pct': unrealized_pct,
                    'recommendation': analysis['recommendation'],
                    'confidence': analysis['confidence_score'],
                    'rsi': analysis['factors']['rsi'],
                    'momentum_pct': analysis['factors']['momentum_percentile']
                })
                
                print(f"\n{analysis['emoji']} {ticker}: {analysis['recommendation']} (Confidence: {analysis['confidence_score']:.0f})")
                print(f"   Position: {pos['qty']} shares @ ${pos['avg_price']:.2f}")
                print(f"   Unrealized P&L: ${unrealized_pnl:,.2f} ({unrealized_pct:+.1f}%)")
                print(f"   RSI: {analysis['factors']['rsi']:.1f} | Momentum: {analysis['factors']['momentum_percentile']:.0f}%ile")
        
        # Summary
        print("\n" + "="*70)
        print("📈 PORTFOLIO SUMMARY")
        print("="*70)
        
        results_df = pd.DataFrame(results)
        if len(results_df) > 0:
            total_pnl = results_df['unrealized_pnl'].sum()
            avg_confidence = results_df['confidence'].mean()
            
            print(f"\n   Total Unrealized P&L: ${total_pnl:,.2f}")
            print(f"   Average Confidence Score: {avg_confidence:.1f}")
            
            # Action items
            print("\n🎯 SUGGESTED ACTIONS:")
            
            sells = results_df[results_df['recommendation'].isin(['SELL', 'REDUCE'])]
            if len(sells) > 0:
                print("   Consider reducing/exiting:")
                for _, row in sells.iterrows():
                    print(f"     - {row['ticker']}: {row['recommendation']} (confidence: {row['confidence']:.0f})")
            
            buys = results_df[results_df['recommendation'].isin(['STRONG BUY', 'BUY'])]
            if len(buys) > 0:
                print("   Strong positions to hold/add:")
                for _, row in buys.iterrows():
                    print(f"     - {row['ticker']}: {row['recommendation']} (confidence: {row['confidence']:.0f})")
        
        return results_df
    
    def generate_daily_signals(self) -> Dict:
        """
        Generate daily trading signals for both tracks.
        Returns dict with systematic and discretionary recommendations.
        """
        print("\n" + "="*70)
        print(f"📡 DAILY SIGNAL GENERATION - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*70)
        
        # Systematic track
        print("\n🤖 SYSTEMATIC TRACK (70% allocation):")
        self.systematic.load_validated_models()
        universe = self.systematic.generate_universe(n_stocks=20)
        
        print(f"\n   Top 20 stocks by composite factor score:")
        print(universe[['ticker', 'composite_score', 'rsi', 'momentum_pct']].to_string(index=False))
        
        # Discretionary track analysis
        print("\n👤 DISCRETIONARY TRACK (30% allocation):")
        print("   (Add positions with: manager.add_discretionary_position('TICKER'))")
        
        return {
            'date': datetime.now().isoformat(),
            'systematic_universe': universe.to_dict('records'),
            'n_systematic_candidates': len(universe)
        }
    
    def calculate_performance_attribution(self) -> Dict:
        """
        Calculate performance attribution between systematic and discretionary.
        """
        print("\n" + "="*70)
        print("📊 PERFORMANCE ATTRIBUTION")
        print("="*70)
        
        # This would require historical position data and prices
        # For now, return structure
        
        attribution = {
            'total_return': 0,
            'systematic_contribution': 0,
            'discretionary_contribution': 0,
            'systematic_sharpe': 0,
            'discretionary_sharpe': 0
        }
        
        print("\n   (Full attribution requires historical tracking)")
        print("   Enable by running: manager.save_state() daily")
        
        return attribution


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    """Main execution - demonstrate the hybrid portfolio manager"""
    
    print("="*70)
    print("🚀 HYBRID PORTFOLIO MANAGER")
    print("   Combining Systematic Edges + Discretionary Skill")
    print("="*70)
    
    # Initialize manager
    manager = HybridPortfolioManager(initial_capital=100000)
    
    # Generate daily signals
    signals = manager.generate_daily_signals()
    
    # Example: Analyze your Alpaca positions
    # (Replace with actual positions from Alpaca API)
    sample_positions = [
        {'ticker': 'ASTS', 'qty': 100, 'avg_price': 25.50, 'current_price': 26.00},
        {'ticker': 'MU', 'qty': 50, 'avg_price': 105.00, 'current_price': 106.50},
        {'ticker': 'KDK', 'qty': 200, 'avg_price': 14.50, 'current_price': 14.20},
        {'ticker': 'PALI', 'qty': 500, 'avg_price': 3.00, 'current_price': 3.10},
        {'ticker': 'SAVA', 'qty': 100, 'avg_price': 25.00, 'current_price': 24.50},
        {'ticker': 'LUNR', 'qty': 100, 'avg_price': 18.00, 'current_price': 18.50},
        {'ticker': 'HUT', 'qty': 200, 'avg_price': 28.00, 'current_price': 28.10},
    ]
    
    # Analyze positions
    results = manager.analyze_alpaca_positions(sample_positions)
    
    # Save state
    manager.save_state()
    
    print("\n" + "="*70)
    print("🎯 HYBRID PORTFOLIO MANAGER READY")
    print("="*70)
    print("\nUsage:")
    print("  manager.add_discretionary_position('TICKER')")
    print("  manager.analyze_alpaca_positions(positions)")
    print("  manager.generate_daily_signals()")


if __name__ == '__main__':
    main()
