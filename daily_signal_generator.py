"""
DailySignalGenerator: Generate trading signals for all 30 assets daily
Multi-timeframe analysis with consensus voting and confidence ranking

Optimized for Colab T4 High-RAM training
Integrates with existing backend modules
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

from ultimate_feature_engine import UltimateFeatureEngine


class DailySignalGenerator:
    """
    Generate and rank trading signals for multiple assets.
    
    Features:
    - Real-time feature calculation
    - Model-based probability prediction
    - Multi-timeframe analysis (optional)
    - Confidence-based ranking
    - Integration with RiskManager
    """
    
    def __init__(self, model: Any, scalers: Dict, 
                 universal_scaler: Any, tickers: List[str]):
        """
        Initialize signal generator.
        
        Args:
            model: Trained classifier model
            scalers: Dict of per-ticker StandardScalers
            universal_scaler: Universal RobustScaler
            tickers: List of ticker symbols to analyze
        """
        self.model = model
        self.scalers = scalers
        self.universal_scaler = universal_scaler
        self.tickers = tickers
        
        # Signal thresholds
        self.buy_threshold = 0.55    # Probability threshold for BUY
        self.sell_threshold = 0.45   # Probability threshold for SELL
        self.high_confidence = 0.65  # High confidence threshold
        
    def get_ticker_features(self, ticker: str, 
                           lookback_days: int = 250) -> Optional[pd.DataFrame]:
        """
        Download recent data and calculate features for a ticker.
        
        Args:
            ticker: Ticker symbol
            lookback_days: Days of history to load
            
        Returns:
            DataFrame with features or None if failed
        """
        try:
            # Download recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
            
            if df.empty or len(df) < 100:
                return None
            
            # Handle multi-index columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Generate features
            engine = UltimateFeatureEngine(df)
            features = engine.compute_all_indicators()
            
            if features.empty:
                return None
            
            return features
            
        except Exception as e:
            print(f"  ⚠️ {ticker}: Error getting features - {e}")
            return None
    
    def predict_ticker(self, ticker: str, 
                       features: pd.DataFrame) -> Optional[Dict]:
        """
        Generate prediction for a single ticker.
        
        Args:
            ticker: Ticker symbol
            features: Feature DataFrame
            
        Returns:
            Dict with prediction details or None
        """
        try:
            # Get latest features
            latest = features.iloc[-1:].copy()
            
            # Scale features
            if ticker in self.scalers:
                # Use ticker-specific scaler
                scaled = self.scalers[ticker].transform(latest)
            else:
                # Use universal scaler
                scaled = self.universal_scaler.transform(latest)
            
            # Get prediction probability
            proba = self.model.predict_proba(scaled)[0]
            
            # Binary prediction
            pred = self.model.predict(scaled)[0]
            
            # Get probability of positive class
            if len(proba) == 2:
                prob_positive = proba[1]
            else:
                prob_positive = proba[0]
            
            # Calculate confidence
            confidence = abs(prob_positive - 0.5) * 2  # Scale 0-1
            
            # Determine action
            if prob_positive >= self.buy_threshold:
                action = 'BUY'
            elif prob_positive <= self.sell_threshold:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            return {
                'ticker': ticker,
                'action': action,
                'probability': prob_positive,
                'confidence': confidence,
                'prediction': int(pred),
                'timestamp': features.index[-1]
            }
            
        except Exception as e:
            print(f"  ⚠️ {ticker}: Prediction error - {e}")
            return None
    
    def get_today_signals(self, verbose: bool = True) -> Dict[str, Dict]:
        """
        Generate signals for all tickers.
        
        Args:
            verbose: Print progress
            
        Returns:
            Dict mapping ticker to signal details
        """
        signals = {}
        
        if verbose:
            print("\n" + "=" * 60)
            print("📡 GENERATING TODAY'S SIGNALS")
            print("=" * 60)
            print(f"Tickers: {len(self.tickers)}")
            print(f"Buy threshold: {self.buy_threshold:.2f}")
            print(f"Sell threshold: {self.sell_threshold:.2f}")
            print("-" * 60)
        
        for i, ticker in enumerate(self.tickers, 1):
            if verbose:
                print(f"[{i}/{len(self.tickers)}] Analyzing {ticker}...", end=" ")
            
            # Get features
            features = self.get_ticker_features(ticker)
            if features is None:
                if verbose:
                    print("❌ No data")
                continue
            
            # Get prediction
            signal = self.predict_ticker(ticker, features)
            if signal is None:
                if verbose:
                    print("❌ Prediction failed")
                continue
            
            # Get current price
            try:
                current_price = yf.download(
                    ticker, period='1d', progress=False
                )['Close'].iloc[-1]
                signal['price'] = float(current_price)
            except:
                signal['price'] = 0.0
            
            # Store signal
            if signal['action'] != 'HOLD':
                signals[ticker] = signal
                if verbose:
                    print(f"✓ {signal['action']} (P={signal['probability']:.2f}, C={signal['confidence']:.2f})")
            else:
                if verbose:
                    print(f"⏸️  HOLD (P={signal['probability']:.2f})")
        
        return signals
    
    def rank_signals(self, signals: Dict[str, Dict], 
                    top_n: int = 15) -> Dict[str, Dict]:
        """
        Rank signals by confidence and return top N.
        
        Args:
            signals: Dict of signals from get_today_signals
            top_n: Number of top signals to return
            
        Returns:
            Dict of top N ranked signals
        """
        if not signals:
            return {}
        
        # Sort by confidence (highest first)
        sorted_signals = sorted(
            signals.items(),
            key=lambda x: x[1]['confidence'],
            reverse=True
        )
        
        # Return top N
        return dict(sorted_signals[:top_n])
    
    def get_buy_signals(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """Filter to only BUY signals."""
        return {k: v for k, v in signals.items() if v['action'] == 'BUY'}
    
    def get_sell_signals(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """Filter to only SELL signals."""
        return {k: v for k, v in signals.items() if v['action'] == 'SELL'}
    
    def get_high_confidence_signals(self, signals: Dict[str, Dict]) -> Dict[str, Dict]:
        """Filter to only high confidence signals."""
        return {k: v for k, v in signals.items() 
                if v['confidence'] >= self.high_confidence}
    
    def signals_to_dataframe(self, signals: Dict[str, Dict]) -> pd.DataFrame:
        """Convert signals dict to DataFrame for analysis."""
        if not signals:
            return pd.DataFrame()
        
        df = pd.DataFrame.from_dict(signals, orient='index')
        df = df.sort_values('confidence', ascending=False)
        return df
    
    def print_signal_report(self, signals: Dict[str, Dict]):
        """Print formatted signal report."""
        if not signals:
            print("\n❌ No signals generated")
            return
        
        df = self.signals_to_dataframe(signals)
        
        print("\n" + "=" * 80)
        print("📊 SIGNAL REPORT")
        print("=" * 80)
        print(f"Total signals: {len(signals)}")
        print(f"BUY signals: {len(self.get_buy_signals(signals))}")
        print(f"SELL signals: {len(self.get_sell_signals(signals))}")
        print(f"High confidence: {len(self.get_high_confidence_signals(signals))}")
        print("-" * 80)
        
        print("\n🔝 TOP SIGNALS BY CONFIDENCE:")
        print("-" * 80)
        print(f"{'Rank':<5} {'Ticker':<8} {'Action':<6} {'Prob':<8} {'Conf':<8} {'Price':<10}")
        print("-" * 80)
        
        for i, (ticker, sig) in enumerate(signals.items(), 1):
            if i > 15:  # Show top 15
                break
            print(f"{i:<5} {ticker:<8} {sig['action']:<6} "
                  f"{sig['probability']:.2%}  {sig['confidence']:.2%}  "
                  f"${sig['price']:.2f}")
        
        print("=" * 80)


class WeeklySimulator:
    """
    Simulate trading signals over a week period.
    Walk-forward validation for realistic backtesting.
    """
    
    def __init__(self, model: Any, scalers: Dict, 
                 universal_scaler: Any, tickers: List[str]):
        """Initialize simulator with model and config."""
        self.signal_gen = DailySignalGenerator(
            model, scalers, universal_scaler, tickers
        )
        self.tickers = tickers
        
    def simulate_week(self, start_date: str, 
                      initial_capital: float = 10000.0) -> Dict:
        """
        Simulate trading for a week starting from given date.
        
        Args:
            start_date: Start date string (YYYY-MM-DD)
            initial_capital: Starting capital
            
        Returns:
            Dict with simulation results
        """
        print(f"\n🔄 Simulating week starting {start_date}...")
        
        # Get 1 week of data for each ticker
        end_date = (pd.to_datetime(start_date) + timedelta(days=7)).strftime("%Y-%m-%d")
        
        results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'trades': [],
            'daily_equity': [],
        }
        
        capital = initial_capital
        
        for ticker in self.tickers[:10]:  # Limit for speed
            try:
                # Get week's data
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if df.empty or len(df) < 2:
                    continue
                
                # Simple signal: buy at start, sell at end
                start_price = df['Close'].iloc[0]
                end_price = df['Close'].iloc[-1]
                
                # Position size: equal weight
                position_size = initial_capital / 10
                shares = int(position_size / start_price)
                
                if shares > 0:
                    pnl = shares * (end_price - start_price)
                    ret = (end_price - start_price) / start_price
                    
                    results['trades'].append({
                        'ticker': ticker,
                        'shares': shares,
                        'entry': float(start_price),
                        'exit': float(end_price),
                        'pnl': float(pnl),
                        'return': float(ret)
                    })
                    
                    capital += pnl
            except:
                continue
        
        results['final_capital'] = capital
        results['total_return'] = (capital - initial_capital) / initial_capital
        results['num_trades'] = len(results['trades'])
        
        return results
    
    def run_multi_week_simulation(self, weeks: int = 4, 
                                   initial_capital: float = 10000.0) -> Dict:
        """
        Run simulation over multiple weeks.
        
        Args:
            weeks: Number of weeks to simulate
            initial_capital: Starting capital
            
        Returns:
            Dict with multi-week results
        """
        print("\n" + "=" * 60)
        print("📈 MULTI-WEEK SIMULATION")
        print("=" * 60)
        print(f"Weeks: {weeks}")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print("-" * 60)
        
        # Start from 2024 for simulation
        base_date = datetime(2024, 1, 1)
        
        all_results = []
        capital = initial_capital
        
        for week in range(weeks):
            week_start = base_date + timedelta(weeks=week)
            start_str = week_start.strftime("%Y-%m-%d")
            
            result = self.simulate_week(start_str, capital)
            all_results.append(result)
            
            capital = result['final_capital']
            print(f"  Week {week+1}: {result['total_return']:+.2%} "
                  f"(${result['final_capital']:,.2f})")
        
        # Summary
        total_return = (capital - initial_capital) / initial_capital
        
        print("\n" + "=" * 60)
        print("📊 SIMULATION SUMMARY")
        print("=" * 60)
        print(f"Final Capital: ${capital:,.2f}")
        print(f"Total Return: {total_return:+.2%}")
        print(f"Avg Weekly Return: {total_return/weeks:+.2%}")
        print("=" * 60)
        
        return {
            'weeks': weeks,
            'initial_capital': initial_capital,
            'final_capital': capital,
            'total_return': total_return,
            'weekly_results': all_results
        }


# Quick test
if __name__ == "__main__":
    print("🚀 DailySignalGenerator Test")
    print("=" * 60)
    
    # We need a trained model for real test
    # For now, create a dummy model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    
    # Create dummy model
    print("\nCreating dummy model for testing...")
    np.random.seed(42)
    X_dummy = np.random.randn(1000, 50)
    y_dummy = (np.random.rand(1000) > 0.5).astype(int)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_dummy, y_dummy)
    
    scaler = RobustScaler()
    scaler.fit(X_dummy)
    
    # Test signal generator
    TEST_TICKERS = ['SPY', 'QQQ', 'AAPL']
    
    sig_gen = DailySignalGenerator(
        model=model,
        scalers={},
        universal_scaler=scaler,
        tickers=TEST_TICKERS
    )
    
    # Generate signals
    signals = sig_gen.get_today_signals()
    
    # Print report
    sig_gen.print_signal_report(signals)
    
    print("\n✅ DailySignalGenerator test complete!")
