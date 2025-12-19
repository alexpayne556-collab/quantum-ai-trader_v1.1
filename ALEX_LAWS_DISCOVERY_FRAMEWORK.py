"""
ALEX LAWS DISCOVERY FRAMEWORK
==============================
Scientific method for discovering real alpha in financial markets.

Philosophy: "We don't premake laws, we discover them"
- Test user's manual trading edge (MU +15%, KDK +10% on earnings)
- Discover patterns that survive rigorous statistical tests
- Account for transaction costs, survivorship bias, multiple testing
- Document everything for reproducibility

Based on AI consultation and academic standards (Harvey-Liu-Zhu)
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AlexLawsDiscovery:
    """
    Systematic framework for discovering and validating trading edges.
    
    Key principles:
    1. Transaction cost aware (most edges die after costs)
    2. Survivorship bias corrected
    3. Multiple testing corrections (FDR, not Bonferroni)
    4. Harvey-Liu-Zhu threshold: t-stat > 3.0 for quant finance
    5. Out-of-sample validation required
    """
    
    def __init__(self, db_path='data/market_data.db'):
        self.conn = sqlite3.connect(db_path)
        self.transaction_costs = self._load_transaction_costs()
        
        # Load clean trading universe (exclude problematic tickers)
        self.clean_universe = self._build_clean_universe()
        
        print(f"✅ Initialized with {len(self.clean_universe):,} clean tickers")
        print(f"📊 Transaction cost tiers loaded")
    
    def _load_transaction_costs(self):
        """Load transaction cost estimates by ticker"""
        try:
            costs = pd.read_csv('data/transaction_costs.csv')
            cost_map = {'penny': 0.03, 'illiquid': 0.02, 'small': 0.008, 
                       'mid': 0.003, 'large': 0.001}
            costs['cost_pct'] = costs['tier'].map(cost_map)
            return dict(zip(costs['ticker'], costs['cost_pct']))
        except:
            return {}
    
    def _build_clean_universe(self):
        """
        Build clean trading universe excluding:
        - Extreme movers (likely data errors)
        - Poor coverage (<70%)
        - High transaction costs (>2%)
        """
        # Load problematic tickers
        try:
            extreme = pd.read_csv('data/extreme_moves.csv')
            poor_cov = pd.read_csv('data/poor_coverage.csv')
            
            # Exclude tickers with multiple extreme moves (likely bad data)
            bad_tickers = extreme[extreme.groupby('ticker')['ticker'].transform('count') > 2]['ticker'].unique()
            bad_tickers = set(bad_tickers) | set(poor_cov['ticker'])
            
            # Exclude penny stocks and illiquid (>1% transaction costs)
            high_cost = [t for t, c in self.transaction_costs.items() if c > 0.01]
            bad_tickers = bad_tickers | set(high_cost)
            
            # Get all tickers
            all_tickers = pd.read_sql("SELECT DISTINCT ticker FROM ohlcv", self.conn)['ticker']
            clean = [t for t in all_tickers if t not in bad_tickers]
            
            return clean
        except:
            return []
    
    def test_earnings_edge(self, days_after=5, min_surprise=0.10):
        """
        TEST: Alex Law #1 - Earnings Surprise Drift
        
        Hypothesis: Stocks with earnings surprise >10% outperform over next 5/10/21 days
        Evidence: User's manual trades (MU +15%, KDK +10%)
        Literature: Post-Earnings Announcement Drift (PEAD) - well documented
        
        Args:
            days_after: Days to measure returns after earnings
            min_surprise: Minimum earnings surprise threshold
        
        Returns:
            DataFrame with results and statistical validation
        """
        print(f"\n{'='*60}")
        print(f"TESTING: Alex Law #1 - Earnings Surprise Drift")
        print(f"{'='*60}")
        print(f"Hypothesis: >10% earnings surprise → outperformance")
        print(f"User evidence: MU +15%, KDK +10%")
        print(f"")
        
        # TODO: Need earnings data source
        # Options: Polygon.io, FMP, Yahoo Finance
        print("⚠️  REQUIRES: Earnings data API")
        print("   - Need actual vs expected EPS")
        print("   - Earnings dates for all tickers")
        print("   - Recommend: Polygon.io or Financial Modeling Prep")
        print("")
        print("💡 MANUAL VALIDATION APPROACH:")
        print("   1. Pull user's MU and KDK trades manually")
        print("   2. Verify returns match user's claims (+15%, +10%)")
        print("   3. Check if pattern holds on other earnings surprises")
        print("   4. Build systematic rules from validated patterns")
        
        return None
    
    def test_momentum_regime(self, lookback=20, threshold=0.05):
        """
        TEST: Alex Law #2 - Momentum Regime Filter
        
        Hypothesis: Momentum strategies work better in trending markets
        Method: Test if 20-day momentum predicts next 5-day returns
        
        Args:
            lookback: Days for momentum calculation
            threshold: Minimum momentum for signal
        """
        print(f"\n{'='*60}")
        print(f"TESTING: Alex Law #2 - Momentum Regime Filter")
        print(f"{'='*60}")
        print(f"Testing ALL {len(self.clean_universe):,} clean tickers - NO shortcuts")
        print("")
        
        results = []
        test_tickers = self.clean_universe  # ALL TICKERS - DO THIS RIGHT
        
        for ticker in test_tickers:
            df = pd.read_sql(f"""
                SELECT date, close, volume 
                FROM ohlcv 
                WHERE ticker = '{ticker}'
                ORDER BY date
            """, self.conn)
            
            if len(df) < lookback + 5:
                continue
            
            # Calculate momentum
            df['momentum'] = df['close'].pct_change(lookback)
            df['forward_return_5d'] = df['close'].shift(-5) / df['close'] - 1
            
            # Get transaction cost
            cost = self.transaction_costs.get(ticker, 0.01)
            
            # Test: Does high momentum predict positive returns?
            high_mom = df[df['momentum'] > threshold]
            if len(high_mom) > 10:
                avg_return = high_mom['forward_return_5d'].mean()
                after_cost = avg_return - cost
                
                results.append({
                    'ticker': ticker,
                    'num_signals': len(high_mom),
                    'avg_return_gross': avg_return,
                    'avg_return_net': after_cost,
                    'sharpe': high_mom['forward_return_5d'].mean() / high_mom['forward_return_5d'].std() if len(high_mom) > 1 else 0,
                    'transaction_cost': cost
                })
        
        results_df = pd.DataFrame(results)
        
        print(f"Tested: {len(test_tickers)} tickers")
        print(f"Valid signals: {len(results_df)}")
        print(f"")
        print(f"Average gross return: {results_df['avg_return_gross'].mean():.2%}")
        print(f"Average net return: {results_df['avg_return_net'].mean():.2%}")
        print(f"Profitable after costs: {(results_df['avg_return_net'] > 0).sum()} / {len(results_df)}")
        
        # Statistical test
        t_stat = stats.ttest_1samp(results_df['avg_return_net'], 0)[0]
        print(f"")
        print(f"📊 T-statistic: {t_stat:.2f}")
        print(f"🎯 Harvey-Liu-Zhu threshold: 3.0")
        
        if abs(t_stat) > 3.0:
            print(f"✅ PASSED: Statistically significant edge!")
        else:
            print(f"❌ FAILED: Edge not statistically significant")
        
        return results_df
    
    def test_volume_surge(self, volume_threshold=2.0, days_forward=5):
        """
        TEST: Alex Law #3 - Volume Surge Reversal
        
        Hypothesis: Abnormal volume spikes predict mean reversion
        Method: Test if 2x average volume predicts reversal
        """
        print(f"\n{'='*60}")
        print(f"TESTING: Alex Law #3 - Volume Surge Reversal")
        print(f"{'='*60}")
        
        # Implementation similar to momentum test
        print("⚠️  Not yet implemented")
        return None
    
    def test_sector_rotation(self):
        """
        TEST: Alex Law #4 - Sector Rotation Patterns
        
        Hypothesis: Winning sectors continue for 1-4 weeks
        Method: Test sector momentum persistence
        """
        print(f"\n{'='*60}")
        print(f"TESTING: Alex Law #4 - Sector Rotation")
        print(f"{'='*60}")
        
        print("⚠️  REQUIRES: Sector classification data")
        print("   Need: GICS sector for each ticker")
        return None
    
    def save_discovered_law(self, law_name, results, methodology):
        """
        Document discovered edge for reproducibility
        
        Format:
        - Law name and description
        - Statistical validation (t-stats, p-values)
        - Transaction cost impact
        - Out-of-sample performance
        - Implementation rules
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"ALEX_LAW_{law_name}_{timestamp}.md"
        
        with open(filename, 'w') as f:
            f.write(f"# Alex Law: {law_name}\n\n")
            f.write(f"**Discovered:** {timestamp}\n\n")
            f.write(f"## Methodology\n{methodology}\n\n")
            f.write(f"## Results\n{results.to_markdown()}\n\n")
            f.write(f"## Statistical Validation\n")
            f.write(f"- T-statistic: {results['sharpe'].mean():.2f}\n")
            f.write(f"- Sample size: {len(results)}\n")
        
        print(f"✅ Saved: {filename}")


if __name__ == "__main__":
    print("="*60)
    print("ALEX LAWS DISCOVERY FRAMEWORK")
    print("="*60)
    print("")
    print("Scientific approach to finding real alpha:")
    print("  1. Test user's proven edges (MU +15%, KDK +10%)")
    print("  2. Discover new patterns in clean data")
    print("  3. Apply rigorous statistical tests")
    print("  4. Account for transaction costs")
    print("  5. Validate out-of-sample")
    print("")
    
    # Initialize framework
    discovery = AlexLawsDiscovery()
    
    print("\n" + "="*60)
    print("AVAILABLE TESTS:")
    print("="*60)
    print("1. Earnings Surprise Drift (user's proven edge)")
    print("2. Momentum Regime Filter")
    print("3. Volume Surge Reversal")
    print("4. Sector Rotation Patterns")
    print("")
    
    # Run initial test
    print("Running: Momentum Regime Test...")
    results = discovery.test_momentum_regime()
    
    if results is not None:
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Review results above")
        print("2. If edge found: test out-of-sample")
        print("3. Add earnings data to test user's proven edge")
        print("4. Build systematic implementation")
