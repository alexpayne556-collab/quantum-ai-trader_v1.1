#!/usr/bin/env python3
"""
ADVANCED WINNER VALIDATION
===========================
Additional statistical tests for the 6 A+ winners + 4 A-grade signals.

Tests:
1. Bootstrap Confidence Intervals (10,000 samples)
2. Signal Correlation Analysis
3. Drawdown Analysis
4. Turnover & Capacity
5. Sub-Period Stability (rolling Sharpe)
6. Market Stress Testing

Run:
    python WINNER_ADVANCED_VALIDATION.py
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
from pathlib import Path
import pickle

warnings.filterwarnings('ignore')


# ============================================================================
# LOAD DATA AND SIGNALS FROM GAUNTLET
# ============================================================================

# Import from main gauntlet
from WINNER_VALIDATION_GAUNTLET import (
    WinnerValidationGauntlet,
    SIGNAL_REGISTRY,
    signal_multi_indicator_oversold,
    signal_vix_turbulence,
    signal_vix_percentile,
    signal_ma_distance,
    signal_weekly_reversal,
    signal_vix_mean_reversion,
    signal_bollinger_mr,
    signal_post_large_move,
    signal_oil_equity,
    signal_zscore_mr,
)


# Top 10 signals to validate further
TOP_SIGNALS = [
    {'id': 'H27E', 'name': 'Multi-Indicator Oversold', 'grade': 'A+'},
    {'id': 'H128', 'name': 'VIX Turbulence', 'grade': 'A+'},
    {'id': 'H21', 'name': 'VIX Percentile', 'grade': 'A+'},
    {'id': 'H27C', 'name': 'MA Distance', 'grade': 'A+'},
    {'id': 'H16', 'name': 'Weekly Reversal', 'grade': 'A+'},
    {'id': 'H20', 'name': 'VIX Mean Reversion', 'grade': 'A+'},
    {'id': 'H19', 'name': 'Bollinger Band MR', 'grade': 'A'},
    {'id': 'H27', 'name': 'Post-Large-Move Reversal', 'grade': 'A'},
    {'id': 'H62', 'name': 'Oil-Equity Relationship', 'grade': 'A'},
    {'id': 'H27B', 'name': 'Z-Score Mean Reversion', 'grade': 'A'},
]


class AdvancedValidation:
    """Advanced validation for top signals."""
    
    def __init__(self):
        self.gauntlet = WinnerValidationGauntlet()
        self.gauntlet.download_data()
        self.signals_df = None
        self.returns = None
        
    def prepare_signals(self, hold_period: int = 21):
        """Generate all signals and returns."""
        data = self.gauntlet.data['SPY'].copy()
        self.returns = data['close'].pct_change(hold_period).shift(-hold_period)
        
        signals = {}
        for sig in TOP_SIGNALS:
            signals[sig['id']] = self.gauntlet.generate_signal(sig['id'], data)
        
        self.signals_df = pd.DataFrame(signals)
        print(f"✓ Generated {len(TOP_SIGNALS)} signals with {len(data)} data points")
        
    def bootstrap_confidence_intervals(self, n_bootstrap: int = 10000):
        """Calculate bootstrap confidence intervals for each signal's spread."""
        print("\n" + "="*60)
        print("1. BOOTSTRAP CONFIDENCE INTERVALS (10,000 samples)")
        print("="*60)
        
        results = []
        
        for sig in TOP_SIGNALS:
            hyp_id = sig['id']
            signal = self.signals_df[hyp_id]
            
            # Get returns for signal=1 vs signal=0
            long_mask = signal == 1
            long_returns = self.returns[long_mask].dropna().values
            other_returns = self.returns[~long_mask].dropna().values
            
            if len(long_returns) < 20:
                continue
            
            # Bootstrap
            spreads = []
            for _ in range(n_bootstrap):
                # Resample with replacement
                long_sample = np.random.choice(long_returns, size=len(long_returns), replace=True)
                other_sample = np.random.choice(other_returns, size=len(other_returns), replace=True)
                spread = long_sample.mean() - other_sample.mean()
                spreads.append(spread)
            
            spreads = np.array(spreads) * (252 / 21)  # Annualize
            
            ci_lower = np.percentile(spreads, 2.5)
            ci_upper = np.percentile(spreads, 97.5)
            mean_spread = spreads.mean()
            std_spread = spreads.std()
            
            # Is it significantly positive?
            significant = ci_lower > 0
            
            results.append({
                'id': hyp_id,
                'name': sig['name'],
                'mean_spread': mean_spread,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': std_spread,
                'significant': significant,
            })
            
            status = "✓ SIGNIFICANT" if significant else "✗ Not significant"
            print(f"  {hyp_id}: {mean_spread:+.1%} [{ci_lower:+.1%}, {ci_upper:+.1%}] {status}")
        
        return pd.DataFrame(results)
    
    def signal_correlation_analysis(self):
        """Analyze correlation between signals."""
        print("\n" + "="*60)
        print("2. SIGNAL CORRELATION ANALYSIS")
        print("="*60)
        
        # Calculate correlation matrix
        corr_matrix = self.signals_df.corr()
        
        print("\nSignal Correlation Matrix:")
        print("-" * 40)
        
        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                corr = corr_matrix.iloc[i, j]
                if abs(corr) > 0.5:
                    high_corr.append((col1, col2, corr))
        
        if high_corr:
            print("\n⚠ Highly Correlated Pairs (|corr| > 0.5):")
            for col1, col2, corr in sorted(high_corr, key=lambda x: -abs(x[2])):
                print(f"  {col1} ↔ {col2}: {corr:.2f}")
        else:
            print("\n✓ No highly correlated pairs (all |corr| < 0.5)")
        
        # Find uncorrelated signals (best for ensemble)
        low_corr = []
        for i in range(len(corr_matrix.columns)):
            avg_corr = corr_matrix.iloc[i].abs().mean()
            low_corr.append((corr_matrix.columns[i], avg_corr))
        
        print("\n📊 Most Independent Signals (lowest avg correlation):")
        for sig_id, avg in sorted(low_corr, key=lambda x: x[1])[:5]:
            print(f"  {sig_id}: avg|corr|={avg:.2f}")
        
        return corr_matrix
    
    def drawdown_analysis(self):
        """Analyze maximum drawdowns when following each signal."""
        print("\n" + "="*60)
        print("3. DRAWDOWN ANALYSIS")
        print("="*60)
        
        data = self.gauntlet.data['SPY'].copy()
        daily_ret = data['close'].pct_change()
        
        results = []
        
        for sig in TOP_SIGNALS:
            hyp_id = sig['id']
            signal = self.signals_df[hyp_id]
            
            # Strategy returns: when signal=1, use market return; otherwise cash
            strat_ret = daily_ret * signal.shift(1)  # Shift for execution delay
            strat_cum = (1 + strat_ret.fillna(0)).cumprod()
            
            # Calculate drawdown
            rolling_max = strat_cum.expanding().max()
            drawdown = (strat_cum - rolling_max) / rolling_max
            max_dd = drawdown.min()
            
            # Calculate CAGR
            years = len(strat_cum) / 252
            cagr = (strat_cum.iloc[-1] ** (1/years)) - 1 if years > 0 else 0
            
            # Calmar ratio
            calmar = cagr / abs(max_dd) if max_dd != 0 else 0
            
            results.append({
                'id': hyp_id,
                'name': sig['name'],
                'max_drawdown': max_dd,
                'cagr': cagr,
                'calmar_ratio': calmar,
            })
            
            print(f"  {hyp_id}: MaxDD={max_dd:.1%}, CAGR={cagr:.1%}, Calmar={calmar:.2f}")
        
        return pd.DataFrame(results)
    
    def turnover_analysis(self):
        """Analyze signal turnover and trading frequency."""
        print("\n" + "="*60)
        print("4. TURNOVER & CAPACITY ANALYSIS")
        print("="*60)
        
        results = []
        
        for sig in TOP_SIGNALS:
            hyp_id = sig['id']
            signal = self.signals_df[hyp_id]
            
            # Count signal changes
            changes = signal.diff().abs().sum()
            
            # Calculate turnover (% of days with position change)
            turnover = changes / len(signal)
            
            # Average days in trade
            in_trade = signal == 1
            trade_lengths = []
            current_length = 0
            for val in in_trade:
                if val:
                    current_length += 1
                else:
                    if current_length > 0:
                        trade_lengths.append(current_length)
                    current_length = 0
            
            avg_trade_length = np.mean(trade_lengths) if trade_lengths else 0
            
            # Trades per year
            years = len(signal) / 252
            trades_per_year = len(trade_lengths) / years if years > 0 else 0
            
            results.append({
                'id': hyp_id,
                'name': sig['name'],
                'turnover': turnover,
                'avg_trade_days': avg_trade_length,
                'trades_per_year': trades_per_year,
            })
            
            print(f"  {hyp_id}: {trades_per_year:.0f} trades/yr, avg {avg_trade_length:.0f} days")
        
        return pd.DataFrame(results)
    
    def rolling_sharpe_stability(self, window: int = 252):
        """Calculate rolling Sharpe to check stability."""
        print("\n" + "="*60)
        print("5. ROLLING SHARPE STABILITY (1-year windows)")
        print("="*60)
        
        data = self.gauntlet.data['SPY'].copy()
        daily_ret = data['close'].pct_change()
        
        results = []
        
        for sig in TOP_SIGNALS:
            hyp_id = sig['id']
            signal = self.signals_df[hyp_id]
            
            # Strategy returns
            strat_ret = daily_ret * signal.shift(1)
            
            # Rolling Sharpe
            rolling_mean = strat_ret.rolling(window).mean() * 252
            rolling_std = strat_ret.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_mean / rolling_std
            
            # Calculate stability metrics
            avg_sharpe = rolling_sharpe.mean()
            min_sharpe = rolling_sharpe.min()
            pct_positive = (rolling_sharpe > 0).mean()
            volatility = rolling_sharpe.std()
            
            results.append({
                'id': hyp_id,
                'name': sig['name'],
                'avg_rolling_sharpe': avg_sharpe,
                'min_rolling_sharpe': min_sharpe,
                'pct_positive_periods': pct_positive,
                'sharpe_volatility': volatility,
            })
            
            status = "✓ STABLE" if pct_positive > 0.7 else "⚠ UNSTABLE"
            print(f"  {hyp_id}: avg={avg_sharpe:.2f}, min={min_sharpe:.2f}, {pct_positive:.0%} positive {status}")
        
        return pd.DataFrame(results)
    
    def market_stress_test(self):
        """Test performance during major market stress events."""
        print("\n" + "="*60)
        print("6. MARKET STRESS TEST")
        print("="*60)
        
        # Define stress periods
        stress_periods = [
            ('2011 EU Crisis', '2011-07-01', '2011-10-31'),
            ('2015 China Crash', '2015-08-01', '2015-09-30'),
            ('2018 Vol Shock', '2018-01-15', '2018-02-15'),
            ('2018 Q4 Selloff', '2018-10-01', '2018-12-31'),
            ('2020 COVID', '2020-02-15', '2020-03-31'),
            ('2022 Bear Market', '2022-01-01', '2022-10-31'),
        ]
        
        data = self.gauntlet.data['SPY'].copy()
        daily_ret = data['close'].pct_change()
        
        all_results = []
        
        print("\nPerformance During Stress Periods:")
        print("-" * 60)
        
        for period_name, start, end in stress_periods:
            mask = (data.index >= start) & (data.index <= end)
            market_ret = daily_ret[mask].sum()
            
            print(f"\n{period_name} ({start} to {end})")
            print(f"  SPY Return: {market_ret:.1%}")
            
            for sig in TOP_SIGNALS:
                hyp_id = sig['id']
                signal = self.signals_df[hyp_id]
                
                strat_ret = (daily_ret * signal.shift(1))[mask].sum()
                alpha = strat_ret - market_ret
                
                all_results.append({
                    'period': period_name,
                    'id': hyp_id,
                    'market_return': market_ret,
                    'strategy_return': strat_ret,
                    'alpha': alpha,
                })
                
                status = "↑" if alpha > 0 else "↓"
                print(f"    {hyp_id}: {strat_ret:+.1%} (α={alpha:+.1%} {status})")
        
        # Summarize average alpha during stress
        df = pd.DataFrame(all_results)
        print("\n📊 Average Alpha During Stress Periods:")
        for sig in TOP_SIGNALS:
            avg_alpha = df[df['id'] == sig['id']]['alpha'].mean()
            status = "✓ CRISIS ALPHA" if avg_alpha > 0 else "✗ No crisis alpha"
            print(f"  {sig['id']}: {avg_alpha:+.1%} {status}")
        
        return df
    
    def run_all_tests(self):
        """Run all advanced validation tests."""
        print("\n" + "="*70)
        print("ADVANCED VALIDATION SUITE")
        print("="*70)
        print(f"Testing {len(TOP_SIGNALS)} top signals from gauntlet")
        
        self.prepare_signals()
        
        # Run all tests
        bootstrap = self.bootstrap_confidence_intervals()
        correlation = self.signal_correlation_analysis()
        drawdown = self.drawdown_analysis()
        turnover = self.turnover_analysis()
        stability = self.rolling_sharpe_stability()
        stress = self.market_stress_test()
        
        # Save results
        results = {
            'bootstrap': bootstrap,
            'correlation': correlation,
            'drawdown': drawdown,
            'turnover': turnover,
            'stability': stability,
            'stress': stress,
        }
        
        cache_path = Path('./hypothesis_data/')
        pd.to_pickle(results, cache_path / 'advanced_validation_results.pkl')
        
        # Create final ranking
        print("\n" + "="*70)
        print("FINAL COMBINED RANKING")
        print("="*70)
        
        ranking = []
        for sig in TOP_SIGNALS:
            hyp_id = sig['id']
            
            # Get scores from each test
            boot_row = bootstrap[bootstrap['id'] == hyp_id].iloc[0] if len(bootstrap[bootstrap['id'] == hyp_id]) > 0 else None
            dd_row = drawdown[drawdown['id'] == hyp_id].iloc[0] if len(drawdown[drawdown['id'] == hyp_id]) > 0 else None
            stab_row = stability[stability['id'] == hyp_id].iloc[0] if len(stability[stability['id'] == hyp_id]) > 0 else None
            stress_df = stress[stress['id'] == hyp_id]
            
            score = 0
            
            # Bootstrap significant +2
            if boot_row is not None and boot_row['significant']:
                score += 2
            
            # Good Calmar +1
            if dd_row is not None and dd_row['calmar_ratio'] > 1:
                score += 1
            
            # Stable Sharpe +1
            if stab_row is not None and stab_row['pct_positive_periods'] > 0.7:
                score += 1
            
            # Crisis alpha +1
            if len(stress_df) > 0 and stress_df['alpha'].mean() > 0:
                score += 1
            
            ranking.append({
                'id': hyp_id,
                'name': sig['name'],
                'grade': sig['grade'],
                'total_score': score,
                'bootstrap_sig': boot_row['significant'] if boot_row is not None else False,
                'calmar': dd_row['calmar_ratio'] if dd_row is not None else 0,
                'stability': stab_row['pct_positive_periods'] if stab_row is not None else 0,
                'crisis_alpha': stress_df['alpha'].mean() if len(stress_df) > 0 else 0,
            })
        
        ranking_df = pd.DataFrame(ranking).sort_values('total_score', ascending=False)
        ranking_df.to_csv(cache_path / 'FINAL_SIGNAL_RANKING.csv', index=False)
        
        print("\n🏆 FINAL RANKINGS (by combined score):")
        print("-" * 60)
        for _, row in ranking_df.iterrows():
            print(f"  {row['total_score']}/5: [{row['grade']}] {row['id']} - {row['name']}")
            print(f"        Bootstrap:{'✓' if row['bootstrap_sig'] else '✗'} Calmar:{row['calmar']:.2f} Stable:{row['stability']:.0%} CrisisAlpha:{row['crisis_alpha']:+.1%}")
        
        return ranking_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    validator = AdvancedValidation()
    ranking = validator.run_all_tests()
