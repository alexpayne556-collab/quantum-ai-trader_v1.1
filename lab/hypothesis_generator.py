#!/usr/bin/env python3
"""
HYPOTHESIS GENERATOR - Automated Discovery System

This module generates testable hypotheses from data WITHOUT presupposing laws.
We DISCOVER patterns, not prescribe them.

Scientific Method:
1. OBSERVE: Look at data, find anomalies
2. QUESTION: Why does this happen?
3. HYPOTHESIZE: Propose mechanism
4. PREDICT: What should we see if hypothesis is true?
5. TEST: Run experiment
6. ANALYZE: Did prediction match reality?
7. REFINE or REJECT: Update hypothesis based on results
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime
import itertools


@dataclass
class Hypothesis:
    """A testable hypothesis about market behavior"""
    id: str
    name: str
    description: str
    mathematical_form: str
    null_hypothesis: str
    alternative_hypothesis: str
    testable_predictions: List[str]
    required_data: List[str]
    expected_effect_size: float
    minimum_sample_size: int
    confidence_level: float
    generated_at: datetime
    status: str = "untested"  # untested, validated, rejected, inconclusive
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'mathematical_form': self.mathematical_form,
            'null_hypothesis': self.null_hypothesis,
            'alternative_hypothesis': self.alternative_hypothesis,
            'predictions': self.testable_predictions,
            'required_data': self.required_data,
            'expected_effect': self.expected_effect_size,
            'min_n': self.minimum_sample_size,
            'confidence': self.confidence_level,
            'status': self.status,
            'generated': self.generated_at.isoformat()
        }


class HypothesisGenerator:
    """
    Automatically generate testable hypotheses from observed data.
    
    Process:
    1. Scan data for statistical anomalies
    2. Formulate potential explanations
    3. Generate testable predictions
    4. Output structured hypothesis for testing
    """
    
    def __init__(self):
        self.hypotheses = []
        self.hypothesis_counter = 0
    
    def scan_for_patterns(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame = None
    ) -> List[Hypothesis]:
        """
        Scan data for statistical patterns and generate hypotheses.
        
        This is EXPLORATORY - we're looking for anomalies, not confirming biases.
        """
        hypotheses = []
        
        # Pattern 1: Autocorrelation (momentum/mean-reversion)
        autocorr_hyp = self._test_autocorrelation_pattern(returns)
        if autocorr_hyp:
            hypotheses.extend(autocorr_hyp)
        
        # Pattern 2: Volatility clustering
        vol_hyp = self._test_volatility_clustering(returns)
        if vol_hyp:
            hypotheses.extend(vol_hyp)
        
        # Pattern 3: Cross-sectional patterns
        if returns.shape[1] > 1:
            cross_hyp = self._test_cross_sectional_patterns(returns)
            if cross_hyp:
                hypotheses.extend(cross_hyp)
        
        # Pattern 4: Feature-return relationships
        if features is not None:
            feature_hyp = self._test_feature_relationships(returns, features)
            if feature_hyp:
                hypotheses.extend(feature_hyp)
        
        self.hypotheses.extend(hypotheses)
        return hypotheses
    
    def _test_autocorrelation_pattern(self, returns: pd.DataFrame) -> List[Hypothesis]:
        """Look for momentum or mean-reversion patterns"""
        hypotheses = []
        
        for col in returns.columns[:min(10, len(returns.columns))]:  # Sample first 10
            series = returns[col].dropna()
            
            if len(series) < 100:
                continue
            
            # Test multiple lags
            for lag in [1, 5, 20, 60]:
                if len(series) <= lag:
                    continue
                
                # Calculate autocorrelation
                autocorr = series.autocorr(lag=lag)
                
                # Statistical significance (Ljung-Box test)
                lb_result = acorr_ljungbox(series, lags=[lag], return_df=True)
                lb_pvalue = lb_result['lb_pvalue'].iloc[0]
                
                # Generate hypothesis if pattern is strong
                if abs(autocorr) > 0.05 and lb_pvalue < 0.05:
                    self.hypothesis_counter += 1
                    
                    if autocorr > 0:
                        name = f"Momentum (lag={lag})"
                        description = f"Returns exhibit positive autocorrelation at lag {lag} - past winners continue winning"
                        math_form = f"ρ({lag}) = {autocorr:.3f} > 0"
                    else:
                        name = f"Mean Reversion (lag={lag})"
                        description = f"Returns exhibit negative autocorrelation at lag {lag} - past winners become losers"
                        math_form = f"ρ({lag}) = {autocorr:.3f} < 0"
                    
                    hyp = Hypothesis(
                        id=f"H{self.hypothesis_counter:04d}",
                        name=name,
                        description=description,
                        mathematical_form=math_form,
                        null_hypothesis=f"ρ({lag}) = 0 (no autocorrelation)",
                        alternative_hypothesis=f"ρ({lag}) ≠ 0 (autocorrelation exists)",
                        testable_predictions=[
                            f"Autocorrelation at lag {lag} should be {autocorr:.3f} ± 0.05 in new data",
                            f"Strategy buying past winners (lag {lag}) should outperform if momentum exists",
                            f"Ljung-Box test should reject null (p < 0.05)"
                        ],
                        required_data=[
                            "Daily/weekly returns",
                            f"Minimum {100 + lag} observations",
                            "Out-of-sample validation period"
                        ],
                        expected_effect_size=abs(autocorr),
                        minimum_sample_size=max(100, lag * 5),
                        confidence_level=0.95,
                        generated_at=datetime.now()
                    )
                    
                    hypotheses.append(hyp)
        
        return hypotheses
    
    def _test_volatility_clustering(self, returns: pd.DataFrame) -> List[Hypothesis]:
        """Look for volatility clustering (GARCH effects)"""
        hypotheses = []
        
        for col in returns.columns[:min(10, len(returns.columns))]:
            series = returns[col].dropna()
            
            if len(series) < 100:
                continue
            
            # Calculate squared returns (proxy for volatility)
            squared_returns = series ** 2
            
            # Test autocorrelation in squared returns
            autocorr_sq = squared_returns.autocorr(lag=1)
            
            if autocorr_sq > 0.1:  # Threshold for meaningful clustering
                self.hypothesis_counter += 1
                
                hyp = Hypothesis(
                    id=f"H{self.hypothesis_counter:04d}",
                    name="Volatility Clustering",
                    description="High volatility periods follow high volatility (GARCH effect)",
                    mathematical_form=f"Corr(ε²_t, ε²_t-1) = {autocorr_sq:.3f}",
                    null_hypothesis="Volatility is constant (homoscedastic)",
                    alternative_hypothesis="Volatility clusters (heteroscedastic)",
                    testable_predictions=[
                        "ARCH test should reject null (p < 0.05)",
                        "GARCH(1,1) model should fit better than constant variance",
                        "α + β in GARCH should be close to 1 (persistence)"
                    ],
                    required_data=[
                        "Daily returns",
                        "Minimum 252 observations (1 year)",
                        "Squared returns for variance estimation"
                    ],
                    expected_effect_size=autocorr_sq,
                    minimum_sample_size=252,
                    confidence_level=0.95,
                    generated_at=datetime.now()
                )
                
                hypotheses.append(hyp)
        
        return hypotheses
    
    def _test_cross_sectional_patterns(self, returns: pd.DataFrame) -> List[Hypothesis]:
        """Look for patterns across stocks (cross-sectional momentum)"""
        hypotheses = []
        
        # Calculate cross-sectional correlation
        corr_matrix = returns.corr()
        mean_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        if mean_corr > 0.3:  # Stocks move together
            self.hypothesis_counter += 1
            
            hyp = Hypothesis(
                id=f"H{self.hypothesis_counter:04d}",
                name="Cross-Sectional Correlation",
                description="Stocks move together (systematic risk factor)",
                mathematical_form=f"Mean pairwise correlation = {mean_corr:.3f}",
                null_hypothesis="Stocks move independently",
                alternative_hypothesis="Stocks are correlated (common factor)",
                testable_predictions=[
                    "Principal Component Analysis should show 1-3 dominant factors",
                    "Market beta (CAPM) should explain >30% of variance",
                    "Correlation should increase during market stress"
                ],
                required_data=[
                    "Returns for multiple stocks",
                    "Market index (SPY) for beta calculation",
                    "Minimum 100 observations"
                ],
                expected_effect_size=mean_corr,
                minimum_sample_size=100,
                confidence_level=0.95,
                generated_at=datetime.now()
            )
            
            hypotheses.append(hyp)
        
        return hypotheses
    
    def _test_feature_relationships(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame
    ) -> List[Hypothesis]:
        """Look for relationships between features and future returns"""
        hypotheses = []
        
        # For each feature, test correlation with future returns
        for feature_name in features.columns[:10]:  # Test first 10 features
            for ret_col in returns.columns[:5]:  # Test first 5 stocks
                
                # Align data
                aligned = pd.concat([
                    features[feature_name],
                    returns[ret_col].shift(-1)  # Next period return
                ], axis=1).dropna()
                
                if len(aligned) < 50:
                    continue
                
                # Calculate correlation
                corr, p_value = stats.pearsonr(aligned.iloc[:, 0], aligned.iloc[:, 1])
                
                if abs(corr) > 0.1 and p_value < 0.05:
                    self.hypothesis_counter += 1
                    
                    direction = "positively" if corr > 0 else "negatively"
                    
                    hyp = Hypothesis(
                        id=f"H{self.hypothesis_counter:04d}",
                        name=f"{feature_name} → Future Returns",
                        description=f"{feature_name} is {direction} related to next-period returns",
                        mathematical_form=f"Corr({feature_name}_t, Return_t+1) = {corr:.3f}",
                        null_hypothesis=f"{feature_name} has no predictive power",
                        alternative_hypothesis=f"{feature_name} predicts future returns",
                        testable_predictions=[
                            f"Regression R² should be {corr**2:.3f} ± 0.05",
                            f"Long/short strategy based on {feature_name} should have Sharpe > 0.5",
                            f"Effect should persist in out-of-sample data"
                        ],
                        required_data=[
                            f"{feature_name} values",
                            "Next-period returns",
                            "Minimum 100 observations",
                            "Train/test split for validation"
                        ],
                        expected_effect_size=abs(corr),
                        minimum_sample_size=100,
                        confidence_level=0.95,
                        generated_at=datetime.now()
                    )
                    
                    hypotheses.append(hyp)
        
        return hypotheses
    
    def generate_null_hypotheses(self) -> List[Hypothesis]:
        """
        Generate null hypotheses to test as controls.
        
        These SHOULD fail - if they pass, something is wrong with our testing.
        """
        null_hyps = []
        
        # Null 1: Random trading should lose money
        self.hypothesis_counter += 1
        null_hyps.append(Hypothesis(
            id=f"H{self.hypothesis_counter:04d}",
            name="Random Trading Fails",
            description="Random buy/sell decisions should have zero expected return (minus costs)",
            mathematical_form="E[R_random] = -transaction_costs",
            null_hypothesis="Random trading is profitable",
            alternative_hypothesis="Random trading has zero or negative returns",
            testable_predictions=[
                "1000 random strategies should have mean return ≈ -0.1% (costs)",
                "Win rate should be ≈ 50%",
                "Sharpe ratio should be ≈ 0"
            ],
            required_data=["Historical returns", "Transaction cost estimate"],
            expected_effect_size=0.0,
            minimum_sample_size=1000,
            confidence_level=0.95,
            generated_at=datetime.now()
        ))
        
        # Null 2: Past performance doesn't predict future (weak EMH)
        self.hypothesis_counter += 1
        null_hyps.append(Hypothesis(
            id=f"H{self.hypothesis_counter:04d}",
            name="Weak Efficient Market Hypothesis",
            description="Past prices contain no information about future returns",
            mathematical_form="Corr(R_t, R_t+k) = 0 for all k > 0",
            null_hypothesis="Markets are weak-form efficient",
            alternative_hypothesis="Past prices predict future returns",
            testable_predictions=[
                "Autocorrelation should be statistically zero",
                "Technical analysis strategies should fail",
                "Random walk model should fit data"
            ],
            required_data=["Price history", "Multiple time horizons"],
            expected_effect_size=0.0,
            minimum_sample_size=1000,
            confidence_level=0.95,
            generated_at=datetime.now()
        ))
        
        self.hypotheses.extend(null_hyps)
        return null_hyps
    
    def export_hypotheses(self, filename: str = None) -> pd.DataFrame:
        """Export all hypotheses to DataFrame/CSV"""
        if not self.hypotheses:
            return pd.DataFrame()
        
        df = pd.DataFrame([h.to_dict() for h in self.hypotheses])
        
        if filename:
            df.to_csv(filename, index=False)
            print(f"✅ Exported {len(df)} hypotheses to {filename}")
        
        return df
    
    def summary(self) -> str:
        """Print summary of generated hypotheses"""
        if not self.hypotheses:
            return "No hypotheses generated yet."
        
        lines = []
        lines.append("=" * 80)
        lines.append("HYPOTHESIS GENERATION SUMMARY")
        lines.append("=" * 80)
        lines.append(f"\nTotal hypotheses generated: {len(self.hypotheses)}")
        
        # Count by type
        types = {}
        for h in self.hypotheses:
            type_name = h.name.split('(')[0].strip()
            types[type_name] = types.get(type_name, 0) + 1
        
        lines.append("\nHypotheses by type:")
        for type_name, count in sorted(types.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  {type_name}: {count}")
        
        lines.append("\n" + "-" * 80)
        lines.append("READY FOR TESTING")
        lines.append("-" * 80)
        
        untested = [h for h in self.hypotheses if h.status == "untested"]
        lines.append(f"\nUntested hypotheses: {len(untested)}")
        
        if untested:
            lines.append("\nTop 5 hypotheses to test first:")
            for h in sorted(untested, key=lambda x: x.expected_effect_size, reverse=True)[:5]:
                lines.append(f"\n  {h.id}: {h.name}")
                lines.append(f"    Expected effect: {h.expected_effect_size:.3f}")
                lines.append(f"    Min sample: {h.minimum_sample_size}")
                lines.append(f"    Prediction: {h.testable_predictions[0]}")
        
        lines.append("\n" + "=" * 80)
        
        return "\n".join(lines)


def example_hypothesis_generation():
    """Example of how to use the hypothesis generator"""
    
    print("EXAMPLE: Automated Hypothesis Generation\n")
    
    # Generate fake data
    np.random.seed(42)
    n_days = 1000
    n_stocks = 10
    
    # Create returns with some momentum
    returns = pd.DataFrame(
        np.random.randn(n_days, n_stocks) * 0.02,
        columns=[f"STOCK_{i}" for i in range(n_stocks)]
    )
    
    # Add momentum to first few stocks
    for i in range(3):
        for t in range(1, n_days):
            returns.iloc[t, i] += 0.1 * returns.iloc[t-1, i]
    
    # Add volatility clustering
    returns.iloc[:, 3] = np.random.randn(n_days) * (1 + 0.5 * np.abs(np.random.randn(n_days)))
    
    # Generate hypotheses
    generator = HypothesisGenerator()
    
    print("Scanning data for patterns...")
    hypotheses = generator.scan_for_patterns(returns)
    
    print(f"\n✅ Generated {len(hypotheses)} hypotheses from data\n")
    
    # Export
    Path('../lab/hypotheses').mkdir(parents=True, exist_ok=True)
    df = generator.export_hypotheses('../lab/hypotheses/generated_hypotheses.csv')
    
    # Print summary
    print(generator.summary())
    
    # Show first few hypotheses
    print("\n" + "=" * 80)
    print("SAMPLE HYPOTHESES")
    print("=" * 80)
    
    for h in hypotheses[:3]:
        print(f"\n{h.id}: {h.name}")
        print(f"Description: {h.description}")
        print(f"Mathematical form: {h.mathematical_form}")
        print(f"H0: {h.null_hypothesis}")
        print(f"H1: {h.alternative_hypothesis}")
        print(f"Predictions:")
        for pred in h.testable_predictions:
            print(f"  - {pred}")


if __name__ == '__main__':
    example_hypothesis_generation()
