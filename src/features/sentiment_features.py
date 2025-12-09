"""
Sentiment Feature Engineering (Perplexity Q9)
==============================================
Process news sentiment scores with smoothing, trend detection, and divergence analysis.

Features from EODHD Sentiment API:
- Raw sentiment scores (0-100 scale, 50=neutral)
- News article counts
- Sentiment categories (positive/negative/neutral)

Transformations:
1. 5-day MA smoothing (reduce noise)
2. Trend: score - MA (momentum detection)
3. Divergence: price up + sentiment down = bearish signal
4. Noise filter: <5 articles → set to neutral

Author: Quantum AI Trader
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentimentFeatures:
    """
    Process and engineer sentiment features for trading signals.
    
    Perplexity Q9 Implementation:
    - Smoothing window: 5 days (reduce day-to-day noise)
    - Trend detection: current - MA (positive = improving sentiment)
    - Divergence detection: price-sentiment conflict (contrarian signal)
    - Noise threshold: <5 articles = insufficient data → neutral
    """
    
    def __init__(
        self,
        smoothing_window: int = 5,
        noise_threshold: int = 5,
        divergence_lookback: int = 10
    ):
        """
        Initialize sentiment feature engineer.
        
        Args:
            smoothing_window: Days for moving average (default 5)
            noise_threshold: Minimum articles to trust sentiment (default 5)
            divergence_lookback: Days to check for price-sentiment divergence (default 10)
        """
        self.smoothing_window = smoothing_window
        self.noise_threshold = noise_threshold
        self.divergence_lookback = divergence_lookback
    
    def compute_smoothed_sentiment(
        self,
        sentiment_score: pd.Series,
        article_count: pd.Series
    ) -> pd.Series:
        """
        Compute smoothed sentiment with noise filtering.
        
        Steps:
        1. Apply noise filter: <5 articles → set to neutral (50)
        2. Compute 5-day moving average
        3. Fill early NaN values with raw score
        
        Args:
            sentiment_score: Raw sentiment scores (0-100, 50=neutral)
            article_count: Number of articles per period
            
        Returns:
            Smoothed sentiment scores
        """
        # Apply noise filter
        filtered_sentiment = sentiment_score.copy()
        low_article_mask = article_count < self.noise_threshold
        filtered_sentiment[low_article_mask] = 50.0  # Neutral
        
        # Compute moving average
        smoothed = filtered_sentiment.rolling(
            window=self.smoothing_window,
            min_periods=1
        ).mean()
        
        return smoothed
    
    def compute_sentiment_trend(
        self,
        sentiment_score: pd.Series,
        smoothed_sentiment: pd.Series
    ) -> pd.Series:
        """
        Compute sentiment trend (momentum).
        
        Formula: Current Score - 5d MA
        
        Interpretation:
        - Positive = sentiment improving (above recent average)
        - Negative = sentiment deteriorating (below recent average)
        - Large absolute values = rapid sentiment shift
        
        Args:
            sentiment_score: Raw sentiment scores
            smoothed_sentiment: Smoothed sentiment (5d MA)
            
        Returns:
            Sentiment trend values
        """
        trend = sentiment_score - smoothed_sentiment
        return trend
    
    def detect_divergence(
        self,
        price: pd.Series,
        sentiment_score: pd.Series
    ) -> pd.Series:
        """
        Detect price-sentiment divergence (contrarian signal).
        
        Logic:
        - Bullish divergence: price falling + sentiment improving
        - Bearish divergence: price rising + sentiment deteriorating
        
        Returns:
        - +1 = bullish divergence (potential bottom)
        -  0 = no divergence
        - -1 = bearish divergence (potential top)
        
        Args:
            price: Price series
            sentiment_score: Sentiment scores
            
        Returns:
            Divergence signals (-1, 0, +1)
        """
        # Compute price trend (percentage change over lookback)
        price_change = price.pct_change(self.divergence_lookback)
        
        # Compute sentiment trend (absolute change over lookback)
        sentiment_change = sentiment_score.diff(self.divergence_lookback)
        
        # Initialize divergence series
        divergence = pd.Series(0, index=price.index)
        
        # Bearish divergence: price up + sentiment down
        bearish_mask = (price_change > 0.02) & (sentiment_change < -5)
        divergence[bearish_mask] = -1
        
        # Bullish divergence: price down + sentiment up
        bullish_mask = (price_change < -0.02) & (sentiment_change > 5)
        divergence[bullish_mask] = 1
        
        return divergence
    
    def compute_sentiment_volatility(
        self,
        sentiment_score: pd.Series,
        window: int = 10
    ) -> pd.Series:
        """
        Compute sentiment volatility (uncertainty measure).
        
        High volatility = conflicting news / uncertainty
        Low volatility = consensus view
        
        Args:
            sentiment_score: Sentiment scores
            window: Rolling window for std calculation
            
        Returns:
            Sentiment volatility
        """
        volatility = sentiment_score.rolling(window=window, min_periods=3).std()
        volatility = volatility.fillna(0)
        return volatility
    
    def compute_sentiment_extreme(
        self,
        sentiment_score: pd.Series,
        upper_threshold: float = 70,
        lower_threshold: float = 30
    ) -> pd.Series:
        """
        Detect extreme sentiment (contrarian signal).
        
        Extreme bullish sentiment (>70) often precedes tops.
        Extreme bearish sentiment (<30) often precedes bottoms.
        
        Args:
            sentiment_score: Sentiment scores
            upper_threshold: Bullish extreme threshold (default 70)
            lower_threshold: Bearish extreme threshold (default 30)
            
        Returns:
            Extreme sentiment indicator (-1, 0, +1)
        """
        extreme = pd.Series(0, index=sentiment_score.index)
        extreme[sentiment_score > upper_threshold] = 1  # Extreme bullish
        extreme[sentiment_score < lower_threshold] = -1  # Extreme bearish
        return extreme
    
    def compute_all_features(
        self,
        df: pd.DataFrame,
        sentiment_col: str = 'sentiment_score',
        article_col: str = 'article_count',
        price_col: str = 'Close'
    ) -> pd.DataFrame:
        """
        Compute all sentiment features.
        
        Args:
            df: DataFrame with sentiment and price data
            sentiment_col: Column name for sentiment scores
            article_col: Column name for article counts
            price_col: Column name for price
            
        Returns:
            DataFrame with original data + sentiment features
        """
        result = df.copy()
        
        # Validate columns
        required = [sentiment_col, article_col, price_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Compute features
        result['sentiment_smoothed'] = self.compute_smoothed_sentiment(
            df[sentiment_col], df[article_col]
        )
        
        result['sentiment_trend'] = self.compute_sentiment_trend(
            df[sentiment_col], result['sentiment_smoothed']
        )
        
        result['sentiment_divergence'] = self.detect_divergence(
            df[price_col], df[sentiment_col]
        )
        
        result['sentiment_volatility'] = self.compute_sentiment_volatility(
            df[sentiment_col]
        )
        
        result['sentiment_extreme'] = self.compute_sentiment_extreme(
            df[sentiment_col]
        )
        
        # Additional derived features
        result['sentiment_acceleration'] = result['sentiment_trend'].diff()
        
        result['sentiment_regime'] = pd.cut(
            result['sentiment_smoothed'],
            bins=[0, 40, 60, 100],
            labels=['bearish', 'neutral', 'bullish']
        )
        
        return result
    
    @staticmethod
    def get_feature_names() -> list:
        """Get list of all sentiment feature names."""
        return [
            'sentiment_smoothed',
            'sentiment_trend',
            'sentiment_divergence',
            'sentiment_volatility',
            'sentiment_extreme',
            'sentiment_acceleration',
            'sentiment_regime'
        ]
    
    @staticmethod
    def get_feature_descriptions() -> Dict[str, str]:
        """Get descriptions of all sentiment features."""
        return {
            'sentiment_smoothed': '5-day MA of sentiment (noise reduced)',
            'sentiment_trend': 'Current - MA (positive = improving)',
            'sentiment_divergence': 'Price-sentiment conflict (-1/0/+1)',
            'sentiment_volatility': '10d std of sentiment (uncertainty)',
            'sentiment_extreme': 'Extreme sentiment flag (-1/0/+1)',
            'sentiment_acceleration': 'Change in sentiment trend',
            'sentiment_regime': 'Categorical: bearish/neutral/bullish'
        }


# ============================================================================
# TEST HARNESS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("SENTIMENT FEATURES - TEST (Perplexity Q9)")
    print("=" * 80)
    
    # Generate synthetic sentiment + price data
    np.random.seed(42)
    n_days = 100
    
    # Simulate price with trend
    price = 100 + np.cumsum(np.random.normal(0.2, 1, n_days))
    
    # Simulate sentiment with mean-reverting behavior
    sentiment_base = 50 + np.cumsum(np.random.normal(0, 3, n_days))
    sentiment_base = np.clip(sentiment_base, 0, 100)
    
    # Add sentiment patterns
    # Days 20-30: Extreme bullish sentiment (potential top)
    sentiment_base[20:30] += 25
    sentiment_base = np.clip(sentiment_base, 0, 100)
    
    # Days 50-60: Bearish divergence (price up, sentiment down)
    price[50:60] += 5  # Price rises
    sentiment_base[50:60] -= 15  # Sentiment falls
    
    # Days 70-80: Extreme bearish sentiment (potential bottom)
    sentiment_base[70:80] = 25
    
    # Simulate article counts (some days have low coverage)
    article_counts = np.random.poisson(10, n_days)
    low_coverage_days = np.random.choice(n_days, size=15, replace=False)
    article_counts[low_coverage_days] = np.random.randint(0, 5, size=15)
    
    df = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=n_days, freq='D'),
        'Close': price,
        'sentiment_score': sentiment_base,
        'article_count': article_counts
    })
    
    print(f"\n📊 Dataset: {len(df)} days")
    print(f"   Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    print(f"   Sentiment range: {df['sentiment_score'].min():.1f} - {df['sentiment_score'].max():.1f}")
    print(f"   Article count range: {df['article_count'].min()} - {df['article_count'].max()}")
    print(f"   Low coverage days (<5 articles): {(df['article_count'] < 5).sum()}")
    
    # Compute features
    print("\n" + "=" * 80)
    print("COMPUTING SENTIMENT FEATURES")
    print("=" * 80)
    
    sentiment_eng = SentimentFeatures(
        smoothing_window=5,
        noise_threshold=5,
        divergence_lookback=10
    )
    
    df_features = sentiment_eng.compute_all_features(df)
    
    feature_names = SentimentFeatures.get_feature_names()
    print(f"\n✅ Computed {len(feature_names)} sentiment features")
    
    # Show descriptions
    print("\n" + "=" * 80)
    print("FEATURE DESCRIPTIONS")
    print("=" * 80)
    
    descriptions = SentimentFeatures.get_feature_descriptions()
    for feat, desc in descriptions.items():
        print(f"  {feat:<30} {desc}")
    
    # Analyze specific periods
    print("\n" + "=" * 80)
    print("PATTERN DETECTION ANALYSIS")
    print("=" * 80)
    
    # Days 20-30: Extreme bullish sentiment
    extreme_bull = df_features.loc[20:30]
    print("\nDays 20-30 (Injected Extreme Bullish Sentiment):")
    print(f"  Avg Sentiment: {extreme_bull['sentiment_score'].mean():.1f}")
    print(f"  Extreme Bullish Flags: {(extreme_bull['sentiment_extreme'] == 1).sum()} days")
    print(f"  Interpretation: Potential top (contrarian signal)")
    
    # Days 50-60: Bearish divergence
    divergence_period = df_features.loc[50:60]
    price_change = (df.loc[60, 'Close'] - df.loc[50, 'Close']) / df.loc[50, 'Close'] * 100
    sentiment_change = df.loc[60, 'sentiment_score'] - df.loc[50, 'sentiment_score']
    print("\nDays 50-60 (Injected Bearish Divergence):")
    print(f"  Price Change: +{price_change:.1f}%")
    print(f"  Sentiment Change: {sentiment_change:.1f} points")
    print(f"  Bearish Divergence Flags: {(divergence_period['sentiment_divergence'] == -1).sum()} days")
    print(f"  Interpretation: Price up but sentiment down (warning signal)")
    
    # Days 70-80: Extreme bearish sentiment
    extreme_bear = df_features.loc[70:80]
    print("\nDays 70-80 (Injected Extreme Bearish Sentiment):")
    print(f"  Avg Sentiment: {extreme_bear['sentiment_score'].mean():.1f}")
    print(f"  Extreme Bearish Flags: {(extreme_bear['sentiment_extreme'] == -1).sum()} days")
    print(f"  Interpretation: Potential bottom (contrarian signal)")
    
    # Noise filter effectiveness
    print("\n" + "=" * 80)
    print("NOISE FILTER ANALYSIS")
    print("=" * 80)
    
    low_coverage = df_features[df_features['article_count'] < 5]
    neutralized_count = (low_coverage['sentiment_smoothed'] == 50.0).sum()
    print(f"\n  Days with <5 articles: {len(low_coverage)}")
    print(f"  Days neutralized to 50.0: {neutralized_count}")
    print(f"  Filter effectiveness: {neutralized_count / len(low_coverage) * 100:.1f}%")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    numeric_features = [f for f in feature_names if f != 'sentiment_regime']
    summary = df_features[numeric_features].describe()
    print(summary.T[['mean', 'std', 'min', 'max']].round(2))
    
    # Regime distribution
    print("\n" + "=" * 80)
    print("SENTIMENT REGIME DISTRIBUTION")
    print("=" * 80)
    
    regime_counts = df_features['sentiment_regime'].value_counts()
    print(regime_counts)
    
    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)
    
    checks_passed = 0
    checks_total = 0
    
    # Smoothed sentiment in [0, 100]
    checks_total += 1
    smooth_min = df_features['sentiment_smoothed'].min()
    smooth_max = df_features['sentiment_smoothed'].max()
    if 0 <= smooth_min and smooth_max <= 100:
        print(f"✅ Smoothed sentiment range: [{smooth_min:.1f}, {smooth_max:.1f}]")
        checks_passed += 1
    else:
        print(f"❌ Smoothed sentiment out of range: [{smooth_min:.1f}, {smooth_max:.1f}]")
    
    # Divergence in [-1, 0, 1]
    checks_total += 1
    div_values = df_features['sentiment_divergence'].unique()
    if set(div_values).issubset({-1, 0, 1}):
        print(f"✅ Divergence values: {sorted(div_values)}")
        checks_passed += 1
    else:
        print(f"❌ Invalid divergence values: {div_values}")
    
    # Extreme in [-1, 0, 1]
    checks_total += 1
    extreme_values = df_features['sentiment_extreme'].unique()
    if set(extreme_values).issubset({-1, 0, 1}):
        print(f"✅ Extreme values: {sorted(extreme_values)}")
        checks_passed += 1
    else:
        print(f"❌ Invalid extreme values: {extreme_values}")
    
    # Detected extreme bullish (days 20-30)
    checks_total += 1
    detected_extreme_bull = (extreme_bull['sentiment_extreme'] == 1).any()
    if detected_extreme_bull:
        print(f"✅ Detected extreme bullish sentiment (days 20-30)")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect extreme bullish sentiment")
    
    # Detected divergence (days 50-60)
    checks_total += 1
    detected_divergence = (divergence_period['sentiment_divergence'] == -1).any()
    if detected_divergence:
        print(f"✅ Detected bearish divergence (days 50-60)")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect bearish divergence")
    
    # Detected extreme bearish (days 70-80)
    checks_total += 1
    detected_extreme_bear = (extreme_bear['sentiment_extreme'] == -1).any()
    if detected_extreme_bear:
        print(f"✅ Detected extreme bearish sentiment (days 70-80)")
        checks_passed += 1
    else:
        print(f"❌ Failed to detect extreme bearish sentiment")
    
    print(f"\n{'=' * 80}")
    print(f"✅ Sentiment Features: {checks_passed}/{checks_total} checks passed")
    print("=" * 80)
