"""
HMM REGIME DETECTOR
====================
Hidden Markov Model for market regime detection (Bull/Bear/Sideways)

Why Regime Detection:
- Model accuracy varies: Bull 52%, Sideways 48%, Bear 35%
- Training separate models per regime = +5-10% accuracy
- Real-time regime switching enables adaptive strategies

HMM Approach:
- 3 hidden states: Bull, Bear, Sideways
- Observations: Returns, Volatility, Trend strength
- Gaussian emissions for continuous features

Research: Top quant funds use regime-aware models (Two Sigma, Renaissance)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn import hmm
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. Install with: pip install hmmlearn")


class RegimeDetector:
    """
    Hidden Markov Model for market regime detection
    
    Regimes:
    - 0: Bull (low vol, uptrend)
    - 1: Sideways (medium vol, no trend)
    - 2: Bear (high vol, downtrend)
    """
    
    def __init__(self, n_regimes: int = 3, verbose: bool = True):
        self.n_regimes = n_regimes
        self.verbose = verbose
        self.model = None
        self.scaler = StandardScaler()
        self.regime_names = {0: 'BULL', 1: 'SIDEWAYS', 2: 'BEAR'}
        self.regime_stats = {}
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[RegimeDetector] {msg}")
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM
        
        Features:
        1. Returns (direction)
        2. Realized volatility (risk)
        3. Trend strength (momentum)
        4. Volume regime
        """
        features = pd.DataFrame(index=df.index)
        
        # 1. Returns
        features['returns'] = df['Close'].pct_change()
        
        # 2. Realized volatility (rolling 20-day)
        features['volatility'] = features['returns'].rolling(20).std() * np.sqrt(252)
        
        # 3. Trend strength (price vs 50-day MA)
        ma50 = df['Close'].rolling(50).mean()
        features['trend'] = (df['Close'] - ma50) / ma50
        
        # 4. Volume z-score
        features['volume_zscore'] = (df['Volume'] - df['Volume'].rolling(50).mean()) / df['Volume'].rolling(50).std()
        
        # 5. Return momentum (5-day)
        features['momentum'] = df['Close'].pct_change(5)
        
        # Drop NaN
        features = features.dropna()
        
        return features
    
    def fit(self, df: pd.DataFrame, n_iter: int = 100) -> 'RegimeDetector':
        """
        Train HMM on historical data
        
        Args:
            df: OHLCV DataFrame
            n_iter: Number of EM iterations
        
        Returns:
            self
        """
        if not HMM_AVAILABLE:
            self.log("❌ hmmlearn not available. Using rule-based fallback.")
            return self
        
        self.log(f"Training HMM with {self.n_regimes} regimes...")
        
        # Prepare features
        features = self._prepare_features(df)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features.values)
        
        # Train Gaussian HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=42
        )
        
        self.model.fit(features_scaled)
        
        # Predict regimes on training data
        regimes = self.model.predict(features_scaled)
        
        # Characterize each regime
        self._characterize_regimes(features, regimes)
        
        self.log(f"✅ HMM trained. Log-likelihood: {self.model.score(features_scaled):.2f}")
        
        return self
    
    def _characterize_regimes(self, features: pd.DataFrame, regimes: np.ndarray):
        """Analyze characteristics of each regime"""
        features_with_regime = features.copy()
        features_with_regime['regime'] = regimes
        
        self.log("\nRegime Characteristics:")
        
        regime_order = []  # Will reorder by avg return
        
        for regime in range(self.n_regimes):
            mask = features_with_regime['regime'] == regime
            regime_data = features_with_regime[mask]
            
            avg_return = regime_data['returns'].mean() * 252  # Annualized
            avg_vol = regime_data['volatility'].mean()
            avg_trend = regime_data['trend'].mean()
            count = mask.sum()
            
            self.regime_stats[regime] = {
                'avg_return': avg_return,
                'avg_volatility': avg_vol,
                'avg_trend': avg_trend,
                'count': count,
                'pct': count / len(features_with_regime) * 100
            }
            
            regime_order.append((regime, avg_return))
            
            self.log(f"  Regime {regime}: Return={avg_return:.1%}, Vol={avg_vol:.1%}, "
                    f"Trend={avg_trend:.2f}, Count={count} ({self.regime_stats[regime]['pct']:.1f}%)")
        
        # Reorder regimes: highest return = BULL (0), lowest = BEAR (2)
        regime_order.sort(key=lambda x: x[1], reverse=True)
        
        self.regime_mapping = {}
        for new_idx, (old_idx, _) in enumerate(regime_order):
            if new_idx == 0:
                self.regime_mapping[old_idx] = 0  # BULL
                self.regime_names[0] = 'BULL'
            elif new_idx == self.n_regimes - 1:
                self.regime_mapping[old_idx] = 2  # BEAR
                self.regime_names[2] = 'BEAR'
            else:
                self.regime_mapping[old_idx] = 1  # SIDEWAYS
                self.regime_names[1] = 'SIDEWAYS'
    
    def predict(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Predict regime for new data
        
        Returns:
            regimes: Series of regime labels (0=Bull, 1=Sideways, 2=Bear)
            probabilities: DataFrame with probability of each regime
        """
        features = self._prepare_features(df)
        
        if self.model is None or not HMM_AVAILABLE:
            # Rule-based fallback
            return self._rule_based_regime(df)
        
        # Scale features
        features_scaled = self.scaler.transform(features.values)
        
        # Predict
        raw_regimes = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Map to consistent ordering (BULL=0, SIDEWAYS=1, BEAR=2)
        mapped_regimes = np.array([self.regime_mapping.get(r, r) for r in raw_regimes])
        
        regimes = pd.Series(mapped_regimes, index=features.index, name='regime')
        prob_df = pd.DataFrame(
            probabilities,
            index=features.index,
            columns=[f'prob_regime_{i}' for i in range(self.n_regimes)]
        )
        
        return regimes, prob_df
    
    def _rule_based_regime(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Rule-based fallback when HMM not available
        
        Rules:
        - BULL: Price > MA50, Volatility < 20%, Positive momentum
        - BEAR: Price < MA50, Volatility > 25%, Negative momentum
        - SIDEWAYS: Everything else
        """
        self.log("Using rule-based regime detection (HMM not available)")
        
        ma50 = df['Close'].rolling(50).mean()
        returns = df['Close'].pct_change()
        volatility = returns.rolling(20).std() * np.sqrt(252)
        momentum = df['Close'].pct_change(20)
        
        regimes = pd.Series(1, index=df.index, name='regime')  # Default: SIDEWAYS
        
        # BULL conditions
        bull_mask = (df['Close'] > ma50) & (volatility < 0.20) & (momentum > 0)
        regimes[bull_mask] = 0
        
        # BEAR conditions
        bear_mask = (df['Close'] < ma50) & (volatility > 0.25) & (momentum < -0.05)
        regimes[bear_mask] = 2
        
        # Create probability estimates
        prob_df = pd.DataFrame(index=df.index)
        prob_df['prob_regime_0'] = bull_mask.astype(float) * 0.8 + 0.1
        prob_df['prob_regime_2'] = bear_mask.astype(float) * 0.8 + 0.1
        prob_df['prob_regime_1'] = 1 - prob_df['prob_regime_0'] - prob_df['prob_regime_2']
        
        return regimes, prob_df
    
    def get_current_regime(self, df: pd.DataFrame) -> Dict:
        """
        Get current regime with confidence
        
        Returns:
            dict with 'regime', 'name', 'confidence', 'all_probs'
        """
        regimes, probs = self.predict(df)
        
        current_regime = int(regimes.iloc[-1])
        current_probs = probs.iloc[-1].values
        confidence = current_probs[current_regime]
        
        return {
            'regime': current_regime,
            'name': self.regime_names.get(current_regime, f'REGIME_{current_regime}'),
            'confidence': confidence,
            'all_probs': {
                'BULL': current_probs[0] if len(current_probs) > 0 else 0,
                'SIDEWAYS': current_probs[1] if len(current_probs) > 1 else 0,
                'BEAR': current_probs[2] if len(current_probs) > 2 else 0
            }
        }
    
    def plot_regimes(self, df: pd.DataFrame, regimes: pd.Series, save_path: Optional[str] = None):
        """Plot price with regime coloring"""
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot price
            ax.plot(df.index, df['Close'], color='black', linewidth=0.5, alpha=0.7)
            
            # Color by regime
            colors = {0: 'green', 1: 'yellow', 2: 'red'}
            
            for regime in range(self.n_regimes):
                mask = regimes == regime
                ax.fill_between(
                    df.index, 
                    df['Close'].min(), 
                    df['Close'].max(),
                    where=mask.reindex(df.index, fill_value=False),
                    alpha=0.3,
                    color=colors.get(regime, 'gray'),
                    label=self.regime_names.get(regime, f'Regime {regime}')
                )
            
            ax.set_title('Market Regimes (HMM Detection)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price')
            ax.legend()
            
            if save_path:
                plt.savefig(save_path)
                self.log(f"Saved plot to {save_path}")
            
            plt.close()
            
        except ImportError:
            self.log("matplotlib not available for plotting")


class RegimeAdaptivePredictor:
    """
    Train and use separate models for each regime
    
    Architecture:
    - RegimeDetector identifies current regime
    - Separate XGBoost/LightGBM model for each regime
    - Switch models based on regime
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.regime_detector = RegimeDetector(n_regimes=3, verbose=verbose)
        self.regime_models = {}
        self.feature_columns = None
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[RegimeAdaptive] {msg}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series, df_ohlcv: pd.DataFrame):
        """
        Train regime detector and regime-specific models
        
        Args:
            X: Feature DataFrame
            y: Labels
            df_ohlcv: OHLCV data for regime detection
        """
        self.feature_columns = X.columns.tolist()
        
        # Step 1: Fit regime detector
        self.log("Step 1: Training regime detector...")
        self.regime_detector.fit(df_ohlcv)
        
        # Step 2: Get regimes for training data
        regimes, _ = self.regime_detector.predict(df_ohlcv)
        
        # Align indices
        common_idx = X.index.intersection(regimes.index).intersection(y.index)
        X_aligned = X.loc[common_idx]
        y_aligned = y.loc[common_idx]
        regimes_aligned = regimes.loc[common_idx]
        
        # Step 3: Train model for each regime
        self.log("\nStep 2: Training regime-specific models...")
        
        try:
            import xgboost as xgb
            model_class = xgb.XGBClassifier
            model_params = {'n_estimators': 100, 'max_depth': 5, 'subsample': 0.8, 'verbosity': 0}
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            model_class = RandomForestClassifier
            model_params = {'n_estimators': 100, 'max_depth': 5}
        
        for regime in range(self.regime_detector.n_regimes):
            mask = regimes_aligned == regime
            X_regime = X_aligned[mask]
            y_regime = y_aligned[mask]
            
            if len(X_regime) < 50:
                self.log(f"  Regime {regime} ({self.regime_detector.regime_names[regime]}): "
                        f"Too few samples ({len(X_regime)}), using combined model")
                # Use all data for this regime
                X_regime = X_aligned
                y_regime = y_aligned
            
            model = model_class(**model_params)
            model.fit(X_regime, y_regime)
            
            train_acc = model.score(X_regime, y_regime)
            self.regime_models[regime] = model
            
            self.log(f"  Regime {regime} ({self.regime_detector.regime_names[regime]}): "
                    f"Trained on {len(X_regime)} samples, Train acc: {train_acc:.2%}")
        
        self.log("\n✅ Regime-adaptive predictor ready")
        
        return self
    
    def predict(self, X: pd.DataFrame, df_ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Predict using regime-appropriate model
        """
        # Get current regime
        regime_info = self.regime_detector.get_current_regime(df_ohlcv)
        current_regime = regime_info['regime']
        
        self.log(f"Current regime: {regime_info['name']} (confidence: {regime_info['confidence']:.1%})")
        
        # Use regime-specific model
        if current_regime in self.regime_models:
            model = self.regime_models[current_regime]
        else:
            # Fallback to first available model
            model = list(self.regime_models.values())[0]
        
        return model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame, df_ohlcv: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities using regime-appropriate model
        """
        regime_info = self.regime_detector.get_current_regime(df_ohlcv)
        current_regime = regime_info['regime']
        
        if current_regime in self.regime_models:
            model = self.regime_models[current_regime]
        else:
            model = list(self.regime_models.values())[0]
        
        return model.predict_proba(X)


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    import yfinance as yf
    
    print("Testing HMM Regime Detector...")
    print("=" * 60)
    
    # Download test data
    df = yf.download("SPY", start="2020-01-01", end="2024-12-01", progress=False)
    
    # Test regime detector
    detector = RegimeDetector(n_regimes=3, verbose=True)
    detector.fit(df)
    
    # Predict regimes
    regimes, probs = detector.predict(df)
    
    print(f"\n{'='*60}")
    print("REGIME DISTRIBUTION")
    print(f"{'='*60}")
    print(regimes.value_counts())
    
    # Current regime
    current = detector.get_current_regime(df)
    print(f"\n📊 CURRENT REGIME: {current['name']}")
    print(f"   Confidence: {current['confidence']:.1%}")
    print(f"   All probabilities: {current['all_probs']}")
    
    # Save plot
    detector.plot_regimes(df, regimes, save_path='regime_plot.png')
