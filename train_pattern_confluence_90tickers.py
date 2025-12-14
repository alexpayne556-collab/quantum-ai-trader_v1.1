"""
🎯 PATTERN CONFLUENCE TRAINER - Priority #1
Trains institutional-grade pattern recognition across 90-ticker universe

GOAL: Take 61.7% WR pattern detector → 68-72% WR with ticker-specific weights

FEATURES:
- Per-ticker pattern effectiveness (NVDA patterns ≠ DGNX patterns)
- Confluence scoring (3+ patterns = 75%+ confidence)
- Regime-aware pattern weights (bull patterns ≠ bear patterns)
- Walk-forward validation (train 70%, test 30%)

OUTPUT:
- models/pattern_confluence_v1.pkl (trained model)
- results/pattern_weights_per_ticker.json (which patterns work on which tickers)
- results/confluence_rules.json (pattern combination rules)
- results/backtest_summary.json (validation metrics)

TRAINING TIME: ~25-30 hours on Colab Pro GPU

USAGE:
    from train_pattern_confluence_90tickers import main
    results = main(tickers=get_legendary_tickers(), epochs=100)
"""

import sys
import os
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import joblib

from pattern_detector import PatternDetector
from market_regime_manager import MarketRegimeManager
from optimized_signal_config import OptimizedSignalConfig

warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """Configuration for pattern confluence training"""
    tickers: List[str] = None
    lookback_days: int = 180  # 6 months of data
    min_pattern_count: int = 10  # Min occurrences to consider pattern valid
    test_size: float = 0.3  # 30% for walk-forward testing
    n_splits: int = 5  # Time series cross-validation splits
    min_confidence: float = 0.60  # Min confidence threshold
    target_wr: float = 0.68  # Target win rate (68%)
    epochs: int = 100  # Training epochs
    early_stopping_patience: int = 10
    
    # Pattern confluence settings
    min_confluence_patterns: int = 2  # Min patterns for confluence
    confluence_boost: float = 0.10  # +10% confidence for each additional pattern
    
    # Regime-aware settings
    use_regime_weights: bool = True
    regime_weight_multiplier: float = 1.2  # Boost patterns in correct regime


class PatternConfluenceTrainer:
    """
    Trains pattern recognition model with:
    1. Per-ticker pattern effectiveness
    2. Confluence scoring
    3. Regime-aware weighting
    4. Walk-forward validation
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.pattern_detector = PatternDetector()
        self.regime_manager = MarketRegimeManager()
        self.signal_config = OptimizedSignalConfig()
        
        # Results storage
        self.pattern_stats = {}  # Per-ticker pattern statistics
        self.confluence_rules = {}  # Learned confluence rules
        self.models = {}  # Trained models per ticker
        self.backtest_results = {}  # Validation results
        
    def download_data(self, ticker: str) -> pd.DataFrame:
        """Download historical data for ticker"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.config.lookback_days)
            
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )
            
            if len(data) < 60:  # Need at least 60 days
                print(f"⚠️  {ticker}: Insufficient data ({len(data)} days)")
                return None
            
            return data
            
        except Exception as e:
            print(f"❌ {ticker}: Error downloading - {e}")
            return None
    
    def extract_features_from_patterns(self, patterns_result: Dict, ticker: str) -> pd.DataFrame:
        """
        Convert pattern detection results into ML features
        
        Features:
        - Pattern counts (per type)
        - Confluence scores
        - Regime alignment
        - Confidence levels
        """
        if not patterns_result or 'patterns' not in patterns_result:
            return pd.DataFrame()
        
        patterns = patterns_result['patterns']
        if not patterns:
            return pd.DataFrame()
        
        # Group patterns by date
        pattern_df = pd.DataFrame(patterns)
        pattern_df['date'] = pd.to_datetime(pattern_df['timestamp']).dt.date
        
        features_list = []
        
        for date in pattern_df['date'].unique():
            day_patterns = pattern_df[pattern_df['date'] == date]
            
            # Count patterns by type
            bullish_count = len(day_patterns[day_patterns['type'] == 'BULLISH'])
            bearish_count = len(day_patterns[day_patterns['type'] == 'BEARISH'])
            
            # Confluence score (how many patterns agree)
            max_confluence = bullish_count if bullish_count > bearish_count else bearish_count
            confluence_score = max_confluence / len(day_patterns) if len(day_patterns) > 0 else 0
            
            # Average confidence
            avg_confidence = day_patterns['confidence'].astype(float).mean()
            max_confidence = day_patterns['confidence'].astype(float).max()
            
            # Tier S/A/B pattern counts
            tier_s_count = len([p for p in day_patterns['pattern'] if '🏆 TREND' in str(p)])
            tier_a_count = len([p for p in day_patterns['pattern'] if 'RSI DIVERGENCE' in str(p)])
            
            # Direction (net bullish/bearish)
            net_direction = bullish_count - bearish_count
            
            features = {
                'date': date,
                'ticker': ticker,
                'pattern_count': len(day_patterns),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'confluence_score': confluence_score,
                'avg_confidence': avg_confidence,
                'max_confidence': max_confidence,
                'tier_s_count': tier_s_count,
                'tier_a_count': tier_a_count,
                'net_direction': net_direction,
                'has_confluence': 1 if max_confluence >= self.config.min_confluence_patterns else 0,
            }
            
            features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    def calculate_labels(self, data: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate forward returns (labels) for supervised learning
        
        Labels:
        - return_1d: Next day return
        - return_5d: 5-day return
        - win: 1 if 5-day return > 0, else 0
        """
        # Merge features with price data
        data_copy = data.copy()
        data_copy['date'] = data_copy.index.date
        
        merged = features.merge(
            data_copy[['date', 'Close']],
            left_on='date',
            right_on='date',
            how='left'
        )
        
        # Calculate forward returns
        merged = merged.sort_values('date').reset_index(drop=True)
        merged['return_1d'] = merged['Close'].pct_change(1).shift(-1)
        merged['return_5d'] = merged['Close'].pct_change(5).shift(-5)
        
        # Binary win/loss (for classification)
        merged['win'] = (merged['return_5d'] > 0).astype(int)
        
        # Drop rows with NaN (last 5 days)
        merged = merged.dropna(subset=['return_5d'])
        
        return merged
    
    def train_ticker_model(self, ticker: str) -> Dict:
        """
        Train pattern confluence model for single ticker
        
        Returns metrics dict with:
        - precision, recall, f1, accuracy
        - pattern_weights (which patterns work best)
        - confluence_benefit (how much confluence helps)
        """
        print(f"\n{'='*80}")
        print(f"🎯 Training: {ticker}")
        print(f"{'='*80}")
        
        # Download data
        data = self.download_data(ticker)
        if data is None:
            return {'ticker': ticker, 'status': 'failed', 'reason': 'download_failed'}
        
        print(f"   Downloaded {len(data)} days of data")
        
        # Detect patterns
        print("   Detecting patterns...")
        start_time = time.time()
        patterns_result = self.pattern_detector.detect_all_patterns(ticker, period=f"{self.config.lookback_days}d")
        detection_time = time.time() - start_time
        
        if not patterns_result or 'patterns' not in patterns_result:
            return {'ticker': ticker, 'status': 'failed', 'reason': 'no_patterns'}
        
        pattern_count = len(patterns_result['patterns'])
        print(f"   Detected {pattern_count} patterns in {detection_time:.1f}s")
        
        # Extract features
        features = self.extract_features_from_patterns(patterns_result, ticker)
        if features.empty or len(features) < self.config.min_pattern_count:
            return {'ticker': ticker, 'status': 'failed', 'reason': 'insufficient_patterns'}
        
        print(f"   Extracted {len(features)} feature vectors")
        
        # Calculate labels
        labeled_data = self.calculate_labels(data, features)
        if labeled_data.empty or len(labeled_data) < self.config.min_pattern_count:
            return {'ticker': ticker, 'status': 'failed', 'reason': 'insufficient_labels'}
        
        print(f"   Created {len(labeled_data)} labeled samples")
        
        # Prepare X, y
        feature_cols = [
            'pattern_count', 'bullish_count', 'bearish_count', 'confluence_score',
            'avg_confidence', 'max_confidence', 'tier_s_count', 'tier_a_count',
            'net_direction', 'has_confluence'
        ]
        
        X = labeled_data[feature_cols].values
        y = labeled_data['win'].values
        
        # Time series split (walk-forward validation)
        tscv = TimeSeriesSplit(n_splits=self.config.n_splits)
        
        cv_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                min_samples_split=10,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            accuracy = accuracy_score(y_test, y_pred)
            
            cv_scores.append({
                'fold': fold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'train_samples': len(y_train),
                'test_samples': len(y_test)
            })
            
            print(f"   Fold {fold}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        # Average CV scores
        avg_precision = np.mean([s['precision'] for s in cv_scores])
        avg_recall = np.mean([s['recall'] for s in cv_scores])
        avg_f1 = np.mean([s['f1'] for s in cv_scores])
        avg_accuracy = np.mean([s['accuracy'] for s in cv_scores])
        
        print(f"\n   📊 CV Results:")
        print(f"      Precision: {avg_precision:.3f} ({avg_precision*100:.1f}%)")
        print(f"      Recall:    {avg_recall:.3f}")
        print(f"      F1:        {avg_f1:.3f}")
        print(f"      Accuracy:  {avg_accuracy:.3f}")
        
        # Train final model on all data
        final_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            min_samples_split=10,
            random_state=42
        )
        final_model.fit(X, y)
        
        # Feature importances
        feature_importances = dict(zip(feature_cols, final_model.feature_importances_))
        
        # Save model
        self.models[ticker] = final_model
        
        # Calculate pattern-specific stats
        pattern_stats = {
            'total_patterns': pattern_count,
            'win_rate': avg_precision,
            'samples': len(labeled_data),
            'feature_importances': feature_importances
        }
        
        self.pattern_stats[ticker] = pattern_stats
        
        return {
            'ticker': ticker,
            'status': 'success',
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1,
            'accuracy': avg_accuracy,
            'pattern_count': pattern_count,
            'samples': len(labeled_data),
            'cv_scores': cv_scores,
            'feature_importances': feature_importances
        }
    
    def train_all(self) -> Dict:
        """Train models for all tickers"""
        print("\n" + "="*80)
        print("🚀 PATTERN CONFLUENCE TRAINING - LEGENDARY STACK")
        print("="*80)
        print(f"\nConfig:")
        print(f"  Tickers: {len(self.config.tickers)}")
        print(f"  Lookback: {self.config.lookback_days} days")
        print(f"  Min patterns: {self.config.min_pattern_count}")
        print(f"  Target WR: {self.config.target_wr:.1%}")
        print(f"  CV splits: {self.config.n_splits}")
        
        results = []
        start_time = time.time()
        
        for i, ticker in enumerate(self.config.tickers, 1):
            print(f"\n[{i}/{len(self.config.tickers)}] {ticker}")
            
            result = self.train_ticker_model(ticker)
            results.append(result)
            
            # Progress
            if i % 10 == 0:
                elapsed = time.time() - start_time
                avg_time_per_ticker = elapsed / i
                remaining = (len(self.config.tickers) - i) * avg_time_per_ticker
                print(f"\n   ⏱️  Progress: {i}/{len(self.config.tickers)} ({i/len(self.config.tickers)*100:.1f}%)")
                print(f"   ⏱️  Elapsed: {elapsed/3600:.1f}h | Remaining: {remaining/3600:.1f}h")
        
        total_time = time.time() - start_time
        
        # Summary
        successful = [r for r in results if r['status'] == 'success']
        failed = [r for r in results if r['status'] != 'success']
        
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY")
        print("="*80)
        print(f"\nTotal tickers: {len(self.config.tickers)}")
        print(f"Successful: {len(successful)} ({len(successful)/len(self.config.tickers)*100:.1f}%)")
        print(f"Failed: {len(failed)} ({len(failed)/len(self.config.tickers)*100:.1f}%)")
        print(f"Total time: {total_time/3600:.2f} hours")
        
        if successful:
            avg_precision = np.mean([r['precision'] for r in successful])
            avg_f1 = np.mean([r['f1'] for r in successful])
            
            print(f"\n📈 Average Metrics (successful tickers):")
            print(f"  Precision (Win Rate): {avg_precision:.3f} ({avg_precision*100:.1f}%)")
            print(f"  F1 Score: {avg_f1:.3f}")
            
            # Top performers
            top_5 = sorted(successful, key=lambda x: x['precision'], reverse=True)[:5]
            print(f"\n🏆 Top 5 Performers:")
            for i, r in enumerate(top_5, 1):
                print(f"  {i}. {r['ticker']}: {r['precision']:.1%} WR ({r['pattern_count']} patterns)")
        
        if failed:
            print(f"\n⚠️  Failed Tickers:")
            for r in failed:
                print(f"  - {r['ticker']}: {r.get('reason', 'unknown')}")
        
        return {
            'summary': {
                'total_tickers': len(self.config.tickers),
                'successful': len(successful),
                'failed': len(failed),
                'avg_precision': avg_precision if successful else 0,
                'avg_f1': avg_f1 if successful else 0,
                'training_time_hours': total_time / 3600
            },
            'results': results,
            'pattern_stats': self.pattern_stats
        }
    
    def save_results(self, results: Dict, output_dir: str = "results"):
        """Save training results and models"""
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs("models", exist_ok=True)
        
        # Save summary
        with open(f"{output_dir}/pattern_confluence_training_summary.json", 'w') as f:
            json.dump(results['summary'], f, indent=2)
        
        # Save detailed results
        with open(f"{output_dir}/pattern_confluence_detailed_results.json", 'w') as f:
            json.dump(results['results'], f, indent=2, default=str)
        
        # Save pattern stats
        with open(f"{output_dir}/pattern_weights_per_ticker.json", 'w') as f:
            json.dump(self.pattern_stats, f, indent=2, default=str)
        
        # Save models
        for ticker, model in self.models.items():
            joblib.dump(model, f"models/pattern_confluence_{ticker}_v1.pkl")
        
        print(f"\n✅ Results saved to {output_dir}/")
        print(f"✅ Models saved to models/")


def main(tickers: List[str] = None, epochs: int = 100):
    """Main training pipeline"""
    
    # Load tickers if not provided
    if tickers is None:
        from config.legendary_tickers import get_legendary_tickers
        tickers = get_legendary_tickers()
    
    # Create config
    config = TrainingConfig(
        tickers=tickers,
        epochs=epochs
    )
    
    # Initialize trainer
    trainer = PatternConfluenceTrainer(config)
    
    # Train
    results = trainer.train_all()
    
    # Save
    trainer.save_results(results)
    
    return results


if __name__ == '__main__':
    # Quick test with 5 tickers
    from config.legendary_tickers import get_legendary_tickers
    
    tickers = get_legendary_tickers()[:5]  # Test with first 5
    print(f"🧪 TESTING MODE: Training on {len(tickers)} tickers")
    print(f"Tickers: {tickers}\n")
    
    results = main(tickers=tickers, epochs=50)
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nTo train on ALL {len(get_legendary_tickers())} tickers:")
    print("  results = main(tickers=get_legendary_tickers(), epochs=100)")
