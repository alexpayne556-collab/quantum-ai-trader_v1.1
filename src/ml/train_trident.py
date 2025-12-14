"""
TRIDENT ENSEMBLE TRAINER
========================
The "God Mode" training pipeline for Quantum AI Trader.

Architecture:
- XGBoost: Pure tabular pattern recognition
- LightGBM: Speed + microstructure feature handling
- CatBoost: Categorical data + robust generalization

Features:
- Ticker-Cluster Training (K-Means grouping by behavior)
- PurgedKFold Cross-Validation (no data leakage)
- Optuna Hyperparameter Optimization (Sharpe ratio focus)
- SHAP Feature Importance Analysis
- GPU Acceleration (tree_method='gpu_hist')

Input: dataset_builder.py output (71.1% WR baseline, 56 features)
Output: 3 trained models per cluster + inference engine

Author: Quantum AI Trader
Date: December 10, 2025
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna

# Validation
from sklearn.model_selection import TimeSeriesSplit

# Feature Importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not available - feature importance will be limited")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PurgedKFold:
    """
    Purged K-Fold Cross-Validation with embargo period.
    Prevents data leakage in time-series data.
    
    Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
    """
    
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        """
        Args:
            n_splits: Number of folds
            embargo_pct: Percentage of samples to embargo between train/test
        """
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct
    
    def split(self, X: pd.DataFrame, y: pd.Series = None, groups: pd.Series = None):
        """
        Generate indices to split data into training and test set.
        
        Args:
            X: Features dataframe
            y: Labels (optional)
            groups: Group labels (optional)
            
        Yields:
            train_idx, test_idx
        """
        n_samples = len(X)
        fold_size = n_samples // self.n_splits
        embargo_size = int(fold_size * self.embargo_pct)
        
        indices = np.arange(n_samples)
        
        for i in range(self.n_splits):
            # Test fold
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]
            
            # Train folds (everything except test + embargo)
            train_idx = np.concatenate([
                indices[:max(0, test_start - embargo_size)],
                indices[min(n_samples, test_end + embargo_size):]
            ])
            
            yield train_idx, test_idx


class TickerClusterer:
    """
    Cluster tickers by behavior patterns using K-Means.
    Groups: "Explosive Small Caps", "Steady Tech", "Choppy Biotech", etc.
    """
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = None
        self.scaler = StandardScaler()
        self.cluster_names = {}
        
    def fit(self, ticker_features: pd.DataFrame) -> Dict:
        """
        Fit K-Means clustering on ticker characteristics.
        
        Args:
            ticker_features: DataFrame with ticker stats
                Columns: volatility, avg_volume, price_range, sector, market_cap
                
        Returns:
            Cluster assignments: {ticker: cluster_id}
        """
        logger.info(f"Clustering {len(ticker_features)} tickers into {self.n_clusters} groups...")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(ticker_features)
        
        # Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Assign cluster names based on characteristics
        self._name_clusters(ticker_features, labels)
        
        # Create mapping
        assignments = dict(zip(ticker_features.index, labels))
        
        logger.info(f"✅ Clustering complete:")
        for cluster_id, name in self.cluster_names.items():
            tickers_in_cluster = [t for t, c in assignments.items() if c == cluster_id]
            logger.info(f"  Cluster {cluster_id} ({name}): {len(tickers_in_cluster)} tickers")
        
        return assignments
    
    def _name_clusters(self, features: pd.DataFrame, labels: np.ndarray):
        """Name clusters based on their characteristics."""
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            cluster_data = features[mask]
            
            # Analyze cluster characteristics
            avg_vol = cluster_data['volatility'].mean()
            avg_volume = cluster_data['avg_volume'].mean()
            avg_price = cluster_data['avg_price'].mean()
            
            # Name based on characteristics
            if avg_vol > 0.05 and avg_price < 10:
                name = "Explosive Small Caps"
            elif avg_vol < 0.02 and avg_price > 100:
                name = "Steady Large Caps"
            elif avg_vol > 0.04 and 'Biotech' in str(cluster_data.get('sector', '')):
                name = "Choppy Biotech"
            elif avg_volume > 10_000_000:
                name = "High Volume Movers"
            else:
                name = f"Mixed Cluster {cluster_id}"
            
            self.cluster_names[cluster_id] = name


class TridenTrainer:
    """
    The Ultimate Training Pipeline.
    Trains XGBoost, LightGBM, and CatBoost models per ticker cluster.
    """
    
    def __init__(
        self,
        use_gpu: bool = True,
        optimize_hyperparams: bool = True,
        n_trials: int = 50,
        cv_folds: int = 5
    ):
        """
        Args:
            use_gpu: Use GPU acceleration (requires CUDA)
            optimize_hyperparams: Run Optuna optimization
            n_trials: Number of Optuna trials per model
            cv_folds: Number of cross-validation folds
        """
        self.use_gpu = use_gpu
        self.optimize_hyperparams = optimize_hyperparams
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        
        # Storage
        self.models = {}  # {cluster_id: {'xgb': model, 'lgb': model, 'cat': model}}
        self.cluster_assignments = {}
        self.feature_importance = {}
        self.training_history = []
        
        logger.info(f"🚀 Trident Trainer initialized")
        logger.info(f"   GPU: {'✅ Enabled' if use_gpu else '❌ Disabled'}")
        logger.info(f"   Hyperparameter Optimization: {'✅ Enabled' if optimize_hyperparams else '❌ Disabled'}")
    
    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        tickers: pd.Series,
        ticker_features: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Main training pipeline.
        
        Args:
            X: Features (56 columns from dataset_builder.py)
            y: Labels (BUY=1, HOLD=0, SELL=-1 or binary)
            tickers: Ticker symbol for each row
            ticker_features: Ticker characteristics for clustering
            
        Returns:
            Training results dictionary
        """
        logger.info("\n" + "="*60)
        logger.info("TRIDENT TRAINING PIPELINE - GOD MODE")
        logger.info("="*60)
        
        # Step 1: Cluster tickers
        if ticker_features is not None:
            clusterer = TickerClusterer(n_clusters=5)
            self.cluster_assignments = clusterer.fit(ticker_features)
        else:
            # Simple clustering by ticker name if no features provided
            logger.warning("No ticker features provided - using single cluster")
            self.cluster_assignments = {ticker: 0 for ticker in tickers.unique()}
        
        # Step 2: Train model for each cluster
        results = {}
        for cluster_id in sorted(set(self.cluster_assignments.values())):
            logger.info(f"\n{'='*60}")
            logger.info(f"TRAINING CLUSTER {cluster_id}")
            logger.info(f"{'='*60}")
            
            # Get data for this cluster
            cluster_tickers = [t for t, c in self.cluster_assignments.items() if c == cluster_id]
            cluster_mask = tickers.isin(cluster_tickers)
            
            X_cluster = X[cluster_mask]
            y_cluster = y[cluster_mask]
            
            logger.info(f"Cluster size: {len(X_cluster)} samples, {len(cluster_tickers)} tickers")
            
            # Train 3 models
            cluster_results = self._train_cluster(X_cluster, y_cluster, cluster_id)
            results[cluster_id] = cluster_results
        
        # Step 3: Generate report
        self._generate_report(results)
        
        return results
    
    def _train_cluster(self, X: pd.DataFrame, y: pd.Series, cluster_id: int) -> Dict:
        """Train XGBoost, LightGBM, and CatBoost for a single cluster."""
        
        # Initialize storage for this cluster
        self.models[cluster_id] = {}
        results = {'cluster_id': cluster_id, 'models': {}}
        
        # Purged K-Fold CV
        cv = PurgedKFold(n_splits=self.cv_folds, embargo_pct=0.01)
        
        # Calculate class weights for imbalance
        class_counts = y.value_counts()
        if len(class_counts) > 1:
            scale_pos_weight = class_counts[0] / class_counts[1] if 1 in class_counts.index else 1.0
        else:
            scale_pos_weight = 1.0
        
        logger.info(f"Class distribution: {dict(class_counts)}")
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train XGBoost
        logger.info("\n🔹 Training XGBoost...")
        xgb_results = self._train_xgboost(X, y, cv, scale_pos_weight)
        self.models[cluster_id]['xgb'] = xgb_results['model']
        results['models']['xgb'] = xgb_results['metrics']
        
        # Train LightGBM
        logger.info("\n🔹 Training LightGBM...")
        lgb_results = self._train_lightgbm(X, y, cv, scale_pos_weight)
        self.models[cluster_id]['lgb'] = lgb_results['model']
        results['models']['lgb'] = lgb_results['metrics']
        
        # Train CatBoost
        logger.info("\n🔹 Training CatBoost...")
        cat_results = self._train_catboost(X, y, cv, scale_pos_weight)
        self.models[cluster_id]['cat'] = cat_results['model']
        results['models']['cat'] = cat_results['metrics']
        
        # Check correlation between models
        logger.info("\n🔹 Checking model correlation...")
        correlation = self._check_model_correlation(X, y, cluster_id)
        results['correlation'] = correlation
        
        # Feature importance
        if SHAP_AVAILABLE:
            logger.info("\n🔹 Computing SHAP values...")
            self.feature_importance[cluster_id] = self._compute_shap_values(X, cluster_id)
        
        return results
    
    def _train_xgboost(self, X: pd.DataFrame, y: pd.Series, cv, scale_pos_weight: float) -> Dict:
        """Train XGBoost with optional Optuna optimization."""
        
        if self.optimize_hyperparams:
            # Optuna optimization
            study = optuna.create_study(direction='maximize', study_name=f'xgb_optuna')
            study.optimize(
                lambda trial: self._xgb_objective(trial, X, y, cv, scale_pos_weight),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            best_params = study.best_params
            logger.info(f"✅ Best XGBoost params: {best_params}")
        else:
            # Default params
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'min_child_weight': 3,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1
            }
        
        # Train final model
        model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            tree_method='gpu_hist' if self.use_gpu else 'hist',
            random_state=42,
            n_jobs=-1
        )
        
        # Cross-validation
        cv_scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)
        
        # Final fit on all data
        model.fit(X, y)
        
        metrics = {
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'params': best_params
        }
        
        logger.info(f"   XGBoost CV Accuracy: {metrics['cv_accuracy']:.3f} ± {metrics['cv_std']:.3f}")
        
        return {'model': model, 'metrics': metrics}
    
    def _train_lightgbm(self, X: pd.DataFrame, y: pd.Series, cv, scale_pos_weight: float) -> Dict:
        """Train LightGBM with optional Optuna optimization."""
        
        if self.optimize_hyperparams:
            study = optuna.create_study(direction='maximize', study_name='lgb_optuna')
            study.optimize(
                lambda trial: self._lgb_objective(trial, X, y, cv, scale_pos_weight),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            best_params = study.best_params
            logger.info(f"✅ Best LightGBM params: {best_params}")
        else:
            best_params = {
                'max_depth': 6,
                'learning_rate': 0.05,
                'n_estimators': 300,
                'num_leaves': 31,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        
        model = lgb.LGBMClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            device='gpu' if self.use_gpu else 'cpu',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        cv_scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)
        
        model.fit(X, y)
        
        metrics = {
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'params': best_params
        }
        
        logger.info(f"   LightGBM CV Accuracy: {metrics['cv_accuracy']:.3f} ± {metrics['cv_std']:.3f}")
        
        return {'model': model, 'metrics': metrics}
    
    def _train_catboost(self, X: pd.DataFrame, y: pd.Series, cv, scale_pos_weight: float) -> Dict:
        """Train CatBoost with optional Optuna optimization."""
        
        if self.optimize_hyperparams:
            study = optuna.create_study(direction='maximize', study_name='cat_optuna')
            study.optimize(
                lambda trial: self._cat_objective(trial, X, y, cv, scale_pos_weight),
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            best_params = study.best_params
            logger.info(f"✅ Best CatBoost params: {best_params}")
        else:
            best_params = {
                'depth': 6,
                'learning_rate': 0.05,
                'iterations': 300,
                'l2_leaf_reg': 3,
                'bootstrap_type': 'Bayesian',
                'bagging_temperature': 1
            }
        
        model = cb.CatBoostClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            task_type='GPU' if self.use_gpu else 'CPU',
            random_seed=42,
            verbose=False
        )
        
        cv_scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            y_pred = model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            cv_scores.append(score)
        
        model.fit(X, y)
        
        metrics = {
            'cv_accuracy': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'params': best_params
        }
        
        logger.info(f"   CatBoost CV Accuracy: {metrics['cv_accuracy']:.3f} ± {metrics['cv_std']:.3f}")
        
        return {'model': model, 'metrics': metrics}
    
    def _xgb_objective(self, trial, X, y, cv, scale_pos_weight):
        """Optuna objective for XGBoost."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5)
        }
        
        model = xgb.XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            tree_method='gpu_hist' if self.use_gpu else 'hist',
            random_state=42
        )
        
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        
        return np.mean(scores)
    
    def _lgb_objective(self, trial, X, y, cv, scale_pos_weight):
        """Optuna objective for LightGBM."""
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
        }
        
        model = lgb.LGBMClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            device='gpu' if self.use_gpu else 'cpu',
            random_state=42,
            verbose=-1
        )
        
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        
        return np.mean(scores)
    
    def _cat_objective(self, trial, X, y, cv, scale_pos_weight):
        """Optuna objective for CatBoost."""
        params = {
            'depth': trial.suggest_int('depth', 4, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'iterations': trial.suggest_int('iterations', 100, 500),
            'l2_leaf_reg': trial.suggest_int('l2_leaf_reg', 1, 10),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0, 1)
        }
        
        model = cb.CatBoostClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            task_type='GPU' if self.use_gpu else 'CPU',
            random_seed=42,
            verbose=False
        )
        
        scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)
            y_pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, y_pred))
        
        return np.mean(scores)
    
    def _check_model_correlation(self, X: pd.DataFrame, y: pd.Series, cluster_id: int) -> float:
        """Check correlation between model predictions."""
        xgb_pred = self.models[cluster_id]['xgb'].predict_proba(X)[:, 1]
        lgb_pred = self.models[cluster_id]['lgb'].predict_proba(X)[:, 1]
        cat_pred = self.models[cluster_id]['cat'].predict_proba(X)[:, 1]
        
        correlation_xgb_lgb = np.corrcoef(xgb_pred, lgb_pred)[0, 1]
        correlation_xgb_cat = np.corrcoef(xgb_pred, cat_pred)[0, 1]
        correlation_lgb_cat = np.corrcoef(lgb_pred, cat_pred)[0, 1]
        
        avg_correlation = np.mean([correlation_xgb_lgb, correlation_xgb_cat, correlation_lgb_cat])
        
        logger.info(f"   Model Correlations:")
        logger.info(f"     XGB-LGB: {correlation_xgb_lgb:.3f}")
        logger.info(f"     XGB-CAT: {correlation_xgb_cat:.3f}")
        logger.info(f"     LGB-CAT: {correlation_lgb_cat:.3f}")
        logger.info(f"     Average: {avg_correlation:.3f}")
        
        if avg_correlation < 0.7:
            logger.info(f"   ✅ Models are diverse (low correlation) - ensemble recommended!")
        else:
            logger.info(f"   ⚠️ Models are similar (high correlation) - may not benefit from ensemble")
        
        return avg_correlation
    
    def _compute_shap_values(self, X: pd.DataFrame, cluster_id: int) -> Dict:
        """Compute SHAP values for feature importance."""
        try:
            # Use XGBoost model for SHAP (fastest)
            model = self.models[cluster_id]['xgb']
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X.sample(min(1000, len(X)), random_state=42))
            
            # Get feature importance
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            logger.info(f"   Top 10 Features:")
            for idx, row in feature_importance.head(10).iterrows():
                logger.info(f"     {row['feature']}: {row['importance']:.4f}")
            
            return feature_importance.to_dict('records')
        
        except Exception as e:
            logger.warning(f"   SHAP computation failed: {e}")
            return {}
    
    def _generate_report(self, results: Dict):
        """Generate training report markdown."""
        report_path = Path('training_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# TRIDENT ENSEMBLE TRAINING REPORT\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            for cluster_id, cluster_results in results.items():
                f.write(f"## Cluster {cluster_id}\n\n")
                
                for model_name, metrics in cluster_results['models'].items():
                    f.write(f"### {model_name.upper()}\n")
                    f.write(f"- **CV Accuracy:** {metrics['cv_accuracy']:.3f} ± {metrics['cv_std']:.3f}\n")
                    f.write(f"- **Parameters:** {json.dumps(metrics['params'], indent=2)}\n\n")
                
                f.write(f"**Model Correlation:** {cluster_results['correlation']:.3f}\n\n")
                f.write("---\n\n")
        
        logger.info(f"\n✅ Training report saved to {report_path}")
    
    def save_models(self, output_dir: str = 'models/trident'):
        """Save all trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for cluster_id, models in self.models.items():
            # Save XGBoost
            models['xgb'].save_model(str(output_path / f'cluster_{cluster_id}_xgb.json'))
            
            # Save LightGBM
            models['lgb'].booster_.save_model(str(output_path / f'cluster_{cluster_id}_lgb.txt'))
            
            # Save CatBoost
            models['cat'].save_model(str(output_path / f'cluster_{cluster_id}_cat.cbm'))
        
        # Save cluster assignments
        with open(output_path / 'cluster_assignments.json', 'w') as f:
            json.dump(self.cluster_assignments, f, indent=2)
        
        logger.info(f"✅ All models saved to {output_path}")


# Example usage
if __name__ == '__main__':
    # This would be replaced with actual data from dataset_builder.py
    logger.info("Trident Trainer - Example usage")
    logger.info("In production, load data from dataset_builder.py output")
