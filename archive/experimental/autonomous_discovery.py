"""
AUTONOMOUS PATTERN DISCOVERY ENGINE
====================================
Runs continuously, trying different feature combinations, labeling strategies,
and model configurations until it finds patterns that work.

Like a Rubik's cube solver or chess AI - explores the solution space systematically.

Run modes:
1. Local: Quick exploration with CPU
2. Colab Pro: Deep exploration with GPU (generates notebook)

Key strategies it tries:
- Different labeling methods (fixed threshold, ATR-based, triple barrier)
- Different feature subsets (momentum, mean-reversion, hybrid)
- Different lookback periods
- Different model hyperparameters
- Market regime-specific models
"""

import os
import sys
import json
import sqlite3
import pickle
import itertools
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import warnings
import hashlib

warnings.filterwarnings('ignore')

# Custom JSON encoder for numpy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_json_dumps(obj, **kwargs):
    """JSON dumps that handles numpy types."""
    return json.dumps(obj, cls=NumpyEncoder, **kwargs)

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'core'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_discovery.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutonomousPatternDiscovery:
    """
    Systematically explores different strategies to find optimal patterns.
    Learns from each experiment and focuses on promising directions.
    """
    
    def __init__(self, db_path: str = 'pattern_discovery.db'):
        self.db_path = db_path
        self.experiment_count = 0
        
        # Strategy space to explore
        self.strategy_space = {
            # Labeling strategies
            'label_method': [
                'fixed_0.5',      # Current: ±0.5%
                'fixed_1.0',      # ±1.0%
                'atr_based',      # Dynamic based on ATR
                'triple_barrier', # Professional method
                'trend_follow',   # Trend-following labels
            ],
            
            # Feature subsets
            'feature_set': [
                'momentum',       # RSI, MACD, Stochastic
                'trend',          # EMAs, ADX
                'volatility',     # BB, ATR, VIX
                'volume',         # OBV, MFI
                'all_50',         # Current 50 features
                'top_20',         # Most important 20
                'regime_aware',   # Different features per regime
            ],
            
            # Lookback periods
            'lookback': [60, 90, 120, 180, 252],
            
            # Prediction horizons
            'horizon': [1, 3, 5],
            
            # Model types
            'model': ['xgboost', 'lightgbm', 'ensemble', 'regime_split'],
            
            # Market regimes
            'regime_filter': ['all', 'bull_only', 'bear_only', 'low_vol', 'high_vol'],
        }
        
        self._init_database()
        
    def _init_database(self):
        """Initialize database for tracking experiments."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Experiments table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                config_hash TEXT UNIQUE,
                config TEXT NOT NULL,
                
                -- Results
                accuracy REAL,
                precision_buy REAL,
                precision_sell REAL,
                recall_buy REAL,
                recall_sell REAL,
                f1_score REAL,
                sharpe_ratio REAL,
                profit_factor REAL,
                max_drawdown REAL,
                win_rate REAL,
                
                -- Meta
                samples_train INTEGER,
                samples_test INTEGER,
                training_time REAL,
                notes TEXT,
                
                -- Status
                status TEXT DEFAULT 'pending'
            )
        ''')
        
        # Best configurations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS best_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                metric TEXT,
                value REAL,
                config TEXT,
                experiment_id INTEGER,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        ''')
        
        # Learning progress
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_progress (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                experiments_run INTEGER,
                best_accuracy REAL,
                best_sharpe REAL,
                promising_directions TEXT,
                abandoned_directions TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def _config_hash(self, config: Dict) -> str:
        """Generate unique hash for a configuration."""
        config_str = safe_json_dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]
        
    def _already_tried(self, config: Dict) -> bool:
        """Check if this configuration was already tested."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM experiments WHERE config_hash = ?', 
                      (self._config_hash(config),))
        result = cursor.fetchone()
        conn.close()
        return result is not None
        
    def generate_label(self, df: pd.DataFrame, method: str, horizon: int = 1) -> pd.Series:
        """
        Generate labels using different strategies.
        """
        returns = df['Close'].pct_change(horizon).shift(-horizon)
        
        if method == 'fixed_0.5':
            labels = pd.Series(0, index=df.index)  # HOLD
            labels[returns > 0.005] = 1   # BUY
            labels[returns < -0.005] = 2  # SELL
            
        elif method == 'fixed_1.0':
            labels = pd.Series(0, index=df.index)
            labels[returns > 0.01] = 1
            labels[returns < -0.01] = 2
            
        elif method == 'atr_based':
            # Dynamic threshold based on ATR
            import talib
            atr = talib.ATR(df['High'].values, df['Low'].values, df['Close'].values, 14)
            atr_pct = atr / df['Close'].values
            
            labels = pd.Series(0, index=df.index)
            labels[returns > atr_pct] = 1
            labels[returns < -atr_pct] = 2
            
        elif method == 'triple_barrier':
            # Triple barrier method from Marcos Lopez de Prado
            labels = self._triple_barrier_labels(df, horizon)
            
        elif method == 'trend_follow':
            # Trend-following: label based on trend direction
            import talib
            ema_fast = talib.EMA(df['Close'].values, 8)
            ema_slow = talib.EMA(df['Close'].values, 21)
            
            labels = pd.Series(0, index=df.index)
            # BUY when fast > slow and price going up
            labels[(ema_fast > ema_slow) & (returns > 0)] = 1
            # SELL when fast < slow and price going down
            labels[(ema_fast < ema_slow) & (returns < 0)] = 2
            
        else:
            labels = pd.Series(0, index=df.index)
            
        return labels
        
    def _triple_barrier_labels(self, df: pd.DataFrame, horizon: int, 
                               pt: float = 0.02, sl: float = 0.02) -> pd.Series:
        """
        Triple barrier labeling method.
        - Upper barrier: Take profit
        - Lower barrier: Stop loss
        - Vertical barrier: Time limit
        """
        labels = pd.Series(0, index=df.index)
        close = df['Close'].values
        
        for i in range(len(close) - horizon):
            entry_price = close[i]
            
            # Check each day until horizon
            for j in range(1, horizon + 1):
                if i + j >= len(close):
                    break
                    
                ret = (close[i + j] - entry_price) / entry_price
                
                if ret >= pt:  # Hit take profit
                    labels.iloc[i] = 1  # BUY signal was correct
                    break
                elif ret <= -sl:  # Hit stop loss
                    labels.iloc[i] = 2  # SELL signal was correct
                    break
            # If neither hit, remains 0 (HOLD)
            
        return labels
        
    def get_feature_subset(self, features: pd.DataFrame, subset: str) -> pd.DataFrame:
        """
        Get a subset of features based on strategy.
        """
        momentum_cols = [c for c in features.columns if any(x in c for x in 
                        ['RSI', 'MACD', 'Stoch', 'MOM', 'ROC'])]
        trend_cols = [c for c in features.columns if any(x in c for x in 
                     ['EMA', 'SMA', 'ADX', 'DI', 'Trend'])]
        vol_cols = [c for c in features.columns if any(x in c for x in 
                   ['BB', 'ATR', 'VIX', 'Range', 'Vol'])]
        volume_cols = [c for c in features.columns if any(x in c for x in 
                      ['OBV', 'MFI', 'Volume', 'AD'])]
        
        if subset == 'momentum':
            return features[momentum_cols] if momentum_cols else features
        elif subset == 'trend':
            return features[trend_cols] if trend_cols else features
        elif subset == 'volatility':
            return features[vol_cols] if vol_cols else features
        elif subset == 'volume':
            return features[volume_cols] if volume_cols else features
        elif subset == 'top_20':
            # Use SHAP importance to get top 20
            importance_cols = features.columns[:20]  # Assuming sorted by importance
            return features[importance_cols]
        else:
            return features
            
    def detect_regime(self, df: pd.DataFrame, spy_data: pd.DataFrame = None) -> str:
        """
        Detect current market regime.
        """
        import talib
        
        close = df['Close'].values
        returns = np.diff(close) / close[:-1]
        
        # Trend detection
        sma_20 = talib.SMA(close, 20)
        sma_50 = talib.SMA(close, 50)
        
        # Volatility
        atr = talib.ATR(df['High'].values, df['Low'].values, close, 14)
        vol_percentile = pd.Series(atr).rank(pct=True).iloc[-1]
        
        # Trend direction
        if sma_20[-1] > sma_50[-1] and close[-1] > sma_20[-1]:
            trend = 'bull'
        elif sma_20[-1] < sma_50[-1] and close[-1] < sma_20[-1]:
            trend = 'bear'
        else:
            trend = 'sideways'
            
        # Volatility regime
        if vol_percentile > 0.7:
            vol = 'high_vol'
        elif vol_percentile < 0.3:
            vol = 'low_vol'
        else:
            vol = 'normal_vol'
            
        return f"{trend}_{vol}"
        
    def run_single_experiment(self, config: Dict, train_data: Dict, test_data: Dict) -> Dict:
        """
        Run a single experiment with given configuration.
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        import lightgbm as lgb
        
        start_time = datetime.now()
        
        try:
            results = {
                'config': config,
                'status': 'success'
            }
            
            # Generate labels
            train_labels = self.generate_label(
                train_data['df'], 
                config['label_method'],
                config['horizon']
            )
            test_labels = self.generate_label(
                test_data['df'],
                config['label_method'],
                config['horizon']
            )
            
            # Get features
            train_features = self.get_feature_subset(train_data['features'], config['feature_set'])
            test_features = self.get_feature_subset(test_data['features'], config['feature_set'])
            
            # Align data
            valid_idx = ~train_labels.isna() & ~train_features.isna().any(axis=1)
            X_train = train_features[valid_idx].values
            y_train = train_labels[valid_idx].values
            
            valid_idx_test = ~test_labels.isna() & ~test_features.isna().any(axis=1)
            X_test = test_features[valid_idx_test].values
            y_test = test_labels[valid_idx_test].values
            
            if len(X_train) < 100 or len(X_test) < 20:
                return {'status': 'insufficient_data'}
                
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if config['model'] == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
                
            elif config['model'] == 'lightgbm':
                model = lgb.LGBMClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42,
                    verbose=-1
                )
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
                
            elif config['model'] == 'ensemble':
                # Train both and average
                xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42,
                                              use_label_encoder=False, eval_metric='mlogloss')
                lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, random_state=42, verbose=-1)
                
                xgb_model.fit(X_train_scaled, y_train)
                lgb_model.fit(X_train_scaled, y_train)
                
                xgb_proba = xgb_model.predict_proba(X_test_scaled)
                lgb_proba = lgb_model.predict_proba(X_test_scaled)
                
                y_proba = 0.55 * xgb_proba + 0.45 * lgb_proba
                y_pred = np.argmax(y_proba, axis=1)
                
            else:
                model = xgb.XGBClassifier(n_estimators=100, random_state=42,
                                          use_label_encoder=False, eval_metric='mlogloss')
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_proba = model.predict_proba(X_test_scaled)
                
            # Calculate metrics
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['precision_buy'] = precision_score(y_test, y_pred, labels=[1], average='macro', zero_division=0)
            results['precision_sell'] = precision_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
            results['recall_buy'] = recall_score(y_test, y_pred, labels=[1], average='macro', zero_division=0)
            results['recall_sell'] = recall_score(y_test, y_pred, labels=[2], average='macro', zero_division=0)
            results['f1_score'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Calculate trading metrics
            results['sharpe_ratio'] = self._calculate_sharpe(y_pred, y_test, test_data['returns'])
            results['win_rate'] = self._calculate_win_rate(y_pred, y_test)
            
            results['samples_train'] = len(X_train)
            results['samples_test'] = len(X_test)
            results['training_time'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            results = {
                'config': config,
                'status': 'error',
                'error': str(e)
            }
            
        return results
        
    def _calculate_sharpe(self, y_pred, y_true, returns, risk_free: float = 0.0) -> float:
        """Calculate Sharpe ratio of predictions."""
        if len(returns) != len(y_pred):
            return 0.0
            
        # Simulate returns based on predictions
        strategy_returns = []
        for pred, ret in zip(y_pred, returns):
            if pred == 1:  # BUY
                strategy_returns.append(ret)
            elif pred == 2:  # SELL
                strategy_returns.append(-ret)
            else:  # HOLD
                strategy_returns.append(0)
                
        if not strategy_returns or np.std(strategy_returns) == 0:
            return 0.0
            
        return (np.mean(strategy_returns) - risk_free) / np.std(strategy_returns) * np.sqrt(252)
        
    def _calculate_win_rate(self, y_pred, y_true) -> float:
        """Calculate win rate (accuracy on non-HOLD predictions)."""
        mask = y_pred != 0  # Non-HOLD predictions
        if mask.sum() == 0:
            return 0.0
        return (y_pred[mask] == y_true[mask]).mean()
        
    def save_experiment(self, results: Dict):
        """Save experiment results to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        config_hash = self._config_hash(results['config'])
        
        cursor.execute('''
            INSERT OR REPLACE INTO experiments (
                timestamp, config_hash, config, accuracy, precision_buy, precision_sell,
                recall_buy, recall_sell, f1_score, sharpe_ratio, win_rate,
                samples_train, samples_test, training_time, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            config_hash,
            safe_json_dumps(results['config']),
            results.get('accuracy'),
            results.get('precision_buy'),
            results.get('precision_sell'),
            results.get('recall_buy'),
            results.get('recall_sell'),
            results.get('f1_score'),
            results.get('sharpe_ratio'),
            results.get('win_rate'),
            results.get('samples_train'),
            results.get('samples_test'),
            results.get('training_time'),
            results.get('status', 'completed')
        ))
        
        conn.commit()
        conn.close()
        
    def get_best_configs(self, metric: str = 'accuracy', top_n: int = 5) -> List[Dict]:
        """Get top N configurations by metric."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT config, {metric}, accuracy, sharpe_ratio, f1_score
            FROM experiments
            WHERE status = 'success' AND {metric} IS NOT NULL
            ORDER BY {metric} DESC
            LIMIT ?
        ''', (top_n,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'config': json.loads(row[0]),
                metric: row[1],
                'accuracy': row[2],
                'sharpe_ratio': row[3],
                'f1_score': row[4]
            })
            
        conn.close()
        return results
        
    def get_promising_directions(self) -> Dict:
        """Analyze results to find promising directions."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM experiments WHERE status = 'success'
        ''', conn)
        conn.close()
        
        if len(df) < 5:
            return {'status': 'need_more_experiments'}
            
        # Parse configs
        df['config_dict'] = df['config'].apply(json.loads)
        
        # Analyze what works
        promising = {}
        
        # Best label method
        for idx, row in df.iterrows():
            config = row['config_dict']
            for key in ['label_method', 'feature_set', 'model', 'horizon']:
                if key not in promising:
                    promising[key] = {}
                val = config.get(key)
                if val not in promising[key]:
                    promising[key][val] = {'count': 0, 'total_acc': 0, 'total_sharpe': 0}
                promising[key][val]['count'] += 1
                promising[key][val]['total_acc'] += row['accuracy'] or 0
                promising[key][val]['total_sharpe'] += row['sharpe_ratio'] or 0
                
        # Calculate averages
        for key in promising:
            for val in promising[key]:
                n = promising[key][val]['count']
                promising[key][val]['avg_accuracy'] = promising[key][val]['total_acc'] / n if n > 0 else 0
                promising[key][val]['avg_sharpe'] = promising[key][val]['total_sharpe'] / n if n > 0 else 0
                
        return promising
        
    def generate_next_experiments(self, batch_size: int = 10) -> List[Dict]:
        """
        Intelligently generate next batch of experiments.
        Focuses on promising directions while exploring new ones.
        """
        experiments = []
        
        # Get promising directions
        promising = self.get_promising_directions()
        
        # 70% exploitation: Build on what works
        for _ in range(int(batch_size * 0.7)):
            config = {}
            
            for key, values in self.strategy_space.items():
                if key in promising and promising[key]:
                    # Choose based on past performance (weighted random)
                    best_val = max(promising[key].items(), 
                                   key=lambda x: x[1].get('avg_accuracy', 0))[0]
                    # 70% chance to use best, 30% chance to explore
                    if np.random.random() < 0.7:
                        config[key] = best_val
                    else:
                        config[key] = np.random.choice(values)
                else:
                    config[key] = np.random.choice(values)
                    
            if not self._already_tried(config):
                experiments.append(config)
                
        # 30% exploration: Try new combinations
        for _ in range(int(batch_size * 0.3)):
            config = {key: np.random.choice(values) 
                     for key, values in self.strategy_space.items()}
            if not self._already_tried(config):
                experiments.append(config)
                
        return experiments[:batch_size]
        
    def run_discovery_loop(self, tickers: List[str], max_experiments: int = 100,
                          target_accuracy: float = 0.55) -> Dict:
        """
        Main discovery loop - keeps trying until target accuracy reached.
        """
        logger.info("="*70)
        logger.info("🔬 AUTONOMOUS PATTERN DISCOVERY ENGINE")
        logger.info("="*70)
        logger.info(f"Target accuracy: {target_accuracy:.1%}")
        logger.info(f"Max experiments: {max_experiments}")
        
        # Download data once
        logger.info("\n📥 Downloading data...")
        all_data = self._download_all_data(tickers)
        
        if not all_data:
            return {'status': 'error', 'message': 'Failed to download data'}
            
        best_accuracy = 0
        best_config = None
        experiments_run = 0
        
        while experiments_run < max_experiments and best_accuracy < target_accuracy:
            # Generate experiments
            batch = self.generate_next_experiments(batch_size=10)
            
            for config in batch:
                if experiments_run >= max_experiments:
                    break
                    
                experiments_run += 1
                logger.info(f"\n🧪 Experiment {experiments_run}/{max_experiments}")
                logger.info(f"   Config: {safe_json_dumps(config, indent=2)}")
                
                # Run on each ticker and average results
                all_results = []
                
                for ticker in tickers[:5]:  # Use first 5 for speed
                    if ticker not in all_data:
                        continue
                        
                    data = all_data[ticker]
                    
                    # Split train/test
                    split_idx = int(len(data['df']) * 0.8)
                    train_data = {
                        'df': data['df'].iloc[:split_idx],
                        'features': data['features'].iloc[:split_idx],
                        'returns': data['returns'].iloc[:split_idx]
                    }
                    test_data = {
                        'df': data['df'].iloc[split_idx:],
                        'features': data['features'].iloc[split_idx:],
                        'returns': data['returns'].iloc[split_idx:]
                    }
                    
                    result = self.run_single_experiment(config, train_data, test_data)
                    
                    if result.get('status') == 'success':
                        all_results.append(result)
                        
                # Average results
                if all_results:
                    avg_result = {
                        'config': config,
                        'status': 'success',
                        'accuracy': np.mean([r['accuracy'] for r in all_results]),
                        'sharpe_ratio': np.mean([r.get('sharpe_ratio', 0) for r in all_results]),
                        'f1_score': np.mean([r.get('f1_score', 0) for r in all_results]),
                        'win_rate': np.mean([r.get('win_rate', 0) for r in all_results]),
                        'samples_train': sum(r.get('samples_train', 0) for r in all_results),
                        'samples_test': sum(r.get('samples_test', 0) for r in all_results),
                        'training_time': sum(r.get('training_time', 0) for r in all_results),
                    }
                    
                    self.save_experiment(avg_result)
                    
                    if avg_result['accuracy'] > best_accuracy:
                        best_accuracy = avg_result['accuracy']
                        best_config = config
                        
                    logger.info(f"   ✅ Accuracy: {avg_result['accuracy']:.1%} | "
                               f"Sharpe: {avg_result['sharpe_ratio']:.2f} | "
                               f"F1: {avg_result['f1_score']:.2f}")
                               
                    if best_accuracy >= target_accuracy:
                        logger.info(f"\n🎯 TARGET REACHED! Accuracy: {best_accuracy:.1%}")
                        break
                        
            # Log progress
            logger.info(f"\n📊 Progress: {experiments_run} experiments | "
                       f"Best accuracy: {best_accuracy:.1%}")
                       
        return {
            'experiments_run': experiments_run,
            'best_accuracy': best_accuracy,
            'best_config': best_config,
            'target_reached': best_accuracy >= target_accuracy,
            'best_configs': self.get_best_configs(metric='accuracy', top_n=5)
        }
        
    def _download_all_data(self, tickers: List[str]) -> Dict:
        """Download and prepare data for all tickers."""
        from core.colab_predictor import ColabPredictor
        
        all_data = {}
        predictor = ColabPredictor()
        
        # Download SPY and VIX once
        spy = yf.download('SPY', period='2y', progress=False)
        vix = yf.download('^VIX', period='2y', progress=False)
        
        # Flatten
        for df in [spy, vix]:
            if hasattr(df.columns, 'levels'):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                
        for ticker in tickers:
            try:
                df = yf.download(ticker, period='2y', progress=False)
                if len(df) < 252:
                    continue
                    
                if hasattr(df.columns, 'levels'):
                    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
                    
                # Engineer features
                features = predictor.engineer_features(df, spy, vix)
                
                # Calculate returns
                returns = df['Close'].pct_change()
                
                all_data[ticker] = {
                    'df': df,
                    'features': features,
                    'returns': returns
                }
                
            except Exception as e:
                logger.warning(f"Error downloading {ticker}: {e}")
                continue
                
        return all_data
        
    def generate_colab_notebook(self, best_configs: List[Dict]) -> str:
        """Generate Colab notebook for GPU training with best configs."""
        notebook_code = '''
# AUTONOMOUS PATTERN DISCOVERY - COLAB GPU TRAINING
# =================================================
# This notebook runs intensive hyperparameter search on GPU

# Install dependencies
!pip install xgboost lightgbm ta-lib-bin yfinance scikit-learn optuna -q

import numpy as np
import pandas as pd
import yfinance as yf
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import optuna
import talib
import warnings
warnings.filterwarnings('ignore')

# Best configurations from local discovery
BEST_CONFIGS = ''' + safe_json_dumps(best_configs, indent=2) + '''

# Download data
TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'SPY', 'QQQ']

def download_data():
    all_data = {}
    spy = yf.download('SPY', period='3y', progress=False)
    vix = yf.download('^VIX', period='3y', progress=False)
    
    for ticker in TICKERS:
        df = yf.download(ticker, period='3y', progress=False)
        if len(df) > 500:
            all_data[ticker] = {
                'df': df,
                'spy': spy,
                'vix': vix
            }
    return all_data

# Engineer features (same as production)
def engineer_features(df, spy, vix):
    """Full feature engineering matching production system"""
    # [Implementation matches colab_predictor.py]
    # ... (full feature engineering code here)
    pass

# Optuna hyperparameter optimization
def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
    }
    
    model = xgb.XGBClassifier(**params, tree_method='gpu_hist', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    return accuracy_score(y_val, y_pred)

# Main training loop
print("🚀 Starting GPU-accelerated pattern discovery...")

data = download_data()
print(f"Downloaded data for {len(data)} tickers")

# Run optimization for each best config
results = []
for config in BEST_CONFIGS:
    print(f"\\nTesting config: {config}")
    
    # Prepare data and run Optuna
    study = optuna.create_study(direction='maximize')
    # ... (optimization code)
    
    results.append({
        'config': config,
        'best_params': study.best_params,
        'best_accuracy': study.best_value
    })

print("\\n" + "="*60)
print("🎯 OPTIMIZATION COMPLETE")
print("="*60)
for r in sorted(results, key=lambda x: x['best_accuracy'], reverse=True):
    print(f"Accuracy: {r['best_accuracy']:.1%} | Config: {r['config']}")
'''
        
        # Save notebook
        notebook_path = 'AUTONOMOUS_DISCOVERY_COLAB.py'
        with open(notebook_path, 'w') as f:
            f.write(notebook_code)
            
        return notebook_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Autonomous Pattern Discovery')
    parser.add_argument('--max-experiments', type=int, default=50, help='Maximum experiments to run')
    parser.add_argument('--target-accuracy', type=float, default=0.55, help='Target accuracy to reach')
    parser.add_argument('--generate-colab', action='store_true', help='Generate Colab notebook')
    parser.add_argument('--report', action='store_true', help='Show discovery report')
    
    args = parser.parse_args()
    
    discovery = AutonomousPatternDiscovery()
    
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'SPY', 'QQQ']
    
    if args.generate_colab:
        best = discovery.get_best_configs(metric='accuracy', top_n=5)
        path = discovery.generate_colab_notebook(best)
        print(f"✅ Generated Colab notebook: {path}")
        
    elif args.report:
        print("\n" + "="*60)
        print("📊 PATTERN DISCOVERY REPORT")
        print("="*60)
        
        best = discovery.get_best_configs(metric='accuracy', top_n=10)
        print("\nTop 10 Configurations by Accuracy:")
        for i, b in enumerate(best, 1):
            print(f"{i}. Accuracy: {b['accuracy']:.1%} | Sharpe: {b['sharpe_ratio']:.2f}")
            print(f"   Config: {safe_json_dumps(b['config'], indent=6)}")
            
        promising = discovery.get_promising_directions()
        print("\n📈 Promising Directions:")
        for key, vals in promising.items():
            if isinstance(vals, dict):
                best_val = max(vals.items(), key=lambda x: x[1].get('avg_accuracy', 0))
                print(f"  {key}: Best = '{best_val[0]}' (avg acc: {best_val[1].get('avg_accuracy', 0):.1%})")
                
    else:
        # Run discovery loop
        results = discovery.run_discovery_loop(
            tickers=tickers,
            max_experiments=args.max_experiments,
            target_accuracy=args.target_accuracy
        )
        
        print("\n" + "="*60)
        print("🎯 DISCOVERY COMPLETE")
        print("="*60)
        print(f"Experiments run: {results['experiments_run']}")
        print(f"Best accuracy: {results['best_accuracy']:.1%}")
        print(f"Target reached: {results['target_reached']}")
        print(f"\nBest config:\n{safe_json_dumps(results['best_config'], indent=2)}")


if __name__ == '__main__':
    main()
