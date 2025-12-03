"""
🔬 BAYESIAN HYPERPARAMETER OPTIMIZATION
=======================================
3-10x faster than grid search for XGBoost/LightGBM optimization.

Uses Optuna for intelligent parameter space exploration.
Based on Perplexity research findings.
"""

import optuna
import numpy as np
import pandas as pd
from typing import Dict, Callable
import logging

logger = logging.getLogger("BayesianOpt")

class BayesianOptimizer:
    """
    Bayesian optimization for trading model hyperparameters.
    
    Much faster than grid search:
    - Grid search: tries every combination
    - Bayesian: learns which parameters work and focuses there
    
    Result: 3-10x speedup with better parameters
    """
    
    # Optimal ranges from research
    LIGHTGBM_SPACE = {
        'max_depth': (9, 12),  # Sweet spot for patterns
        'learning_rate': (0.01, 0.1, 'log'),  # Log scale
        'num_leaves': (20, 100),
        'subsample': (0.6, 0.9),
        'colsample_bytree': (0.6, 0.9),
        'min_child_weight': (5, 8),  # Financial data specific
        'reg_alpha': (0.1, 1.0),  # L1 regularization
        'reg_lambda': (1, 10),  # L2 regularization
    }
    
    XGBOOST_SPACE = {
        'max_depth': (9, 12),
        'learning_rate': (0.01, 0.1, 'log'),
        'subsample': (0.6, 0.9),
        'colsample_bytree': (0.6, 0.9),
        'min_child_weight': (5, 8),
        'gamma': (0, 5),
        'reg_alpha': (0.1, 1.0),
        'reg_lambda': (1, 10),
    }
    
    def __init__(self, model_type: str = 'lightgbm'):
        self.model_type = model_type
        self.study = None
        self.best_params = None
        
        logger.info(f"🔬 Bayesian Optimizer initialized for {model_type}")
    
    def optimize(self, 
                 objective_func: Callable,  # Function to optimize
                 n_trials: int = 100,  # Number of optimization iterations
                 timeout: int = 3600,  # Max time (seconds)
                 direction: str = 'maximize') -> Dict:
        """
        Run Bayesian optimization.
        
        Args:
            objective_func: Function(trial) -> score to maximize/minimize
            n_trials: Number of trials (100 is good balance)
            timeout: Max time in seconds
            direction: 'maximize' or 'minimize'
            
        Returns:
            Best parameters found
        """
        
        logger.info(f"🚀 Starting Bayesian optimization")
        logger.info(f"   Trials: {n_trials}")
        logger.info(f"   Timeout: {timeout}s")
        logger.info(f"   Direction: {direction}")
        
        # Create study
        self.study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=42),  # Tree-structured Parzen Estimator
            pruner=optuna.pruners.MedianPruner()  # Stop bad trials early
        )
        
        # Run optimization
        self.study.optimize(
            objective_func,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        # Get best parameters
        self.best_params = self.study.best_params
        
        logger.info(f"\n✅ Optimization complete!")
        logger.info(f"   Best value: {self.study.best_value:.6f}")
        logger.info(f"   Best params:")
        for key, value in self.best_params.items():
            logger.info(f"      {key}: {value}")
        
        return self.best_params
    
    def create_objective(self, 
                        train_func: Callable,  # Function(params) -> model
                        eval_func: Callable,   # Function(model, data) -> score
                        train_data,
                        val_data) -> Callable:
        """
        Create objective function for optimization.
        
        Args:
            train_func: Function that trains model with params
            eval_func: Function that evaluates model
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Objective function for Optuna
        """
        
        param_space = (self.LIGHTGBM_SPACE if self.model_type == 'lightgbm' 
                      else self.XGBOOST_SPACE)
        
        def objective(trial):
            # Sample parameters from space
            params = {}
            for name, bounds in param_space.items():
                if len(bounds) == 2:
                    # Integer or float range
                    if isinstance(bounds[0], int):
                        params[name] = trial.suggest_int(name, bounds[0], bounds[1])
                    else:
                        params[name] = trial.suggest_float(name, bounds[0], bounds[1])
                elif len(bounds) == 3 and bounds[2] == 'log':
                    # Log scale (for learning rate)
                    params[name] = trial.suggest_float(
                        name, bounds[0], bounds[1], log=True
                    )
            
            # Train model
            model = train_func(params, train_data)
            
            # Evaluate
            score = eval_func(model, val_data)
            
            return score
        
        return objective
    
    def get_importance(self) -> Dict[str, float]:
        """Get parameter importance scores"""
        if self.study is None:
            return {}
        
        importance = optuna.importance.get_param_importances(self.study)
        
        logger.info("\n📊 Parameter Importance:")
        for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {param}: {score:.4f}")
        
        return importance
    
    def plot_optimization_history(self):
        """Visualize optimization progress"""
        if self.study is None:
            return None
        
        import plotly.graph_objects as go
        
        trials = self.study.trials
        values = [t.value for t in trials if t.value is not None]
        
        # Best value so far at each trial
        best_so_far = []
        current_best = float('-inf') if self.study.direction == optuna.study.StudyDirection.MAXIMIZE else float('inf')
        
        for val in values:
            if self.study.direction == optuna.study.StudyDirection.MAXIMIZE:
                current_best = max(current_best, val)
            else:
                current_best = min(current_best, val)
            best_so_far.append(current_best)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=values,
            mode='markers',
            name='Trial Value',
            marker=dict(size=5, color='lightblue')
        ))
        
        fig.add_trace(go.Scatter(
            y=best_so_far,
            mode='lines',
            name='Best Value',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Bayesian Optimization History',
            xaxis_title='Trial',
            yaxis_title='Objective Value',
            template='plotly_dark'
        )
        
        return fig
    
    def plot_param_importances(self):
        """Visualize parameter importances"""
        if self.study is None:
            return None
        
        import plotly.graph_objects as go
        
        importance = self.get_importance()
        
        fig = go.Figure(go.Bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation='h',
            marker=dict(color='lightgreen')
        ))
        
        fig.update_layout(
            title='Parameter Importances',
            xaxis_title='Importance',
            yaxis_title='Parameter',
            template='plotly_dark'
        )
        
        return fig

# ============================================================================
# ENSEMBLE WEIGHT OPTIMIZATION
# ============================================================================

class EnsembleWeightOptimizer(BayesianOptimizer):
    """
    Optimize ensemble weights using Bayesian optimization.
    
    Much better than manual tuning!
    """
    
    def __init__(self, module_names: list):
        super().__init__('ensemble')
        self.module_names = module_names
        
        # Create parameter space for weights
        self.weight_space = {
            f'weight_{name}': (0.01, 0.50) for name in module_names
        }
    
    def optimize_weights(self, 
                        backtest_func: Callable,  # Function(weights) -> sharpe_ratio
                        n_trials: int = 50) -> Dict[str, float]:
        """
        Find optimal ensemble weights.
        
        Args:
            backtest_func: Function that backtests with given weights
            n_trials: Number of optimization trials
            
        Returns:
            Optimal weights dictionary
        """
        
        def objective(trial):
            # Sample weights
            weights = {}
            for name in self.module_names:
                weights[name] = trial.suggest_float(
                    f'weight_{name}', 0.01, 0.50
                )
            
            # Normalize to sum to 1.0
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}
            
            # Evaluate
            sharpe_ratio = backtest_func(weights)
            
            return sharpe_ratio
        
        # Run optimization
        self.optimize(
            objective_func=objective,
            n_trials=n_trials,
            direction='maximize'
        )
        
        # Get normalized weights
        raw_weights = {
            name: self.best_params[f'weight_{name}']
            for name in self.module_names
        }
        total = sum(raw_weights.values())
        optimal_weights = {k: v/total for k, v in raw_weights.items()}
        
        logger.info("\n🎯 Optimal Ensemble Weights:")
        for name, weight in sorted(optimal_weights.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"   {name}: {weight:.3f}")
        
        return optimal_weights

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

def example_train_function(params, train_data):
    """Example training function"""
    from lightgbm import LGBMClassifier
    
    model = LGBMClassifier(**params, n_estimators=100, random_state=42)
    model.fit(train_data['X'], train_data['y'])
    
    return model

def example_eval_function(model, val_data):
    """Example evaluation function"""
    predictions = model.predict_proba(val_data['X'])[:, 1]
    
    # Calculate Sharpe ratio or custom metric
    returns = (predictions > 0.5).astype(int) * val_data['y']
    sharpe = np.mean(returns) / (np.std(returns) + 1e-6)
    
    return sharpe

if __name__ == '__main__':
    # Example: Optimize LightGBM parameters
    optimizer = BayesianOptimizer('lightgbm')
    
    # Create dummy data
    train_data = {
        'X': np.random.randn(1000, 20),
        'y': np.random.randint(0, 2, 1000)
    }
    val_data = {
        'X': np.random.randn(200, 20),
        'y': np.random.randint(0, 2, 200)
    }
    
    # Create objective
    objective = optimizer.create_objective(
        train_func=example_train_function,
        eval_func=example_eval_function,
        train_data=train_data,
        val_data=val_data
    )
    
    # Optimize
    best_params = optimizer.optimize(objective, n_trials=20)
    
    # Show importance
    optimizer.get_importance()

