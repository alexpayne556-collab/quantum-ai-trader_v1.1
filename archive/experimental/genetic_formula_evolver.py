"""
GeneticFormulaEvolver: Evolve trading formulas using DEAP
Discovers complex patterns from 50+ indicators

Optimized for Colab T4 High-RAM
Uses genetic programming to find optimal indicator combinations
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
import warnings
import operator
import random

warnings.filterwarnings('ignore')

# DEAP imports with error handling
try:
    from deap import base, creator, tools, gp, algorithms
    HAS_DEAP = True
except ImportError:
    HAS_DEAP = False
    print("⚠️ DEAP not installed. Run: pip install deap")


class GeneticFormulaEvolver:
    """
    Evolve trading formulas using genetic programming.
    
    Features:
    - 50+ indicator inputs from UltimateFeatureEngine
    - Mathematical primitives: add, sub, mul, div, sqrt, tanh, sin
    - Fitness: correlation with future returns
    - Evolution: 200-500 population, 50-100 generations
    - Output: Top 10 discovered formulas
    """
    
    def __init__(self, features_df: pd.DataFrame, max_features: int = 30):
        """
        Initialize evolver with feature data.
        
        Args:
            features_df: DataFrame with indicator features (from UltimateFeatureEngine)
            max_features: Maximum number of features to use (for speed)
        """
        if not HAS_DEAP:
            raise ImportError("DEAP library required. Install with: pip install deap")
        
        self.features = features_df.dropna()
        
        # Select top features by variance (most informative)
        if len(self.features.columns) > max_features:
            variances = self.features.var().sort_values(ascending=False)
            top_cols = variances.head(max_features).index.tolist()
            self.features = self.features[top_cols]
        
        self.feature_names = list(self.features.columns)
        self.n_features = len(self.feature_names)
        
        # Initialize DEAP
        self._setup_deap()
        
        print(f"✓ GeneticFormulaEvolver initialized with {self.n_features} features")
    
    def _setup_deap(self):
        """Setup DEAP genetic programming primitives."""
        # Create fitness and individual classes (only once)
        if not hasattr(creator, 'FitnessMax'):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, 'Individual'):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Create primitive set
        self.pset = gp.PrimitiveSet("MAIN", self.n_features)
        
        # Rename arguments to feature names
        for i, name in enumerate(self.feature_names):
            # Clean name for valid Python identifier
            clean_name = name.replace(' ', '_').replace('-', '_')
            self.pset.renameArguments(**{f"ARG{i}": clean_name})
        
        # Add mathematical primitives
        self.pset.addPrimitive(operator.add, 2, name='add')
        self.pset.addPrimitive(operator.sub, 2, name='sub')
        self.pset.addPrimitive(operator.mul, 2, name='mul')
        self.pset.addPrimitive(self._protected_div, 2, name='div')
        self.pset.addPrimitive(self._protected_sqrt, 1, name='sqrt')
        self.pset.addPrimitive(np.tanh, 1, name='tanh')
        self.pset.addPrimitive(np.sin, 1, name='sin')
        self.pset.addPrimitive(np.cos, 1, name='cos')
        self.pset.addPrimitive(self._protected_log, 1, name='log')
        self.pset.addPrimitive(np.abs, 1, name='abs')
        self.pset.addPrimitive(operator.neg, 1, name='neg')
        
        # Add ephemeral constants
        self.pset.addEphemeralConstant("rand_const", 
                                        lambda: random.uniform(-1, 1))
        
        # Setup toolbox
        self.toolbox = base.Toolbox()
        
        # Expression generation
        self.toolbox.register("expr", gp.genHalfAndHalf, 
                             pset=self.pset, min_=2, max_=5)
        self.toolbox.register("individual", tools.initIterate, 
                             creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, 
                             list, self.toolbox.individual)
        
        # Genetic operators
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, 
                             expr=self.toolbox.expr, pset=self.pset)
        self.toolbox.register("select", tools.selTournament, tournsize=5)
        
        # Compile function
        self.toolbox.register("compile", gp.compile, pset=self.pset)
    
    @staticmethod
    def _protected_div(a, b):
        """Division protected against divide by zero."""
        try:
            result = a / (b + 1e-10)
            return np.clip(result, -1e6, 1e6)
        except:
            return 0.0
    
    @staticmethod
    def _protected_sqrt(a):
        """Square root protected against negative numbers."""
        return np.sqrt(np.abs(a))
    
    @staticmethod
    def _protected_log(a):
        """Log protected against non-positive numbers."""
        return np.log(np.abs(a) + 1e-10)
    
    def _evaluate_formula(self, individual, future_returns: np.ndarray) -> Tuple[float]:
        """
        Evaluate fitness of a formula.
        
        Fitness = correlation between formula output and future returns
        """
        try:
            # Compile individual to callable function
            func = self.toolbox.compile(expr=individual)
            
            # Calculate predictions for each row
            predictions = []
            feature_values = self.features.values
            
            for row in feature_values:
                try:
                    pred = func(*row)
                    # Handle infinity/nan
                    if np.isnan(pred) or np.isinf(pred):
                        pred = 0.0
                    predictions.append(pred)
                except:
                    predictions.append(0.0)
            
            predictions = np.array(predictions)
            
            # Calculate correlation with future returns
            # Ignore if all same value
            if np.std(predictions) < 1e-10:
                return (-1.0,)
            
            # Pearson correlation
            correlation = np.corrcoef(predictions, future_returns)[0, 1]
            
            if np.isnan(correlation):
                return (-1.0,)
            
            # Return correlation as fitness (higher = better)
            return (max(correlation, -1.0),)
            
        except Exception as e:
            return (-1.0,)
    
    def evolve(self, future_returns: np.ndarray, 
               pop_size: int = 200, 
               generations: int = 50,
               cx_prob: float = 0.7,
               mut_prob: float = 0.3,
               verbose: bool = True) -> Tuple[List, Any]:
        """
        Evolve population to discover trading formulas.
        
        Args:
            future_returns: Target returns to predict
            pop_size: Population size (200-500 recommended)
            generations: Number of generations (50-100 recommended)
            cx_prob: Crossover probability
            mut_prob: Mutation probability
            verbose: Print progress
            
        Returns:
            hall_of_fame: Top 10 best individuals
            logbook: Evolution statistics
        """
        print("\n" + "=" * 60)
        print("🧬 GENETIC FORMULA EVOLUTION")
        print("=" * 60)
        print(f"Population: {pop_size}")
        print(f"Generations: {generations}")
        print(f"Features: {self.n_features}")
        print(f"Samples: {len(self.features)}")
        print("=" * 60)
        
        # Ensure future_returns aligns with features
        if len(future_returns) != len(self.features):
            future_returns = future_returns[:len(self.features)]
        
        # Register evaluation function
        self.toolbox.register("evaluate", self._evaluate_formula, 
                             future_returns=future_returns)
        
        # Create initial population
        population = self.toolbox.population(n=pop_size)
        
        # Hall of fame (keep top 10)
        hof = tools.HallOfFame(10)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        stats.register("min", np.min)
        stats.register("std", np.std)
        
        # Run evolution
        if verbose:
            print("\nEvolution Progress:")
            print("-" * 40)
        
        population, logbook = algorithms.eaSimple(
            population, self.toolbox,
            cxpb=cx_prob,
            mutpb=mut_prob,
            ngen=generations,
            stats=stats,
            halloffame=hof,
            verbose=verbose
        )
        
        print("\n" + "=" * 60)
        print("🏆 TOP DISCOVERED FORMULAS")
        print("=" * 60)
        
        for i, ind in enumerate(hof, 1):
            fitness = ind.fitness.values[0]
            formula_str = str(ind)
            # Truncate if too long
            if len(formula_str) > 80:
                formula_str = formula_str[:77] + "..."
            print(f"\n{i}. Correlation: {fitness:.4f}")
            print(f"   Formula: {formula_str}")
        
        return hof, logbook
    
    def get_formula_predictions(self, individual) -> np.ndarray:
        """
        Get predictions from a evolved formula.
        
        Args:
            individual: Evolved individual from hall of fame
            
        Returns:
            Array of predictions
        """
        func = self.toolbox.compile(expr=individual)
        predictions = []
        
        for row in self.features.values:
            try:
                pred = func(*row)
                if np.isnan(pred) or np.isinf(pred):
                    pred = 0.0
                predictions.append(pred)
            except:
                predictions.append(0.0)
        
        return np.array(predictions)
    
    def formula_to_string(self, individual) -> str:
        """Convert evolved formula to readable string."""
        return str(individual)
    
    def create_ensemble_signal(self, hof: List, 
                               threshold: float = 0.0) -> np.ndarray:
        """
        Create ensemble signal from top formulas.
        
        Args:
            hof: Hall of fame (list of best individuals)
            threshold: Threshold for buy signal
            
        Returns:
            Binary signal array (1=buy, 0=hold)
        """
        # Get predictions from all formulas
        all_preds = []
        for ind in hof:
            preds = self.get_formula_predictions(ind)
            all_preds.append(preds)
        
        # Average predictions
        avg_pred = np.mean(all_preds, axis=0)
        
        # Generate binary signal
        signal = (avg_pred > threshold).astype(int)
        
        return signal


def evolve_trading_formulas(features_df: pd.DataFrame, 
                            returns: np.ndarray,
                            pop_size: int = 200,
                            generations: int = 50) -> Dict:
    """
    Convenience function to evolve trading formulas.
    
    Args:
        features_df: DataFrame with indicator features
        returns: Future returns to predict
        pop_size: Population size
        generations: Number of generations
        
    Returns:
        Dictionary with results
    """
    evolver = GeneticFormulaEvolver(features_df)
    hof, logbook = evolver.evolve(returns, pop_size, generations)
    
    # Generate ensemble signal
    signal = evolver.create_ensemble_signal(hof)
    
    return {
        'hall_of_fame': hof,
        'logbook': logbook,
        'evolver': evolver,
        'ensemble_signal': signal,
        'top_formula': str(hof[0]) if hof else None,
        'top_correlation': hof[0].fitness.values[0] if hof else 0.0
    }


# Quick test
if __name__ == "__main__":
    print("🚀 GeneticFormulaEvolver Test")
    print("=" * 60)
    
    if not HAS_DEAP:
        print("❌ DEAP not installed. Run: pip install deap")
        exit(1)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Simulate indicator features
    features = pd.DataFrame({
        'RSI': np.random.uniform(20, 80, n_samples),
        'MACD': np.random.normal(0, 1, n_samples),
        'EMA_Ribbon': np.random.uniform(-0.1, 0.1, n_samples),
        'ADX': np.random.uniform(10, 50, n_samples),
        'Volume_Ratio': np.random.uniform(0.5, 2.0, n_samples),
        'BB_Position': np.random.uniform(0, 1, n_samples),
        'ATR_Ratio': np.random.uniform(0.01, 0.05, n_samples),
        'Return_1d': np.random.normal(0, 0.02, n_samples),
    })
    
    # Create synthetic target (correlated with some features)
    future_returns = (
        0.3 * features['RSI'] / 100 +
        0.2 * features['MACD'] +
        0.1 * features['EMA_Ribbon'] +
        np.random.normal(0, 0.5, n_samples)
    )
    
    # Run evolution (small test)
    evolver = GeneticFormulaEvolver(features)
    hof, logbook = evolver.evolve(
        future_returns.values,
        pop_size=50,
        generations=10,
        verbose=True
    )
    
    print("\n✅ GeneticFormulaEvolver test complete!")
    print(f"Top formula correlation: {hof[0].fitness.values[0]:.4f}")
