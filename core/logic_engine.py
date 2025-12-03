"""
LOGIC ENGINE: SYMBOLIC REGRESSION & GENETIC PROGRAMMING
=========================================================
Find the EXACT mathematical formula for trading success.

Why Symbolic Regression:
- Standard ML is a "Black Box" - gives answer but no explanation
- Symbolic Regression evolves READABLE math formulas
- You can audit, understand, and trust the rules

Example Output:
Instead of: model.predict(features) → 0.73
You get: IF (RSI > 70 AND MACD_Cross < 0) THEN Sell_Prob = 0.85

Research Finding:
- PySR can discover physical laws from data
- Applied to trading: finds exact equations governing price movements
- Example: Price_Change = a × Volume^0.5 × Trend + b × VIX^2

Tools:
1. PySR (Parallel Symbolic Regression) - State of the art
2. DEAP (Genetic Programming) - Flexible evolution of strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import warnings
warnings.filterwarnings('ignore')

# Try to import PySR
try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False
    print("Note: PySR not installed. Install with: pip install pysr")

# Try to import DEAP
try:
    from deap import base, creator, tools, gp, algorithms
    import operator
    DEAP_AVAILABLE = True
except ImportError:
    DEAP_AVAILABLE = False
    print("Note: DEAP not installed. Install with: pip install deap")


class SymbolicRegressionEngine:
    """
    Discover mathematical formulas that predict price movement
    
    Uses PySR (Julia backend) for state-of-the-art symbolic regression
    """
    
    def __init__(self, verbose: bool = True, max_complexity: int = 20):
        self.verbose = verbose
        self.max_complexity = max_complexity
        self.model = None
        self.best_equation = None
        self.equations_history = []
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[SymbolicRegression] {msg}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        niterations: int = 100,
        populations: int = 30
    ) -> 'SymbolicRegressionEngine':
        """
        Find mathematical equation that predicts y from X
        
        Args:
            X: Feature DataFrame (RSI, MACD, Volume, etc.)
            y: Target (forward returns or labels)
            niterations: Number of evolutionary iterations
            populations: Number of parallel populations
        """
        if not PYSR_AVAILABLE:
            self.log("⚠️ PySR not available. Using fallback equation discovery.")
            return self._fallback_discovery(X, y)
        
        self.log(f"Starting Symbolic Regression with {niterations} iterations...")
        self.log(f"Features: {list(X.columns)}")
        
        # Configure PySR
        self.model = PySRRegressor(
            niterations=niterations,
            populations=populations,
            binary_operators=["+", "-", "*", "/", "^"],
            unary_operators=["sin", "cos", "exp", "log", "abs", "sqrt"],
            complexity_of_operators={
                "+": 1, "-": 1, "*": 1, "/": 2,
                "^": 2, "sin": 3, "cos": 3, "exp": 3,
                "log": 3, "abs": 1, "sqrt": 2
            },
            maxsize=self.max_complexity,
            loss="loss(prediction, target) = (prediction - target)^2",
            batching=True,
            batch_size=100,
            turbo=True,
            progress=self.verbose,
            verbosity=1 if self.verbose else 0
        )
        
        # Fit
        self.model.fit(X.values, y.values, variable_names=list(X.columns))
        
        # Get best equation
        self.best_equation = self.model.get_best()
        self.equations_history = self.model.equations_
        
        self.log(f"\n✅ Best equation found:")
        self.log(f"   {self.best_equation}")
        
        return self
    
    def _fallback_discovery(self, X: pd.DataFrame, y: pd.Series) -> 'SymbolicRegressionEngine':
        """
        Simple correlation-based equation discovery when PySR not available
        """
        self.log("Using correlation-based equation discovery...")
        
        # Calculate correlations
        correlations = {}
        for col in X.columns:
            corr = np.corrcoef(X[col].values, y.values)[0, 1]
            if not np.isnan(corr):
                correlations[col] = corr
        
        # Sort by absolute correlation
        sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        
        # Build simple linear equation from top features
        equation_parts = []
        for feature, corr in sorted_features[:5]:
            sign = "+" if corr > 0 else "-"
            equation_parts.append(f"{sign} {abs(corr):.3f} * {feature}")
        
        self.best_equation = " ".join(equation_parts)
        
        self.log(f"\n✅ Discovered equation (simplified):")
        self.log(f"   Signal = {self.best_equation}")
        
        # Store feature importance
        self.feature_importance = correlations
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using discovered equation"""
        if self.model is not None and PYSR_AVAILABLE:
            return self.model.predict(X.values)
        else:
            # Use correlation-based prediction
            prediction = np.zeros(len(X))
            for feature, corr in self.feature_importance.items():
                if feature in X.columns:
                    prediction += corr * X[feature].values
            return prediction
    
    def get_equation_string(self) -> str:
        """Get human-readable equation"""
        return str(self.best_equation)
    
    def get_simplified_rules(self) -> List[str]:
        """Convert equation to simple trading rules"""
        rules = []
        
        if hasattr(self, 'feature_importance'):
            for feature, corr in sorted(
                self.feature_importance.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]:
                if corr > 0.1:
                    rules.append(f"IF {feature} HIGH → BUY (correlation: {corr:.2f})")
                elif corr < -0.1:
                    rules.append(f"IF {feature} HIGH → SELL (correlation: {corr:.2f})")
        
        return rules


class GeneticProgrammingEngine:
    """
    Evolve trading strategies using Genetic Programming
    
    Creates readable rules like:
    IF (RSI > 70 AND MACD_Cross < 0) THEN SELL
    IF (Price > EMA20 AND Volume > Avg) THEN BUY
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.toolbox = None
        self.best_individual = None
        self.best_fitness = None
        self.population_history = []
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[GeneticProgramming] {msg}")
    
    def _setup_primitives(self, feature_names: List[str]):
        """Setup primitive set for GP"""
        if not DEAP_AVAILABLE:
            return None
        
        # Create primitive set
        pset = gp.PrimitiveSetTyped("MAIN", [float] * len(feature_names), float)
        
        # Rename arguments to feature names
        for i, name in enumerate(feature_names):
            pset.renameArguments(**{f"ARG{i}": name})
        
        # Add operators
        pset.addPrimitive(operator.add, [float, float], float, name="add")
        pset.addPrimitive(operator.sub, [float, float], float, name="sub")
        pset.addPrimitive(operator.mul, [float, float], float, name="mul")
        pset.addPrimitive(self._protected_div, [float, float], float, name="div")
        pset.addPrimitive(self._if_then_else, [float, float, float], float, name="if_gt")
        pset.addPrimitive(max, [float, float], float, name="max")
        pset.addPrimitive(min, [float, float], float, name="min")
        pset.addPrimitive(abs, [float], float, name="abs")
        pset.addPrimitive(self._protected_sqrt, [float], float, name="sqrt")
        
        # Add constants
        pset.addEphemeralConstant("const", lambda: np.random.uniform(-1, 1), float)
        
        return pset
    
    def _protected_div(self, x: float, y: float) -> float:
        """Division protected against zero"""
        if abs(y) < 1e-10:
            return 0.0
        return x / y
    
    def _protected_sqrt(self, x: float) -> float:
        """Square root protected against negative"""
        return np.sqrt(abs(x))
    
    def _if_then_else(self, condition: float, true_val: float, false_val: float) -> float:
        """If-then-else function"""
        return true_val if condition > 0 else false_val
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_generations: int = 50,
        population_size: int = 300,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.2
    ) -> 'GeneticProgrammingEngine':
        """
        Evolve trading strategies
        
        Args:
            X: Feature DataFrame
            y: Target (forward returns or labels)
            n_generations: Number of evolutionary generations
            population_size: Size of population
        """
        if not DEAP_AVAILABLE:
            self.log("⚠️ DEAP not available. Using rule-based fallback.")
            return self._fallback_rules(X, y)
        
        feature_names = list(X.columns)
        self.log(f"Starting Genetic Programming with {n_generations} generations...")
        self.log(f"Features: {feature_names}")
        
        # Setup
        pset = self._setup_primitives(feature_names)
        
        # Create fitness and individual types
        if not hasattr(creator, "FitnessMax"):
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        
        # Create toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=4)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.expr)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("compile", gp.compile, pset=pset)
        
        # Fitness function
        def evaluate(individual):
            func = self.toolbox.compile(expr=individual)
            
            try:
                predictions = []
                for i in range(len(X)):
                    row = X.iloc[i].values
                    pred = func(*row)
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                
                # Correlation with target as fitness
                if np.std(predictions) < 1e-10:
                    return (0.0,)
                
                corr = np.corrcoef(predictions, y.values)[0, 1]
                if np.isnan(corr):
                    return (0.0,)
                
                # Penalize complexity
                complexity_penalty = len(individual) * 0.001
                
                return (corr - complexity_penalty,)
            except:
                return (0.0,)
        
        self.toolbox.register("evaluate", evaluate)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("mate", gp.cxOnePoint)
        self.toolbox.register("mutate", gp.mutUniform, expr=self.toolbox.expr, pset=pset)
        
        # Size limits
        self.toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        self.toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=10))
        
        # Create population
        pop = self.toolbox.population(n=population_size)
        
        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)
        
        # Run evolution
        pop, logbook = algorithms.eaSimple(
            pop, self.toolbox,
            cxpb=crossover_prob,
            mutpb=mutation_prob,
            ngen=n_generations,
            stats=stats,
            verbose=self.verbose
        )
        
        # Get best individual
        self.best_individual = tools.selBest(pop, k=1)[0]
        self.best_fitness = self.best_individual.fitness.values[0]
        
        self.log(f"\n✅ Best strategy found (fitness: {self.best_fitness:.4f}):")
        self.log(f"   {self.best_individual}")
        
        return self
    
    def _fallback_rules(self, X: pd.DataFrame, y: pd.Series) -> 'GeneticProgrammingEngine':
        """Simple rule discovery when DEAP not available"""
        self.log("Using correlation-based rule discovery...")
        
        rules = []
        
        # Find best feature thresholds
        for col in X.columns:
            # Try different thresholds
            for percentile in [25, 50, 75]:
                threshold = X[col].quantile(percentile / 100)
                
                above_mask = X[col] > threshold
                below_mask = X[col] <= threshold
                
                if above_mask.sum() > 10 and below_mask.sum() > 10:
                    above_return = y[above_mask].mean()
                    below_return = y[below_mask].mean()
                    
                    if above_return > below_return + 0.001:
                        rules.append({
                            'rule': f"IF {col} > {threshold:.4f} THEN BUY",
                            'expected_return': above_return,
                            'samples': above_mask.sum()
                        })
                    elif below_return > above_return + 0.001:
                        rules.append({
                            'rule': f"IF {col} <= {threshold:.4f} THEN BUY",
                            'expected_return': below_return,
                            'samples': below_mask.sum()
                        })
        
        # Sort by expected return
        rules.sort(key=lambda x: x['expected_return'], reverse=True)
        
        self.discovered_rules = rules[:10]
        
        self.log(f"\n✅ Discovered {len(self.discovered_rules)} rules:")
        for rule in self.discovered_rules[:5]:
            self.log(f"   {rule['rule']} (E[r]={rule['expected_return']:.4f})")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict using evolved strategy"""
        if self.best_individual is not None and DEAP_AVAILABLE:
            func = self.toolbox.compile(expr=self.best_individual)
            predictions = []
            for i in range(len(X)):
                row = X.iloc[i].values
                try:
                    pred = func(*row)
                    predictions.append(pred)
                except:
                    predictions.append(0.0)
            return np.array(predictions)
        else:
            # Use rule-based prediction
            if hasattr(self, 'discovered_rules') and self.discovered_rules:
                return self._rule_based_predict(X)
            return np.zeros(len(X))
    
    def _rule_based_predict(self, X: pd.DataFrame) -> np.ndarray:
        """Simple rule-based prediction"""
        scores = np.zeros(len(X))
        
        for rule in self.discovered_rules:
            rule_str = rule['rule']
            expected_return = rule['expected_return']
            
            # Parse simple rules
            if " > " in rule_str:
                parts = rule_str.split(" > ")
                feature = parts[0].replace("IF ", "")
                threshold = float(parts[1].split(" ")[0])
                
                if feature in X.columns:
                    mask = X[feature] > threshold
                    scores[mask] += expected_return
            
            elif " <= " in rule_str:
                parts = rule_str.split(" <= ")
                feature = parts[0].replace("IF ", "")
                threshold = float(parts[1].split(" ")[0])
                
                if feature in X.columns:
                    mask = X[feature] <= threshold
                    scores[mask] += expected_return
        
        return scores
    
    def get_strategy_string(self) -> str:
        """Get human-readable strategy"""
        if self.best_individual is not None:
            return str(self.best_individual)
        elif hasattr(self, 'discovered_rules'):
            return "\n".join([r['rule'] for r in self.discovered_rules[:5]])
        return "No strategy discovered"


class LogicEngine:
    """
    Combined Logic Engine using both Symbolic Regression and Genetic Programming
    
    Finds:
    1. Mathematical equations (PySR)
    2. Trading rules (GP/DEAP)
    
    Then combines them for maximum predictive power
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.sr_engine = SymbolicRegressionEngine(verbose=verbose)
        self.gp_engine = GeneticProgrammingEngine(verbose=verbose)
        self.combined_model = None
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[LogicEngine] {msg}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        use_sr: bool = True,
        use_gp: bool = True
    ) -> 'LogicEngine':
        """
        Discover trading logic using both methods
        
        Args:
            X: Feature DataFrame
            y: Target
            use_sr: Use Symbolic Regression
            use_gp: Use Genetic Programming
        """
        self.log("=" * 60)
        self.log("LOGIC ENGINE: Discovering Trading Rules")
        self.log("=" * 60)
        
        if use_sr:
            self.log("\n--- Symbolic Regression ---")
            self.sr_engine.fit(X, y, niterations=50)
        
        if use_gp:
            self.log("\n--- Genetic Programming ---")
            self.gp_engine.fit(X, y, n_generations=30, population_size=100)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Combined prediction from both engines"""
        predictions = []
        
        if hasattr(self.sr_engine, 'best_equation') and self.sr_engine.best_equation:
            sr_pred = self.sr_engine.predict(X)
            predictions.append(sr_pred)
        
        if hasattr(self.gp_engine, 'best_individual') or hasattr(self.gp_engine, 'discovered_rules'):
            gp_pred = self.gp_engine.predict(X)
            predictions.append(gp_pred)
        
        if predictions:
            # Average predictions
            return np.mean(predictions, axis=0)
        
        return np.zeros(len(X))
    
    def get_trading_rules(self) -> Dict[str, any]:
        """Get all discovered trading rules"""
        return {
            'equation': self.sr_engine.get_equation_string() if hasattr(self.sr_engine, 'best_equation') else None,
            'simplified_rules': self.sr_engine.get_simplified_rules() if hasattr(self.sr_engine, 'get_simplified_rules') else [],
            'evolved_strategy': self.gp_engine.get_strategy_string() if hasattr(self.gp_engine, 'get_strategy_string') else None
        }


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    import yfinance as yf
    
    print("=" * 60)
    print("TESTING LOGIC ENGINE (Symbolic Regression + GP)")
    print("=" * 60)
    
    # Download test data
    df = yf.download("SPY", start="2022-01-01", end="2024-12-01", progress=False)
    
    # Create features
    features = pd.DataFrame(index=df.index)
    features['returns'] = df['Close'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()
    features['momentum'] = df['Close'].pct_change(20)
    features['rsi'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                         df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Target: 5-day forward return
    target = df['Close'].pct_change(5).shift(-5)
    
    # Drop NaN
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    features = features.loc[valid_idx]
    target = target.loc[valid_idx]
    
    print(f"\nFeatures: {list(features.columns)}")
    print(f"Samples: {len(features)}")
    
    # Test Logic Engine
    engine = LogicEngine(verbose=True)
    engine.fit(features, target, use_sr=True, use_gp=True)
    
    # Get predictions
    predictions = engine.predict(features)
    
    print(f"\n{'='*60}")
    print("DISCOVERED TRADING RULES")
    print(f"{'='*60}")
    
    rules = engine.get_trading_rules()
    
    if rules['equation']:
        print(f"\n📐 Mathematical Equation:")
        print(f"   {rules['equation']}")
    
    if rules['simplified_rules']:
        print(f"\n📋 Simplified Rules:")
        for rule in rules['simplified_rules']:
            print(f"   • {rule}")
    
    if rules['evolved_strategy']:
        print(f"\n🧬 Evolved Strategy:")
        print(f"   {rules['evolved_strategy']}")
