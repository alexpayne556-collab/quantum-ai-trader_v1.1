"""
EXECUTION ENGINE: SOFT ACTOR-CRITIC (SAC) REINFORCEMENT LEARNING
==================================================================
The final piece - manages risk and executes trades.

Why SAC:
- Older RL (DQN, PPO) is "brittle" - finds one narrow strategy
- SAC optimizes for ENTROPY + Profit
- Asks: "What's the most profitable move that keeps options open?"
- Incredibly robust to market crashes and regime changes

The Secret Sauce - Reward Function:
- DON'T train to maximize raw profit (takes crazy risks)
- Train to maximize DIFFERENTIAL SHARPE RATIO
- Forces smooth equity curve, punishes volatility

Architecture:
- Actor: Decides position size (-1 to +1)
- Critic: Evaluates state-action value
- Entropy: Encourages exploration

Research: SAC is state-of-the-art for continuous action spaces (trading)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

# Try to import stable-baselines3
try:
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    print("Note: stable-baselines3 not installed. Install with: pip install stable-baselines3")

# Try to import gymnasium
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        print("Note: gymnasium not installed. Install with: pip install gymnasium")


class TradingEnvironment:
    """
    Custom Trading Environment for Reinforcement Learning
    
    State: [position, returns, volatility, momentum, signal_strength, portfolio_value]
    Action: Position size from -1 (full short) to +1 (full long)
    Reward: Differential Sharpe Ratio (smooth returns, penalize volatility)
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        signals: Optional[np.ndarray] = None,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        max_position: float = 1.0
    ):
        """
        Args:
            df: OHLCV DataFrame
            features: Feature DataFrame (from other engines)
            signals: Predicted signals from Visual/Logic engines
            initial_capital: Starting capital
            transaction_cost: Cost per trade as fraction
            max_position: Maximum position size
        """
        self.df = df
        self.features = features
        self.signals = signals if signals is not None else np.zeros(len(df))
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        
        # State dimensions
        self.n_features = len(features.columns) if features is not None else 0
        self.state_dim = self.n_features + 5  # features + position, returns, vol, momentum, portfolio
        
        # Action space: continuous [-1, 1]
        self.action_space_low = -max_position
        self.action_space_high = max_position
        
        # Reset
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = 50  # Start after warmup period
        self.position = 0.0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.holdings = 0.0
        
        # Track history for Sharpe calculation
        self.returns_history = deque(maxlen=20)
        self.portfolio_history = [self.initial_capital]
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        if self.current_step >= len(self.df):
            return np.zeros(self.state_dim)
        
        state = []
        
        # Add features
        if self.features is not None and self.current_step < len(self.features):
            feat_values = self.features.iloc[self.current_step].values
            state.extend(feat_values)
        else:
            state.extend([0] * self.n_features)
        
        # Add position
        state.append(self.position)
        
        # Recent returns
        if len(self.returns_history) > 0:
            state.append(np.mean(self.returns_history))
        else:
            state.append(0)
        
        # Volatility
        if len(self.returns_history) >= 5:
            state.append(np.std(self.returns_history))
        else:
            state.append(0)
        
        # Momentum (5-day)
        if self.current_step >= 5:
            momentum = (self.df['Close'].iloc[self.current_step] - 
                       self.df['Close'].iloc[self.current_step - 5]) / self.df['Close'].iloc[self.current_step - 5]
            state.append(momentum)
        else:
            state.append(0)
        
        # Normalized portfolio value
        state.append(self.portfolio_value / self.initial_capital - 1)
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: Position size [-1, 1]
        
        Returns:
            state, reward, done, info
        """
        if self.current_step >= len(self.df) - 1:
            return self._get_state(), 0, True, {}
        
        # Clip action to valid range
        action = np.clip(action, self.action_space_low, self.action_space_high)
        
        # Get prices
        current_price = self.df['Close'].iloc[self.current_step]
        next_price = self.df['Close'].iloc[self.current_step + 1]
        
        # Calculate position change
        old_position = self.position
        new_position = action
        position_change = abs(new_position - old_position)
        
        # Transaction cost
        trade_cost = position_change * self.portfolio_value * self.transaction_cost
        
        # Update holdings based on new position
        target_holdings_value = new_position * self.portfolio_value
        target_shares = target_holdings_value / current_price
        
        # Calculate return
        price_return = (next_price - current_price) / current_price
        position_return = new_position * price_return
        
        # Update portfolio value
        old_portfolio = self.portfolio_value
        self.portfolio_value = self.portfolio_value * (1 + position_return) - trade_cost
        
        # Calculate actual return
        actual_return = (self.portfolio_value - old_portfolio) / old_portfolio
        self.returns_history.append(actual_return)
        self.portfolio_history.append(self.portfolio_value)
        
        # Update position
        self.position = new_position
        
        # Calculate reward (Differential Sharpe Ratio)
        reward = self._calculate_differential_sharpe_reward(actual_return)
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'return': actual_return,
            'trade_cost': trade_cost
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_differential_sharpe_reward(self, current_return: float) -> float:
        """
        Differential Sharpe Ratio reward
        
        Encourages:
        - Positive returns
        - Low volatility
        - Consistency
        
        Formula:
        dS/dt = (μ_t - μ_{t-1}) / σ - (σ_t - σ_{t-1}) / σ^2 * (r_t - μ)
        
        Simplified: reward = return / volatility - volatility_penalty
        """
        if len(self.returns_history) < 2:
            return current_return * 100  # Simple return-based reward initially
        
        returns = np.array(self.returns_history)
        mean_return = np.mean(returns)
        std_return = np.std(returns) + 1e-8
        
        # Sharpe-like reward
        sharpe_reward = mean_return / std_return
        
        # Penalty for high volatility
        volatility_penalty = std_return * 10
        
        # Penalty for large drawdowns
        max_portfolio = max(self.portfolio_history)
        current_drawdown = (max_portfolio - self.portfolio_value) / max_portfolio
        drawdown_penalty = current_drawdown * 5
        
        # Combined reward
        reward = sharpe_reward * 100 - volatility_penalty - drawdown_penalty
        
        # Bonus for profit
        if self.portfolio_value > self.initial_capital:
            profit_bonus = (self.portfolio_value / self.initial_capital - 1) * 10
            reward += profit_bonus
        
        return reward
    
    def get_performance_metrics(self) -> Dict:
        """Calculate performance metrics"""
        returns = np.diff(self.portfolio_history) / self.portfolio_history[:-1]
        
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        if len(returns) > 0:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            max_drawdown = np.min(np.minimum.accumulate(self.portfolio_history) / 
                                  np.maximum.accumulate(self.portfolio_history)) - 1
        else:
            sharpe = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_portfolio': self.portfolio_value,
            'num_trades': len(self.portfolio_history)
        }


# Only define GymTradingEnv if gymnasium is available
if GYM_AVAILABLE:
    class GymTradingEnv(gym.Env):
        """
        Gymnasium-compatible wrapper for stable-baselines3
        """
        
        def __init__(
            self,
            df: pd.DataFrame,
            features: pd.DataFrame,
            signals: Optional[np.ndarray] = None,
            initial_capital: float = 100000
        ):
            super().__init__()
            
            self.env = TradingEnvironment(df, features, signals, initial_capital)
            
            # Define spaces
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
            
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.env.state_dim,),
                dtype=np.float32
            )
        
        def reset(self, seed=None):
            if seed is not None:
                np.random.seed(seed)
            state = self.env.reset()
            return state, {}
        
        def step(self, action):
            state, reward, done, info = self.env.step(action[0])
            return state, reward, done, False, info
        
        def render(self):
            pass
else:
    GymTradingEnv = None


class ExecutionEngine:
    """
    SAC-based Execution Engine for position sizing and risk management
    
    Uses Soft Actor-Critic to learn:
    - When to enter trades
    - How much to size positions
    - When to exit
    
    Optimizes for Differential Sharpe Ratio (smooth, consistent returns)
    """
    
    def __init__(
        self,
        verbose: bool = True,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001
    ):
        self.verbose = verbose
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.model = None
        self.env = None
        self.training_history = []
        
    def log(self, msg: str):
        if self.verbose:
            print(f"[ExecutionEngine] {msg}")
    
    def train(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        signals: Optional[np.ndarray] = None,
        total_timesteps: int = 50000,
        learning_rate: float = 3e-4
    ) -> 'ExecutionEngine':
        """
        Train SAC agent on historical data
        
        Args:
            df: OHLCV DataFrame
            features: Feature DataFrame
            signals: Optional signal predictions from other engines
            total_timesteps: Training steps
        """
        self.log("=" * 60)
        self.log("TRAINING SAC EXECUTION ENGINE")
        self.log("=" * 60)
        
        if not SB3_AVAILABLE or not GYM_AVAILABLE:
            self.log("⚠️ stable-baselines3 or gymnasium not available. Using simple rules.")
            return self._train_simple_rules(df, features, signals)
        
        # Create environment
        self.log("Creating trading environment...")
        
        def make_env():
            return GymTradingEnv(df, features, signals, self.initial_capital)
        
        env = DummyVecEnv([make_env])
        
        # Create SAC model
        self.log("Initializing SAC model...")
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=100000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1 if self.verbose else 0
        )
        
        # Train
        self.log(f"Training for {total_timesteps} timesteps...")
        self.model.learn(total_timesteps=total_timesteps)
        
        # Evaluate
        self.log("\nEvaluating trained agent...")
        self._evaluate(df, features, signals)
        
        return self
    
    def _train_simple_rules(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        signals: Optional[np.ndarray]
    ) -> 'ExecutionEngine':
        """Simple rule-based position sizing when SAC not available"""
        self.log("Training simple position sizing rules...")
        
        # Learn volatility-based position sizing
        returns = df['Close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Calculate optimal position sizes based on historical performance
        self.vol_target = 0.15 / np.sqrt(252)  # Target 15% annual vol
        self.base_position = 0.5
        
        self.log("✅ Simple rules learned:")
        self.log(f"   Target volatility: {self.vol_target * np.sqrt(252):.1%} annual")
        self.log(f"   Base position: {self.base_position:.1%}")
        
        return self
    
    def _evaluate(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        signals: Optional[np.ndarray]
    ):
        """Evaluate trained agent"""
        env = TradingEnvironment(df, features, signals, self.initial_capital)
        state = env.reset()
        
        done = False
        while not done:
            if self.model is not None:
                action, _ = self.model.predict(state, deterministic=True)
                state, _, done, _ = env.step(action[0])
            else:
                # Simple rule-based action
                action = self._simple_action(state)
                state, _, done, _ = env.step(action)
        
        metrics = env.get_performance_metrics()
        
        self.log(f"\n📊 EVALUATION RESULTS:")
        self.log(f"   Total Return: {metrics['total_return']:.2%}")
        self.log(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        self.log(f"   Max Drawdown: {metrics['max_drawdown']:.2%}")
        self.log(f"   Final Portfolio: ${metrics['final_portfolio']:,.2f}")
        
        self.training_history.append(metrics)
    
    def _simple_action(self, state: np.ndarray) -> float:
        """Simple rule-based action when SAC not available"""
        # Extract features from state
        if len(state) > 5:
            momentum = state[-2] if len(state) >= 2 else 0
            recent_return = state[-4] if len(state) >= 4 else 0
        else:
            momentum = 0
            recent_return = 0
        
        # Simple momentum-based position
        if momentum > 0.02:
            position = min(0.8, 0.5 + momentum * 5)
        elif momentum < -0.02:
            position = max(-0.8, -0.5 + momentum * 5)
        else:
            position = 0.0
        
        return position
    
    def predict_position(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        signals: Optional[np.ndarray] = None
    ) -> float:
        """
        Predict optimal position for current market state
        
        Returns:
            Position size from -1 (full short) to +1 (full long)
        """
        env = TradingEnvironment(df, features, signals, self.initial_capital)
        state = env._get_state()
        
        if self.model is not None:
            action, _ = self.model.predict(state, deterministic=True)
            return float(action[0])
        else:
            return self._simple_action(state)
    
    def get_position_size_dollars(
        self,
        df: pd.DataFrame,
        features: pd.DataFrame,
        signals: Optional[np.ndarray] = None,
        portfolio_value: float = None
    ) -> Dict:
        """
        Get recommended position size in dollars
        
        Returns:
            Dict with position recommendation
        """
        if portfolio_value is None:
            portfolio_value = self.initial_capital
        
        position_pct = self.predict_position(df, features, signals)
        position_dollars = position_pct * portfolio_value
        
        # Get current price
        current_price = df['Close'].iloc[-1]
        shares = int(position_dollars / current_price)
        
        return {
            'position_pct': position_pct,
            'position_dollars': position_dollars,
            'shares': shares,
            'direction': 'LONG' if position_pct > 0 else 'SHORT' if position_pct < 0 else 'FLAT',
            'confidence': abs(position_pct)
        }
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model is not None:
            self.model.save(path)
            self.log(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        if SB3_AVAILABLE:
            self.model = SAC.load(path)
            self.log(f"Model loaded from {path}")


# =============================================================================
# QUICK TEST
# =============================================================================
if __name__ == "__main__":
    import yfinance as yf
    
    print("=" * 60)
    print("TESTING EXECUTION ENGINE (SAC Reinforcement Learning)")
    print("=" * 60)
    
    # Download test data
    df = yf.download("SPY", start="2022-01-01", end="2024-12-01", progress=False)
    
    # Create simple features
    features = pd.DataFrame(index=df.index)
    features['returns'] = df['Close'].pct_change()
    features['volatility'] = features['returns'].rolling(20).std()
    features['momentum'] = df['Close'].pct_change(20)
    features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    features = features.dropna()
    
    # Align df with features
    df = df.loc[features.index]
    
    print(f"\nData: {len(df)} samples")
    print(f"Features: {list(features.columns)}")
    
    # Test Execution Engine
    engine = ExecutionEngine(verbose=True, initial_capital=100000)
    
    # Train (with reduced timesteps for quick test)
    engine.train(df, features, total_timesteps=10000)
    
    # Get position recommendation
    print(f"\n{'='*60}")
    print("POSITION RECOMMENDATION")
    print(f"{'='*60}")
    
    recommendation = engine.get_position_size_dollars(df, features, portfolio_value=100000)
    
    print(f"   Direction: {recommendation['direction']}")
    print(f"   Position %: {recommendation['position_pct']:.1%}")
    print(f"   Position $: ${recommendation['position_dollars']:,.2f}")
    print(f"   Shares: {recommendation['shares']}")
    print(f"   Confidence: {recommendation['confidence']:.1%}")
