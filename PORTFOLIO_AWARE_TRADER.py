"""
📊 PORTFOLIO-AWARE SWING TRADER
Understands YOUR watchlist, YOUR portfolio, and makes context-aware decisions

Features:
✅ Trains specifically on YOUR watchlist tickers
✅ Knows what you currently hold (positions, entry prices, P&L)
✅ Makes portfolio-level decisions (add, hold, trim, exit)
✅ Position sizing based on portfolio allocation
✅ Sector diversification management
✅ Exit signals for existing positions
✅ Entry signals for watchlist opportunities
✅ Risk management across entire portfolio
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import pickle
from pathlib import Path

# Import sector-aware components
from SECTOR_AWARE_SWING_TRADER import (
    SectorAwareSwingTrader,
    SectorAwareRecommendation,
    SECTOR_MAPPING,
    REVERSE_SECTOR_MAP
)

# ML imports
try:
    import xgboost as xgb
    import lightgbm as lgb
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    HAS_ML = True
except ImportError:
    HAS_ML = False


# ============================================================================
# POSITION & PORTFOLIO TRACKING
# ============================================================================

@dataclass
class Position:
    """Track a held position"""
    ticker: str
    entry_price: float
    shares: int
    entry_date: datetime
    sector: str
    current_price: float = 0.0
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    
    @property
    def market_value(self) -> float:
        return self.current_price * self.shares
    
    @property
    def cost_basis(self) -> float:
        return self.entry_price * self.shares
    
    @property
    def pnl_dollars(self) -> float:
        return (self.current_price - self.entry_price) * self.shares
    
    @property
    def pnl_percent(self) -> float:
        return (self.current_price / self.entry_price - 1) * 100
    
    @property
    def days_held(self) -> int:
        return (datetime.now() - self.entry_date).days
    
    def to_dict(self):
        return {
            'ticker': self.ticker,
            'entry_price': self.entry_price,
            'shares': self.shares,
            'entry_date': self.entry_date.isoformat(),
            'sector': self.sector,
            'current_price': self.current_price,
            'stop_loss': self.stop_loss,
            'target_price': self.target_price
        }


@dataclass
class Portfolio:
    """Track entire portfolio"""
    cash: float
    positions: List[Position] = field(default_factory=list)
    max_position_size: float = 0.20  # Max 20% per position
    max_sector_allocation: float = 0.40  # Max 40% per sector
    
    @property
    def total_equity(self) -> float:
        """Total portfolio value"""
        return self.cash + sum(p.market_value for p in self.positions)
    
    @property
    def invested_capital(self) -> float:
        """Amount currently invested"""
        return sum(p.market_value for p in self.positions)
    
    @property
    def cash_percent(self) -> float:
        """Cash as % of total"""
        return self.cash / self.total_equity * 100
    
    @property
    def total_pnl(self) -> float:
        """Total unrealized P&L"""
        return sum(p.pnl_dollars for p in self.positions)
    
    @property
    def total_pnl_percent(self) -> float:
        """Total P&L %"""
        invested = sum(p.cost_basis for p in self.positions)
        if invested == 0:
            return 0
        return (sum(p.pnl_dollars for p in self.positions) / invested) * 100
    
    def get_sector_allocation(self) -> Dict[str, float]:
        """Get % allocation by sector"""
        sector_values = {}
        for pos in self.positions:
            sector = pos.sector
            if sector not in sector_values:
                sector_values[sector] = 0
            sector_values[sector] += pos.market_value
        
        # Convert to percentages
        total = self.total_equity
        return {sector: (value / total * 100) for sector, value in sector_values.items()}
    
    def get_position(self, ticker: str) -> Optional[Position]:
        """Get position by ticker"""
        for pos in self.positions:
            if pos.ticker == ticker:
                return pos
        return None
    
    def has_position(self, ticker: str) -> bool:
        """Check if ticker is held"""
        return self.get_position(ticker) is not None
    
    def can_add_position(self, ticker: str, dollars: float) -> Tuple[bool, str]:
        """Check if we can add a new position"""
        sector = REVERSE_SECTOR_MAP.get(ticker, 'UNKNOWN')
        
        # Check position size limit
        position_pct = dollars / self.total_equity * 100
        if position_pct > self.max_position_size * 100:
            return False, f"Position would be {position_pct:.1f}% (max {self.max_position_size*100:.0f}%)"
        
        # Check sector allocation limit
        sector_alloc = self.get_sector_allocation()
        current_sector_pct = sector_alloc.get(sector, 0)
        new_sector_pct = (sector_alloc.get(sector, 0) * self.total_equity / 100 + dollars) / self.total_equity * 100
        
        if new_sector_pct > self.max_sector_allocation * 100:
            return False, f"Sector {sector} would be {new_sector_pct:.1f}% (max {self.max_sector_allocation*100:.0f}%)"
        
        # Check if we have cash
        if dollars > self.cash:
            return False, f"Insufficient cash (${self.cash:,.2f} available)"
        
        return True, "OK"
    
    def to_dict(self):
        return {
            'cash': self.cash,
            'positions': [p.to_dict() for p in self.positions],
            'max_position_size': self.max_position_size,
            'max_sector_allocation': self.max_sector_allocation
        }
    
    def save(self, filepath: str = "portfolio.json"):
        """Save portfolio to file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"✅ Portfolio saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = "portfolio.json"):
        """Load portfolio from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            positions = []
            for p in data['positions']:
                positions.append(Position(
                    ticker=p['ticker'],
                    entry_price=p['entry_price'],
                    shares=p['shares'],
                    entry_date=datetime.fromisoformat(p['entry_date']),
                    sector=p['sector'],
                    current_price=p['current_price'],
                    stop_loss=p.get('stop_loss'),
                    target_price=p.get('target_price')
                ))
            
            portfolio = cls(
                cash=data['cash'],
                positions=positions,
                max_position_size=data.get('max_position_size', 0.20),
                max_sector_allocation=data.get('max_sector_allocation', 0.40)
            )
            
            print(f"✅ Portfolio loaded from {filepath}")
            return portfolio
            
        except FileNotFoundError:
            print(f"⚠️ No saved portfolio found. Creating new portfolio.")
            return cls(cash=100000)  # Default $100k


# ============================================================================
# PORTFOLIO-AWARE TRADING DECISIONS
# ============================================================================

@dataclass
class PortfolioAction:
    """Recommended action with portfolio context"""
    ticker: str
    action: str  # BUY_NEW, ADD_TO, HOLD, TRIM, SELL, WAIT
    confidence: float
    reasoning: List[str]
    
    # Trade details
    suggested_dollars: Optional[float] = None
    suggested_shares: Optional[int] = None
    current_position_size: Optional[float] = None  # % of portfolio
    target_position_size: Optional[float] = None
    
    # Risk management
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None
    
    # Context
    sector: str = ""
    sector_allocation: float = 0.0  # Current sector %
    days_held: Optional[int] = None
    pnl_percent: Optional[float] = None


class PortfolioAwareTrader:
    """
    Trading system that understands YOUR portfolio and watchlist
    """
    
    def __init__(self, watchlist: List[str], portfolio: Optional[Portfolio] = None):
        """
        Initialize with YOUR watchlist and portfolio
        
        Args:
            watchlist: List of tickers you're tracking
            portfolio: Your current portfolio (or creates new one)
        """
        self.watchlist = watchlist
        self.portfolio = portfolio or Portfolio(cash=100000)
        
        # Initialize sector-aware trader
        self.trader = SectorAwareSwingTrader()
        self.trader.load_ml_ensemble()
        
        # Model storage
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        print(f"✅ Portfolio-Aware Trader initialized")
        print(f"   Watchlist: {len(watchlist)} tickers")
        print(f"   Portfolio: ${self.portfolio.total_equity:,.2f} total")
        print(f"   Cash: ${self.portfolio.cash:,.2f} ({self.portfolio.cash_percent:.1f}%)")
        print(f"   Positions: {len(self.portfolio.positions)}")
    
    def train_on_watchlist(self, period='2y'):
        """
        Train ML models specifically on YOUR watchlist tickers
        This makes the system familiar with the stocks you care about
        """
        print(f"\n{'='*100}")
        print(f"🎓 TRAINING ON YOUR WATCHLIST ({len(self.watchlist)} tickers)")
        print(f"{'='*100}\n")
        
        if not HAS_ML:
            print("⚠️ ML libraries not available. Skipping training.")
            return False
        
        # Collect training data from watchlist
        X_list, y_list = [], []
        
        for i, ticker in enumerate(self.watchlist, 1):
            print(f"[{i}/{len(self.watchlist)}] Collecting data for {ticker}...")
            
            try:
                df = yf.download(ticker, period=period, interval='1d', progress=False)
                if len(df) < 200:
                    print(f"   ⚠️ Insufficient data ({len(df)} days)")
                    continue
                
                # Engineer features (simplified for demo)
                window_size = 60
                horizon = 5
                
                df = df.copy()
                df['Return'] = df['Close'].pct_change(horizon).shift(-horizon)
                
                for j in range(window_size, len(df) - horizon):
                    window = df.iloc[j-window_size:j]
                    future_return = df['Return'].iloc[j]
                    
                    if pd.isna(future_return):
                        continue
                    
                    # Label: BUY=0, HOLD=1, SELL=2
                    if future_return > 0.03:
                        label = 0
                    elif future_return < -0.03:
                        label = 2
                    else:
                        label = 1
                    
                    # Calculate basic features
                    close = window['Close'].values
                    features = self._calculate_basic_features(close)
                    
                    if features:
                        X_list.append(list(features.values()))
                        y_list.append(label)
                
                print(f"   ✅ Collected {len(X_list)} samples")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        if len(X_list) < 100:
            print(f"\n❌ Insufficient training data ({len(X_list)} samples)")
            return False
        
        # Train models
        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\n🎓 Training ML ensemble on {len(X_train)} samples...")
        
        # Train each model
        for name, model in self.trader.ml_models.items():
            print(f"   Training {name}...")
            model.fit(X_train_scaled, y_train)
            acc = model.score(X_test_scaled, y_test)
            print(f"   ✅ {name}: {acc*100:.1f}% accuracy")
        
        # Save models
        self.trader.scaler = scaler
        self._save_models()
        
        print(f"\n✅ Training complete! Models trained on your watchlist.")
        return True
    
    def _calculate_basic_features(self, close):
        """Calculate basic features for training"""
        if len(close) < 20:
            return None
        
        features = {}
        features['price_mean'] = np.mean(close)
        features['price_std'] = np.std(close)
        features['momentum_5'] = (close[-1] - close[-5]) / (close[-5] + 1e-8) if len(close) >= 5 else 0
        features['momentum_10'] = (close[-1] - close[-10]) / (close[-10] + 1e-8) if len(close) >= 10 else 0
        features['volatility'] = np.std(np.diff(close) / (close[:-1] + 1e-8))
        features['ma_5'] = np.mean(close[-5:]) if len(close) >= 5 else close[-1]
        features['ma_20'] = np.mean(close[-20:]) if len(close) >= 20 else close[-1]
        
        return features
    
    def _save_models(self):
        """Save trained models"""
        for name, model in self.trader.ml_models.items():
            filepath = self.models_dir / f"{name}_watchlist.pkl"
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        
        # Save scaler
        with open(self.models_dir / "scaler.pkl", 'wb') as f:
            pickle.dump(self.trader.scaler, f)
        
        print(f"✅ Models saved to {self.models_dir}")
    
    def _load_models(self):
        """Load trained models"""
        try:
            for name in self.trader.ml_models.keys():
                filepath = self.models_dir / f"{name}_watchlist.pkl"
                with open(filepath, 'rb') as f:
                    self.trader.ml_models[name] = pickle.load(f)
            
            with open(self.models_dir / "scaler.pkl", 'rb') as f:
                self.trader.scaler = pickle.load(f)
            
            print(f"✅ Models loaded from {self.models_dir}")
            return True
        except FileNotFoundError:
            print(f"⚠️ No saved models found. Train first with train_on_watchlist()")
            return False
    
    def update_portfolio_prices(self):
        """Update current prices for all positions"""
        print(f"\n📊 Updating portfolio prices...")
        
        for pos in self.portfolio.positions:
            try:
                df = yf.download(pos.ticker, period='1d', interval='1d', progress=False)
                if len(df) > 0:
                    pos.current_price = float(df['Close'].iloc[-1])
                    print(f"   {pos.ticker}: ${pos.current_price:.2f} ({pos.pnl_percent:+.1f}%)")
            except:
                print(f"   ⚠️ {pos.ticker}: Could not update price")
        
        print(f"\n💰 Portfolio Value: ${self.portfolio.total_equity:,.2f}")
        print(f"   P&L: ${self.portfolio.total_pnl:,.2f} ({self.portfolio.total_pnl_percent:+.1f}%)\n")
    
    def analyze_portfolio_and_watchlist(self):
        """
        Analyze entire portfolio + watchlist
        Provides actionable recommendations for each ticker
        """
        print(f"\n{'='*100}")
        print(f"📊 PORTFOLIO & WATCHLIST ANALYSIS")
        print(f"{'='*100}\n")
        
        # Update prices first
        self.update_portfolio_prices()
        
        # Analyze each ticker
        actions = []
        
        print(f"\n{'='*100}")
        print(f"📈 ANALYZING HELD POSITIONS")
        print(f"{'='*100}\n")
        
        # 1. Analyze existing positions
        for pos in self.portfolio.positions:
            action = self._analyze_position(pos)
            actions.append(action)
            self._print_action(action)
        
        print(f"\n{'='*100}")
        print(f"👀 ANALYZING WATCHLIST OPPORTUNITIES")
        print(f"{'='*100}\n")
        
        # 2. Analyze watchlist tickers (not held)
        for ticker in self.watchlist:
            if not self.portfolio.has_position(ticker):
                action = self._analyze_watchlist_ticker(ticker)
                if action:
                    actions.append(action)
                    self._print_action(action)
        
        # 3. Portfolio summary
        self._print_portfolio_summary(actions)
        
        return actions
    
    def _analyze_position(self, pos: Position) -> PortfolioAction:
        """Analyze an existing position - should we hold, trim, or sell?"""
        
        # Get sector-aware recommendation
        rec = self.trader.recommend(pos.ticker, full_analysis=False)
        
        sector_alloc = self.portfolio.get_sector_allocation()
        position_pct = pos.market_value / self.portfolio.total_equity * 100
        
        reasoning = []
        
        # Check P&L
        if pos.pnl_percent > 10:
            reasoning.append(f"✅ Strong profit: +{pos.pnl_percent:.1f}%")
        elif pos.pnl_percent < -5:
            reasoning.append(f"⚠️ Loss: {pos.pnl_percent:.1f}%")
        
        # Check hold duration
        if pos.days_held > rec.expected_hold:
            reasoning.append(f"⏰ Held {pos.days_held} days (expected {rec.expected_hold})")
        
        # Check ML signal
        if rec.action == 'SELL':
            reasoning.append(f"🔴 ML model says SELL ({rec.adjusted_confidence*100:.0f}%)")
        elif rec.action == 'BUY':
            reasoning.append(f"🟢 ML model still bullish ({rec.adjusted_confidence*100:.0f}%)")
        
        # Check sector
        if not rec.sector_favored:
            reasoning.append(f"⚠️ Sector {pos.sector} not in favor")
        
        # Check stop loss
        if pos.stop_loss and pos.current_price <= pos.stop_loss:
            reasoning.append(f"🛑 Stop loss hit (${pos.stop_loss:.2f})")
        
        # Check target
        if pos.target_price and pos.current_price >= pos.target_price:
            reasoning.append(f"🎯 Target reached (${pos.target_price:.2f})")
        
        # Determine action
        if pos.stop_loss and pos.current_price <= pos.stop_loss:
            action = "SELL"
            confidence = 0.95
        elif pos.target_price and pos.current_price >= pos.target_price:
            if rec.action == 'SELL':
                action = "SELL"
                confidence = 0.90
            else:
                action = "TRIM"
                confidence = 0.75
        elif rec.action == 'SELL' and rec.adjusted_confidence > 0.7:
            action = "SELL"
            confidence = rec.adjusted_confidence
        elif pos.pnl_percent < -8 and rec.action != 'BUY':
            action = "SELL"
            confidence = 0.80
            reasoning.append("📉 Cut losses")
        elif position_pct > self.portfolio.max_position_size * 100 * 1.5:
            action = "TRIM"
            confidence = 0.70
            reasoning.append(f"⚖️ Position too large ({position_pct:.1f}%)")
        else:
            action = "HOLD"
            confidence = rec.adjusted_confidence
        
        return PortfolioAction(
            ticker=pos.ticker,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            current_position_size=position_pct,
            sector=pos.sector,
            sector_allocation=sector_alloc.get(pos.sector, 0),
            days_held=pos.days_held,
            pnl_percent=pos.pnl_percent,
            stop_loss=pos.stop_loss,
            target_price=pos.target_price
        )
    
    def _analyze_watchlist_ticker(self, ticker: str) -> Optional[PortfolioAction]:
        """Analyze a watchlist ticker - should we buy?"""
        
        # Get sector-aware recommendation
        rec = self.trader.recommend(ticker, full_analysis=False)
        
        sector = rec.sector
        sector_alloc = self.portfolio.get_sector_allocation()
        
        reasoning = []
        
        # Check ML signal
        if rec.action == 'BUY':
            reasoning.append(f"🟢 ML says BUY ({rec.adjusted_confidence*100:.0f}%)")
        else:
            return None  # Don't show if not a BUY
        
        # Check sector rotation
        if rec.sector_favored:
            reasoning.append(f"✅ Sector {sector} in favor ({rec.rotation_stage})")
        else:
            reasoning.append(f"⚠️ Sector {sector} not favored")
        
        # Check sector strength
        if rec.sector_strength > 70:
            reasoning.append(f"💪 Strong sector ({rec.sector_strength:.0f}/100)")
        
        # Calculate position size
        available_cash = self.portfolio.cash
        suggested_dollars = min(
            available_cash * 0.20,  # Max 20% of cash
            self.portfolio.total_equity * self.portfolio.max_position_size  # Max position size
        )
        
        # Check if we can add
        can_add, reason = self.portfolio.can_add_position(ticker, suggested_dollars)
        
        if not can_add:
            reasoning.append(f"❌ Cannot add: {reason}")
            action = "WAIT"
            confidence = 0.0
        elif rec.adjusted_confidence > 0.70:
            action = "BUY_NEW"
            confidence = rec.adjusted_confidence
        else:
            action = "WAIT"
            confidence = rec.adjusted_confidence
            reasoning.append(f"⏸️ Confidence too low ({rec.adjusted_confidence*100:.0f}%)")
        
        # Get current price
        try:
            df = yf.download(ticker, period='1d', interval='1d', progress=False)
            current_price = float(df['Close'].iloc[-1])
            suggested_shares = int(suggested_dollars / current_price)
        except:
            current_price = 0
            suggested_shares = 0
        
        return PortfolioAction(
            ticker=ticker,
            action=action,
            confidence=confidence,
            reasoning=reasoning,
            suggested_dollars=suggested_dollars,
            suggested_shares=suggested_shares,
            sector=sector,
            sector_allocation=sector_alloc.get(sector, 0),
            stop_loss=rec.stop_loss,
            target_price=rec.target_price
        )
    
    def _print_action(self, action: PortfolioAction):
        """Print action recommendation"""
        
        # Action emoji
        emoji_map = {
            'BUY_NEW': '🟢',
            'ADD_TO': '🟢',
            'HOLD': '🟡',
            'TRIM': '🟠',
            'SELL': '🔴',
            'WAIT': '⚪'
        }
        
        emoji = emoji_map.get(action.action, '⚪')
        
        print(f"{emoji} {action.ticker} - {action.action} ({action.confidence*100:.0f}%)")
        
        if action.current_position_size:
            print(f"   Position: {action.current_position_size:.1f}% of portfolio")
        
        if action.pnl_percent is not None:
            pnl_color = "🟢" if action.pnl_percent > 0 else "🔴"
            print(f"   P&L: {pnl_color} {action.pnl_percent:+.1f}%")
        
        if action.days_held:
            print(f"   Days Held: {action.days_held}")
        
        if action.suggested_dollars:
            print(f"   Suggested: ${action.suggested_dollars:,.0f} ({action.suggested_shares} shares)")
        
        print(f"   Sector: {action.sector} ({action.sector_allocation:.1f}% allocated)")
        
        for reason in action.reasoning:
            print(f"   • {reason}")
        
        print()
    
    def _print_portfolio_summary(self, actions: List[PortfolioAction]):
        """Print portfolio summary"""
        
        print(f"\n{'='*100}")
        print(f"📊 PORTFOLIO SUMMARY")
        print(f"{'='*100}\n")
        
        print(f"💰 Total Value: ${self.portfolio.total_equity:,.2f}")
        print(f"   Cash: ${self.portfolio.cash:,.2f} ({self.portfolio.cash_percent:.1f}%)")
        print(f"   Invested: ${self.portfolio.invested_capital:,.2f}")
        print(f"   P&L: ${self.portfolio.total_pnl:,.2f} ({self.portfolio.total_pnl_percent:+.1f}%)\n")
        
        # Sector allocation
        print(f"📊 Sector Allocation:")
        sector_alloc = self.portfolio.get_sector_allocation()
        for sector in sorted(sector_alloc.keys(), key=lambda x: sector_alloc[x], reverse=True):
            pct = sector_alloc[sector]
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            print(f"   {sector:20} {bar} {pct:.1f}%")
        
        # Action summary
        print(f"\n📈 Recommended Actions:")
        action_counts = {}
        for action in actions:
            if action.action not in action_counts:
                action_counts[action.action] = 0
            action_counts[action.action] += 1
        
        for action_type, count in sorted(action_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   {action_type}: {count}")
        
        print(f"\n{'='*100}\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*100)
    print("📊 PORTFOLIO-AWARE SWING TRADER - DEMO")
    print("="*100 + "\n")
    
    # Define YOUR watchlist
    MY_WATCHLIST = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',  # Tech
        'JPM', 'BAC', 'GS',  # Finance
        'XOM', 'CVX',  # Energy
        'JNJ', 'UNH',  # Healthcare
        'WMT', 'HD'  # Consumer
    ]
    
    # Create or load portfolio
    try:
        portfolio = Portfolio.load("my_portfolio.json")
    except:
        # Create demo portfolio with some positions
        portfolio = Portfolio(cash=50000)
        portfolio.positions = [
            Position(
                ticker='AAPL',
                entry_price=275.00,
                shares=100,
                entry_date=datetime(2025, 11, 15),
                sector='TECH',
                current_price=278.78,
                stop_loss=265.00,
                target_price=290.00
            ),
            Position(
                ticker='MSFT',
                entry_price=480.00,
                shares=50,
                entry_date=datetime(2025, 11, 20),
                sector='TECH',
                current_price=483.16,
                stop_loss=470.00,
                target_price=500.00
            )
        ]
    
    # Initialize trader
    trader = PortfolioAwareTrader(
        watchlist=MY_WATCHLIST,
        portfolio=portfolio
    )
    
    # Train on your watchlist (optional - only do once)
    print("\nWould you like to train models on your watchlist? (Takes a few minutes)")
    print("Type 'yes' to train, or press Enter to skip...")
    # For demo, skip training
    # trader.train_on_watchlist(period='1y')
    
    # Analyze portfolio and watchlist
    actions = trader.analyze_portfolio_and_watchlist()
    
    # Save portfolio
    trader.portfolio.save("my_portfolio.json")
    
    print("\n✅ Analysis complete!")
