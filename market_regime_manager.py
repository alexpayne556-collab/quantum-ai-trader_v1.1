"""
MARKET REGIME MANAGER v1.0
Adaptive bear market defense for live Windsurf trading
Integrates with your existing bot to detect crashes and protect capital

USAGE:
1. Save as: market_regime_manager.py
2. In your Windsurf bot:
   from market_regime_manager import MarketRegimeManager
   manager = MarketRegimeManager()
   regime = manager.get_current_regime()
   rules = manager.get_trading_rules(regime)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf


class MarketRegimeManager:
    """
    Real-time market regime detection for live trading.
    Monitors SPY to detect BULL/CORRECTION/BEAR/CRASH conditions.
    Returns adaptive trading rules (entry threshold, position size, hold time).
    """

    def __init__(self, cache_path: str = "./regime_cache"):
        self.cache_path = cache_path
        self.spy_data: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
        self.regime_history: list[Dict[str, float]] = []

    def download_spy_data(self, lookback_days: int = 365) -> Optional[pd.DataFrame]:
        """Download SPY data with caching."""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            spy_data = yf.download("SPY", start=start_date, end=end_date, progress=False)
            self.spy_data = spy_data
            self.last_update = datetime.now()
            return spy_data
        except Exception as exc:  # pragma: no cover - network errors
            print(f"❌ Error downloading SPY data: {exc}")
            return None

    def calculate_market_regime(self, spy_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect current market regime based on SPY position and volatility.

        Returns dict with regime, price, SMAs, volatility proxy, and confidence.
        """
        try:
            if spy_data is None:
                if self.spy_data is None:
                    self.download_spy_data()
                spy_data = self.spy_data

            if spy_data is None or spy_data.empty:
                raise ValueError("SPY data unavailable")

            # Handle MultiIndex columns from yfinance
            if isinstance(spy_data.columns, pd.MultiIndex):
                spy_data.columns = spy_data.columns.get_level_values(0)
            
            close = spy_data["Close"]
            
            # Ensure we get scalar values, not Series
            def get_scalar(val):
                """Extract scalar from potential Series/array"""
                if hasattr(val, 'item'):
                    return float(val.item())
                elif hasattr(val, 'iloc'):
                    return float(val.iloc[0]) if len(val) > 0 else 0.0
                return float(val)

            current_price = get_scalar(close.iloc[-1])
            sma_200 = get_scalar(close.rolling(200).mean().iloc[-1])
            price_vs_sma_pct = (current_price / sma_200 - 1) * 100 if sma_200 > 0 else 0

            returns = close.pct_change()
            historical_vol = returns.rolling(20).std() * np.sqrt(252) * 100
            vix_proxy = get_scalar(historical_vol.iloc[-1])

            sma_50 = get_scalar(close.rolling(50).mean().iloc[-1])
            trend_strength = (sma_50 / sma_200 - 1) * 100 if sma_200 > 0 else 0

            if price_vs_sma_pct > 2 and vix_proxy < 20:
                regime = "BULL"
                confidence = min(100, (price_vs_sma_pct / 2) * 50 + (20 - vix_proxy) * 2.5)
            elif price_vs_sma_pct > -2 and 20 <= vix_proxy < 30:
                regime = "CORRECTION"
                confidence = min(100, abs(price_vs_sma_pct) * 25 + (vix_proxy - 20) * 5)
            elif price_vs_sma_pct > -8 and 25 <= vix_proxy < 35:
                regime = "BEAR"
                confidence = min(100, abs(price_vs_sma_pct) * 10 + (vix_proxy - 25) * 5)
            elif price_vs_sma_pct <= -8 or vix_proxy >= 35:
                regime = "CRASH"
                confidence = 95
            else:
                regime = "NEUTRAL"
                confidence = 50

            regime_info = {
                "regime": regime,
                "price": float(current_price),
                "sma_200": float(sma_200),
                "sma_50": float(sma_50),
                "price_vs_sma": float(price_vs_sma_pct),
                "trend_strength": float(trend_strength),
                "vix_proxy": float(vix_proxy),
                "confidence": float(confidence),
                "timestamp": datetime.now(),
            }

            self.regime_history.append(regime_info)
            return regime_info

        except Exception as exc:  # pragma: no cover - just defensive logging
            print(f"❌ Error calculating market regime: {exc}")
            return {
                "regime": "BULL",
                "price": 0,
                "sma_200": 0,
                "sma_50": 0,
                "price_vs_sma": 0,
                "trend_strength": 0,
                "vix_proxy": 0,
                "confidence": 0,
                "timestamp": datetime.now(),
            }

    def get_adaptive_rules(self, regime: str) -> Dict:
        """Return adaptive trading rules for each market regime."""

        rules_map = {
            "BULL": {
                "regime": "BULL",
                "status": "🟢 BULL - NORMAL TRADING",
                "entry_threshold": 80,
                "position_pct": 0.05,  # Reduced from 0.10 for capital protection
                "hold_days": 10,
                "use_trailing_stop": False,
                "max_open_trades": 5,  # Increased for bull markets
                "max_drawdown_pct": 25,
                "can_trade": True,
                "description": "Price > SMA200, Low vol. Full trading mode.",
            },
            "CORRECTION": {
                "regime": "CORRECTION",
                "status": "🟡 CORRECTION - CAUTIOUS",
                "entry_threshold": 82,
                "position_pct": 0.03,  # Reduced from 0.07 for capital protection
                "hold_days": 8,
                "use_trailing_stop": True,
                "max_open_trades": 2,  # Reduced for safety
                "max_drawdown_pct": 15,
                "can_trade": True,
                "description": "Price ~SMA200, Moderate vol. Reduce position size.",
            },
            "BEAR": {
                "regime": "BEAR",
                "status": "🔴 BEAR - SELECTIVE",
                "entry_threshold": 85,
                "position_pct": 0.02,  # Reduced from 0.05 for capital protection
                "hold_days": 5,
                "use_trailing_stop": True,
                "max_open_trades": 1,  # Very conservative
                "max_drawdown_pct": 10,
                "can_trade": True,
                "description": "Price < SMA200, High vol. Only best signals. 50% position size.",
            },
            "CRASH": {
                "regime": "CRASH",
                "status": "⛔ CRASH - NO TRADING",
                "entry_threshold": 100,
                "position_pct": 0.00,  # Zero position size
                "hold_days": 0,
                "use_trailing_stop": False,
                "max_open_trades": 0,  # No trading allowed
                "max_drawdown_pct": 0,
                "can_trade": False,  # Trading disabled
                "description": "Market crash detected. Trading disabled. Preserve capital.",
            },
            "NEUTRAL": {
                "regime": "NEUTRAL",
                "status": "⚪ NEUTRAL - NORMAL",
                "entry_threshold": 81,
                "position_pct": 0.04,  # Conservative default
                "hold_days": 9,
                "use_trailing_stop": False,
                "max_open_trades": 2,
                "max_drawdown_pct": 20,
                "can_trade": True,
                "description": "Unclear regime. Use default rules.",
            },
        }

        return rules_map.get(regime, rules_map["NEUTRAL"])

    def get_current_regime(self) -> Dict:
        """Get current market regime (main entry point for bot)."""
        return self.calculate_market_regime()

    def should_trade(self) -> bool:
        """Simple boolean: is it safe to trade right now?"""
        regime_info = self.get_current_regime()
        return regime_info["regime"] != "CRASH"

    def enforce_trading_halt(self) -> Dict:
        """
        Enforce trading halt based on market regime
        Returns dict with halt status and regime information
        """
        regime_info = self.get_current_regime()
        rules = self.get_adaptive_rules(regime_info["regime"])

        halt_info = {
            "should_halt": not rules["can_trade"],
            "regime": regime_info["regime"],
            "reason": f"{regime_info['regime']} regime detected",
            "position_pct": rules["position_pct"],
            "max_open_trades": rules["max_open_trades"],
            "confidence": regime_info["confidence"],
            "description": rules["description"]
        }

        if halt_info["should_halt"]:
            print(f"🚫 TRADING HALT ENFORCED: {halt_info['reason']}")
            print(f"   Position Size: {halt_info['position_pct']:.1%}")
            print(f"   Max Open Trades: {halt_info['max_open_trades']}")
            print(f"   Confidence: {halt_info['confidence']:.0f}%")

        return halt_info

    def get_trading_rules(self, regime: Optional[str] = None) -> Dict:
        """Get adaptive rules for regime (main entry point for bot)."""
        if regime is None:
            regime_info = self.get_current_regime()
            regime = regime_info["regime"]

        return self.get_adaptive_rules(regime)

    def print_status(self) -> None:
        """Print current market status to console."""
        regime_info = self.get_current_regime()
        rules = self.get_adaptive_rules(regime_info["regime"])

        print("\n" + "=" * 70)
        print("MARKET REGIME STATUS")
        print("=" * 70)
        print(f"\n{rules['status']}")
        print(f"Confidence: {regime_info['confidence']:.0f}%")
        print(f"\nPrice: ${regime_info['price']:.2f}")
        print(f"SMA200: ${regime_info['sma_200']:.2f}")
        print(f"Distance: {regime_info['price_vs_sma']:+.1f}%")
        print(f"Volatility: {regime_info['vix_proxy']:.1f}% (annualized)")
        print(f"\nTrading Rules:")
        print(f"  Entry Threshold: {rules['entry_threshold']}")
        print(f"  Position Size: {rules['position_pct'] * 100:.0f}% of account")
        print(f"  Max Hold Days: {rules['hold_days']}")
        print(f"  Max Drawdown: {rules['max_drawdown_pct']:.0f}%")
        print(f"\n{rules['description']}")
        print("=" * 70 + "\n")

    def get_regime_history(self, lookback_hours: int = 24) -> pd.DataFrame:
        """Get regime history for last N hours."""
        if not self.regime_history:
            return pd.DataFrame()

        history_df = pd.DataFrame(self.regime_history)
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        history_df = history_df[history_df["timestamp"] >= cutoff_time]

        return history_df


if __name__ == "__main__":
    manager = MarketRegimeManager()

    regime_info = manager.get_current_regime()
    print(f"\nCurrent Regime: {regime_info['regime']}")
    print(f"Confidence: {regime_info['confidence']:.0f}%")

    rules = manager.get_trading_rules()
    print(f"\nEntry Threshold: {rules['entry_threshold']}")
    print(f"Position Size: {rules['position_pct'] * 100:.0f}%")
    print(f"Max Hold Days: {rules['hold_days']}")

    if manager.should_trade():
        print("\n✅ Trading enabled")
    else:
        print("\n⛔ Trading disabled - market in crash")

    manager.print_status()
