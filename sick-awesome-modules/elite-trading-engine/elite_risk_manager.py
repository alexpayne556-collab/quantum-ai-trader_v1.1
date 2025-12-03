import logging
import math
from typing import List, Dict, Any

from backend.modules.elite.elite_data_fetcher import EliteDataFetcher
from backend.modules.elite.elite_signal_generator import EliteSignalGenerator


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


class EliteRiskManager:
    """Calculate position sizes for elite trading signals based on risk rules.

    This module enforces per-trade risk and total portfolio heat constraints to
    prevent oversized positions and excessive aggregate risk.
    """

    def __init__(
        self,
        account_value: float,
        risk_per_trade: float = 0.02,
        max_portfolio_heat: float = 0.10,
    ) -> None:
        """Initialize the risk manager.

        Args:
            account_value: Total account value in account currency.
            risk_per_trade: Fraction of account to risk per trade (default 0.02 = 2%).
            max_portfolio_heat: Maximum fraction of account allocated across
                all open positions (default 0.10 = 10%).
        """
        if account_value <= 0:
            raise ValueError("account_value must be positive")
        if not (0 < risk_per_trade <= 1):
            raise ValueError("risk_per_trade must be between 0 and 1")
        if not (0 < max_portfolio_heat <= 1):
            raise ValueError("max_portfolio_heat must be between 0 and 1")

        self.account_value = float(account_value)
        self.risk_per_trade = float(risk_per_trade)
        self.max_portfolio_heat = float(max_portfolio_heat)

    # ------------------------------------------------------------------
    # Advanced sizing utilities
    # ------------------------------------------------------------------
    def calculate_kelly_size(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        signal_confidence: float,
    ) -> float:
        """Calculate Half-Kelly position size, scaled by confidence.

        Formula: Kelly % = (Win Rate × Avg Win - Loss Rate × Avg Loss) / Avg Win

        Safety modifications:
        - Half-Kelly (× 0.5)
        - Scaled by signal confidence (0–1)
        - Clipped to [1%, 10%]
        """
        win_rate = max(0.0, min(1.0, float(win_rate)))
        loss_rate = 1.0 - win_rate
        avg_win_pct = float(avg_win_pct)
        avg_loss_pct = float(avg_loss_pct)
        signal_confidence = max(0.0, min(1.0, float(signal_confidence)))

        if avg_win_pct <= 0:
            return 0.01

        kelly_pct = (win_rate * avg_win_pct - loss_rate * abs(avg_loss_pct)) / avg_win_pct
        kelly_pct *= 0.5  # Half-Kelly
        kelly_pct *= signal_confidence

        kelly_pct = max(0.01, min(0.10, kelly_pct))
        return kelly_pct

    def calculate_atr_adjusted_size(
        self,
        base_size_pct: float,
        current_atr: float,
        avg_atr: float,
        price: float,
    ) -> float:
        """Adjust position size based on volatility (ATR).

        - High volatility (ATR > average) → reduce size
        - Low volatility (ATR < average) → increase size
        """
        if price <= 0 or current_atr <= 0 or avg_atr <= 0:
            return max(0.01, min(0.15, base_size_pct))

        atr_pct = (current_atr / price) * 100.0
        avg_atr_pct = (avg_atr / price) * 100.0
        if avg_atr_pct <= 0:
            volatility_ratio = 1.0
        else:
            volatility_ratio = atr_pct / avg_atr_pct

        if volatility_ratio > 1.5:
            adjusted_size = base_size_pct * 0.5
        elif volatility_ratio > 1.2:
            adjusted_size = base_size_pct * 0.75
        elif volatility_ratio < 0.8:
            adjusted_size = base_size_pct * 1.25
        else:
            adjusted_size = base_size_pct

        adjusted_size = max(0.01, min(0.15, adjusted_size))
        return adjusted_size

    def calculate_regime_adjusted_size(self, base_size_pct: float, market_regime: str) -> float:
        """Adjust position size based on market regime.

        Regimes:
        - BULL: normal sizing
        - VOLATILE: reduce 25%
        - CRISIS: reduce 50%
        - BEAR: reduce 40%
        """
        regime_multipliers = {
            "BULL": 1.0,
            "VOLATILE": 0.75,
            "CRISIS": 0.5,
            "BEAR": 0.6,
        }
        multiplier = regime_multipliers.get(market_regime, 0.75)
        return base_size_pct * multiplier

    def check_portfolio_correlation(
        self,
        new_ticker: str,
        existing_positions: List[str],
        correlation_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Simplified correlation check using sector groupings.

        Returns a dict with:
            {'allowed': bool, 'correlation': float, 'conflicting_tickers': List[str]}
        """
        if not existing_positions:
            return {"allowed": True, "correlation": 0.0, "conflicting_tickers": []}

        sector_map = {
            "SPY": "INDEX",
            "QQQ": "INDEX",
            "IWM": "INDEX",
            "AAPL": "TECH",
            "MSFT": "TECH",
            "GOOGL": "TECH",
            "META": "TECH",
            "NVDA": "TECH",
            "AMD": "TECH",
            "INTC": "TECH",
            "TSLA": "AUTO",
            "F": "AUTO",
            "GM": "AUTO",
            "JPM": "FINANCE",
            "BAC": "FINANCE",
            "WFC": "FINANCE",
        }

        new_sector = sector_map.get(new_ticker, "OTHER")
        conflicting: List[str] = []
        for ticker in existing_positions:
            if sector_map.get(ticker, "OTHER") == new_sector and new_sector != "OTHER":
                conflicting.append(ticker)

        if len(conflicting) >= 2:
            return {
                "allowed": False,
                "correlation": max(correlation_threshold, 0.8),
                "conflicting_tickers": conflicting,
            }

        return {"allowed": True, "correlation": 0.3, "conflicting_tickers": conflicting}

    def calculate_position_size(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal position size using Kelly, volatility, and regime.

        Returns the signal dict enriched with:
            shares, position_value, position_size_pct, risk_amount,
            sizing_method, kelly_pct
        """
        ticker = str(signal.get("ticker", ""))
        try:
            entry_price = float(signal.get("entry_price", 0.0))
        except (TypeError, ValueError):
            entry_price = 0.0

        if entry_price <= 0:
            raise ValueError(f"Invalid entry_price for {ticker}")

        confidence = float(signal.get("confidence", 0.7))
        strategy = str(signal.get("strategy", "MEAN_REVERSION"))

        # Strategy-specific win rates (can be refined with backtest stats)
        win_rates = {
            "MEAN_REVERSION": 0.76,
            "MOMENTUM_BREAKOUT": 0.64,
            "ICHIMOKU_CLOUD": 0.70,
            "VOLUME_SPIKE_REVERSAL": 0.75,
            "BOLLINGER_SQUEEZE": 0.72,
        }
        win_rate = win_rates.get(strategy, 0.70)
        avg_win_pct = 0.05
        avg_loss_pct = -0.02

        kelly_size = self.calculate_kelly_size(win_rate, avg_win_pct, avg_loss_pct, confidence)

        # Volatility adjustment if ATR is available
        current_atr = float(signal.get("atr_14", 0.0))
        avg_atr = float(signal.get("atr_14_avg", current_atr or 0.0))
        if current_atr > 0 and avg_atr > 0:
            base_size_pct = self.calculate_atr_adjusted_size(
                kelly_size, current_atr=current_atr, avg_atr=avg_atr, price=entry_price
            )
        else:
            base_size_pct = kelly_size

        # Regime-aware adjustment (placeholder regime; integrate VIX later)
        market_regime = str(signal.get("market_regime", "BULL"))
        regime_adjusted_size = self.calculate_regime_adjusted_size(base_size_pct, market_regime)

        # Simple correlation check (no existing positions wired here yet)
        existing_positions: List[str] = signal.get("existing_positions", [])
        correlation_check = self.check_portfolio_correlation(ticker, existing_positions)
        if not correlation_check["allowed"]:
            raise ValueError(
                f"Position rejected for {ticker} due to high sector correlation with {correlation_check['conflicting_tickers']}"
            )

        position_size_pct = regime_adjusted_size
        position_value = self.account_value * position_size_pct
        shares = math.floor(position_value / entry_price)

        if shares < 1:
            shares = 1
            position_value = entry_price
            position_size_pct = position_value / self.account_value

        # Risk calculation
        stop_loss = signal.get("stop_loss")
        if stop_loss is not None:
            try:
                stop_loss_value = float(stop_loss)
            except (TypeError, ValueError):
                stop_loss_value = None
        else:
            stop_loss_value = None

        if stop_loss_value is not None and stop_loss_value < entry_price:
            risk_per_share = entry_price - stop_loss_value
            risk_amount = risk_per_share * shares
        else:
            # Fallback: use configured per-trade risk on positioned capital
            risk_amount = position_value * self.risk_per_trade

        enriched = dict(signal)
        enriched.update(
            {
                "shares": int(shares),
                "position_value": float(position_value),
                "position_size_pct": float(position_size_pct * 100.0),
                "risk_amount": float(risk_amount),
                "sizing_method": "Kelly + Volatility + Regime",
                "kelly_pct": float(kelly_size * 100.0),
            }
        )

        logger.info(
            "Position sized for %s: %d shares (%.2f%% of account)",
            ticker,
            shares,
            position_size_pct * 100.0,
        )

        return enriched

    def check_portfolio_heat(self, active_positions: List[Dict[str, Any]]) -> float:
        """Calculate total portfolio heat from active positions.

        Heat is defined as the sum of risk_amount across positions
        divided by account value.

        Args:
            active_positions: List of position dicts with risk_amount.

        Returns:
            Total heat as a fraction of account (e.g., 0.08 = 8%).
        """
        total_risk = 0.0
        for pos in active_positions:
            try:
                risk_amount = float(pos.get("risk_amount", 0.0))
            except (TypeError, ValueError):
                risk_amount = 0.0
            total_risk += risk_amount

        heat = total_risk / self.account_value
        return heat

    def size_positions(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply position sizing to a list of trading signals.

        Args:
            signals: List of signal dictionaries from EliteSignalGenerator.

        Returns:
            List of enriched signal dictionaries including position sizing
            fields, filtered and adjusted to respect the portfolio heat cap.
        """
        sized_signals: List[Dict[str, Any]] = []
        current_heat = 0.0  # fraction of account currently allocated

        max_heat_value = self.account_value * self.max_portfolio_heat
        per_trade_risk_amount = self.account_value * self.risk_per_trade

        for signal in signals:
            entry_price = float(signal.get("entry_price", 0.0))
            if entry_price <= 0:
                logger.warning("Invalid entry_price for %s; skipping", signal.get("ticker"))
                continue

            stop_loss = signal.get("stop_loss")

            if stop_loss is not None:
                try:
                    stop_loss_value = float(stop_loss)
                except (TypeError, ValueError):
                    logger.warning("Invalid stop_loss for %s; treating as no stop.", signal.get("ticker"))
                    stop_loss_value = None
            else:
                stop_loss_value = None

            if stop_loss_value is not None:
                risk_per_share = entry_price - stop_loss_value
                if risk_per_share <= 0:
                    logger.warning(
                        "Non-positive risk_per_share for %s (entry=%s, stop=%s); skipping",
                        signal.get("ticker"),
                        entry_price,
                        stop_loss_value,
                    )
                    continue
            else:
                # Mean reversion without explicit stop: 2% of entry as ATR proxy
                risk_per_share = entry_price * 0.02

            raw_shares = per_trade_risk_amount / risk_per_share
            shares = math.floor(raw_shares)

            if shares <= 0:
                logger.warning(
                    "Computed zero shares for %s at entry %.2f; skipping",
                    signal.get("ticker"),
                    entry_price,
                )
                continue

            position_value = shares * entry_price
            proposed_heat_value = current_heat * self.account_value + position_value

            if proposed_heat_value > max_heat_value:
                # Need to reduce position size to stay within max_portfolio_heat
                remaining_heat_value = max_heat_value - current_heat * self.account_value
                if remaining_heat_value <= 0:
                    logger.warning(
                        "Max portfolio heat reached; skipping signal for %s",
                        signal.get("ticker"),
                    )
                    continue

                max_affordable_shares = math.floor(remaining_heat_value / entry_price)
                if max_affordable_shares <= 0:
                    logger.warning(
                        "Remaining portfolio heat too small for %s; skipping",
                        signal.get("ticker"),
                    )
                    continue

                logger.warning(
                    "Reducing position size for %s from %d to %d shares to respect portfolio heat.",
                    signal.get("ticker"),
                    shares,
                    max_affordable_shares,
                )
                shares = max_affordable_shares
                position_value = shares * entry_price

            # Update heat
            current_heat = proposed_heat_value / self.account_value

            risk_amount = per_trade_risk_amount
            risk_pct_of_account = (risk_amount / self.account_value) * 100.0
            portfolio_allocation_pct = (position_value / self.account_value) * 100.0

            enriched_signal = dict(signal)
            enriched_signal.update(
                {
                    "shares": int(shares),
                    "position_value": float(position_value),
                    "risk_amount": float(risk_amount),
                    "risk_pct_of_account": float(risk_pct_of_account),
                    "portfolio_allocation_pct": float(portfolio_allocation_pct),
                }
            )

            sized_signals.append(enriched_signal)

if __name__ == "__main__":
    # End-to-end smoke test: fetch data → generate signals → size top positions
    fetcher = EliteDataFetcher()
    data = fetcher.fetch_data(["SPY", "QQQ", "AAPL"], days=60)

    signal_gen = EliteSignalGenerator(min_confidence=0.70, backtest_mode=True)
    signals = signal_gen.generate_signals(data)

    risk_mgr = EliteRiskManager(account_value=10_000.0, risk_per_trade=0.02)

    print("\n" + "=" * 60)
    print("RISK MANAGEMENT TEST")
    print("=" * 60)

    for sig in signals[:3]:  # Top 3 signals
        sized = risk_mgr.calculate_position_size(sig)

        print(f"\n{sized['ticker']} - {sized.get('strategy', 'UNKNOWN')}")
        print(f"  Confidence: {sized.get('confidence', 0.0):.0%}")
        print(f"  Entry: ${sized['entry_price']:.2f}")
        print(f"  Kelly %: {sized.get('kelly_pct', 0.0):.1f}%")
        print(f"  Position size: {sized.get('position_size_pct', 0.0):.1f}%")
        print(f"  Shares: {sized['shares']}")
        print(f"  Position value: ${sized['position_value']:.2f}")
        print(f"  Risk: ${sized['risk_amount']:.2f}")
        print(f"  Method: {sized.get('sizing_method', 'N/A')}")

    print("\n" + "=" * 60)
    print(" RISK MANAGER WORKING")
    print("=" * 60)
