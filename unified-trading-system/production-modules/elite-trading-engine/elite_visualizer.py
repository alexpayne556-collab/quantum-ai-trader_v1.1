import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


class EliteVisualizer:
    """Visualization helpers for Elite trading modules.

    Provides simple, clear matplotlib charts for:
    - Per-ticker price/volume/RSI with signals overlaid
    - Backtest equity curve and drawdown
    """

    def __init__(self) -> None:
        # No special state for now; placeholder for future themes
        self._default_save_path = "charts/"

    # ------------------------------------------------------------------
    # Ticker visualization
    # ------------------------------------------------------------------
    def visualize_ticker(
        self,
        ticker: str,
        data: pd.DataFrame,
        signals: List[Dict[str, Any]],
        days: int = 60,
        save_path: str = "charts/",
    ) -> str:
        """Create a 3-panel chart for a single ticker.

        Panel 1: Price + SMAs + Bollinger Bands + BUY signals
        Panel 2: Volume bars
        Panel 3: RSI(2) and RSI(14)
        """
        if data.empty:
            raise ValueError("Data is empty in visualize_ticker")

        df = data[data["ticker"] == ticker].copy()
        if df.empty:
            raise ValueError(f"No data for ticker {ticker}")

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").tail(days)

        ticker_signals = [s for s in signals if s.get("ticker") == ticker]

        # Prepare figure
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

        # Panel 1: Price + SMAs + BB + BUY markers
        ax1.plot(df["date"], df["close"], label="Close", linewidth=2, color="black")

        if "sma_50" in df.columns:
            ax1.plot(df["date"], df["sma_50"], label="SMA 50", alpha=0.7, color="blue")
        if "sma_200" in df.columns:
            ax1.plot(df["date"], df["sma_200"], label="SMA 200", alpha=0.7, color="orange")

        if {"bb_lower", "bb_upper"}.issubset(df.columns):
            ax1.fill_between(
                df["date"],
                df["bb_lower"],
                df["bb_upper"],
                alpha=0.1,
                color="gray",
                label="Bollinger Bands",
            )

        for sig in ticker_signals:
            sig_date = pd.to_datetime(sig["date"])
            entry_price = float(sig.get("entry_price", 0.0))
            if entry_price <= 0:
                continue
            if sig.get("signal") == "BUY":
                ax1.scatter(sig_date, entry_price, color="green", marker="^", s=80, zorder=5)
                ax1.text(
                    sig_date,
                    entry_price * 0.98,
                    str(sig.get("strategy", ""))[:10],
                    fontsize=8,
                    ha="center",
                    va="top",
                )

        ax1.set_ylabel("Price ($)")
        ax1.set_title(f"{ticker} - Price and Signals")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        # Panel 2: Volume bars (green/red)
        closes = df["close"].values
        colors: List[str] = ["gray"]
        for i in range(1, len(closes)):
            colors.append("green" if closes[i] >= closes[i - 1] else "red")

        ax2.bar(df["date"], df["volume"], color=colors, alpha=0.5)
        ax2.set_ylabel("Volume")
        ax2.set_title("Volume")
        ax2.grid(alpha=0.3)

        # Panel 3: RSI
        if "rsi_2" in df.columns:
            ax3.plot(df["date"], df["rsi_2"], label="RSI(2)", linewidth=2, color="purple")
        if "rsi_14" in df.columns:
            ax3.plot(df["date"], df["rsi_14"], label="RSI(14)", alpha=0.7, color="brown")

        ax3.axhline(70, color="red", linestyle="--", alpha=0.5, label="Overbought")
        ax3.axhline(30, color="green", linestyle="--", alpha=0.5, label="Oversold")
        ax3.set_ylabel("RSI")
        ax3.set_xlabel("Date")
        ax3.set_title("RSI Indicators")
        ax3.legend(loc="upper left")
        ax3.grid(alpha=0.3)

        plt.tight_layout()

        os.makedirs(save_path, exist_ok=True)
        last_date = df["date"].iloc[-1]
        filename = os.path.join(save_path, f"{ticker}_{last_date.strftime('%Y%m%d')}.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

        return filename

    # ------------------------------------------------------------------
    # Equity curve visualization
    # ------------------------------------------------------------------
    def plot_equity_curve(self, backtest_results: Dict[str, Any], save_path: str = "charts/") -> str:
        """Plot backtest equity curve with drawdown shading."""
        equity_curve = backtest_results.get("equity_curve") or backtest_results.get("equity")
        dates = backtest_results.get("dates")
        if not equity_curve or not dates:
            raise ValueError("Backtest results must contain 'equity_curve' and 'dates'")

        eq_series = pd.Series(equity_curve, index=pd.to_datetime(dates))
        running_max = eq_series.expanding().max()
        drawdown = (eq_series - running_max) / running_max * 100.0

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Equity curve
        ax1.plot(eq_series.index, eq_series.values, linewidth=2, label="Portfolio Value")
        initial_capital = float(backtest_results.get("initial_capital", eq_series.iloc[0]))
        ax1.axhline(initial_capital, color="gray", linestyle="--", alpha=0.5, label="Starting Capital")
        ax1.set_ylabel("Portfolio Value ($)")
        ax1.set_title("Backtest Equity Curve")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        # Drawdown panel
        ax2.fill_between(eq_series.index, drawdown, 0, where=drawdown < 0, color="red", alpha=0.3)
        ax2.plot(eq_series.index, drawdown, color="red", linewidth=1)
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.set_title("Drawdown")
        ax2.grid(alpha=0.3)

        plt.tight_layout()

        os.makedirs(save_path, exist_ok=True)
        filename = os.path.join(save_path, "equity_curve.png")
        plt.savefig(filename, dpi=150)
        plt.close(fig)

        return filename


if __name__ == "__main__":
    from backend.modules.elite.elite_data_fetcher import EliteDataFetcher
    from backend.modules.elite.elite_signal_generator import EliteSignalGenerator

    print("=" * 60)
    print("ELITE VISUALIZER TEST")
    print("=" * 60)

    fetcher = EliteDataFetcher()
    data = fetcher.fetch_data(["SPY", "AAPL"], days=90)

    gen = EliteSignalGenerator(min_confidence=0.70, backtest_mode=True)
    signals = gen.generate_signals(data)

    print(f"\nCreating charts for {len(signals)} signals...")

    visualizer = EliteVisualizer()

    for ticker in ["SPY", "AAPL"]:
        if (data["ticker"] == ticker).any():
            filename = visualizer.visualize_ticker(ticker, data, signals, days=90)
            print(f"✅ Chart saved: {filename}")

    import os as _os
    import json as _json

    if _os.path.exists("elite_backtest_results.json"):
        with open("elite_backtest_results.json", "r", encoding="utf-8") as f:
            results = _json.load(f)
        filename = visualizer.plot_equity_curve(results)
        print(f"✅ Equity curve saved: {filename}")

    print("\n✅ VISUALIZER WORKING")
