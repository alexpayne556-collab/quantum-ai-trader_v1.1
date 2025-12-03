import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, date, time
from typing import List, Dict, Any

import numpy as np
import pandas as pd

try:
    from .elite_signal_generator import EliteSignalGenerator
    from .elite_risk_manager import EliteRiskManager
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    elite_dir = Path(__file__).parent
    sys.path.insert(0, str(elite_dir))
    from elite_signal_generator import EliteSignalGenerator
    from elite_risk_manager import EliteRiskManager


logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


SLIPPAGE_PCT = 0.0005  # 0.05% per trade


@dataclass
class Position:
    ticker: str
    entry_date: datetime
    entry_price: float
    shares: int
    stop_loss: float | None
    target_price: float
    days_held: int = 0
    consecutive_up_days: int = 0


class EliteBacktestEngine:
    """Backtest engine for the elite mean reversion strategy.

    Walks historical data day by day, enters positions on next open after
    signals, exits on rule-based conditions, and computes performance metrics.
    """

    def __init__(self, initial_capital: float = 3000.0) -> None:
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital for the backtest.
        """
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        self.initial_capital = float(initial_capital)

    def run_backtest(
        self,
        data: pd.DataFrame,
        signal_generator: EliteSignalGenerator,
        risk_manager: EliteRiskManager,
    ) -> Dict[str, Any]:
        """Run backtest on historical data.

        Args:
            data: Historical OHLCV+indicator DataFrame from EliteDataFetcher.
            signal_generator: Configured EliteSignalGenerator instance.
            risk_manager: Configured EliteRiskManager instance.

        Returns:
            Results dictionary with full performance metrics and trade list.
        """
        if data.empty:
            raise ValueError("Input data is empty")

        df = data.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

        dates = sorted(df["date"].unique())

        cash = self.initial_capital
        equity_curve: List[float] = []
        open_positions: List[Position] = []
        trades: List[Dict[str, Any]] = []

        # For 2-consecutive-up-days logic we track per-ticker previous close
        prev_close: Dict[str, float] = {}

        for i, current_date in enumerate(dates):
            day_data = df[df["date"] == current_date]

            # Log progress every 10 days
            if i % 10 == 0:
                logger.info("Processing date %s (%d/%d)", current_date.date(), i + 1, len(dates))

            # 1) Exit checks at today's close
            new_open_positions: List[Position] = []
            for pos in open_positions:
                ticker_data = day_data[day_data["ticker"] == pos.ticker]
                if ticker_data.empty:
                    new_open_positions.append(pos)
                    continue

                row = ticker_data.iloc[-1]

                # Update days_held and consecutive_up_days
                pos.days_held += 1
                prev_c = prev_close.get(pos.ticker)
                today_close = float(row["close"])
                if prev_c is not None and today_close > prev_c:
                    pos.consecutive_up_days += 1
                else:
                    pos.consecutive_up_days = 0
                prev_close[pos.ticker] = today_close

                if self._check_exit(pos, row):
                    exit_price = today_close * (1 - SLIPPAGE_PCT)
                    pnl = (exit_price - pos.entry_price) * pos.shares
                    cash += pnl

                    trades.append(
                        {
                            "ticker": pos.ticker,
                            "entry_date": pos.entry_date,
                            "exit_date": datetime.combine(current_date.date(), time.min),
                            "entry_price": pos.entry_price,
                            "exit_price": exit_price,
                            "shares": pos.shares,
                            "pnl": pnl,
                            "hold_days": pos.days_held,
                        }
                    )
                else:
                    new_open_positions.append(pos)

            open_positions = new_open_positions

            # 2) Entry checks - signals based on data up to and including current_date
            # We will enter at next day's open, so skip on last date
            if i < len(dates) - 1:
                history_to_date = df[df["date"] <= current_date]
                signals = signal_generator.generate_signals(history_to_date)
                # Filter signals for this specific date (close of current_date)
                signals_today = [s for s in signals if pd.to_datetime(s["date"]).normalize() == current_date]

                # Size positions and record to enter next day
                next_date = dates[i + 1]
                next_day_data = df[df["date"] == next_date]

                for sig in signals_today:
                    try:
                        sized = risk_manager.calculate_position_size(sig)
                    except ValueError:
                        continue

                    ticker = sized["ticker"]
                    ticker_next = next_day_data[next_day_data["ticker"] == ticker]
                    if ticker_next.empty:
                        continue

                    row_next = ticker_next.iloc[0]
                    # Enter at next day's open + slippage
                    entry_price = float(row_next["open"]) * (1 + SLIPPAGE_PCT)
                    shares = sized["shares"]
                    cost = entry_price * shares

                    if cost > cash:
                        logger.warning("Insufficient cash to enter %s; skipping", ticker)
                        continue

                    cash -= cost
                    open_positions.append(
                        Position(
                            ticker=ticker,
                            entry_date=datetime.combine(next_date.date(), time.min),
                            entry_price=entry_price,
                            shares=shares,
                            stop_loss=sig.get("stop_loss"),
                            target_price=sig.get("target_price", entry_price),
                        )
                    )

            # 3) Update equity curve at end of day
            equity = cash
            for pos in open_positions:
                ticker_data = day_data[day_data["ticker"] == pos.ticker]
                if ticker_data.empty:
                    continue
                last_close = float(ticker_data.iloc[-1]["close"])
                equity += last_close * pos.shares

            equity_curve.append(equity)

        # Force-close any remaining open positions at the final date's close
        if open_positions:
            last_date = dates[-1]
            last_day_data = df[df["date"] == last_date]
            for pos in open_positions:
                ticker_data = last_day_data[last_day_data["ticker"] == pos.ticker]
                if ticker_data.empty:
                    continue
                row = ticker_data.iloc[-1]
                last_close = float(row["close"])
                exit_price = last_close * (1 - SLIPPAGE_PCT)
                pnl = (exit_price - pos.entry_price) * pos.shares
                cash += pnl

                trades.append(
                    {
                        "ticker": pos.ticker,
                        "entry_date": pos.entry_date,
                        "exit_date": datetime.combine(last_date.date(), time.min),
                        "entry_price": pos.entry_price,
                        "exit_price": exit_price,
                        "shares": pos.shares,
                        "pnl": pnl,
                        "hold_days": pos.days_held,
                    }
                )

            # Recompute final equity including forced exits
            equity_curve[-1] = cash

        results = self._calculate_metrics(trades, equity_curve, dates)
        return results

    def _check_exit(self, position: Position, current_row: pd.Series) -> bool:
        """Returns True if position should be closed based on exit rules.

        Exit checks:
        1. RSI(2) > 40 (target reached)
        2. 2 consecutive up closes
        3. 5 days max hold
        """
        if position.days_held >= 5:
            return True

        rsi_2 = float(current_row.get("rsi_2", 0.0))
        if rsi_2 > 40.0:
            return True

        if position.consecutive_up_days >= 2:
            return True

        return False

    def _calculate_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: List[float],
        dates: List[pd.Timestamp],
    ) -> Dict[str, Any]:
        """Calculate all performance metrics for the backtest."""
        equity_series = pd.Series(equity_curve, index=dates)

        total_trades = len(trades)
        if total_trades == 0:
            drawdown_analysis = self._analyze_drawdowns(equity_curve, dates)
            return {
                "initial_capital": self.initial_capital,
                "final_capital": float(equity_series.iloc[-1]) if not equity_series.empty else self.initial_capital,
                "total_return_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "largest_win": 0.0,
                "largest_loss": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "profit_factor": 0.0,
                "avg_hold_days": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "expectancy": 0.0,
                "drawdown_analysis": drawdown_analysis,
                "monte_carlo": {
                    "median_return": 0.0,
                    "worst_case_dd": 0.0,
                    "best_case_return": 0.0,
                    "probability_of_profit": 0.0,
                    "simulations_run": 0,
                },
                "trades": [],
                "equity_curve": equity_curve,
                "dates": dates,
            }

        # Ensure each trade has a pnl_pct field for expectancy & Monte Carlo
        for t in trades:
            if "pnl_pct" not in t:
                entry_value = t.get("entry_price", 0.0) * t.get("shares", 0)
                if entry_value > 0:
                    t["pnl_pct"] = (t.get("pnl", 0.0) / entry_value) * 100.0
                else:
                    t["pnl_pct"] = 0.0

        pnls = np.array([t["pnl"] for t in trades], dtype=float)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]

        winning_trades = int((pnls > 0).sum())
        losing_trades = int((pnls < 0).sum())

        win_rate = (winning_trades / total_trades) if total_trades > 0 else 0.0

        avg_win = float(wins.mean()) if wins.size > 0 else 0.0
        avg_loss = float(losses.mean()) if losses.size > 0 else 0.0
        largest_win = float(wins.max()) if wins.size > 0 else 0.0
        largest_loss = float(losses.min()) if losses.size > 0 else 0.0

        total_wins = wins.sum() if wins.size > 0 else 0.0
        total_losses = losses.sum() if losses.size > 0 else 0.0
        profit_factor = total_wins / abs(total_losses) if total_losses < 0 else 0.0

        # Max drawdown (percent, negative value)
        running_max = equity_series.cummax()
        drawdowns = (equity_series - running_max) / running_max * 100.0
        max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

        # Sharpe & Sortino (daily returns, annualized with 252 trading days)
        returns = equity_series.pct_change().dropna()
        if not returns.empty and returns.std() != 0:
            sharpe_ratio = float((returns.mean() / returns.std()) * math.sqrt(252))
        else:
            sharpe_ratio = 0.0

        sortino_ratio = self._calculate_sortino_ratio(returns)

        # Average hold days
        hold_days = np.array([t["hold_days"] for t in trades], dtype=float)
        avg_hold_days = float(hold_days.mean()) if hold_days.size > 0 else 0.0

        final_capital = float(equity_series.iloc[-1]) if not equity_series.empty else self.initial_capital
        total_return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100.0

        calmar_ratio = self._calculate_calmar_ratio(total_return_pct, abs(max_drawdown))
        expectancy = self._calculate_expectancy(trades)
        drawdown_analysis = self._analyze_drawdowns(equity_curve, dates)
        monte_carlo = self.run_monte_carlo(trades, n_simulations=1000)

        return {
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return_pct": float(total_return_pct),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": float(win_rate),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "profit_factor": float(profit_factor),
            "avg_hold_days": avg_hold_days,
            "sortino_ratio": float(sortino_ratio),
            "calmar_ratio": float(calmar_ratio),
            "expectancy": float(expectancy),
            "drawdown_analysis": drawdown_analysis,
            "monte_carlo": monte_carlo,
            "trades": trades,
            "equity_curve": equity_curve,
            "dates": dates,
        }

    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Sortino ratio using downside deviation only (annualized)."""
        if returns.empty:
            return 0.0

        downside = returns[returns < risk_free_rate]
        if downside.empty:
            return 0.0

        downside_std = downside.std()
        if downside_std == 0:
            return 0.0

        mean_ret = returns.mean()
        sortino = (mean_ret - risk_free_rate) / downside_std * math.sqrt(252)
        return float(sortino)

    def _calculate_calmar_ratio(self, total_return_pct: float, max_drawdown_pct: float) -> float:
        """Calmar ratio: total return / max drawdown (absolute)."""
        if max_drawdown_pct == 0:
            return 0.0
        return float(abs(total_return_pct) / max_drawdown_pct)

    def _calculate_expectancy(self, trades: List[Dict[str, Any]]) -> float:
        """Expectancy: average profit per trade in percentage terms."""
        if not trades:
            return 0.0

        pnl_pcts = [t.get("pnl_pct", 0.0) for t in trades]
        winning = [p for p in pnl_pcts if p > 0]
        losing = [p for p in pnl_pcts if p <= 0]

        win_rate = len(winning) / len(trades) if trades else 0.0
        loss_rate = 1.0 - win_rate

        avg_win_pct = float(np.mean(winning)) if winning else 0.0
        avg_loss_pct = float(np.mean([abs(p) for p in losing])) if losing else 0.0

        expectancy = (win_rate * avg_win_pct) - (loss_rate * avg_loss_pct)
        return float(expectancy)

    def _analyze_drawdowns(self, equity_curve: List[float], dates: List) -> Dict[str, Any]:
        """Detailed drawdown analysis on the equity curve."""
        if len(equity_curve) < 2:
            return {
                "max_drawdown_pct": 0.0,
                "avg_drawdown_pct": 0.0,
                "max_drawdown_duration": 0,
                "avg_recovery_days": 0.0,
                "drawdown_periods": [],
            }

        equity_series = pd.Series(equity_curve, index=dates)
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        in_drawdown = False
        drawdown_start = None
        drawdown_periods: List[Dict[str, Any]] = []

        for i, dd in enumerate(drawdown):
            if dd < 0 and not in_drawdown:
                in_drawdown = True
                drawdown_start = i
            elif dd == 0 and in_drawdown:
                in_drawdown = False
                duration = i - drawdown_start
                max_dd_in_period = drawdown[drawdown_start:i].min()
                drawdown_periods.append(
                    {
                        "start_idx": drawdown_start,
                        "end_idx": i,
                        "duration_days": int(duration),
                        "max_drawdown_pct": float(max_dd_in_period * 100.0),
                    }
                )

        max_drawdown_pct = abs(float(drawdown.min() * 100.0))
        avg_drawdown_pct = (
            abs(float(drawdown[drawdown < 0].mean() * 100.0)) if (drawdown < 0).any() else 0.0
        )

        durations = [p["duration_days"] for p in drawdown_periods]
        max_drawdown_duration = max(durations) if durations else 0
        avg_recovery_days = float(np.mean(durations)) if durations else 0.0

        return {
            "max_drawdown_pct": max_drawdown_pct,
            "avg_drawdown_pct": avg_drawdown_pct,
            "max_drawdown_duration": int(max_drawdown_duration),
            "avg_recovery_days": avg_recovery_days,
            "drawdown_periods": drawdown_periods,
        }

    def run_monte_carlo(self, trades: List[Dict[str, Any]], n_simulations: int = 1000) -> Dict[str, Any]:
        """Monte Carlo simulation: randomize trade order many times to test robustness."""
        if not trades:
            return {
                "median_return": 0.0,
                "worst_case_dd": 0.0,
                "best_case_return": 0.0,
                "probability_of_profit": 0.0,
                "simulations_run": 0,
            }

        results: List[Dict[str, float]] = []
        for _ in range(n_simulations):
            shuffled = trades.copy()
            np.random.shuffle(shuffled)

            equity = self.initial_capital
            peak = equity
            max_dd = 0.0

            for trade in shuffled:
                pnl_pct = float(trade.get("pnl_pct", 0.0)) / 100.0
                pnl = pnl_pct * equity
                equity += pnl
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)

            final_return = (equity - self.initial_capital) / self.initial_capital * 100.0
            results.append({"final_return": final_return, "max_drawdown": max_dd * 100.0})

        returns = [r["final_return"] for r in results]
        drawdowns = [r["max_drawdown"] for r in results]

        median_return = float(np.median(returns)) if returns else 0.0
        worst_case_dd = float(np.percentile(drawdowns, 95)) if drawdowns else 0.0
        best_case_return = float(np.percentile(returns, 95)) if returns else 0.0
        probability_of_profit = (
            float(sum(1 for r in returns if r > 0) / len(returns) * 100.0) if returns else 0.0
        )

        return {
            "median_return": median_return,
            "worst_case_dd": worst_case_dd,
            "best_case_return": best_case_return,
            "probability_of_profit": probability_of_profit,
            "simulations_run": n_simulations,
        }

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate a professional backtest report string from results dict."""
        m = results
        dd = m.get("drawdown_analysis", {})
        mc = m.get("monte_carlo", {})

        report = f"""
═══════════════════════════════════════════════════════════════════════════════
ELITE BACKTEST REPORT
═══════════════════════════════════════════════════════════════════════════════

CAPITAL
-------
Initial: ${m['initial_capital']:,.2f}
Final: ${m['final_capital']:,.2f}
Total Return: {m['total_return_pct']:+.1f}%

TRADE STATISTICS
----------------
Total Trades: {m['total_trades']}
Winning Trades: {m['winning_trades']} ({m['win_rate']:.1%})
Losing Trades: {m['losing_trades']}
Avg Win (P&L): ${m['avg_win']:,.2f}
Avg Loss (P&L): ${m['avg_loss']:,.2f}
Profit Factor: {m['profit_factor']:.2f}
Expectancy: {m['expectancy']:+.2f}% per trade

RISK METRICS
------------
Max Drawdown: {dd.get('max_drawdown_pct', abs(m['max_drawdown'])):.2f}%
Avg Drawdown: {dd.get('avg_drawdown_pct', 0.0):.2f}%
Longest Drawdown: {dd.get('max_drawdown_duration', 0)} days
Avg Recovery Time: {dd.get('avg_recovery_days', 0.0):.0f} days

RISK-ADJUSTED RETURNS
---------------------
Sharpe Ratio: {m['sharpe_ratio']:.2f}
Sortino Ratio: {m['sortino_ratio']:.2f}
Calmar Ratio: {m['calmar_ratio']:.2f}

MONTE CARLO ({mc.get('simulations_run', 0)} simulations)
-------------------------------
Median Return: {mc.get('median_return', 0.0):+.1f}%
Best Case (95th %ile): {mc.get('best_case_return', 0.0):+.1f}%
Worst-Case Max DD (95th %ile): {mc.get('worst_case_dd', 0.0):.1f}%
Probability of Profit: {mc.get('probability_of_profit', 0.0):.1f}%

VERDICT
-------
"""

        passed: List[str] = []
        failed: List[str] = []

        if m["win_rate"] >= 0.65:
            passed.append(f"✅ Win Rate: {m['win_rate']:.1%} (target: ≥65%)")
        else:
            failed.append(f"❌ Win Rate: {m['win_rate']:.1%} (target: ≥65%)")

        if m["profit_factor"] >= 1.5:
            passed.append(f"✅ Profit Factor: {m['profit_factor']:.2f} (target: ≥1.5)")
        else:
            failed.append(f"❌ Profit Factor: {m['profit_factor']:.2f} (target: ≥1.5)")

        if dd.get("max_drawdown_pct", abs(m["max_drawdown"])) <= 15.0:
            passed.append(
                f"✅ Max Drawdown: {dd.get('max_drawdown_pct', abs(m['max_drawdown'])):.1f}% (target: ≤15%)"
            )
        else:
            failed.append(
                f"❌ Max Drawdown: {dd.get('max_drawdown_pct', abs(m['max_drawdown'])):.1f}% (target: ≤15%)"
            )

        if m["sharpe_ratio"] >= 1.5:
            passed.append(f"✅ Sharpe Ratio: {m['sharpe_ratio']:.2f} (target: ≥1.5)")
        else:
            failed.append(f"❌ Sharpe Ratio: {m['sharpe_ratio']:.2f} (target: ≥1.5)")

        for p in passed:
            report += p + "\n"
        for f in failed:
            report += f + "\n"

        if not failed:
            report += "\n🏆 ALL CRITERIA PASSED - SYSTEM READY FOR PAPER TRADING"
        else:
            report += f"\n⚠️ {len(failed)} CRITERIA FAILED - OPTIMIZE BEFORE TRADING"

        report += "\n═══════════════════════════════════════════════════════════════════════════════"
        return report


if __name__ == "__main__":
    import sys
    try:
        from .elite_data_fetcher import EliteDataFetcher
    except ImportError:
        from elite_data_fetcher import EliteDataFetcher

    print("=" * 80)
    print("ELITE BACKTEST ENGINE - FULL SYSTEM TEST")
    print("=" * 80)

    tickers = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"]
    print(f"\nBacktesting {len(tickers)} tickers over ~250 trading days")
    print("Initial Capital: $10,000\n")

    fetcher = EliteDataFetcher()
    data = fetcher.fetch_data(tickers, days=250)

    signal_gen = EliteSignalGenerator(min_confidence=0.70, backtest_mode=True)
    risk_mgr = EliteRiskManager(account_value=10_000.0, risk_per_trade=0.02)
    engine = EliteBacktestEngine(initial_capital=10_000.0)

    results = engine.run_backtest(data, signal_gen, risk_mgr)

    report = engine.generate_report(results)
    print(report)

    # Optionally save raw results
    try:
        import json

        with open("elite_backtest_results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print("\n✅ Results saved to elite_backtest_results.json")
    except Exception as exc:  # noqa: BLE001
        print(f"\n⚠️ Could not save results to disk: {exc}")

    sys.exit(0)
