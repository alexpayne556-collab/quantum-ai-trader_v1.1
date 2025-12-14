# quantum_trader.py
# QUANTUM AI TRADER - OPTION E (PRODUCTION READY)
# Copy this entire file into Windsurf and save as quantum_trader.py
# Run: python quantum_trader.py

"""
QUANTUM AI TRADER - OPTION E
Complete standalone trading signal system
No dependencies except: pandas, numpy, yfinance

Install: pip install pandas numpy yfinance

Usage:
  python quantum_trader.py

Output: Creates ensemble_signals_*.csv with all signals
"""


import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")


def _print_banner() -> None:
    """Print startup banner (only when run directly)."""
    print("\n" + "=" * 120)
    print("🚀 QUANTUM AI TRADER - OPTION E (STANDALONE)")
    print("=" * 120)
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("System: Continuous Scoring with Partial Credit")
    print("=" * 120 + "\n")


# ==================================================================================
# TECHNICAL INDICATORS
# ==================================================================================


class TechnicalIndicators:
    """Calculate technical indicators."""

    @staticmethod
    def rsi(closes, period=14):
        """RSI (14-period)."""
        if len(closes) < period + 1:
            return None
        deltas = np.diff(closes[-period - 1 :])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100 if avg_gain > 0 else 0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def ema(closes, period=12):
        """EMA calculation."""
        if len(closes) < period:
            return None
        multiplier = 2 / (period + 1)
        ema_val = np.mean(closes[:period])
        for i in range(period, len(closes)):
            ema_val = closes[i] * multiplier + ema_val * (1 - multiplier)
        return ema_val


# ==================================================================================
# PATTERN SCORING (OPTION E)
# ==================================================================================


class PatternScorer:
    """Score patterns with continuous 0-100 scale and partial credit."""

    @staticmethod
    def pattern_1_gap_volume_range(df):
        """Pattern 1: Gap + Volume + Range (max 100)."""
        if len(df) < 2:
            return 0
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        gap_pct = abs((today["Open"] - yesterday["Close"]) / yesterday["Close"] * 100)
        gap_score = min(gap_pct / 3.0 * 30, 30)

        vol_20_avg = df["Volume"].tail(20).mean()
        vol_ratio = today["Volume"] / vol_20_avg if vol_20_avg > 0 else 0
        vol_score = min(max(0, (vol_ratio - 1.0) / 2.0 * 30), 30)

        daily_range = (
            (today["High"] - today["Low"]) / today["Close"] * 100
            if today["Close"] > 0
            else 0
        )
        range_20_avg = (
            ((df["High"] - df["Low"]) / df["Close"] * 100).tail(20).mean()
        )
        range_ratio = daily_range / range_20_avg if range_20_avg > 0 else 0
        range_score = min(range_ratio / 2.0 * 20, 20)

        intraday_move = (
            abs((today["Close"] - today["Open"]) / today["Open"] * 100)
            if today["Open"] > 0
            else 0
        )
        intra_score = min(intraday_move / 3.0 * 20, 20)

        return min(gap_score + vol_score + range_score + intra_score, 100)

    @staticmethod
    def pattern_2_volume_spike(df):
        """Pattern 2: Volume Spike (max 100)."""
        if len(df) < 21:
            return 0
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        vol_20_avg = df["Volume"].tail(20).mean()
        vol_ratio = today["Volume"] / vol_20_avg if vol_20_avg > 0 else 0
        vol_score = min(max(0, (vol_ratio - 1.0) / 2.0 * 50), 50)

        price_move = (
            abs((today["Close"] - yesterday["Close"]) / yesterday["Close"] * 100)
            if yesterday["Close"] > 0
            else 0
        )
        move_score = min(price_move / 3.0 * 30, 30)

        consecutive_down = sum(
            1
            for i in range(max(0, len(df) - 5), len(df))
            if df.iloc[i]["Close"] < df.iloc[i]["Open"]
        )
        recency_score = 20 if consecutive_down <= 2 else 10

        return min(vol_score + move_score + recency_score, 100)

    @staticmethod
    def pattern_3_momentum_breakout(df):
        """Pattern 3: Momentum Breakout (max 100)."""
        if len(df) < 25:
            return 0
        today = df.iloc[-1]
        high_20 = df["High"].tail(20).max()

        if today["Close"] > high_20:
            breakout_score = 40
        elif today["Close"] > high_20 * 0.98:
            breakout_score = 30
        elif today["Close"] > high_20 * 0.95:
            breakout_score = 20
        elif today["Close"] > high_20 * 0.90:
            breakout_score = 10
        else:
            breakout_score = 0

        rsi = TechnicalIndicators.rsi(df["Close"].values)
        if rsi is None:
            return 0

        if 50 <= rsi <= 85:
            rsi_score = 35
        elif 40 <= rsi < 50 or 85 < rsi <= 95:
            rsi_score = 20
        elif 30 <= rsi < 40 or 95 < rsi <= 100:
            rsi_score = 10
        else:
            rsi_score = 0

        ema_12 = TechnicalIndicators.ema(df["Close"].values, 12)
        ema_26 = TechnicalIndicators.ema(df["Close"].values, 26)

        if ema_12 and ema_26:
            macd = ema_12 - ema_26
            macd_score = 25 if macd > 0 else 0
        else:
            macd_score = 0

        return min(breakout_score + rsi_score + macd_score, 100)

    @staticmethod
    def pattern_4_support_bounce(df):
        """Pattern 4: Support Bounce (max 100)."""
        if len(df) < 51:
            return 0
        today = df.iloc[-1]
        yesterday = df.iloc[-2]
        ma50 = df["Close"].tail(50).mean()

        if today["Close"] > ma50 and yesterday["Close"] < ma50:
            bounce_score = 40
        elif today["Close"] > ma50 and yesterday["Close"] < ma50 * 1.02:
            bounce_score = 30
        elif today["Close"] > ma50:
            bounce_score = 20
        else:
            bounce_score = 0

        vol_20_avg = df["Volume"].tail(20).mean()
        vol_spike = today["Volume"] / vol_20_avg if vol_20_avg > 0 else 1.0
        vol_score = min(max(0, (vol_spike - 1.0) / 2.0 * 30), 30)

        rsi = TechnicalIndicators.rsi(df["Close"].values)
        if rsi is None:
            return 0

        if rsi < 30:
            rsi_score = 30
        elif rsi < 40:
            rsi_score = 20
        elif rsi < 50:
            rsi_score = 10
        else:
            rsi_score = 0

        return min(bounce_score + vol_score + rsi_score, 100)

    @staticmethod
    def pattern_5_gap_fill(df):
        """Pattern 5: Gap Fill (max 100)."""
        if len(df) < 21:
            return 0
        today = df.iloc[-1]
        yesterday = df.iloc[-2]

        gap_up = (today["Open"] - yesterday["Close"]) / yesterday["Close"] * 100
        if gap_up < 0.5:
            return 0

        gap_score = min(gap_up / 5.0 * 30, 30)

        vol_20_avg = df["Volume"].tail(20).mean()
        vol_ratio = today["Volume"] / vol_20_avg if vol_20_avg > 0 else 0
        vol_score = min(max(0, (vol_ratio - 1.0) / 2.0 * 30), 30)

        day_range = today["High"] - today["Low"]
        if day_range > 0:
            close_pos = (today["Close"] - today["Low"]) / day_range
            close_score = max(0, (1.0 - close_pos) * 40)
        else:
            close_score = 0

        return min(gap_score + vol_score + close_score, 100)

    @staticmethod
    def pattern_6_new_highs(df):
        """Pattern 6: New Highs (max 100)."""
        if len(df) < 200:
            return 0
        today = df.iloc[-1]
        ath = df["High"].max()

        if today["Close"] > ath * 0.99:
            return 50
        elif today["Close"] > ath * 0.95:
            return 40
        elif today["Close"] > ath * 0.90:
            return 30
        elif today["Close"] > df["High"].tail(52).max():
            return 20
        return 0


# ==================================================================================
# ENSEMBLE VOTING
# ==================================================================================


class EnsembleVoter:
    """Ensemble voting system."""

    def __init__(self):
        self.scorer = PatternScorer()

    def score_ticker(self, df, symbol, current_date):
        """Score a ticker on a specific date."""
        if len(df) < 2:
            return None

        scores = {
            "gap_volume": self.scorer.pattern_1_gap_volume_range(df),
            "volume_spike": self.scorer.pattern_2_volume_spike(df),
            "momentum": self.scorer.pattern_3_momentum_breakout(df),
            "bounce": self.scorer.pattern_4_support_bounce(df),
            "gap_fill": self.scorer.pattern_5_gap_fill(df),
            "new_highs": self.scorer.pattern_6_new_highs(df),
        }

        # OPTION E: Minimum 3/6 votes to qualify
        voting_patterns = [name for name, score in scores.items() if score > 40]
        vote_count = len(voting_patterns)

        if vote_count < 3:
            return None

        avg_score = np.mean(list(scores.values()))

        # Determine strength
        if avg_score >= 65:
            strength = "VERY_STRONG"
        elif avg_score >= 55:
            strength = "STRONG"
        elif avg_score >= 45:
            strength = "MODERATE"
        else:
            strength = "WEAK"

        return {
            "symbol": symbol,
            "date": current_date.strftime("%Y-%m-%d"),
            "price": round(df["Close"].iloc[-1], 2),
            "gap_volume": round(scores["gap_volume"], 1),
            "volume_spike": round(scores["volume_spike"], 1),
            "momentum": round(scores["momentum"], 1),
            "bounce": round(scores["bounce"], 1),
            "gap_fill": round(scores["gap_fill"], 1),
            "new_highs": round(scores["new_highs"], 1),
            "avg_score": round(avg_score, 1),
            "votes": vote_count,
            "voting_patterns": voting_patterns,
            "strength": strength,
        }


# ==================================================================================
# BACKTEST ENGINE
# ==================================================================================


class BacktestEngine:
    """Backtest the ensemble system."""

    def __init__(self):
        self.voter = EnsembleVoter()

    def run(self, tickers, end_date, lookback_days=365):
        """Run backtest on multiple tickers."""
        start_date = end_date - timedelta(days=lookback_days)
        results = []

        print(f"Testing {len(tickers)} tickers")
        print(
            f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
        )

        for idx, ticker in enumerate(sorted(tickers), 1):
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                if len(df) < 50:
                    print(f"  [{idx:2d}/{len(tickers)}] {ticker:8s} ✗ (insufficient data)")
                    continue

                signals = []
                for i in range(50, len(df)):
                    day_df = df.iloc[: i + 1]
                    score = self.voter.score_ticker(day_df, ticker, day_df.index[-1])
                    if score:
                        signals.append(score)

                print(f"  [{idx:2d}/{len(tickers)}] {ticker:8s} ✓ ({len(signals)} signals)")
                results.extend(signals)

            except Exception:
                print(f"  [{idx:2d}/{len(tickers)}] {ticker:8s} ✗ (error)")

        return results

    @staticmethod
    def analyze(results):
        """Analyze and display results."""
        if not results:
            print("\n❌ No signals generated\n")
            return

        df_results = pd.DataFrame(results)

        print("\n" + "=" * 120)
        print("📊 RESULTS - OPTION E")
        print("=" * 120 + "\n")

        print(f"Total signals: {len(df_results)}")
        print(f"Unique tickers: {df_results['symbol'].nunique()}")
        print(f"Date range: {df_results['date'].min()} to {df_results['date'].max()}")
        print(f"Average score: {df_results['avg_score'].mean():.1f}/100")
        print(f"Average votes: {df_results['votes'].mean():.2f}/6\n")

        print("Strength distribution:")
        for strength in ["VERY_STRONG", "STRONG", "MODERATE", "WEAK"]:
            count = len(df_results[df_results["strength"] == strength])
            if count > 0:
                pct = (count / len(df_results)) * 100
                print(f"  {strength:15s}: {count:4d} ({pct:5.1f}%)")
        print()

        print("Top 20 signals:")
        print(
            f"{'#':<3} {'Date':<12} {'Symbol':<8} {'Price':<10} {'Score':<8} {'Votes':<6} {'Strength'}"
        )
        print("-" * 80)

        top = df_results.nlargest(20, "avg_score")
        for num, (_, row) in enumerate(top.iterrows(), 1):
            print(
                f"{num:<3} {row['date']:<12} {row['symbol']:<8} ${row['price']:<9.2f} "
                f"{row['avg_score']:<8.1f} {row['votes']:<6} {row['strength']}"
            )

        print("\n" + "=" * 120 + "\n")

        # Export
        filename = f"ensemble_signals_optione_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_results.to_csv(filename, index=False)
        print(f"✅ Exported {len(df_results)} signals to {filename}\n")


# ==================================================================================
# MAIN EXECUTION
# ==================================================================================

if __name__ == "__main__":
    _print_banner()
    
    # Tickers to analyze
    TICKERS = [
        "PLTR",
        "HOOD",
        "WDC",
        "SNDK",
        "IREN",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
        "META",
        "AMD",
        "CRM",
        "ADBE",
        "RIOT",
        "MARA",
        "COIN",
        "STX",
        "AVGO",
        "UPST",
        "RBLX",
        "DASH",
        "UBER",
        "ASML",
    ]

    TICKERS = list(set(TICKERS))

    # Run backtest
    engine = BacktestEngine()
    end_date = datetime.now()
    results = engine.run(TICKERS, end_date, lookback_days=365)

    # Analyze
    BacktestEngine.analyze(results)

    print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 120 + "\n")
