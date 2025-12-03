import os
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential


# Load environment variables from .env at project root
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic configuration if not already configured by the application
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class EliteDataFetcher:
    """Fetch OHLCV market data from premium APIs with fallback and enrich with technical indicators.

    Priority of data sources for each ticker:
    1. Polygon
    2. FinancialModelingPrep (FMP)
    3. Twelve Data
    4. yfinance (fallback)
    """

    def __init__(self) -> None:
        """Initialize the EliteDataFetcher with API keys loaded from environment variables."""
        self.polygon_api_key = os.getenv("POLYGON_API_KEY")
        self.fmp_api_key = os.getenv("FINANCIALMODELINGPREP_API_KEY")
        self.twelve_data_api_key = os.getenv("TWELVEDATA_API_KEY")

    def fetch_data(self, tickers: List[str], days: int = 60) -> pd.DataFrame:
        """Fetch OHLCV data and technical indicators for a list of tickers.

        Parameters
        ----------
        tickers : List[str]
            List of ticker symbols to fetch.
        days : int, optional
            Approximate number of recent trading days to keep, by default 60.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns:
            [date, ticker, open, high, low, close, volume,
             rsi_2, rsi_14,
             bb_upper, bb_middle, bb_lower,
             sma_200,
             macd, macd_signal,
             volume_ratio]
        """
        # Add buffer for weekends/holidays to ensure enough data for indicators
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=days * 3)

        frames: List[pd.DataFrame] = []
        for ticker in tickers:
            df_ticker = self._fetch_single_ticker(ticker, start_date, end_date, days)
            if df_ticker is not None and not df_ticker.empty:
                frames.append(df_ticker)
            else:
                logger.warning("No data available for ticker %s after all API attempts", ticker)

        columns = [
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "volume",
            # Core momentum/volatility
            "rsi_2",
            "rsi_14",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "sma_200",
            "macd",
            "macd_signal",
            "volume_ratio",
            # Ichimoku
            "ichimoku_conversion",
            "ichimoku_base",
            "ichimoku_span_a",
            "ichimoku_span_b",
            "ichimoku_lagging",
            # Fibonacci retracements (50 bar)
            "fib_23_6",
            "fib_38_2",
            "fib_50_0",
            "fib_61_8",
            "fib_78_6",
            # Volume / accumulation
            "vwap",
            "vwap_band_upper_1",
            "vwap_band_lower_1",
            "vwap_band_upper_2",
            "vwap_band_lower_2",
            "obv",
            "adl",
            # Advanced momentum
            "stoch_k",
            "stoch_d",
            "williams_r",
            "cci_20",
            "roc_10",
            # Trend strength
            "adx_14",
            "plus_di_14",
            "minus_di_14",
            "aroon_up_25",
            "aroon_down_25",
            "parabolic_sar",
            # Volatility
            "atr_14",
            "hist_vol_20",
            "keltner_middle",
            "keltner_upper",
            "keltner_lower",
            # Pivot points
            "pivot_p",
            "pivot_r1",
            "pivot_r2",
            "pivot_r3",
            "pivot_s1",
            "pivot_s2",
            "pivot_s3",
            # Data quality
            "data_quality_score",
        ]

        if not frames:
            logger.warning("No data fetched for any ticker. Returning empty DataFrame.")
            return pd.DataFrame(columns=columns)

        result = pd.concat(frames, ignore_index=True)
        # Ensure column order
        result = result.reindex(columns=columns)
        return result

    def _fetch_single_ticker(
        self,
        ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        days: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch data for a single ticker using API fallback logic and add indicators."""
        logger.info("Fetching data for ticker %s", ticker)

        fetchers = [
            ("Polygon", self._fetch_from_polygon),
            ("FMP", self._fetch_from_fmp),
            ("TwelveData", self._fetch_from_twelve_data),
            ("yfinance", self._fetch_from_yfinance),
        ]

        df: Optional[pd.DataFrame] = None

        for name, func in fetchers:
            try:
                df = func(ticker, start_date, end_date, days)
                if df is not None and not df.empty:
                    logger.info("Successfully fetched %s data for %s", name, ticker)
                    break
                logger.warning("%s returned no data for %s", name, ticker)
            except Exception as exc:  # noqa: BLE001
                logger.warning("%s API failed for %s: %s", name, ticker, exc)
                df = None

        if df is None or df.empty:
            return None

        # Normalize and sort
        df["ticker"] = ticker
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date")
        # Keep last `days` rows for consistency
        df = df.tail(days)

        # Add technical indicators
        df = self._add_indicators(df)

        # Add pivot points using previous session's OHLC (classical pivots)
        if {"high", "low", "close"}.issubset(df.columns):
            prev_high = df["high"].shift(1)
            prev_low = df["low"].shift(1)
            prev_close = df["close"].shift(1)
            pivot_p = (prev_high + prev_low + prev_close) / 3.0
            r1 = 2 * pivot_p - prev_low
            s1 = 2 * pivot_p - prev_high
            r2 = pivot_p + (prev_high - prev_low)
            s2 = pivot_p - (prev_high - prev_low)
            r3 = prev_high + 2 * (pivot_p - prev_low)
            s3 = prev_low - 2 * (prev_high - pivot_p)
            df["pivot_p"] = pivot_p
            df["pivot_r1"] = r1
            df["pivot_r2"] = r2
            df["pivot_r3"] = r3
            df["pivot_s1"] = s1
            df["pivot_s2"] = s2
            df["pivot_s3"] = s3

        # Compute and attach a per-ticker data quality score (constant across rows)
        quality_score = self._calculate_data_quality_score(df)
        df["data_quality_score"] = float(quality_score)

        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_from_polygon(
        self,
        ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        days: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data from Polygon API."""
        if not self.polygon_api_key:
            logger.warning("POLYGON_API_KEY not set; skipping Polygon for %s", ticker)
            return None

        base_url = "https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        url = base_url.format(
            ticker=ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
        )
        params = {
            "apiKey": self.polygon_api_key,
            "adjusted": "true",
            "sort": "asc",
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        results = data.get("results")
        if not results:
            return None

        records = []
        for item in results:
            # t: timestamp in ms since epoch
            ts = item.get("t")
            if ts is None:
                continue
            date = datetime.utcfromtimestamp(ts / 1000.0).date()
            records.append(
                {
                    "date": date,
                    "open": item.get("o"),
                    "high": item.get("h"),
                    "low": item.get("l"),
                    "close": item.get("c"),
                    "volume": item.get("v"),
                }
            )

        if not records:
            return None

        df = pd.DataFrame.from_records(records)
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_from_fmp(
        self,
        ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        days: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data from FinancialModelingPrep API."""
        if not self.fmp_api_key:
            logger.warning("FINANCIALMODELINGPREP_API_KEY not set; skipping FMP for %s", ticker)
            return None

        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
        params = {"apikey": self.fmp_api_key}

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        historical = data.get("historical")
        if not historical:
            return None

        # FMP returns most recent first; we want chronological
        df = pd.DataFrame(historical)
        df["date"] = pd.to_datetime(df["date"]).dt.date
        df = df.sort_values("date")

        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df.loc[mask]

        df = df.tail(days)

        if df.empty:
            return None

        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )

        df = df[["date", "open", "high", "low", "close", "volume"]]
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_from_twelve_data(
        self,
        ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        days: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data from Twelve Data API."""
        if not self.twelve_data_api_key:
            logger.warning("TWELVEDATA_API_KEY not set; skipping Twelve Data for %s", ticker)
            return None

        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": ticker,
            "interval": "1day",
            "outputsize": str(days * 3),  # buffer for filtering
            "apikey": self.twelve_data_api_key,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        values = data.get("values")
        if not values:
            return None

        df = pd.DataFrame(values)
        # Twelve Data returns strings
        df["datetime"] = pd.to_datetime(df["datetime"]).dt.date
        df = df.rename(
            columns={
                "datetime": "date",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
            }
        )

        # Convert numeric columns
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.sort_values("date")
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        df = df.loc[mask]
        df = df.tail(days)

        if df.empty:
            return None

        df = df[["date", "open", "high", "low", "close", "volume"]]
        return df

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _fetch_from_yfinance(
        self,
        ticker: str,
        start_date: datetime.date,
        end_date: datetime.date,
        days: int,
    ) -> Optional[pd.DataFrame]:
        """Fetch daily OHLCV data from yfinance as a fallback."""
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
            progress=False,
            auto_adjust=False,
        )

        if df is None or df.empty:
            return None

        df = df.reset_index()
        # yfinance 'Date' is Timestamp
        df["date"] = pd.to_datetime(df["Date"]).dt.date
        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        df = df[["date", "open", "high", "low", "close", "volume"]]
        df = df.sort_values("date")
        df = df.tail(days)

        return df if not df.empty else None

    def _add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a comprehensive set of technical indicators to OHLCV data.

        The input DataFrame must contain at least: ['date', 'open', 'high', 'low', 'close', 'volume'].
        """
        df = df.copy()
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        volume = df["volume"].astype(float)

        # === Core RSI ===
        def _rsi(series: pd.Series, period: int) -> pd.Series:
            delta = series.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=period, min_periods=period).mean()
            avg_loss = loss.rolling(window=period, min_periods=period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        df["rsi_2"] = _rsi(close, 2)
        df["rsi_14"] = _rsi(close, 14)

        # === Bollinger Bands (20, 2 std) ===
        window_bb = 20
        rolling_mean = close.rolling(window=window_bb, min_periods=window_bb).mean()
        rolling_std = close.rolling(window=window_bb, min_periods=window_bb).std()
        df["bb_middle"] = rolling_mean
        df["bb_upper"] = rolling_mean + (2 * rolling_std)
        df["bb_lower"] = rolling_mean - (2 * rolling_std)

        # === SMA 200 ===
        df["sma_200"] = close.rolling(window=200, min_periods=200).mean()

        # === MACD (12, 26, 9) ===
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df["macd"] = macd
        df["macd_signal"] = macd_signal

        # === Volume ratio: current volume / 20-day average volume ===
        vol_ma20 = volume.rolling(window=20, min_periods=20).mean()
        df["volume_ratio"] = volume / vol_ma20

        # === Ichimoku Cloud ===
        high_9 = high.rolling(window=9, min_periods=9).max()
        low_9 = low.rolling(window=9, min_periods=9).min()
        conversion = (high_9 + low_9) / 2

        high_26 = high.rolling(window=26, min_periods=26).max()
        low_26 = low.rolling(window=26, min_periods=26).min()
        base = (high_26 + low_26) / 2

        span_a = ((conversion + base) / 2).shift(26)
        high_52 = high.rolling(window=52, min_periods=52).max()
        low_52 = low.rolling(window=52, min_periods=52).min()
        span_b = ((high_52 + low_52) / 2).shift(26)
        lagging = close.shift(-26)

        df["ichimoku_conversion"] = conversion
        df["ichimoku_base"] = base
        df["ichimoku_span_a"] = span_a
        df["ichimoku_span_b"] = span_b
        df["ichimoku_lagging"] = lagging

        # === Fibonacci retracements (50-bar lookback) ===
        lookback = 50
        rolling_high = high.rolling(window=lookback, min_periods=lookback).max()
        rolling_low = low.rolling(window=lookback, min_periods=lookback).min()
        range_ = rolling_high - rolling_low
        df["fib_23_6"] = rolling_high - range_ * 0.236
        df["fib_38_2"] = rolling_high - range_ * 0.382
        df["fib_50_0"] = rolling_high - range_ * 0.5
        df["fib_61_8"] = rolling_high - range_ * 0.618
        df["fib_78_6"] = rolling_high - range_ * 0.786

        # === VWAP and VWAP bands ===
        typical_price = (high + low + close) / 3
        cum_vol = volume.cumsum()
        cum_vp = (typical_price * volume).cumsum()
        vwap = np.where(cum_vol > 0, cum_vp / cum_vol, np.nan)
        df["vwap"] = vwap

        vwap_window = 20
        vwap_series = pd.Series(vwap, index=df.index)
        vwap_std = vwap_series.rolling(window=vwap_window, min_periods=vwap_window).std()
        df["vwap_band_upper_1"] = vwap_series + vwap_std
        df["vwap_band_lower_1"] = vwap_series - vwap_std
        df["vwap_band_upper_2"] = vwap_series + 2 * vwap_std
        df["vwap_band_lower_2"] = vwap_series - 2 * vwap_std

        # === OBV ===
        obv = np.where(close > close.shift(1), volume,
                       np.where(close < close.shift(1), -volume, 0.0))
        df["obv"] = pd.Series(obv, index=df.index).cumsum()

        # === Accumulation/Distribution Line ===
        clv = ((close - low) - (high - close)) / np.where((high - low) != 0, (high - low), np.nan)
        clv = clv.fillna(0.0)
        df["adl"] = (clv * volume).cumsum()

        # === Stochastic Oscillator (14,3,3) ===
        stoch_period = 14
        lowest_low = low.rolling(window=stoch_period, min_periods=stoch_period).min()
        highest_high = high.rolling(window=stoch_period, min_periods=stoch_period).max()
        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan)
        stoch_d = stoch_k.rolling(window=3, min_periods=3).mean()
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d

        # === Williams %R (14) ===
        will_r = -100 * (highest_high - close) / (highest_high - lowest_low)
        df["williams_r"] = will_r

        # === CCI (20) ===
        tp = (high + low + close) / 3
        cci_period = 20
        sma_tp = tp.rolling(window=cci_period, min_periods=cci_period).mean()
        mad = (tp - sma_tp).abs().rolling(window=cci_period, min_periods=cci_period).mean()
        df["cci_20"] = (tp - sma_tp) / (0.015 * mad)

        # === Rate of Change (ROC 10) ===
        df["roc_10"] = close.pct_change(10) * 100

        # === ATR (14) ===
        high_low = high - low
        high_close_prev = (high - close.shift(1)).abs()
        low_close_prev = (low - close.shift(1)).abs()
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        atr = true_range.rolling(window=14, min_periods=14).mean()
        df["atr_14"] = atr

        # === Historical volatility (20-day, annualized) ===
        log_ret = np.log(close / close.shift(1))
        hist_vol = log_ret.rolling(window=20, min_periods=20).std() * np.sqrt(252)
        df["hist_vol_20"] = hist_vol

        # === Keltner Channels (20 EMA, 2 * ATR) ===
        ema20 = close.ewm(span=20, adjust=False).mean()
        df["keltner_middle"] = ema20
        df["keltner_upper"] = ema20 + 2 * atr
        df["keltner_lower"] = ema20 - 2 * atr

        # === ADX (14) with +DI and -DI ===
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        tr_smooth = true_range.rolling(window=14, min_periods=14).sum()
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=14, min_periods=14).sum() / tr_smooth)
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=14, min_periods=14).sum() / tr_smooth)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)).replace([np.inf, -np.inf], np.nan) * 100
        adx = dx.rolling(window=14, min_periods=14).mean()

        df["plus_di_14"] = plus_di
        df["minus_di_14"] = minus_di
        df["adx_14"] = adx

        # === Aroon (25) ===
        aroon_period = 25
        aroon_up = 100 * high.rolling(window=aroon_period, min_periods=aroon_period).apply(
            lambda x: np.argmax(x[::-1]) / (aroon_period - 1) if len(x) == aroon_period else np.nan,
            raw=True,
        )
        aroon_down = 100 * low.rolling(window=aroon_period, min_periods=aroon_period).apply(
            lambda x: np.argmin(x[::-1]) / (aroon_period - 1) if len(x) == aroon_period else np.nan,
            raw=True,
        )
        df["aroon_up_25"] = aroon_up
        df["aroon_down_25"] = aroon_down

        # === Parabolic SAR (simple implementation) ===
        psar = []
        af = 0.02
        max_af = 0.2
        trend_up = True
        ep = low.iloc[0]
        sar = low.iloc[0]
        for i in range(len(df)):
            if i == 0:
                psar.append(np.nan)
                continue
            prev_sar = sar
            if trend_up:
                sar = prev_sar + af * (ep - prev_sar)
                sar = min(sar, low.iloc[i - 1], low.iloc[i])
                if low.iloc[i] < sar:
                    trend_up = False
                    sar = ep
                    ep = low.iloc[i]
                    af = 0.02
            else:
                sar = prev_sar + af * (ep - prev_sar)
                sar = max(sar, high.iloc[i - 1], high.iloc[i])
                if high.iloc[i] > sar:
                    trend_up = True
                    sar = ep
                    ep = high.iloc[i]
                    af = 0.02

            if trend_up and high.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(max_af, af + 0.02)
            elif (not trend_up) and low.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(max_af, af + 0.02)
            psar.append(sar)
        df["parabolic_sar"] = pd.Series(psar, index=df.index)

        return df

    def _calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Score data quality from 0 to 100 for a single-ticker DataFrame."""

        if df.empty:
            return 0.0

        score = 100.0
        total_cells = len(df) * len(df.columns)
        if total_cells > 0:
            missing_pct = df.isna().sum().sum() / total_cells
        else:
            missing_pct = 1.0
        score -= missing_pct * 50

        if "volume" in df.columns and len(df) > 0:
            zero_vol_pct = (df["volume"] == 0).sum() / len(df)
        else:
            zero_vol_pct = 0.0
        score -= zero_vol_pct * 30

        if "close" in df.columns and len(df) > 1:
            price_changes = df["close"].pct_change().abs()
            anomaly_pct = (price_changes > 0.20).sum() / len(df)
        else:
            anomaly_pct = 0.0
        score -= anomaly_pct * 20

        return float(max(0.0, min(100.0, score)))


if __name__ == "__main__":
    fetcher = EliteDataFetcher()
    data = fetcher.fetch_data(["SQQQ", "TQQQ", "GME", "TSLA"], days=60)
    print(f"Fetched {len(data)} rows")
    print(data.tail())
