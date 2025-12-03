import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA


logger = logging.getLogger(__name__)


class EliteForecaster:
    """Real 14-day hybrid forecaster using only working models.

    Models used:
    1. Random Forest (sklearn) - pattern-based prediction
    2. ARIMA (statsmodels) - statistical time series
    3. Linear Regression (sklearn) - trend extrapolation
    """

    def __init__(self) -> None:
        self.rf_model: RandomForestRegressor | None = None
        self.lr_model: LinearRegression | None = None
        self.arima_order = (5, 1, 0)

    def forecast_ticker(self, ticker: str, data: pd.DataFrame, days: int = 14) -> Dict[str, Any]:
        """Generate a real 14-day forecast using RF, ARIMA, and Linear Regression."""

        ticker_data = data[data["ticker"] == ticker].sort_values("date")
        if len(ticker_data) < 30:
            logger.warning("Insufficient data for %s (need 30+ days)", ticker)
            return self._empty_forecast(ticker)

        close_prices = ticker_data["close"].astype(float).values
        current_price = float(close_prices[-1])

        # Model 1: Random Forest
        rf_forecast = self._forecast_random_forest(ticker_data, days)

        # Model 2: ARIMA
        arima_forecast = self._forecast_arima(close_prices, days)

        # Model 3: Linear Regression
        lr_forecast = self._forecast_linear_regression(close_prices, days)

        # Ensemble: RF=50%, ARIMA=30%, LR=20%
        ensemble_forecast = (
            rf_forecast * 0.50 + arima_forecast * 0.30 + lr_forecast * 0.20
        )

        target_14d = float(ensemble_forecast[-1])
        expected_return = (target_14d - current_price) / current_price
        overall_direction = "UP" if target_14d > current_price else "DOWN"

        # Model agreement
        predictions = [rf_forecast[-1], arima_forecast[-1], lr_forecast[-1]]
        up_votes = sum(1 for p in predictions if p > current_price)
        model_agreement = max(up_votes, 3 - up_votes) / 3.0

        # Confidence from agreement
        overall_confidence = 0.60 + (model_agreement - 0.33) * 0.50
        overall_confidence = min(0.90, max(0.50, overall_confidence))

        # Build daily forecast
        forecast_days: List[Dict[str, Any]] = []
        dates = pd.date_range(
            start=ticker_data["date"].iloc[-1] + pd.Timedelta(days=1),
            periods=days,
        )

        for i in range(days):
            day_price = float(ensemble_forecast[i])
            day_direction = "UP" if day_price > current_price else "DOWN"

            # Confidence decays ~3% per day
            day_conf = overall_confidence * (1 - 0.03 * i)
            day_conf = max(0.50, day_conf)

            forecast_days.append(
                {
                    "day": i + 1,
                    "date": dates[i].strftime("%Y-%m-%d"),
                    "price": round(day_price, 2),
                    "direction": day_direction,
                    "confidence": round(day_conf, 2),
                }
            )

        # Graph data
        graph_data = self._prepare_graph_data(
            ticker_data, ensemble_forecast, dates, overall_confidence
        )

        # Supporting evidence
        rf_dir = "UP" if rf_forecast[-1] > current_price else "DOWN"
        arima_dir = "UP" if arima_forecast[-1] > current_price else "DOWN"
        lr_dir = "UP" if lr_forecast[-1] > current_price else "DOWN"

        supporting_evidence = [
            f"Random Forest: {rf_dir} (target ${rf_forecast[-1]:.2f})",
            f"ARIMA: {arima_dir} (target ${arima_forecast[-1]:.2f})",
            f"Linear Regression: {lr_dir} (target ${lr_forecast[-1]:.2f})",
            f"Ensemble: {overall_direction} with {int(model_agreement*100)}% agreement",
        ]

        return {
            "ticker": ticker,
            "current_price": round(current_price, 2),
            "forecast_days": forecast_days,
            "overall_direction": overall_direction,
            "overall_confidence": round(overall_confidence, 2),
            "target_14d": round(target_14d, 2),
            "expected_return_14d": round(expected_return, 4),
            "model_agreement": round(model_agreement, 2),
            "supporting_evidence": supporting_evidence,
            "graph_data": graph_data,
        }

    # ------------------------------------------------------------------
    # Model implementations
    # ------------------------------------------------------------------
    def _forecast_random_forest(self, data: pd.DataFrame, days: int) -> np.ndarray:
        """Random Forest prediction using actual indicator features."""

        X: List[List[float]] = []
        y: List[float] = []

        closes = data["close"].astype(float)
        rsi = data.get("rsi_14", pd.Series([50.0] * len(data)))
        vol_ratio = data.get("volume_ratio", pd.Series([1.0] * len(data)))
        sma_50 = data.get("sma_50", closes)

        for i in range(10, len(data)):
            features = [
                float(closes.iloc[i - 1]),
                float(closes.iloc[i - 5 : i].mean()),
                float(rsi.iloc[i]),
                float(vol_ratio.iloc[i]),
                float(sma_50.iloc[i]),
            ]
            X.append(features)
            y.append(float(closes.iloc[i]))

        if len(X) < 20:
            return self._simple_moving_average(closes.values, days)

        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X, y)

        forecast: List[float] = []
        last_features = X[-1].copy()

        for _ in range(days):
            pred = float(model.predict([last_features])[0])
            forecast.append(pred)

            # Update simple features for next step
            last_features[0] = pred
            # Update 5-day MA approximated from last pred and last_features
            last_features[1] = (last_features[1] * 4 + pred) / 5.0

        return np.array(forecast)

    def _forecast_arima(self, prices: np.ndarray, days: int) -> np.ndarray:
        """ARIMA statistical forecasting."""

        try:
            model = ARIMA(prices, order=self.arima_order)
            fitted = model.fit()
            forecast = fitted.forecast(steps=days)
            return np.array(forecast)
        except Exception as exc:  # noqa: BLE001
            logger.warning("ARIMA failed: %s, using moving average fallback", exc)
            return self._simple_moving_average(prices, days)

    def _forecast_linear_regression(self, prices: np.ndarray, days: int) -> np.ndarray:
        """Linear Regression trend extrapolation using last 30 days."""

        recent = prices[-30:]
        X = np.arange(len(recent)).reshape(-1, 1)
        y = recent

        model = LinearRegression()
        model.fit(X, y)

        future_X = np.arange(len(recent), len(recent) + days).reshape(-1, 1)
        forecast = model.predict(future_X)
        return np.array(forecast)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _simple_moving_average(self, prices: np.ndarray, days: int) -> np.ndarray:
        """Simple moving-average-based extrapolation as a robust fallback."""

        ma_10 = float(np.mean(prices[-10:]))
        trend = (float(prices[-1]) - float(prices[-10])) / 10.0
        forecast = [ma_10 + trend * (i + 1) for i in range(days)]
        return np.array(forecast)

    def _prepare_graph_data(
        self,
        historical: pd.DataFrame,
        forecast: np.ndarray,
        dates: pd.DatetimeIndex,
        confidence: float,
    ) -> Dict[str, Any]:
        """Prepare data for visualization in elite_visualizer."""

        hist_dates = (
            pd.to_datetime(historical["date"]).dt.strftime("%Y-%m-%d").tolist()[-30:]
        )
        hist_prices = historical["close"].astype(float).tolist()[-30:]

        forecast_dates = [d.strftime("%Y-%m-%d") for d in dates]
        forecast_prices = [round(float(p), 2) for p in forecast]

        # Confidence bands widen over horizon
        upper_band = [p * (1 + 0.015 * (i + 1)) for i, p in enumerate(forecast)]
        lower_band = [p * (1 - 0.015 * (i + 1)) for i, p in enumerate(forecast)]

        return {
            "historical_dates": hist_dates,
            "historical_prices": hist_prices,
            "forecast_dates": forecast_dates,
            "forecast_prices": forecast_prices,
            "confidence_upper": [round(float(u), 2) for u in upper_band],
            "confidence_lower": [round(float(l), 2) for l in lower_band],
        }

    def _empty_forecast(self, ticker: str) -> Dict[str, Any]:
        """Return an empty forecast when data is insufficient."""

        return {
            "ticker": ticker,
            "error": "Insufficient data for forecasting (need 30+ days)",
            "forecast_days": [],
            "current_price": 0.0,
            "target_14d": 0.0,
            "expected_return_14d": 0.0,
            "overall_direction": "UNKNOWN",
            "overall_confidence": 0.0,
            "model_agreement": 0.0,
            "supporting_evidence": [],
            "graph_data": {},
        }


if __name__ == "__main__":
    from backend.modules.elite.elite_data_fetcher import EliteDataFetcher

    print("Testing Elite Forecaster...")

    fetcher = EliteDataFetcher()
    data = fetcher.fetch_data(["SPY", "QQQ", "AAPL"], days=90)

    forecaster = EliteForecaster()

    for ticker in ["SPY", "QQQ", "AAPL"]:
        print(f"\n{'='*80}")
        forecast = forecaster.forecast_ticker(ticker, data, days=14)

        if "error" in forecast:
            print(f"{ticker}: {forecast['error']}")
            continue

        print(f"14-Day Forecast for {forecast['ticker']}:")
        print(f"  Current Price: ${forecast['current_price']:.2f}")
        print(f"  14-Day Target: ${forecast['target_14d']:.2f}")
        print(f"  Expected Return: {forecast['expected_return_14d']:.1%}")
        print(f"  Direction: {forecast['overall_direction']}")
        print(f"  Confidence: {forecast['overall_confidence']:.0%}")
        print(f"  Model Agreement: {forecast['model_agreement']:.0%}")

        print(f"\n  Supporting Evidence:")
        for evidence in forecast["supporting_evidence"]:
            print(f"    • {evidence}")

        print(f"\n  First 7 Days:")
        for day in forecast["forecast_days"][:7]:
            print(
                f"    Day {day['day']} ({day['date']}): "
                f"${day['price']:.2f} {day['direction']} ({day['confidence']:.0%})"
            )
