"""
Unit tests for data_pipeline.py
"""
import pytest
import pandas as pd
import asyncio
from data_pipeline import DataValidator, FeatureEngineer, DataCache, StreamingPublisher, DataFetcher, DataPipeline

@pytest.mark.asyncio
async def test_data_validator():
    df = pd.DataFrame({
        "timestamp": ["2023-01-01T09:30:00", "2023-01-01T09:31:00"],
        "open": [100.0, 101.0],
        "high": [102.0, 103.0],
        "low": [99.0, 100.0],
        "close": [101.0, 102.0],
        "volume": [1000, 1100]
    })
    validator = DataValidator()
    validated = validator.validate_data(df)
    assert not validated.empty
    assert "open" in validated.columns

@pytest.mark.asyncio
async def test_feature_engineer():
    df = pd.DataFrame({
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "high": [101]*11,
        "low": [99]*11
    })
    engineer = FeatureEngineer()
    features = engineer.engineer_features(df)
    assert "rsi_14" in features.columns
    assert "macd" in features.columns

@pytest.mark.asyncio
async def test_data_cache(monkeypatch):
    class DummyRedis:
        def __init__(self):
            self.store = {}
        async def get(self, key):
            return self.store.get(key)
        async def set(self, key, value, expire):
            self.store[key] = value
    cache = DataCache(redis_url="redis://localhost:6379/0")
    cache.redis = DummyRedis()
    df = pd.DataFrame({"a": [1,2,3]})
    await cache.set_cached("test", df)
    cached = await cache.get_cached("test")
    assert cached is not None

@pytest.mark.asyncio
async def test_streaming_publisher(monkeypatch):
    class DummyWS:
        async def send(self, msg):
            assert isinstance(msg, str)
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc, tb):
            pass
    monkeypatch.setattr("websockets.connect", lambda url: DummyWS())
    publisher = StreamingPublisher()
    await publisher.publish_streaming("AAPL", {"close": [100,101]})

@pytest.mark.asyncio
async def test_data_fetcher(monkeypatch):
    fetcher = DataFetcher()
    monkeypatch.setattr(fetcher, "fetch_yfinance", lambda t, tf: pd.DataFrame({"timestamp": ["2023-01-01T09:30:00"], "open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000]}))
    monkeypatch.setattr(fetcher, "fetch_finnhub", lambda t, tf: pd.DataFrame())
    monkeypatch.setattr(fetcher, "fetch_polygon", lambda t, tf: pd.DataFrame())
    result = await fetcher.fetch_ohlcv(["AAPL"], "1m")
    assert "AAPL" in result
    assert not result["AAPL"].empty

@pytest.mark.asyncio
async def test_pipeline(monkeypatch):
    pipeline = DataPipeline(redis_url="redis://localhost:6379/0")
    monkeypatch.setattr(pipeline.cache, "get_cached", lambda key, ttl=300: None)
    monkeypatch.setattr(pipeline.cache, "set_cached", lambda key, df, ttl=300: None)
    monkeypatch.setattr(pipeline.fetcher, "fetch_ohlcv", lambda tickers, tf: {"AAPL": pd.DataFrame({"timestamp": ["2023-01-01T09:30:00"], "open": [100], "high": [101], "low": [99], "close": [100], "volume": [1000]})})
    monkeypatch.setattr(pipeline.validator, "validate_data", lambda df: df)
    monkeypatch.setattr(pipeline.engineer, "engineer_features", lambda df: df)
    monkeypatch.setattr(pipeline.publisher, "publish_streaming", lambda ticker, data: None)
    df = await pipeline.process_ticker("AAPL", "1m")
    assert not df.empty
