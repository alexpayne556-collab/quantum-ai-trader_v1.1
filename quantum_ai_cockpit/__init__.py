"""Quantum AI Cockpit package initialization."""

__all__ = [
    "config",
    "signals",
    "utils",
    "screener",
    "portfolio_tracker",
    "position_sizer",
    "data_fetcher",
    # API Keys (from config)
    "FINNHUB_API_KEY",
    "POLYGON_API_KEY",
    "FMP_API_KEY",
    "ALPHAVANTAGE_API_KEY",
    "EODHD_API_TOKEN",
    "NEWSAPI_API_KEY",
    "FRED_API_KEY",
    "OPENAI_API_KEY",
    "get_all_api_keys",
    "validate_api_keys",
    # Data Fetcher (unified data access)
    "DataFetcher",
    "fetch_stock",
    "fetch_quote",
    "fetch_news",
    "fetch_economic",
    # Backtested modules
    "TechnicalIndicators",
    "PatternScorer",
    "EnsembleVoter",
    "BacktestEngine",
    "MarketRegimeManager",
    # Red/Green Detector (Pattern B)
    "RedGreenDetector",
    "scan_for_green_stocks",
    "get_top_candidates",
    # Data Pipeline (ML-ready)
    "DataCleaner",
    "FeatureEngineer",
    # Quantum Forecaster (Hybrid ML)
    "QuantumForecaster",
    "ForecasterConfig",
    "train_forecaster",
    # Backtester (Real Metrics)
    "ForecasterBacktester",
    "quick_backtest",
]

# Export API keys from config for easy access
from .config import (
    FINNHUB_API_KEY,
    POLYGON_API_KEY,
    FMP_API_KEY,
    FINANCIALMODELINGPREP_API_KEY,
    ALPHAVANTAGE_API_KEY,
    EODHD_API_TOKEN,
    TWELVEDATA_API_KEY,
    TIINGO_API_KEY,
    INTRINIO_API_KEY,
    NEWSAPI_API_KEY,
    NEWSDATA_API_KEY,
    MARKETAUX_API_KEY,
    FRED_API_KEY,
    OPENAI_API_KEY,
    get_all_api_keys,
    validate_api_keys,
)

# Export unified data fetcher
from .data_fetcher import (
    DataFetcher,
    fetch_stock,
    fetch_quote,
    fetch_news,
    fetch_economic,
    get_fetcher,
)

# Expose top-level backtested modules (import with try/except for flexibility)
try:
    from quantum_trader import (
        TechnicalIndicators,
        PatternScorer,
        EnsembleVoter,
        BacktestEngine,
    )
except ImportError:
    pass  # quantum_trader not in path

try:
    from market_regime_manager import MarketRegimeManager
except ImportError:
    pass  # market_regime_manager not in path

# Red/Green Detector (Pattern B - Technical Alignment)
try:
    import sys
    from pathlib import Path
    # Add backend to path if not already there
    backend_path = Path(__file__).parent.parent / "backend"
    if str(backend_path) not in sys.path:
        sys.path.insert(0, str(backend_path))
    from red_green_detector import (
        RedGreenDetector,
        scan_for_green_stocks,
        get_top_candidates,
    )
except ImportError:
    pass  # red_green_detector not in path

# Data Pipeline (ML-ready features)
try:
    from data_cleaner import DataCleaner
except ImportError:
    pass  # data_cleaner not in path

try:
    from feature_engineer import FeatureEngineer
except ImportError:
    pass  # feature_engineer not in path

# Quantum Forecaster (Hybrid ML model)
try:
    from quantum_forecaster import (
        QuantumForecaster,
        ForecasterConfig,
        train_forecaster,
    )
except ImportError:
    pass  # quantum_forecaster not in path

# Forecaster Backtester (Real Metrics)
try:
    from forecaster_backtest import (
        ForecasterBacktester,
        quick_backtest,
    )
except ImportError:
    pass  # forecaster_backtest not in path
