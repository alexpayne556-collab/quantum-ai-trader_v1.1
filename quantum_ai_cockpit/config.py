"""Centralized configuration for the Quantum AI Cockpit."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# =============================================================================
# DATA PROVIDER API KEYS
# =============================================================================
FINNHUB_API_KEY: Optional[str] = os.getenv("FINNHUB_API_KEY")
POLYGON_API_KEY: Optional[str] = os.getenv("POLYGON_API_KEY")
FMP_API_KEY: Optional[str] = os.getenv("FMP_API_KEY")
FINANCIALMODELINGPREP_API_KEY: Optional[str] = os.getenv("FINANCIALMODELINGPREP_API_KEY")
ALPHA_VANTAGE_KEY: Optional[str] = os.getenv("ALPHAVANTAGE_API_KEY")
ALPHAVANTAGE_API_KEY: Optional[str] = os.getenv("ALPHAVANTAGE_API_KEY")  # alias
EODHD_API_TOKEN: Optional[str] = os.getenv("EODHD_API_TOKEN")
TWELVEDATA_API_KEY: Optional[str] = os.getenv("TWELVEDATA_API_KEY")
TIINGO_API_KEY: Optional[str] = os.getenv("TIINGO_API_KEY")
INTRINIO_API_KEY: Optional[str] = os.getenv("INTRINIO_API_KEY")

# =============================================================================
# NEWS API KEYS
# =============================================================================
NEWSAPI_API_KEY: Optional[str] = os.getenv("NEWSAPI_API_KEY")
NEWSDATA_API_KEY: Optional[str] = os.getenv("NEWSDATA_API_KEY")
MARKETAUX_API_KEY: Optional[str] = os.getenv("MARKETAUX_API_KEY")

# =============================================================================
# ADDITIONAL DATA PROVIDERS
# =============================================================================
TWELVEDATA_API_KEY: Optional[str] = os.getenv("TWELVEDATA_API_KEY")

# =============================================================================
# ECONOMIC DATA API KEYS
# =============================================================================
FRED_API_KEY: Optional[str] = os.getenv("FRED_API_KEY")

# =============================================================================
# AI/ML API KEYS
# =============================================================================
OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================
SECRET_KEY: Optional[str] = os.getenv("SECRET_KEY")
ENCRYPTION_SALT: Optional[str] = os.getenv("ENCRYPTION_SALT")


def get_all_api_keys() -> Dict[str, Optional[str]]:
    """Return all API keys as a dictionary for validation."""
    return {
        "FINNHUB_API_KEY": FINNHUB_API_KEY,
        "POLYGON_API_KEY": POLYGON_API_KEY,
        "FMP_API_KEY": FMP_API_KEY,
        "ALPHAVANTAGE_API_KEY": ALPHAVANTAGE_API_KEY,
        "EODHD_API_TOKEN": EODHD_API_TOKEN,
        "TWELVEDATA_API_KEY": TWELVEDATA_API_KEY,
        "TIINGO_API_KEY": TIINGO_API_KEY,
        "INTRINIO_API_KEY": INTRINIO_API_KEY,
        "NEWSAPI_API_KEY": NEWSAPI_API_KEY,
        "NEWSDATA_API_KEY": NEWSDATA_API_KEY,
        "MARKETAUX_API_KEY": MARKETAUX_API_KEY,
        "FRED_API_KEY": FRED_API_KEY,
        "OPENAI_API_KEY": OPENAI_API_KEY,
    }


def validate_api_keys() -> None:
    """Print status of all API keys."""
    keys = get_all_api_keys()
    print("\n" + "=" * 50)
    print("API KEY STATUS")
    print("=" * 50)
    for name, value in keys.items():
        status = "✅ SET" if value else "❌ MISSING"
        print(f"  {name:25s} {status}")
    print("=" * 50 + "\n")

SIGNAL_THRESHOLD: int = 80
MIN_VOLUME_RATIO: float = 1.5
STOP_LOSS_PCT: float = 0.12
TARGET_PCT: float = 0.15

STOCK_UNIVERSE: List[str] = [
    # Crypto/Mining
    "MARA",
    "RIOT",
    "CLSK",
    "BITF",
    "HUT",
    "COIN",
    "APLD",
    # Tech small caps
    "SOFI",
    "HOOD",
    "AFRM",
    "UPST",
    "LC",
    "NU",
    # Current holdings
    "MU",
    "NVDA",
    "ORCL",
    # Mining/Materials
    "UUUU",
    "GDX",
    "NEM",
    "AG",
    "FSM",
    # High volatility small caps
    "PLTR",
    "IONQ",
    "RGTI",
    "QUBT",
    "SOUN",
    "BBAI",
    # EV/Energy
    "LCID",
    "RIVN",
    "NIO",
    "PLUG",
    "FCEL",
    # Biotech
    "NVAX",
    "MRNA",
    "SAVA",
    "ATER",
    # Meme/momentum
    "GME",
    "AMC",
    "SPCE",
]
"""Validated stock universe (39 high-volatility tickers)."""
