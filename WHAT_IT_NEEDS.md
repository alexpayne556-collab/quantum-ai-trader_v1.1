# WHAT IT NEEDS

## Requirements to Run `trading_system_unified.py`

### 1. Python Environment

```bash
# Python 3.8+ required
python --version  # Should be 3.8 or higher
```

### 2. Required Python Packages

```bash
# Core requirements
pip install numpy pandas

# ML Models
pip install xgboost lightgbm scikit-learn joblib

# Data fetching
pip install yfinance

# Technical indicators (HIGHLY RECOMMENDED)
pip install TA-Lib
```

**TA-Lib Installation Notes:**
- **Linux:** `sudo apt-get install libta-lib-dev && pip install TA-Lib`
- **macOS:** `brew install ta-lib && pip install TA-Lib`
- **Windows:** Download wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib

If TA-Lib is not available, the system falls back to pure-Python indicators (slightly less accurate).

### 3. Trained Model Files

The system expects these files in `trained_models/colab/`:

```
trained_models/
└── colab/
    ├── xgboost_model.pkl      # XGBoost classifier (~3MB)
    ├── lightgbm_model.pkl     # LightGBM classifier (~10MB)
    ├── scaler.pkl             # StandardScaler (~3KB)
    └── top_features.json      # 51 feature names (~1KB)
```

**CHECK IF MODELS EXIST:**
```bash
ls -la trained_models/colab/
```

**IF MISSING:** The models need to be trained in Google Colab. See `COLAB_PRO_TRAINING_GUIDE.ipynb`.

### 4. Internet Connection

Required for:
- Fetching stock price data via `yfinance`
- Fetching SPY data for correlation features
- Fetching VIX data for volatility context

### 5. API Access (Optional)

No API keys required for basic operation. The system uses `yfinance` which scrapes Yahoo Finance.

---

## Quick Start

### Option A: Command Line

```bash
# Single ticker
python trading_system_unified.py --ticker AAPL

# Multiple tickers
python trading_system_unified.py --tickers AAPL,MSFT,GOOGL,TSLA

# JSON output
python trading_system_unified.py --ticker AAPL --json

# Custom period
python trading_system_unified.py --ticker AAPL --period 1y
```

### Option B: Python Module

```python
from trading_system_unified import UnifiedTradingSystem

# Initialize
system = UnifiedTradingSystem()

# Analyze single ticker
signal = system.analyze('AAPL')

# Print result
print(signal)
# Output: BUY AAPL @ $178.50 | Confidence: 72.3% | SL: $173.20 | TP: $186.40 | R:R = 1:1.5

# Access components
print(f"Signal: {signal.signal}")
print(f"Confidence: {signal.confidence:.1%}")
print(f"Entry Price: ${signal.entry_price:.2f}")
print(f"Stop Loss: ${signal.stop_loss:.2f}")
print(f"Take Profit: ${signal.take_profit:.2f}")
print(f"Probabilities: {signal.probabilities}")
print(f"Patterns: {signal.patterns_detected}")

# Batch analysis
signals = system.batch_analyze(['AAPL', 'MSFT', 'GOOGL'])
for s in signals:
    print(s)
```

### Option C: With Custom Data

```python
import pandas as pd
from trading_system_unified import UnifiedTradingSystem

# Load your own data
df = pd.read_csv('my_stock_data.csv', index_col='Date', parse_dates=True)

# Must have columns: Open, High, Low, Close, Volume
system = UnifiedTradingSystem()
signal = system.analyze('CUSTOM', price_data=df)
```

---

## Input Requirements

### Price Data Format

The system expects OHLCV data with these columns:

| Column | Type | Required |
|--------|------|----------|
| Open | float | Yes |
| High | float | Yes |
| Low | float | Yes |
| Close | float | Yes |
| Volume | int/float | Yes |

Index should be DatetimeIndex.

**Minimum rows required:** 50 (ideally 200+)

### Valid Periods

For yfinance data fetching:
- `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `max`

---

## Output Format

### TradingSignal Object

```python
@dataclass
class TradingSignal:
    ticker: str           # e.g., 'AAPL'
    signal: str           # 'BUY', 'SELL', or 'HOLD'
    confidence: float     # 0.0 to 1.0 (e.g., 0.72 = 72%)
    entry_price: float    # Current price
    stop_loss: float      # Suggested stop loss price
    take_profit: float    # Suggested take profit price
    risk_reward_ratio: float  # e.g., 1.5 means 1:1.5 R:R
    probabilities: Dict   # {'BUY': 0.45, 'HOLD': 0.30, 'SELL': 0.25}
    patterns_detected: List[Dict]  # Detected patterns
    pattern_confluence: int  # +ve = bullish, -ve = bearish
    regime: str           # 'STRONG_BULL', 'BULL', 'SIDEWAYS', 'BEAR', 'STRONG_BEAR'
    timestamp: str        # ISO format timestamp
    metadata: Dict        # Additional info
```

### JSON Output Example

```json
{
  "ticker": "AAPL",
  "signal": "BUY",
  "confidence": 0.723,
  "entry_price": 178.50,
  "stop_loss": 173.20,
  "take_profit": 186.40,
  "risk_reward_ratio": 1.5,
  "probabilities": {
    "HOLD": 0.277,
    "BUY": 0.452,
    "SELL": 0.271
  },
  "patterns_detected": [
    {"pattern": "MACD Bullish Cross", "type": "BULLISH", "confidence": 0.6},
    {"pattern": "EMA Ribbon Bullish", "type": "BULLISH", "confidence": 0.7}
  ],
  "pattern_confluence": 2,
  "regime": "BULL",
  "timestamp": "2024-12-12T10:30:00",
  "metadata": {
    "data_rows": 126,
    "features_used": 51,
    "patterns_bullish": 3,
    "patterns_bearish": 1,
    "atr": 3.25
  }
}
```

---

## Full Installation Script

```bash
#!/bin/bash
# Full setup script

# Create virtual environment
python -m venv trading_env
source trading_env/bin/activate  # Linux/Mac
# trading_env\Scripts\activate   # Windows

# Install dependencies
pip install numpy pandas
pip install xgboost lightgbm scikit-learn joblib
pip install yfinance

# TA-Lib (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y libta-lib-dev
pip install TA-Lib

# Verify models exist
ls trained_models/colab/

# Run test
python trading_system_unified.py --ticker SPY --verbose
```

---

## Summary Checklist

- [ ] Python 3.8+ installed
- [ ] `numpy`, `pandas` installed
- [ ] `xgboost`, `lightgbm`, `scikit-learn`, `joblib` installed
- [ ] `yfinance` installed
- [ ] TA-Lib installed (recommended)
- [ ] Model files present in `trained_models/colab/`
- [ ] Internet connection available
