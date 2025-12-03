# 🚀 Ultimate AI Model Artifacts

## Required Files (Download from Colab)

After running `ULTIMATE_COLAB_TRAINER.ipynb`, download these files:

1. **`ultimate_ai_model.txt`** - The trained LightGBM model (REQUIRED)
2. **`discovered_formulas.json`** - Genetic algorithm discoveries
3. **`feature_importance.csv`** - Feature rankings
4. **`training_summary.json`** - Training report

## How to Download from Colab

1. In Google Colab, click the **📁 folder icon** on the left sidebar
2. You'll see the generated files listed
3. Right-click each file → **Download**
4. Save them to this `models/` folder

## Model Performance

| Metric | Value |
|--------|-------|
| Win Rate | 85.4% |
| Expected Value | +1.56% per trade |
| Validation | 5-fold walk-forward |
| Fold Consistency | 81% - 87% |

## Top Features

1. `MACD_Hist_5_13` - Importance: 2238
2. `Vol_Price_Trend` - Importance: 1059
3. `OBV` - Importance: 1041
4. `ADX` - Importance: 1006
5. `rs_vs_spy_20d` - Importance: 944

## Usage

```python
from ultimate_signal_generator import UltimateSignalGenerator

gen = UltimateSignalGenerator('models/ultimate_ai_model.txt')
signals = gen.scan_all_tickers()
```

Or run directly:
```bash
python ultimate_signal_generator.py
```
