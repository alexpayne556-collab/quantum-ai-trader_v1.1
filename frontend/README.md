# Quantum AI Trader - Web Frontend

Interactive web interface for training and managing the Quantum AI Trading System.

## Features

- **System Control Center**: Load training data, train modules, monitor status
- **Module Training**: Individual or batch training of all trading modules
- **Real-time Visualization**: Interactive charts showing training results
- **Advanced Dashboard**: Integration with existing dashboard components

## Quick Start

1. Install dependencies:
```bash
pip install -r frontend/requirements.txt
```

2. Start the web server:
```bash
python web_frontend.py
```

3. Open http://localhost:5000 in your browser

## Usage

### 1. Load Training Data
- Select tickers to train on
- Set training and test periods
- Click "Load Data" to fetch historical data

### 2. Train Modules
- Train individual modules or all at once
- Monitor progress in real-time
- View results in interactive charts

### 3. View Results
- Forecast accuracy metrics
- Pattern detection hit rates
- Risk management performance
- Training summaries

## Modules

- **Elliott Wave Detector**: Pattern recognition and wave analysis
- **Forecast Engine**: 24-day price forecasting with confidence bands
- **Risk Manager**: Position sizing and drawdown control optimization
- **Watchlist Scanner**: Filter tuning for candidate selection
- **Trade Executor**: Execution validation and safety checks

## API Endpoints

- `GET /api/status`: System status
- `POST /api/load_data`: Load training data
- `POST /api/train_all`: Train all modules
- `POST /api/train_module`: Train specific module
- `GET /api/results/<module>`: Get module results
- `GET /api/chart/<type>`: Get visualization charts

## Integration with Colab

The web frontend integrates seamlessly with the comprehensive training system, allowing you to:

1. Train modules locally via the web interface
2. Upload results to Colab for further analysis
3. Use the same TRAIN_DATA/TEST_DATA structure
4. Export trained models for production deployment

## Architecture

```
web_frontend.py (Flask app)
├── COMPREHENSIVE_SYSTEM_TEST_AND_TRAINER.py (training logic)
├── frontend/templates/ (HTML templates)
├── frontend/static/ (CSS/JS assets)
└── [module files] (risk_manager.py, forecast_engine.py, etc.)
```