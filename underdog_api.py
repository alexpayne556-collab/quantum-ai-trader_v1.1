"""
🚀 UNDERDOG API - Flask Backend for Spark Dashboard Integration

Serves Alpha 76 predictions from trained Colab models.
Plugs seamlessly into existing Spark frontend.

Endpoints:
  GET  /api/underdog/status         - System health check
  GET  /api/underdog/alpha76         - Get Alpha 76 watchlist
  POST /api/underdog/predict         - Get prediction for ticker
  POST /api/underdog/batch-predict   - Batch predictions for multiple tickers
  GET  /api/underdog/regime          - Current market regime
  GET  /api/underdog/top-signals     - Top BUY signals (high confidence)
  
Run: python underdog_api.py
Access: http://localhost:5000/api/underdog/
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import sys
import os
from typing import Dict, List, Optional

# Add src/python to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'python'))

from multi_model_ensemble import MultiModelEnsemble
from feature_engine import FeatureEngine
from regime_classifier import RegimeClassifier

# Initialize Flask
app = Flask(__name__)

# Configure CORS for Spark frontend
cors_origins = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173,http://localhost:8080').split(',')
CORS(app, resources={r"/api/*": {"origins": cors_origins}})

# Alpha 76 Watchlist
ALPHA_76 = [
    # Autonomous & AI Hardware
    'SYM', 'IONQ', 'RGTI', 'QUBT', 'AMBA', 'LAZR', 'INVZ', 'OUST', 'AEVA', 'SERV',
    
    # Space Economy
    'RKLB', 'ASTS', 'LUNR', 'JOBY', 'ACHR', 'PL', 'SPIR', 'IRDM',
    
    # Biotech (Gene Editing & Rare Disease)
    'VKTX', 'NTLA', 'BEAM', 'CRSP', 'EDIT', 'VERV', 'BLUE', 'FATE', 'AKRO', 'KOD',
    'CYTK', 'LEGN', 'RARE', 'SRPT', 'BMRN', 'ALNY',
    
    # Green Energy & Grid
    'FLNC', 'NXT', 'BE', 'ARRY', 'ENPH', 'ENOV', 'QS', 'VST', 'AES',
    
    # Fintech & Digital Assets
    'SOFI', 'COIN', 'HOOD', 'UPST', 'AFRM', 'LC', 'MARA', 'SQ', 'NU',
    
    # Next-Gen Consumer & Software
    'APP', 'DUOL', 'PATH', 'S', 'CELH', 'ONON', 'SOUN', 'FOUR', 'NET', 'GTLB',
    'DDOG', 'SNOW', 'PLTR', 'RBLX', 'U'
]

# Sector mapping
SECTOR_MAP = {
    'SYM': 'Autonomous', 'IONQ': 'Autonomous', 'RGTI': 'Autonomous', 'QUBT': 'Autonomous',
    'AMBA': 'Autonomous', 'LAZR': 'Autonomous', 'INVZ': 'Autonomous', 'OUST': 'Autonomous',
    'AEVA': 'Autonomous', 'SERV': 'Autonomous',
    
    'RKLB': 'Space', 'ASTS': 'Space', 'LUNR': 'Space', 'JOBY': 'Space', 'ACHR': 'Space',
    'PL': 'Space', 'SPIR': 'Space', 'IRDM': 'Space',
    
    'VKTX': 'Biotech', 'NTLA': 'Biotech', 'BEAM': 'Biotech', 'CRSP': 'Biotech', 'EDIT': 'Biotech',
    'VERV': 'Biotech', 'BLUE': 'Biotech', 'FATE': 'Biotech', 'AKRO': 'Biotech', 'KOD': 'Biotech',
    'CYTK': 'Biotech', 'LEGN': 'Biotech', 'RARE': 'Biotech', 'SRPT': 'Biotech', 'BMRN': 'Biotech',
    'ALNY': 'Biotech',
    
    'FLNC': 'Energy', 'NXT': 'Energy', 'BE': 'Energy', 'ARRY': 'Energy', 'ENPH': 'Energy',
    'ENOV': 'Energy', 'QS': 'Energy', 'VST': 'Energy', 'AES': 'Energy',
    
    'SOFI': 'Fintech', 'COIN': 'Fintech', 'HOOD': 'Fintech', 'UPST': 'Fintech', 'AFRM': 'Fintech',
    'LC': 'Fintech', 'MARA': 'Fintech', 'SQ': 'Fintech', 'NU': 'Fintech',
    
    'APP': 'Software', 'DUOL': 'Software', 'PATH': 'Software', 'S': 'Software', 'CELH': 'Software',
    'ONON': 'Software', 'SOUN': 'Software', 'FOUR': 'Software', 'NET': 'Software', 'GTLB': 'Software',
    'DDOG': 'Software', 'SNOW': 'Software', 'PLTR': 'Software', 'RBLX': 'Software', 'U': 'Software'
}

# Global instances (lazy loaded)
ensemble = None
feature_engine = None
regime_classifier = None

def init_models():
    """Initialize models (load from disk after Colab training)"""
    global ensemble, feature_engine, regime_classifier
    
    if ensemble is None:
        ensemble = MultiModelEnsemble(use_gpu=False)
        
        # Try to load trained models
        model_path = 'models/underdog_v1'
        if os.path.exists(model_path):
            try:
                ensemble.load(model_path)
                print(f"✅ Loaded trained ensemble from {model_path}")
            except Exception as e:
                print(f"⚠️ Could not load models: {e}")
                print("   Using untrained ensemble (train on Colab first)")
    
    if feature_engine is None:
        feature_engine = FeatureEngine()
        print("✅ Feature engine initialized")
    
    if regime_classifier is None:
        regime_classifier = RegimeClassifier()
        print("✅ Regime classifier initialized")


def download_data(ticker: str, period: str = '3mo', interval: str = '1h') -> pd.DataFrame:
    """Download recent data for a ticker"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        if len(df) > 0:
            df = df.reset_index()
            # Handle MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df['ticker'] = ticker
            return df
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
    return pd.DataFrame()


def generate_prediction(ticker: str) -> Dict:
    """Generate prediction for a single ticker"""
    init_models()
    
    # Download data
    df = download_data(ticker)
    if len(df) < 50:
        return {
            'ticker': ticker,
            'error': 'Insufficient data',
            'success': False
        }
    
    # Calculate features
    try:
        df_features = feature_engine.calculate_all_features(df)
        df_features = feature_engine.fill_missing_values(df_features)
    except Exception as e:
        return {
            'ticker': ticker,
            'error': f'Feature calculation failed: {str(e)}',
            'success': False
        }
    
    # Get latest features
    feature_cols = feature_engine.get_feature_names()
    X = df_features[feature_cols].iloc[-1:]
    
    # Make prediction
    try:
        if ensemble.is_trained:
            pred = ensemble.predict(X)
        else:
            return {
                'ticker': ticker,
                'error': 'Models not trained - train on Colab Pro first',
                'success': False
            }
    except Exception as e:
        return {
            'ticker': ticker,
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }
    
    # Get current regime
    regime = regime_classifier.classify_regime()
    
    # Check if signal meets regime criteria
    meets_regime = pred['confidence'] >= regime['min_confidence']
    
    # Get current price
    current_price = float(df['close'].iloc[-1])
    
    return {
        'ticker': ticker,
        'sector': SECTOR_MAP.get(ticker, 'Unknown'),
        'signal': pred['signal'],
        'confidence': round(pred['confidence'], 3),
        'agreement': round(pred['agreement'], 3),
        'votes': pred['votes'],
        'vote_counts': pred['vote_counts'],
        'current_price': round(current_price, 2),
        'regime': regime['name'],
        'regime_filter': 'PASS' if meets_regime else 'FAIL',
        'position_size_multiplier': regime['position_size_multiplier'],
        'stop_loss_pct': regime['stop_loss_pct'],
        'timestamp': datetime.now().isoformat(),
        'success': True
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/underdog/status', methods=['GET'])
def status():
    """System health check"""
    init_models()
    
    return jsonify({
        'success': True,
        'status': 'operational',
        'models': {
            'ensemble_trained': ensemble.is_trained if ensemble else False,
            'feature_engine': feature_engine is not None,
            'regime_classifier': regime_classifier is not None
        },
        'watchlist_size': len(ALPHA_76),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/underdog/alpha76', methods=['GET'])
def alpha76_watchlist():
    """Get Alpha 76 watchlist with sector breakdown"""
    sector_breakdown = {}
    for ticker in ALPHA_76:
        sector = SECTOR_MAP.get(ticker, 'Unknown')
        if sector not in sector_breakdown:
            sector_breakdown[sector] = []
        sector_breakdown[sector].append(ticker)
    
    return jsonify({
        'success': True,
        'watchlist': ALPHA_76,
        'total_tickers': len(ALPHA_76),
        'sectors': sector_breakdown,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/underdog/predict', methods=['POST'])
def predict_ticker():
    """Get prediction for single ticker"""
    data = request.get_json()
    ticker = data.get('ticker', '').upper()
    
    if not ticker:
        return jsonify({'success': False, 'error': 'Missing ticker parameter'}), 400
    
    if ticker not in ALPHA_76:
        return jsonify({
            'success': False,
            'error': f'{ticker} not in Alpha 76 watchlist',
            'suggestion': 'Use /api/underdog/alpha76 to see available tickers'
        }), 400
    
    result = generate_prediction(ticker)
    return jsonify(result)


@app.route('/api/underdog/batch-predict', methods=['POST'])
def batch_predict():
    """Batch predictions for multiple tickers"""
    data = request.get_json()
    tickers = data.get('tickers', [])
    
    if not tickers:
        # Default: All Alpha 76
        tickers = ALPHA_76
    else:
        # Validate tickers
        tickers = [t.upper() for t in tickers if t.upper() in ALPHA_76]
    
    if not tickers:
        return jsonify({'success': False, 'error': 'No valid tickers provided'}), 400
    
    predictions = []
    for ticker in tickers[:20]:  # Limit to 20 for performance
        pred = generate_prediction(ticker)
        if pred['success']:
            predictions.append(pred)
    
    return jsonify({
        'success': True,
        'predictions': predictions,
        'total': len(predictions),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/underdog/regime', methods=['GET'])
def current_regime():
    """Get current market regime"""
    init_models()
    
    regime = regime_classifier.classify_regime()
    
    return jsonify({
        'success': True,
        'regime': regime,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/underdog/top-signals', methods=['GET'])
def top_signals():
    """Get top BUY signals (high confidence)"""
    init_models()
    
    # Get regime first
    regime = regime_classifier.classify_regime()
    min_confidence = regime['min_confidence']
    
    # Generate predictions for all Alpha 76
    signals = []
    for ticker in ALPHA_76[:30]:  # Limit to 30 for speed
        pred = generate_prediction(ticker)
        if pred['success'] and pred['signal'] == 'BUY' and pred['confidence'] >= min_confidence:
            signals.append(pred)
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return jsonify({
        'success': True,
        'signals': signals[:10],  # Top 10
        'regime': regime['name'],
        'min_confidence': min_confidence,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/underdog/backtest-summary', methods=['GET'])
def backtest_summary():
    """Return backtest performance summary (if available)"""
    # This will be populated after Colab training
    summary = {
        'training_date': 'Not trained yet',
        'validation_accuracy': 0.0,
        'models': {
            'xgboost': {'accuracy': 0.0, 'roc_auc': 0.0},
            'random_forest': {'accuracy': 0.0, 'roc_auc': 0.0},
            'gradient_boosting': {'accuracy': 0.0, 'roc_auc': 0.0}
        },
        'backtest': {
            'win_rate': 0.0,
            'avg_return': 0.0,
            'total_signals': 0
        }
    }
    
    # Try to load from training results
    results_file = 'models/underdog_v1/training_metrics.json'
    if os.path.exists(results_file):
        try:
            import json
            with open(results_file, 'r') as f:
                summary = json.load(f)
        except:
            pass
    
    return jsonify({
        'success': True,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    })


# ============================================================================
# HEALTH & METADATA
# ============================================================================

@app.route('/api/underdog/', methods=['GET'])
@app.route('/api/underdog', methods=['GET'])
def api_root():
    """API documentation"""
    return jsonify({
        'name': 'Underdog Trading System API',
        'version': '1.0.0',
        'description': 'Alpha 76 predictions from 3-model ensemble',
        'endpoints': {
            'GET /api/underdog/status': 'System health check',
            'GET /api/underdog/alpha76': 'Get Alpha 76 watchlist',
            'POST /api/underdog/predict': 'Predict single ticker (JSON: {ticker: "RKLB"})',
            'POST /api/underdog/batch-predict': 'Batch predictions (JSON: {tickers: [...]})',
            'GET /api/underdog/regime': 'Current market regime',
            'GET /api/underdog/top-signals': 'Top 10 BUY signals',
            'GET /api/underdog/backtest-summary': 'Training & backtest metrics'
        },
        'architecture': {
            'models': 'XGBoost (GPU) + RandomForest + GradientBoosting',
            'features': '49 technical indicators',
            'regime_detection': '10 market regimes (VIX, SPY, yield curve)',
            'horizon': '5-bar forward (5 hours)',
            'training': 'Colab Pro T4 GPU (2-4 hours)'
        },
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 UNDERDOG API - Flask Backend for Spark Dashboard")
    print("="*70)
    print(f"\nAlpha 76 Watchlist: {len(ALPHA_76)} tickers")
    print(f"Sectors: {len(set(SECTOR_MAP.values()))} (Autonomous, Space, Biotech, Energy, Fintech, Software)")
    print(f"\nStarting server on http://localhost:5000")
    print(f"API Root: http://localhost:5000/api/underdog/")
    print(f"\nAvailable Endpoints:")
    print(f"  GET  /api/underdog/status")
    print(f"  GET  /api/underdog/alpha76")
    print(f"  POST /api/underdog/predict")
    print(f"  POST /api/underdog/batch-predict")
    print(f"  GET  /api/underdog/regime")
    print(f"  GET  /api/underdog/top-signals")
    print(f"  GET  /api/underdog/backtest-summary")
    print(f"\nReady for Spark dashboard integration! 🎯")
    print("="*70 + "\n")
    
    # Initialize models on startup
    init_models()
    
    # Start Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
