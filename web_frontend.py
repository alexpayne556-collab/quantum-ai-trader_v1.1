"""
QUANTUM AI TRADER - WEB FRONTEND
================================
Interactive web interface for the Quantum AI Trading System

Features:
- Run comprehensive training for all modules
- View training results and metrics
- Real-time system status
- Interactive charts and visualizations
- Module-specific controls

Built with Flask + Plotly for modern web interface.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our training system
from COMPREHENSIVE_SYSTEM_TEST_AND_TRAINER import (
    run_comprehensive_training,
    prepare_train_test_data,
    train_elliott_wave,
    train_forecast_engine,
    train_risk_manager,
    train_watchlist_scanner,
    validate_trade_executor
)

# Import dashboard components
try:
    from advanced_dashboard import AdvancedDashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))

# Global variables for training data
TRAIN_DATA = None
TEST_DATA = None
TRAINING_RESULTS = {}

# Default tickers
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'NVDA', 'SPY', 'QQQ', 'MU', 'APLD', 'IONQ', 'ANNX', 'TSLA']

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html',
                         tickers=DEFAULT_TICKERS,
                         has_dashboard=HAS_DASHBOARD)

@app.route('/api/status')
def get_status():
    """Get system status"""
    status = {
        'training_data_loaded': TRAIN_DATA is not None and TEST_DATA is not None,
        'tickers_count': len(TRAIN_DATA) if TRAIN_DATA else 0,
        'results_available': len(TRAINING_RESULTS) > 0,
        'modules': list(TRAINING_RESULTS.keys()) if TRAINING_RESULTS else [],
        'timestamp': datetime.now().isoformat()
    }
    return jsonify(status)

@app.route('/api/load_data', methods=['POST'])
def load_data():
    """Load training and test data"""
    global TRAIN_DATA, TEST_DATA

    data = request.get_json()
    tickers = data.get('tickers', DEFAULT_TICKERS)
    train_days = data.get('train_days', 365)
    test_days = data.get('test_days', 90)

    try:
        TRAIN_DATA, TEST_DATA = prepare_train_test_data(tickers, train_days, test_days)
        return jsonify({
            'success': True,
            'message': f'Loaded data for {len(TRAIN_DATA)} tickers',
            'tickers': list(TRAIN_DATA.keys())
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/train_all', methods=['POST'])
def train_all_modules():
    """Run comprehensive training for all modules"""
    global TRAINING_RESULTS

    if TRAIN_DATA is None or TEST_DATA is None:
        return jsonify({
            'success': False,
            'error': 'Training data not loaded. Please load data first.'
        })

    try:
        results = run_comprehensive_training(TRAIN_DATA, TEST_DATA)
        TRAINING_RESULTS = results.get('module_training_results', {})

        return jsonify({
            'success': True,
            'message': 'Training completed successfully',
            'modules_trained': len(TRAINING_RESULTS)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/train_module', methods=['POST'])
def train_single_module():
    """Train a specific module"""
    global TRAINING_RESULTS

    if TRAIN_DATA is None or TEST_DATA is None:
        return jsonify({
            'success': False,
            'error': 'Training data not loaded. Please load data first.'
        })

    data = request.get_json()
    module = data.get('module')

    if not module:
        return jsonify({
            'success': False,
            'error': 'Module name required'
        })

    try:
        if module == 'elliott_wave_detector':
            result = train_elliott_wave(TRAIN_DATA, TEST_DATA)
        elif module == 'forecast_engine':
            result = train_forecast_engine(TRAIN_DATA, TEST_DATA)
        elif module == 'risk_manager':
            result = train_risk_manager(TRAIN_DATA, TEST_DATA)
        elif module == 'watchlist_scanner':
            result = train_watchlist_scanner(TRAIN_DATA, TEST_DATA)
        elif module == 'trade_executor':
            result = validate_trade_executor(TRAIN_DATA, TEST_DATA)
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown module: {module}'
            })

        TRAINING_RESULTS[module] = result

        return jsonify({
            'success': True,
            'message': f'{module} training completed',
            'result': result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/results/<module>')
def get_module_results(module):
    """Get results for a specific module"""
    if module not in TRAINING_RESULTS:
        return jsonify({
            'error': f'No results available for {module}'
        })

    return jsonify(TRAINING_RESULTS[module])

@app.route('/api/chart/<chart_type>')
def get_chart(chart_type):
    """Generate charts for visualization"""
    if not TRAINING_RESULTS:
        return jsonify({'error': 'No training results available'})

    try:
        if chart_type == 'forecast_accuracy':
            # Create forecast accuracy chart
            if 'forecast_engine' in TRAINING_RESULTS:
                fe_results = TRAINING_RESULTS['forecast_engine']
                per_ticker = fe_results.get('per_ticker', {})

                tickers = list(per_ticker.keys())
                accuracies = [r.get('direction_accuracy', 0) for r in per_ticker.values()]
                maes = [r.get('mae', 0) for r in per_ticker.values()]

                fig = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Direction Accuracy', 'Mean Absolute Error'))

                fig.add_trace(go.Bar(x=tickers, y=accuracies, name='Direction Accuracy'),
                            row=1, col=1)
                fig.add_trace(go.Bar(x=tickers, y=maes, name='MAE'),
                            row=1, col=2)

                fig.update_layout(title='Forecast Engine Performance')

        elif chart_type == 'pattern_hit_rates':
            # Create pattern hit rates chart
            if 'elliott_wave_detector' in TRAINING_RESULTS:
                ew_results = TRAINING_RESULTS['elliott_wave_detector']
                per_ticker = ew_results.get('per_ticker', {})

                tickers = list(per_ticker.keys())
                hit_rates = [r.get('pattern_hit_rate', 0) for r in per_ticker.values()]

                fig = go.Figure()
                fig.add_trace(go.Bar(x=tickers, y=hit_rates, name='Pattern Hit Rate'))
                fig.update_layout(title='Elliott Wave Pattern Hit Rates')

        elif chart_type == 'risk_performance':
            # Create risk management performance chart
            if 'risk_manager' in TRAINING_RESULTS:
                rm_results = TRAINING_RESULTS['risk_manager']
                per_ticker = rm_results.get('per_ticker', {})

                tickers = list(per_ticker.keys())
                sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in per_ticker.values()]
                max_dds = [r['metrics'].get('max_drawdown', 0) for r in per_ticker.values()]

                fig = make_subplots(rows=1, cols=2,
                                  subplot_titles=('Sharpe Ratios', 'Max Drawdowns'))

                fig.add_trace(go.Bar(x=tickers, y=sharpes, name='Sharpe Ratio'),
                            row=1, col=1)
                fig.add_trace(go.Bar(x=tickers, y=max_dds, name='Max Drawdown'),
                            row=1, col=2)

                fig.update_layout(title='Risk Manager Performance')

        else:
            return jsonify({'error': f'Unknown chart type: {chart_type}'})

        # Convert to JSON
        chart_json = json.loads(plotly.io.to_json(fig))
        return jsonify(chart_json)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/dashboard')
def dashboard():
    """Advanced dashboard page"""
    if not HAS_DASHBOARD:
        return "Advanced dashboard not available"

    try:
        dashboard = AdvancedDashboard()
        # This would generate the dashboard - simplified for web interface
        return render_template('dashboard.html')
    except Exception as e:
        return f"Dashboard error: {e}"

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(static_dir, exist_ok=True)

    print("🚀 Starting Quantum AI Trader Web Frontend...")
    print("📊 Open http://localhost:5000 in your browser")
    app.run(debug=True, host='0.0.0.0', port=5000)</content>
<parameter name="filePath">e:\quantum-ai-trader-v1.1\web_frontend.py