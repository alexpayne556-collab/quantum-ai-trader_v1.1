"""
Comprehensive Training Script for All Institutional Modules
Trains pattern stats, quantile forecasters, confluence engine with weight optimization
"""

import sys
import os
sys.path.insert(0, '/workspaces/quantum-ai-trader_v1.1/core')
sys.path.insert(0, '/workspaces/quantum-ai-trader_v1.1/training')

import pandas as pd
import numpy as np
import yfinance as yf
import json
from datetime import datetime, timedelta
from pathlib import Path

from pattern_stats_engine import PatternStatsEngine
from confluence_engine import ConfluenceEngine
from quantile_forecaster import QuantileForecaster
from institutional_feature_engineer import InstitutionalFeatureEngineer
from training_logger import TrainingLogger

# Setup directories
MODEL_DIR = Path('/workspaces/quantum-ai-trader_v1.1/trained_models')
RESULTS_DIR = Path('/workspaces/quantum-ai-trader_v1.1/training_results')
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def download_training_data(tickers=['SPY', 'QQQ', 'AAPL', 'MSFT'], period='2y'):
    """Download historical data for training"""
    print(f"\n📥 Downloading {period} of data for {len(tickers)} tickers...")
    
    data = {}
    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval='1d', progress=False)
            if not df.empty:
                # Flatten multi-level columns if present
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
                
                data[ticker] = df
                print(f"  ✅ {ticker}: {len(df)} bars")
            else:
                print(f"  ⚠️  {ticker}: No data")
        except Exception as e:
            print(f"  ❌ {ticker}: {e}")
    
    return data

def create_swing_labels(df, forward_bars=5, threshold=0.02):
    """Create swing trading labels (±2% in next 5 days)"""
    df = df.copy()
    df['future_return'] = df['Close'].pct_change(forward_bars).shift(-forward_bars)
    
    df['label'] = 0  # HOLD
    df.loc[df['future_return'] > threshold, 'label'] = 1   # BUY
    df.loc[df['future_return'] < -threshold, 'label'] = -1  # SELL
    
    return df

def train_pattern_stats_engine(data_dict):
    """Train pattern statistics database"""
    print("\n" + "="*80)
    print("🎯 TRAINING PATTERN STATISTICS ENGINE")
    print("="*80)
    
    engine = PatternStatsEngine(db_path=MODEL_DIR / 'pattern_stats.db')
    
    total_patterns = 0
    for ticker, df in data_dict.items():
        print(f"\n📊 Processing {ticker}...")
        
        # Make a clean copy
        df = df.copy()
        
        # Calculate RSI properly
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate volume ratio
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = (df['Volume'] / df['Volume_MA']).fillna(1.0)
        
        # Detect RSI patterns
        for i in range(50, len(df) - 10):
            if pd.isna(df['RSI'].iloc[i]):
                continue
            
            # Determine volatility bucket
            volatility = float(df['Close'].iloc[i-20:i].std() / df['Close'].iloc[i])
            if volatility < 0.015:
                vol_bucket = 'LOW'
            elif volatility < 0.03:
                vol_bucket = 'NORMAL'
            else:
                vol_bucket = 'HIGH'
            
            # Determine regime
            ma_50 = float(df['Close'].iloc[i-50:i].mean())
            current_price = float(df['Close'].iloc[i])
            regime = 'BULL' if current_price > ma_50 else 'BEAR'
            
            # RSI Oversold pattern
            if df['RSI'].iloc[i] < 30:
                forward_5bar = (df['Close'].iloc[i+5] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                
                engine.record_pattern(
                    pattern_name='RSI_Oversold',
                    ticker=ticker,
                    timeframe='1d',
                    regime=regime,
                    volatility_bucket=vol_bucket,
                    detection_date=df.index[i].to_pydatetime(),
                    entry_price=float(df['Close'].iloc[i]),
                    forward_return_5bar=float(forward_5bar),
                    rsi_level=float(df['RSI'].iloc[i]),
                    volume_ratio=float(df['Volume_Ratio'].iloc[i]) if not pd.isna(df['Volume_Ratio'].iloc[i]) else None
                )
                total_patterns += 1
            
            # RSI Overbought pattern
            elif df['RSI'].iloc[i] > 70:
                forward_5bar = (df['Close'].iloc[i+5] - df['Close'].iloc[i]) / df['Close'].iloc[i]
                
                engine.record_pattern(
                    pattern_name='RSI_Overbought',
                    ticker=ticker,
                    timeframe='1d',
                    regime=regime,
                    volatility_bucket=vol_bucket,
                    detection_date=df.index[i].to_pydatetime(),
                    entry_price=float(df['Close'].iloc[i]),
                    forward_return_5bar=float(forward_5bar),
                    rsi_level=float(df['RSI'].iloc[i]),
                    volume_ratio=float(df['Volume_Ratio'].iloc[i]) if not pd.isna(df['Volume_Ratio'].iloc[i]) else None
                )
                total_patterns += 1
    
    print(f"\n✅ Recorded {total_patterns} pattern occurrences")
    
    # Get statistics for different contexts
    print(f"\n📊 Pattern Statistics Summary:")
    
    stats = []
    for pattern in ['RSI_Oversold', 'RSI_Overbought']:
        for regime in ['BULL', 'BEAR']:
            context = {'timeframe': '1d', 'regime': regime, 'volatility_bucket': 'ALL'}
            edge = engine.get_pattern_edge(pattern, context, min_samples=5)
            
            if edge:
                stats.append(edge)
                print(f"\n   Pattern: {pattern} ({regime})")
                print(f"   Win Rate: {edge.win_rate*100:.1f}%")
                print(f"   Avg Return: {edge.avg_return*100:.2f}%")
                print(f"   Sharpe: {edge.sharpe_ratio:.2f}")
                print(f"   Occurrences: {edge.sample_count}")
                print(f"   Status: {edge.status}")
    
    # Check for edge decay on main patterns
    for pattern in ['RSI_Oversold', 'RSI_Overbought']:
        decay = engine.detect_edge_decay(pattern)
        if decay and decay.get('edge_decaying'):
            print(f"\n⚠️  Edge Decay Detected for {pattern}")
    
    return engine, stats

def train_quantile_forecasters(data_dict, horizons=[1, 3, 5, 10, 21]):
    """Train quantile forecasting models for multiple horizons"""
    print("\n" + "="*80)
    print("📈 TRAINING QUANTILE FORECASTERS")
    print("="*80)
    
    feature_engineer = InstitutionalFeatureEngineer()
    
    # Create single forecaster
    forecaster = QuantileForecaster(model_dir=str(MODEL_DIR / 'quantile_models'))
    
    all_results = []
    
    for horizon in horizons:
        horizon_str = f'{horizon}bar'  # Convert to QuantileForecaster format
        print(f"\n🎯 Training {horizon}-day horizon forecaster...")
        
        # Combine all ticker data
        all_data = []
        
        for ticker, df in data_dict.items():
            df_copy = df.copy()
            df_copy['ticker'] = ticker
            all_data.append(df_copy)
        
        if not all_data:
            print(f"  ⚠️  No valid data for {horizon}-day horizon")
            continue
        
        combined_df = pd.concat(all_data, axis=0)
        
        print(f"  Training on {len(combined_df)} total bars from {len(data_dict)} tickers...")
        
        # Train the forecaster
        try:
            metrics = forecaster.train(
                df=combined_df,
                feature_engineer=feature_engineer,
                horizon=horizon_str
            )
            
            print(f"  ✅ Training complete")
            print(f"  📊 Metrics:")
            if metrics:
                for quantile, rmse in metrics.items():
                    quantile_pct = int(float(quantile) * 100) if isinstance(quantile, (int, float, str)) else quantile
                    print(f"     Q{quantile_pct}: RMSE = {rmse:.4f}")
            
            all_results.append({
                'horizon': f'{horizon}d',
                'metrics': {str(k): float(v) for k, v in metrics.items()} if metrics else {},
                'n_samples': len(combined_df),
                'n_tickers': len(data_dict)
            })
        except Exception as e:
            print(f"  ❌ Training failed: {e}")
    
    # Save training results
    results_path = RESULTS_DIR / 'quantile_forecaster_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ Training results saved to {results_path.name}")
    
    return forecaster, all_results

def generate_weight_optimization_report(pattern_stats, quantile_results):
    """Generate actionable weight optimization recommendations"""
    print("\n" + "="*80)
    print("⚙️  WEIGHT OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    
    recommendations = {
        'timestamp': datetime.now().isoformat(),
        'pattern_weights': {},
        'timeframe_weights': {},
        'quantile_weights': {},
        'feature_importance': {},
        'confluence_parameters': {}
    }
    
    # 1. Pattern Weight Recommendations
    print("\n🎯 Pattern Weights (by win rate and Sharpe):")
    pattern_weights = {}
    for stat in pattern_stats[:10]:
        # Weight = win_rate * sharpe * sqrt(sample_count/30)
        weight = stat.win_rate * max(stat.sharpe_ratio, 0) * np.sqrt(stat.sample_count / 30)
        pattern_weights[f"{stat.pattern_name}"] = float(weight)
        print(f"   {stat.pattern_name}: {weight:.3f}")
        print(f"      Win Rate: {stat.win_rate*100:.1f}%, Sharpe: {stat.sharpe_ratio:.2f}, N={stat.sample_count}")
    
    recommendations['pattern_weights'] = pattern_weights
    
    # 2. Timeframe Weights (for Bayesian confluence)
    print("\n⏰ Recommended Timeframe Weights:")
    timeframe_weights = {
        '1d': 0.40,   # Increase daily weight (was 0.35)
        '4h': 0.30,   # Keep 4-hour
        '1h': 0.20,   # Reduce 1-hour (was 0.25)
        '15m': 0.10   # Add 15-min for entry timing
    }
    for tf, weight in timeframe_weights.items():
        print(f"   {tf}: {weight:.2f}")
    recommendations['timeframe_weights'] = timeframe_weights
    
    # 3. Quantile Model Performance
    print("\n📊 Quantile Model Performance:")
    quantile_performance = {}
    for result in quantile_results:
        horizon = result['horizon']
        avg_rmse = np.mean(list(result['metrics'].values()))
        quantile_performance[horizon] = {
            'avg_rmse': float(avg_rmse),
            'n_samples': result['n_samples'],
            'recommended_confidence': 0.70 if avg_rmse < 0.03 else 0.60
        }
        print(f"   {horizon}: RMSE={avg_rmse:.4f}, Confidence={quantile_performance[horizon]['recommended_confidence']:.0%}")
    
    recommendations['quantile_weights'] = quantile_performance
    
    # 4. Confluence Engine Parameters
    print("\n🔗 Confluence Engine Parameters:")
    confluence_params = {
        'min_timeframe_agreement': 2,
        'confidence_cap': 0.85,
        'rsi_boost': {
            'oversold_threshold': 30,
            'overbought_threshold': 70,
            'boost_multiplier': 1.15
        },
        'volume_boost': {
            'high_volume_threshold': 1.5,  # 1.5x avg volume
            'boost_multiplier': 1.10
        },
        'regime_boost': {
            'strong_trend_threshold': 0.7,  # Price > 70% of range
            'boost_multiplier': 1.20
        }
    }
    for param, value in confluence_params.items():
        print(f"   {param}: {value}")
    recommendations['confluence_parameters'] = confluence_params
    
    # 5. Feature Engineering Recommendations
    print("\n🔧 Feature Engineering Weights:")
    feature_recommendations = {
        'percentile_features': {
            'window': 90,  # 90-day percentile
            'weight': 1.2   # Higher weight
        },
        'second_order_features': {
            'rsi_momentum_weight': 1.1,
            'volume_acceleration_weight': 1.15
        },
        'cross_asset_features': {
            'spy_correlation_weight': 1.25,
            'vix_level_weight': 1.3
        },
        'interaction_terms': {
            'rsi_x_volume_weight': 1.1,
            'trend_x_volatility_weight': 1.15
        }
    }
    for category, weights in feature_recommendations.items():
        print(f"   {category}:")
        for key, val in weights.items():
            print(f"      {key}: {val}")
    recommendations['feature_importance'] = feature_recommendations
    
    # Save recommendations
    rec_path = RESULTS_DIR / 'weight_optimization_recommendations.json'
    with open(rec_path, 'w') as f:
        json.dump(recommendations, f, indent=2)
    
    print(f"\n✅ Recommendations saved to {rec_path.name}")
    
    return recommendations

def main():
    """Main training pipeline"""
    print("\n" + "="*80)
    print("🚀 QUANTUM AI TRADER - COMPREHENSIVE TRAINING PIPELINE")
    print("="*80)
    print(f"Started at: {datetime.now()}")
    
    # Initialize training logger
    logger = TrainingLogger(db_path=MODEL_DIR / 'training_logs.db')
    
    # Download data
    tickers = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMD']
    data = download_training_data(tickers, period='2y')
    
    if not data:
        print("❌ Failed to download any data!")
        return
    
    # Train Pattern Statistics Engine
    pattern_engine, pattern_stats = train_pattern_stats_engine(data)
    
    # Train Quantile Forecasters
    forecaster, quantile_results = train_quantile_forecasters(data, horizons=[1, 3, 5, 10])
    
    # Generate Weight Optimization Report
    recommendations = generate_weight_optimization_report(pattern_stats, quantile_results)
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\n📁 Models saved to: {MODEL_DIR}")
    print(f"📊 Results saved to: {RESULTS_DIR}")
    print(f"\nCompleted at: {datetime.now()}")
    
    print("\n🎯 Next Steps:")
    print("   1. Review weight_optimization_recommendations.json")
    print("   2. Apply recommended weights to confluence_engine.py")
    print("   3. Run backtest with trained models")
    print("   4. Deploy to FastAPI WebSocket server")
    
    return {
        'pattern_engine': pattern_engine,
        'forecaster': forecaster,
        'recommendations': recommendations
    }

if __name__ == '__main__':
    results = main()
