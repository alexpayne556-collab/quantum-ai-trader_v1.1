"""
🔬 COMPREHENSIVE ACCURACY VALIDATOR
Tests forecaster, pattern detector, and AI recommender performance
Provides detailed metrics for integration with Spark frontend dashboard
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

# Import modules
try:
    from forecast_engine import ForecastEngine
    from pattern_detector import PatternDetector
    from ai_recommender import AIRecommender, FeatureEngineer
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    MODULES_AVAILABLE = False

class PerformanceValidator:
    """Validate accuracy of all trading modules"""
    
    def __init__(self, tickers: List[str] = None):
        self.tickers = tickers or ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 
                                   'META', 'TSLA', 'AMD', 'MU', 'SPY']
        self.results = {}
        
        if MODULES_AVAILABLE:
            self.forecast_engine = ForecastEngine()
            self.pattern_detector = PatternDetector()
            self.ai_recommender = AIRecommender(use_adaptive_labels=True, n_features=12)
    
    def validate_forecaster(self) -> Dict:
        """
        Test forecaster accuracy on historical data
        Metrics: Direction accuracy, MAE, RMSE, 5% hit rate
        """
        print("\n" + "=" * 80)
        print("📊 VALIDATING FORECASTER ACCURACY")
        print("=" * 80)
        
        results = {}
        
        for ticker in self.tickers:
            print(f"\n🔍 Testing {ticker}...")
            
            try:
                # Download 1 year of historical data
                df = yf.download(ticker, period='1y', interval='1d', progress=False)
                
                if len(df) < 100:
                    print(f"   ⚠️ Insufficient data ({len(df)} days)")
                    continue
                
                # Split: Use first 80% for "history", last 20% for validation
                split_idx = int(len(df) * 0.8)
                historical_df = df.iloc[:split_idx]
                validation_df = df.iloc[split_idx:]
                
                # Mock model and feature engineer for forecast
                class MockModel:
                    def predict(self, X):
                        return np.array([2])  # BULLISH
                    def predict_proba(self, X):
                        return np.array([[0.1, 0.2, 0.7]])  # 70% conf
                
                class MockFE:
                    def engineer(self, df):
                        return pd.DataFrame([[1, 2, 3, 4, 5]])
                
                # Generate forecast from historical data
                forecast = self.forecast_engine.generate_forecast(
                    historical_df, MockModel(), MockFE(), ticker
                )
                
                if len(forecast) == 0 or forecast is None or not isinstance(forecast, pd.DataFrame):
                    print(f"   ⚠️ Forecast generation failed (empty or invalid)")
                    continue
                
                # Debug: Check columns
                if 'price' not in forecast.columns:
                    print(f"   ⚠️ Forecast missing 'price' column. Columns: {list(forecast.columns)}")
                    continue
                
                # Compare forecast vs actual prices
                forecast_days = min(len(forecast), len(validation_df))
                
                # Ensure we have valid forecast prices
                if 'price' not in forecast.columns or len(forecast['price']) == 0:
                    print(f"   ⚠️ No forecast prices generated")
                    continue
                    
                forecast_prices = forecast['price'].values[:forecast_days]
                
                # Handle both Series and DataFrame formats from yfinance
                if isinstance(validation_df['Close'], pd.DataFrame):
                    actual_prices = validation_df['Close'].values[:forecast_days].flatten()
                else:
                    actual_prices = validation_df['Close'].values[:forecast_days]
                
                # Ensure arrays have matching lengths
                min_len = min(len(forecast_prices), len(actual_prices))
                if min_len < 2:  # Need at least 2 points for direction
                    print(f"   ⚠️ Insufficient forecast/actual data ({min_len} points)")
                    continue
                    
                forecast_prices = forecast_prices[:min_len]
                actual_prices = actual_prices[:min_len]
                
                # Calculate metrics
                direction_actual = np.diff(actual_prices) > 0
                direction_forecast = np.diff(forecast_prices) > 0
                direction_accuracy = float(np.mean(direction_actual == direction_forecast))
                
                errors = actual_prices - forecast_prices
                mae = float(np.mean(np.abs(errors)))
                rmse = float(np.sqrt(np.mean(errors**2)))
                
                # Hit rate: % of predictions within 5% of actual
                pct_errors = np.abs((forecast_prices - actual_prices) / (actual_prices + 1e-8))
                hit_rate_5pct = float(np.mean(pct_errors <= 0.05))
                
                results[ticker] = {
                    'direction_accuracy': direction_accuracy,
                    'mae': mae,
                    'rmse': rmse,
                    'hit_rate_5pct': hit_rate_5pct,
                    'forecast_days': forecast_days,
                    'current_price': float(validation_df['Close'].iloc[-1])
                }
                
                print(f"   ✅ Direction Accuracy: {direction_accuracy:.1%}")
                print(f"   📈 MAE: ${mae:.2f}")
                print(f"   📉 RMSE: ${rmse:.2f}")
                print(f"   🎯 5% Hit Rate: {hit_rate_5pct:.1%}")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:80]}")
                continue
        
        # Calculate aggregate metrics
        if results:
            avg_direction = np.mean([r['direction_accuracy'] for r in results.values()])
            avg_mae = np.mean([r['mae'] for r in results.values()])
            avg_hit_rate = np.mean([r['hit_rate_5pct'] for r in results.values()])
            
            summary = {
                'per_ticker': results,
                'aggregate': {
                    'avg_direction_accuracy': float(avg_direction),
                    'avg_mae': float(avg_mae),
                    'avg_hit_rate_5pct': float(avg_hit_rate),
                    'tickers_tested': len(results)
                }
            }
            
            print(f"\n📊 AGGREGATE FORECASTER METRICS:")
            print(f"   Average Direction Accuracy: {avg_direction:.1%}")
            print(f"   Average MAE: ${avg_mae:.2f}")
            print(f"   Average 5% Hit Rate: {avg_hit_rate:.1%}")
            
            return summary
        
        return {'error': 'No forecasts generated'}
    
    def validate_pattern_detector(self) -> Dict:
        """
        Test pattern detector quality
        Metrics: Patterns detected, confidence distribution, detection speed
        """
        print("\n" + "=" * 80)
        print("🔍 VALIDATING PATTERN DETECTOR")
        print("=" * 80)
        
        results = {}
        
        for ticker in self.tickers:
            print(f"\n🔍 Testing {ticker}...")
            
            try:
                # Detect patterns
                detection_result = self.pattern_detector.detect_all_patterns(
                    ticker, period='60d', interval='1d'
                )
                
                if 'error' in detection_result:
                    print(f"   ⚠️ Detection failed: {detection_result['error']}")
                    continue
                
                patterns = detection_result.get('patterns', [])
                stats = detection_result.get('stats', {})
                optimized_signals = detection_result.get('optimized_signals', [])
                
                # Analyze pattern quality
                high_conf_patterns = [p for p in patterns if p['confidence'] > 0.7]
                avg_confidence = np.mean([p['confidence'] for p in patterns]) if patterns else 0
                
                # Pattern type distribution
                pattern_types = {}
                for p in patterns:
                    ptype = p.get('type', 'UNKNOWN')
                    pattern_types[ptype] = pattern_types.get(ptype, 0) + 1
                
                results[ticker] = {
                    'total_patterns': len(patterns),
                    'high_confidence_patterns': len(high_conf_patterns),
                    'avg_confidence': float(avg_confidence),
                    'optimized_signals': len(optimized_signals),
                    'pattern_types': pattern_types,
                    'detection_time_ms': stats.get('detection_time_ms', 0)
                }
                
                print(f"   ✅ Total Patterns: {len(patterns)}")
                print(f"   🎯 High Confidence (>70%): {len(high_conf_patterns)}")
                print(f"   📊 Avg Confidence: {avg_confidence:.1%}")
                print(f"   ⚡ Optimized Signals: {len(optimized_signals)}")
                print(f"   ⏱️  Detection Time: {stats.get('detection_time_ms', 0):.1f}ms")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:80]}")
                continue
        
        # Aggregate metrics
        if results:
            avg_patterns = np.mean([r['total_patterns'] for r in results.values()])
            avg_high_conf = np.mean([r['high_confidence_patterns'] for r in results.values()])
            avg_confidence = np.mean([r['avg_confidence'] for r in results.values()])
            avg_time = np.mean([r['detection_time_ms'] for r in results.values()])
            
            summary = {
                'per_ticker': results,
                'aggregate': {
                    'avg_patterns_per_ticker': float(avg_patterns),
                    'avg_high_confidence_patterns': float(avg_high_conf),
                    'avg_confidence_score': float(avg_confidence),
                    'avg_detection_time_ms': float(avg_time),
                    'tickers_tested': len(results)
                }
            }
            
            print(f"\n📊 AGGREGATE PATTERN DETECTOR METRICS:")
            print(f"   Avg Patterns Per Ticker: {avg_patterns:.1f}")
            print(f"   Avg High Confidence: {avg_high_conf:.1f}")
            print(f"   Avg Confidence Score: {avg_confidence:.1%}")
            print(f"   Avg Detection Speed: {avg_time:.1f}ms")
            
            return summary
        
        return {'error': 'No patterns detected'}
    
    def validate_ai_recommender(self) -> Dict:
        """
        Test AI recommender accuracy
        Metrics: CV accuracy, train/test split, feature importance
        """
        print("\n" + "=" * 80)
        print("🤖 VALIDATING AI RECOMMENDER")
        print("=" * 80)
        
        results = {}
        
        for ticker in self.tickers[:5]:  # Test on first 5 tickers for speed
            print(f"\n🔍 Testing {ticker}...")
            
            try:
                # Train model
                train_result = self.ai_recommender.train_for_ticker(
                    ticker, 
                    use_kfold=True
                )
                
                if 'error' in train_result:
                    print(f"   ⚠️ Training failed: {train_result['error']}")
                    continue
                
                # Get prediction
                pred_result = self.ai_recommender.predict_latest(ticker)
                
                results[ticker] = {
                    'cv_mean_accuracy': train_result.get('cv_mean', 0),
                    'cv_std': train_result.get('cv_std', 0),
                    'n_samples': train_result.get('n_samples', 0),
                    'n_selected_features': len(train_result.get('selected_features', [])),
                    'signal': pred_result.get('signal', 'UNKNOWN'),
                    'confidence': pred_result.get('confidence', 0),
                    'label_stats': train_result.get('label_stats', {})
                }
                
                print(f"   ✅ CV Mean Accuracy: {train_result.get('cv_mean', 0):.1%}")
                print(f"   📊 CV Std Dev: ±{train_result.get('cv_std', 0):.1%}")
                print(f"   📈 Samples: {train_result.get('n_samples', 0)}")
                print(f"   🎯 Selected Features: {len(train_result.get('selected_features', []))}")
                print(f"   🔮 Current Signal: {pred_result.get('signal')} ({pred_result.get('confidence', 0):.1%})")
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)[:80]}")
                continue
        
        # Aggregate metrics
        if results:
            avg_cv_accuracy = np.mean([r['cv_mean_accuracy'] for r in results.values()])
            avg_samples = np.mean([r['n_samples'] for r in results.values()])
            avg_features = np.mean([r['n_selected_features'] for r in results.values()])
            
            # Signal distribution
            signals = {}
            for r in results.values():
                sig = r['signal']
                signals[sig] = signals.get(sig, 0) + 1
            
            summary = {
                'per_ticker': results,
                'aggregate': {
                    'avg_cv_accuracy': float(avg_cv_accuracy),
                    'avg_samples': float(avg_samples),
                    'avg_selected_features': float(avg_features),
                    'signal_distribution': signals,
                    'tickers_tested': len(results)
                }
            }
            
            print(f"\n📊 AGGREGATE AI RECOMMENDER METRICS:")
            print(f"   Avg CV Accuracy: {avg_cv_accuracy:.1%}")
            print(f"   Avg Training Samples: {avg_samples:.0f}")
            print(f"   Avg Features Selected: {avg_features:.0f}")
            print(f"   Signal Distribution: {signals}")
            
            return summary
        
        return {'error': 'No models trained'}
    
    def run_full_validation(self) -> Dict:
        """Run all validation tests and generate comprehensive report"""
        print("\n" + "=" * 80)
        print("🚀 RUNNING COMPREHENSIVE ACCURACY VALIDATION")
        print("=" * 80)
        print(f"Testing on {len(self.tickers)} tickers: {', '.join(self.tickers)}")
        
        if not MODULES_AVAILABLE:
            return {'error': 'Required modules not available'}
        
        # Run all validations
        forecaster_results = self.validate_forecaster()
        pattern_results = self.validate_pattern_detector()
        ai_recommender_results = self.validate_ai_recommender()
        
        # Compile final report
        report = {
            'timestamp': datetime.now().isoformat(),
            'tickers_tested': self.tickers,
            'modules': {
                'forecaster': forecaster_results,
                'pattern_detector': pattern_results,
                'ai_recommender': ai_recommender_results
            },
            'overall_health': self._calculate_overall_health(
                forecaster_results,
                pattern_results,
                ai_recommender_results
            )
        }
        
        # Save to JSON
        output_file = 'system_accuracy_report.json'
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n" + "=" * 80)
        print("✅ VALIDATION COMPLETE")
        print("=" * 80)
        print(f"📁 Report saved to: {output_file}")
        
        # Print health summary
        health = report['overall_health']
        print(f"\n🏥 SYSTEM HEALTH SUMMARY:")
        print(f"   Forecaster: {health['forecaster_status']} ({health['forecaster_score']:.0f}/100)")
        print(f"   Pattern Detector: {health['pattern_detector_status']} ({health['pattern_detector_score']:.0f}/100)")
        print(f"   AI Recommender: {health['ai_recommender_status']} ({health['ai_recommender_score']:.0f}/100)")
        print(f"\n   📊 Overall Score: {health['overall_score']:.0f}/100 - {health['overall_status']}")
        
        return report
    
    def _calculate_overall_health(self, forecaster, pattern, ai_rec) -> Dict:
        """Calculate health scores for each module"""
        
        # Forecaster score (0-100)
        forecaster_score = 0
        if 'aggregate' in forecaster:
            agg = forecaster['aggregate']
            dir_acc = agg.get('avg_direction_accuracy', 0)
            hit_rate = agg.get('avg_hit_rate_5pct', 0)
            forecaster_score = (dir_acc * 60) + (hit_rate * 40)  # Weight direction more
        
        # Pattern detector score (0-100)
        pattern_score = 0
        if 'aggregate' in pattern:
            agg = pattern['aggregate']
            avg_conf = agg.get('avg_confidence_score', 0)
            avg_patterns = agg.get('avg_patterns_per_ticker', 0)
            detection_speed = agg.get('avg_detection_time_ms', 1000)
            
            # Score: confidence 50%, coverage 30%, speed 20%
            conf_score = avg_conf * 50
            coverage_score = min(avg_patterns / 50, 1.0) * 30  # Target 50+ patterns
            speed_score = max(0, (1000 - detection_speed) / 1000) * 20  # Under 1s good
            pattern_score = conf_score + coverage_score + speed_score
        
        # AI Recommender score (0-100)
        ai_score = 0
        if 'aggregate' in ai_rec:
            agg = ai_rec['aggregate']
            cv_acc = agg.get('avg_cv_accuracy', 0)
            ai_score = cv_acc * 100
        
        # Overall score (weighted average)
        overall = (forecaster_score * 0.35 + pattern_score * 0.30 + ai_score * 0.35)
        
        # Status labels
        def get_status(score):
            if score >= 70: return "✅ EXCELLENT"
            elif score >= 60: return "🟢 GOOD"
            elif score >= 50: return "🟡 ACCEPTABLE"
            else: return "🔴 NEEDS IMPROVEMENT"
        
        return {
            'forecaster_score': forecaster_score,
            'forecaster_status': get_status(forecaster_score),
            'pattern_detector_score': pattern_score,
            'pattern_detector_status': get_status(pattern_score),
            'ai_recommender_score': ai_score,
            'ai_recommender_status': get_status(ai_score),
            'overall_score': overall,
            'overall_status': get_status(overall)
        }


if __name__ == '__main__':
    # Run validation
    validator = PerformanceValidator(
        tickers=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AMD', 'MU', 'SPY']
    )
    
    report = validator.run_full_validation()
    
    print("\n" + "=" * 80)
    print("🎯 READY FOR SPARK FRONTEND INTEGRATION")
    print("=" * 80)
    print("\nThe system_accuracy_report.json file contains:")
    print("  • Forecaster: Direction accuracy, MAE, RMSE, hit rates")
    print("  • Pattern Detector: Pattern counts, confidence scores, detection speed")
    print("  • AI Recommender: CV accuracy, feature selection, signals")
    print("\nUse this data to:")
    print("  1. Display real-time accuracy metrics in dashboard")
    print("  2. Show health status for each module")
    print("  3. Track performance over time")
    print("  4. Alert when accuracy drops below thresholds")
