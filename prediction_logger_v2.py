# ============================================================================
# PREDICTION LOGGER - Track and Validate Forecaster Predictions
# ============================================================================
"""
Track all predictions with:
- Timestamp and ticker
- 7/14/21-day targets
- Confidence and action
- Update with actuals when available
- Calculate rolling accuracy metrics
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

class PredictionLogger:
    """
    Log predictions and track accuracy over time
    
    Usage:
        logger = PredictionLogger()
        logger.log_prediction('AAPL', 'BUY', 0.78, 175.50, 180.00, 168.00, 186.00)
        logger.update_actuals()  # Run daily to update with real prices
        stats = logger.get_accuracy_stats()
    """
    
    def __init__(self, log_file: str = 'prediction_log.json'):
        self.log_file = log_file
        self.predictions = self._load_log()
    
    def _load_log(self) -> List[Dict]:
        """Load existing predictions from file"""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Error loading log: {e}")
                return []
        return []
    
    def _save_log(self):
        """Save predictions to file"""
        with open(self.log_file, 'w') as f:
            json.dump(self.predictions, f, indent=2, default=str)
    
    def log_prediction(
        self,
        ticker: str,
        action: str,
        confidence: float,
        entry_price: float,
        target_7d: float,
        lower_band: float,
        upper_band: float,
        regime: str = 'UNKNOWN',
        reasoning: str = ''
    ) -> str:
        """
        Log a new prediction
        
        Args:
            ticker: Stock symbol
            action: BUY, SELL, HOLD, ABSTAIN
            confidence: 0.0-1.0
            entry_price: Current price at prediction time
            target_7d: 7-day price target
            lower_band: Lower confidence band
            upper_band: Upper confidence band
            regime: Market regime
            reasoning: Why this prediction was made
        
        Returns:
            Prediction ID
        """
        pred_id = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        prediction = {
            'id': pred_id,
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'confidence': confidence,
            'regime': regime,
            'reasoning': reasoning,
            
            # Prices
            'entry_price': entry_price,
            'target_7d': target_7d,
            'target_14d': None,  # Can be filled in later
            'target_21d': None,
            
            # Confidence bands
            'lower_band': lower_band,
            'upper_band': upper_band,
            
            # Actual results (filled in later)
            'actual_7d': None,
            'actual_14d': None,
            'actual_21d': None,
            'actual_7d_return': None,
            'actual_14d_return': None,
            'actual_21d_return': None,
            
            # Validation results
            'correct_7d': None,
            'correct_14d': None,
            'correct_21d': None,
            'within_bands_7d': None,
            
            # Status
            'status': 'OPEN',  # OPEN, PARTIALLY_VALIDATED, VALIDATED
        }
        
        self.predictions.append(prediction)
        self._save_log()
        
        print(f"✅ Logged prediction: {pred_id}")
        print(f"   Action: {action} ({confidence*100:.1f}% confidence)")
        print(f"   Entry: ${entry_price:.2f} → Target: ${target_7d:.2f}")
        print(f"   Range: ${lower_band:.2f} - ${upper_band:.2f}")
        
        return pred_id
    
    def update_actuals(self) -> Dict:
        """
        Update predictions with actual prices
        
        Returns:
            Summary of updates made
        """
        try:
            import yfinance as yf
        except ImportError:
            print("⚠️ yfinance not available")
            return {}
        
        updated = {'7d': 0, '14d': 0, '21d': 0}
        now = datetime.now()
        
        for pred in self.predictions:
            if pred['status'] == 'VALIDATED':
                continue
            
            pred_time = datetime.fromisoformat(pred['timestamp'])
            days_elapsed = (now - pred_time).days
            ticker = pred['ticker']
            
            try:
                df = yf.download(ticker, period='60d', progress=False)
                if len(df) == 0:
                    continue
                
                # Find actual prices at target dates
                for days, key in [(7, '7d'), (14, '14d'), (21, '21d')]:
                    if days_elapsed >= days and pred[f'actual_{key}'] is None:
                        target_date = pred_time + timedelta(days=days)
                        
                        # Find closest trading day
                        mask = df.index >= target_date
                        if mask.any():
                            actual_price = float(df.loc[mask, 'Close'].iloc[0])
                            pred[f'actual_{key}'] = actual_price
                            
                            # Calculate return
                            pred[f'actual_{key}_return'] = (actual_price - pred['entry_price']) / pred['entry_price']
                            
                            # Check if correct
                            if pred['action'] == 'BUY':
                                pred[f'correct_{key}'] = actual_price > pred['entry_price'] * 1.01
                            elif pred['action'] == 'SELL':
                                pred[f'correct_{key}'] = actual_price < pred['entry_price'] * 0.99
                            else:
                                pred[f'correct_{key}'] = abs(actual_price - pred['entry_price']) / pred['entry_price'] < 0.03
                            
                            updated[key] += 1
                
                # Check if 7d within bands
                if pred['actual_7d'] is not None and pred['within_bands_7d'] is None:
                    pred['within_bands_7d'] = pred['lower_band'] <= pred['actual_7d'] <= pred['upper_band']
                
                # Update status
                if all(pred[f'actual_{k}'] is not None for k in ['7d', '14d', '21d']):
                    pred['status'] = 'VALIDATED'
                elif pred['actual_7d'] is not None:
                    pred['status'] = 'PARTIALLY_VALIDATED'
                
            except Exception as e:
                print(f"⚠️ Error updating {ticker}: {e}")
        
        self._save_log()
        print(f"✅ Updated actuals: 7d={updated['7d']}, 14d={updated['14d']}, 21d={updated['21d']}")
        return updated
    
    def get_accuracy_stats(self) -> Dict:
        """
        Calculate accuracy statistics
        
        Returns:
            {
                'total_predictions': int,
                'validated': int,
                'accuracy_7d': float,
                'accuracy_14d': float,
                'accuracy_21d': float,
                'within_bands_rate': float,
                'by_action': {action: {acc, count}},
                'by_confidence': {range: {acc, count}}
            }
        """
        stats = {
            'total_predictions': len(self.predictions),
            'validated': 0,
            'accuracy_7d': None,
            'accuracy_14d': None,
            'accuracy_21d': None,
            'within_bands_rate': None,
            'by_action': {},
            'by_confidence': {}
        }
        
        # Count validated
        correct_7d, total_7d = 0, 0
        correct_14d, total_14d = 0, 0
        correct_21d, total_21d = 0, 0
        within_bands, total_bands = 0, 0
        
        action_stats = {}
        confidence_buckets = {
            '50-60%': {'correct': 0, 'total': 0},
            '60-70%': {'correct': 0, 'total': 0},
            '70-80%': {'correct': 0, 'total': 0},
            '80-90%': {'correct': 0, 'total': 0},
            '90-100%': {'correct': 0, 'total': 0},
        }
        
        for pred in self.predictions:
            action = pred['action']
            conf = pred['confidence']
            
            # Track by action
            if action not in action_stats:
                action_stats[action] = {'correct': 0, 'total': 0}
            
            # 7-day accuracy
            if pred['correct_7d'] is not None:
                total_7d += 1
                if pred['correct_7d']:
                    correct_7d += 1
                    action_stats[action]['correct'] += 1
                action_stats[action]['total'] += 1
                
                # Track by confidence
                if 0.5 <= conf < 0.6:
                    bucket = '50-60%'
                elif 0.6 <= conf < 0.7:
                    bucket = '60-70%'
                elif 0.7 <= conf < 0.8:
                    bucket = '70-80%'
                elif 0.8 <= conf < 0.9:
                    bucket = '80-90%'
                else:
                    bucket = '90-100%'
                
                confidence_buckets[bucket]['total'] += 1
                if pred['correct_7d']:
                    confidence_buckets[bucket]['correct'] += 1
            
            # 14-day accuracy
            if pred['correct_14d'] is not None:
                total_14d += 1
                if pred['correct_14d']:
                    correct_14d += 1
            
            # 21-day accuracy
            if pred['correct_21d'] is not None:
                total_21d += 1
                if pred['correct_21d']:
                    correct_21d += 1
            
            # Within bands
            if pred['within_bands_7d'] is not None:
                total_bands += 1
                if pred['within_bands_7d']:
                    within_bands += 1
            
            if pred['status'] == 'VALIDATED':
                stats['validated'] += 1
        
        # Calculate rates
        stats['accuracy_7d'] = correct_7d / total_7d if total_7d > 0 else None
        stats['accuracy_14d'] = correct_14d / total_14d if total_14d > 0 else None
        stats['accuracy_21d'] = correct_21d / total_21d if total_21d > 0 else None
        stats['within_bands_rate'] = within_bands / total_bands if total_bands > 0 else None
        
        # Action stats
        for action, data in action_stats.items():
            stats['by_action'][action] = {
                'accuracy': data['correct'] / data['total'] if data['total'] > 0 else None,
                'count': data['total']
            }
        
        # Confidence bucket stats
        for bucket, data in confidence_buckets.items():
            stats['by_confidence'][bucket] = {
                'accuracy': data['correct'] / data['total'] if data['total'] > 0 else None,
                'count': data['total']
            }
        
        return stats
    
    def get_recent_predictions(self, days: int = 7) -> List[Dict]:
        """Get predictions from last N days"""
        cutoff = datetime.now() - timedelta(days=days)
        recent = []
        
        for pred in self.predictions:
            pred_time = datetime.fromisoformat(pred['timestamp'])
            if pred_time >= cutoff:
                recent.append(pred)
        
        return sorted(recent, key=lambda x: x['timestamp'], reverse=True)
    
    def print_summary(self):
        """Print formatted summary of prediction performance"""
        stats = self.get_accuracy_stats()
        
        print("\n" + "="*60)
        print("📊 PREDICTION PERFORMANCE SUMMARY")
        print("="*60)
        
        print(f"\nTotal Predictions: {stats['total_predictions']}")
        print(f"Fully Validated:   {stats['validated']}")
        
        print(f"\n📈 ACCURACY BY HORIZON:")
        if stats['accuracy_7d'] is not None:
            print(f"   7-day:  {stats['accuracy_7d']*100:.1f}%")
        if stats['accuracy_14d'] is not None:
            print(f"   14-day: {stats['accuracy_14d']*100:.1f}%")
        if stats['accuracy_21d'] is not None:
            print(f"   21-day: {stats['accuracy_21d']*100:.1f}%")
        
        if stats['within_bands_rate'] is not None:
            print(f"\n🎯 Within Confidence Bands: {stats['within_bands_rate']*100:.1f}%")
        
        print(f"\n📊 ACCURACY BY ACTION:")
        for action, data in stats['by_action'].items():
            if data['count'] > 0:
                acc = data['accuracy']*100 if data['accuracy'] else 0
                print(f"   {action:6s}: {acc:.1f}% ({data['count']} predictions)")
        
        print(f"\n🎚️ ACCURACY BY CONFIDENCE:")
        for bucket, data in stats['by_confidence'].items():
            if data['count'] > 0:
                acc = data['accuracy']*100 if data['accuracy'] else 0
                print(f"   {bucket:8s}: {acc:.1f}% ({data['count']} predictions)")
        
        print("="*60)


# ============================================================================
# QUICK TEST
# ============================================================================

if __name__ == '__main__':
    logger = PredictionLogger('test_predictions.json')
    
    # Log a test prediction
    logger.log_prediction(
        ticker='AAPL',
        action='BUY',
        confidence=0.78,
        entry_price=175.50,
        target_7d=182.00,
        lower_band=170.00,
        upper_band=188.00,
        regime='BULL',
        reasoning='Strong trend, high alignment'
    )
    
    # Print stats
    logger.print_summary()
    
    print("\n✅ Prediction logger test complete!")
