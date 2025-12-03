"""
REAL_DATA_COLLECTOR.py
======================
Collects REAL historical pump events, gaps, and patterns for training

Uses your proven data_orchestrator and data_router infrastructure
Finds actual market events (not synthetic data)
Target: 500+ real pump events for training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealDataCollector:
    """
    Collect real market events for training using data_orchestrator
    """
    
    def __init__(self, start_date='2020-01-01', end_date=None):
        """Initialize with date range"""
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Import your data infrastructure
        try:
            from data_orchestrator import DataOrchestrator_v84
            from data_router import DataRouter
            self.orchestrator = DataOrchestrator_v84()
            self.router = DataRouter()
            logger.info("✅ Data infrastructure loaded")
        except Exception as e:
            logger.warning(f"⚠️ Data infrastructure not available: {e}")
            self.orchestrator = None
            self.router = None
        
        # Storage
        self.pump_events = []
        self.gap_events = []
        self.normal_events = []
    
    def collect_pump_training_data(self, universe: List[str], max_symbols: int = 100) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Collect real pump events from historical data
        
        Pump Definition:
        - 50%+ gain in 1-5 days
        - Volume spike 3x+ average
        - Price <$20 (penny stocks)
        
        Returns: X (features), y (labels)
        """
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📦 COLLECTING REAL PUMP TRAINING DATA")
        logger.info(f"{'='*80}\n")
        
        pump_features = []
        pump_labels = []
        
        symbols_processed = 0
        
        for symbol in universe[:max_symbols]:
            try:
                # Use YOUR data orchestrator to fetch data
                df = self._fetch_data(symbol)
                
                if df is None or len(df) < 30:
                    continue
                
                symbols_processed += 1
                
                # Calculate metrics
                df['returns_1d'] = df['Close'].pct_change()
                df['returns_5d'] = df['Close'].pct_change(periods=5)
                df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
                
                # Identify pump events (50%+ in 5 days, 3x volume)
                pump_mask = (df['returns_5d'] > 0.50) & (df['volume_ratio'] > 3.0)
                pump_indices = df[pump_mask].index
                
                for pump_date in pump_indices:
                    # Get features 24 hours BEFORE pump
                    feature_idx = df.index.get_loc(pump_date)
                    if feature_idx < 20:
                        continue
                    
                    feature_date = df.index[feature_idx - 1]
                    
                    # Extract proactive features
                    features = self._extract_pre_pump_features(df, feature_date)
                    
                    if features:
                        pump_features.append(features)
                        pump_labels.append(1)  # Positive example
                        
                        logger.info(f"✅ Pump: {symbol} on {pump_date.date()} (+{df.loc[pump_date, 'returns_5d']*100:.1f}%)")
                
                # Collect negative examples (no pump)
                normal_mask = ~pump_mask
                normal_indices = df[normal_mask].index[-50:]
                
                for normal_date in normal_indices:
                    feature_idx = df.index.get_loc(normal_date)
                    if feature_idx < 20:
                        continue
                    
                    features = self._extract_pre_pump_features(df, normal_date)
                    if features:
                        pump_features.append(features)
                        pump_labels.append(0)  # Negative example
            
            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue
            
            # Progress update
            if symbols_processed % 10 == 0:
                logger.info(f"Progress: {symbols_processed}/{max_symbols} symbols processed")
        
        # Convert to arrays
        X = pd.DataFrame(pump_features)
        y = np.array(pump_labels)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 DATA COLLECTION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"Symbols processed: {symbols_processed}")
        logger.info(f"Positive examples (pumps): {sum(y)}")
        logger.info(f"Negative examples (normal): {len(y) - sum(y)}")
        logger.info(f"Total examples: {len(y)}")
        logger.info(f"Class ratio: {sum(y)/len(y)*100:.1f}% positive\n")
        
        return X, y
    
    def _fetch_data(self, symbol: str) -> pd.DataFrame:
        """Fetch data using YOUR data orchestrator"""
        try:
            if self.orchestrator:
                # Use your proven data orchestrator
                df = self.orchestrator.get_data(
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    interval='1d'
                )
                return df
            else:
                # Fallback to yfinance
                import yfinance as yf
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)
                return df
        except Exception as e:
            logger.debug(f"Failed to fetch {symbol}: {e}")
            return None
    
    def _extract_pre_pump_features(self, df: pd.DataFrame, date) -> Dict:
        """
        Extract PROACTIVE features from 24 hours BEFORE event
        
        Features (velocity/acceleration - not absolute values):
        - Volume velocity (rate of change)
        - Volume acceleration (2nd derivative)
        - Price stability (accumulation phase)
        - Momentum indicators
        """
        
        # Get last 20 days before date
        feature_idx = df.index.get_loc(date)
        if feature_idx < 20:
            return {}
        
        lookback_data = df.iloc[feature_idx-20:feature_idx+1]
        
        if len(lookback_data) < 20:
            return {}
        
        features = {}
        
        try:
            # Volume features (PROACTIVE - velocity & acceleration)
            volume = lookback_data['Volume'].values
            volume_velocity = np.diff(volume) / (volume[:-1] + 1)
            volume_acceleration = np.diff(volume_velocity)
            
            features['volume_velocity_mean'] = float(np.mean(volume_velocity[-5:]))
            features['volume_velocity_std'] = float(np.std(volume_velocity[-5:]))
            features['volume_acceleration_mean'] = float(np.mean(volume_acceleration[-4:]))
            features['volume_ratio'] = float(volume[-1] / (np.mean(volume[-20:]) + 1))
            
            # Price stability (accumulation = low volatility)
            returns = lookback_data['Close'].pct_change()
            features['price_volatility'] = float(returns.std())
            features['price_stability'] = float(1 / (returns.std() + 0.001))
            
            # Trend features
            sma_20 = lookback_data['Close'].rolling(20).mean().iloc[-1]
            features['sma_20'] = float(sma_20)
            features['price_vs_sma'] = float(lookback_data['Close'].iloc[-1] / sma_20)
            
            # Range features
            features['high_low_ratio'] = float(lookback_data['High'].iloc[-1] / lookback_data['Low'].iloc[-1])
            
            # Momentum
            features['rsi'] = self._calculate_rsi(lookback_data['Close'])
            features['macd'] = self._calculate_macd(lookback_data['Close'])
            
            return features
            
        except Exception as e:
            logger.debug(f"Feature extraction error: {e}")
            return {}
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
            rs = gain / (loss + 0.00001)
            rsi = 100 - (100 / (1 + rs))
            return float(rsi.iloc[-1]) if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _calculate_macd(self, prices):
        """Calculate MACD"""
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            return float(macd.iloc[-1]) if not macd.empty else 0.0
        except:
            return 0.0
    
    def get_penny_stock_universe(self) -> List[str]:
        """Get list of penny stocks prone to pumps"""
        return [
            'SNDL', 'BNGO', 'OCGN', 'GNUS', 'IDEX',
            'NAKD', 'JAGX', 'XSPA', 'TSNP', 'ALPP',
            'ZOM', 'SNDL', 'AEZS', 'ONTX', 'SENS',
            'ATOS', 'TLRY', 'CTRM', 'SOS', 'EBON',
            'MARA', 'RIOT', 'PLUG', 'FCEL', 'BLNK',
            'NIO', 'XPEV', 'LI', 'FSR', 'CCIV',
            'BB', 'AMC', 'GME', 'WKHS', 'RIDE',
            'SPCE', 'PLTR', 'WISH', 'CLOV', 'SOFI'
        ]
    
    def save_training_data(self, X: pd.DataFrame, y: np.ndarray, output_dir: str = 'training_data'):
        """Save collected training data"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        X.to_csv(f'{output_dir}/pump_features.csv', index=False)
        np.save(f'{output_dir}/pump_labels.npy', y)
        
        logger.info(f"✅ Training data saved to {output_dir}/")


if __name__ == '__main__':
    # Example usage
    collector = RealDataCollector(start_date='2020-01-01')
    
    # Get penny stock universe
    universe = collector.get_penny_stock_universe()
    
    # Collect real pump data
    logger.info("Starting data collection...")
    X, y = collector.collect_pump_training_data(universe, max_symbols=50)
    
    # Save
    if len(X) > 0:
        collector.save_training_data(X, y)
        logger.info(f"\n✅ Collection complete! Ready to train with {len(X)} examples")
    else:
        logger.warning("❌ No data collected")

