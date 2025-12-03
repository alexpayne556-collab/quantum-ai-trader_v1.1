"""
🔧 DATA ORCHESTRATOR - SCALAR UTILITIES
========================================
Add these methods to your DataOrchestrator to ensure all modules 
receive properly formatted scalar values (not Series).

This prevents "The truth value of a Series is ambiguous" errors.
"""

import pandas as pd
import numpy as np
from typing import Any, Union

class ScalarExtractor:
    """
    Utility class to extract scalar values from pandas Series/arrays.
    
    Add this to your DataOrchestrator to ensure clean data for all modules.
    """
    
    @staticmethod
    def to_scalar(value: Any) -> Union[float, int, str]:
        """
        Convert any value (Series, array, scalar) to a Python scalar.
        
        Args:
            value: Value to convert (could be Series, array, or already scalar)
            
        Returns:
            Python scalar (float, int, or str)
        """
        # Already a scalar
        if isinstance(value, (int, float, str, bool)):
            return value
        
        # Pandas Series - extract first value
        if isinstance(value, pd.Series):
            if len(value) > 0:
                return ScalarExtractor.to_scalar(value.iloc[0])
            return 0.0
        
        # Numpy array or generic - extract item
        if isinstance(value, (np.ndarray, np.generic)):
            if hasattr(value, 'item'):
                return value.item()
            elif hasattr(value, '__len__') and len(value) > 0:
                return ScalarExtractor.to_scalar(value[0])
            return 0.0
        
        # Fallback - try to convert to float
        try:
            return float(value)
        except:
            return 0.0
    
    @staticmethod
    def ensure_scalar_dict(data_dict: dict) -> dict:
        """
        Ensure all values in a dictionary are scalars.
        
        Useful for module outputs that might contain Series.
        """
        clean_dict = {}
        for key, value in data_dict.items():
            if isinstance(value, dict):
                # Recursively clean nested dicts
                clean_dict[key] = ScalarExtractor.ensure_scalar_dict(value)
            elif isinstance(value, list):
                # Clean lists
                clean_dict[key] = [ScalarExtractor.to_scalar(v) for v in value]
            else:
                # Clean single values
                clean_dict[key] = ScalarExtractor.to_scalar(value)
        
        return clean_dict
    
    @staticmethod
    def get_dataframe_scalar(df: pd.DataFrame, column: str, index: int = -1) -> float:
        """
        Safely extract a scalar value from a DataFrame.
        
        Args:
            df: DataFrame
            column: Column name
            index: Row index (-1 = last row)
            
        Returns:
            Scalar value
        """
        try:
            value = df[column].iloc[index]
            return ScalarExtractor.to_scalar(value)
        except:
            return 0.0
    
    @staticmethod
    def get_last_price(df: pd.DataFrame, column: str = 'close') -> float:
        """
        Get last price from dataframe (common operation).
        
        Args:
            df: OHLCV dataframe
            column: Price column ('close', 'open', etc.)
            
        Returns:
            Last price as scalar
        """
        # Try lowercase
        col = column.lower()
        if col in df.columns:
            return ScalarExtractor.get_dataframe_scalar(df, col, -1)
        
        # Try uppercase
        col = column.upper()
        if col in df.columns:
            return ScalarExtractor.get_dataframe_scalar(df, col, -1)
        
        # Try capitalize
        col = column.capitalize()
        if col in df.columns:
            return ScalarExtractor.get_dataframe_scalar(df, col, -1)
        
        return 0.0

# ============================================================================
# ADD THESE METHODS TO YOUR DataOrchestrator CLASS
# ============================================================================

def add_scalar_utils_to_orchestrator():
    """
    Example of how to add these utilities to your DataOrchestrator.
    
    Add these methods to your DataOrchestrator class:
    """
    
    methods = '''
    # Add to DataOrchestrator class:
    
    def to_scalar(self, value):
        """Convert any value to scalar"""
        return ScalarExtractor.to_scalar(value)
    
    def clean_module_output(self, output: dict) -> dict:
        """Clean module output to ensure all values are scalars"""
        return ScalarExtractor.ensure_scalar_dict(output)
    
    def get_price(self, df: pd.DataFrame, column: str = 'close', index: int = -1) -> float:
        """Get price as scalar from dataframe"""
        return ScalarExtractor.get_dataframe_scalar(df, column, index)
    
    def get_last_close(self, df: pd.DataFrame) -> float:
        """Get last close price"""
        return ScalarExtractor.get_last_price(df, 'close')
    
    def get_volume_ratio(self, df: pd.DataFrame, period: int = 20) -> float:
        """Get volume ratio (current / average) as scalar"""
        if len(df) < period:
            return 1.0
        
        current_vol = self.get_price(df, 'volume', -1)
        avg_vol = df['volume'].iloc[-period:-1].mean()
        avg_vol = self.to_scalar(avg_vol)
        
        if avg_vol > 0:
            return float(current_vol / avg_vol)
        return 1.0
    '''
    
    return methods

# ============================================================================
# EXAMPLE USAGE IN MODULES
# ============================================================================

def example_module_usage():
    """
    Example of how modules should use the DataOrchestrator with scalar utilities.
    """
    
    example = '''
    # In your module (e.g., dark_pool_tracker.py):
    
    from data_orchestrator import DataOrchestrator
    
    orchestrator = DataOrchestrator()
    
    def analyze_ticker(symbol: str, data: pd.DataFrame) -> dict:
        """Analyze ticker with clean scalar values"""
        
        # Get last close price (scalar)
        current_price = orchestrator.get_last_close(data)
        
        # Get volume ratio (scalar)
        volume_ratio = orchestrator.get_volume_ratio(data, period=20)
        
        # Calculate signal
        if volume_ratio > 1.5:  # This will work! Both are scalars
            return {
                'signal': 'BUY',
                'confidence': min(0.7 + (volume_ratio - 1.5) * 0.1, 0.9),
                'volume_ratio': volume_ratio,  # Already scalar
                'current_price': current_price  # Already scalar
            }
        
        return {'signal': 'NEUTRAL', 'confidence': 0.5}
    '''
    
    return example

# ============================================================================
# INTEGRATION WITH BACKTEST
# ============================================================================

def integrate_with_backtest():
    """
    How to integrate scalar utilities with your backtest.
    """
    
    integration = '''
    # In BACKTEST_INSTITUTIONAL_ENSEMBLE.py:
    
    from data_orchestrator import DataOrchestrator
    
    class BacktestEngine:
        def __init__(self):
            # ... existing code ...
            self.orchestrator = DataOrchestrator()
        
        def _generate_signals(self, date, data_dict, capital, positions):
            """Generate signals with clean data"""
            
            for symbol, data in data_dict.items():
                # Get regime with clean scalars
                regime_result = self.regime.detect_regime(data)
                
                # Ensure clean output
                regime_result = self.orchestrator.clean_module_output(regime_result)
                
                # Now safe to use
                if regime_result['regime'] == 'bull':  # Works!
                    # ... continue processing ...
                    pass
    '''
    
    return integration

if __name__ == '__main__':
    print("🔧 DATA ORCHESTRATOR SCALAR UTILITIES")
    print("=" * 80)
    print("\nAdd these methods to your DataOrchestrator:")
    print(add_scalar_utils_to_orchestrator())
    print("\n" + "=" * 80)
    print("\nExample module usage:")
    print(example_module_usage())
    print("\n" + "=" * 80)
    print("\nIntegration with backtest:")
    print(integrate_with_backtest())

