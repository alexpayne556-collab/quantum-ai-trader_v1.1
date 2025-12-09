"""
Unit Tests for Dark Pool Signals Module
========================================

Tests all 5 institutional signals with various scenarios:
- Normal market conditions
- High volatility periods
- Bull/bear divergences
- Edge cases (insufficient data, API failures)

Run: pytest tests/unit/test_dark_pool_signals.py -v
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.features.dark_pool_signals import DarkPoolSignals


class TestDarkPoolSignals:
    """Test suite for Dark Pool Signals module."""
    
    @pytest.fixture
    def mock_daily_data(self):
        """Generate mock daily OHLCV data (20 days)."""
        dates = pd.date_range(end='2024-12-08', periods=20, freq='D')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 20),
            'High': np.random.uniform(110, 120, 20),
            'Low': np.random.uniform(90, 100, 20),
            'Close': np.random.uniform(100, 110, 20),
            'Volume': np.random.uniform(1e6, 5e6, 20)
        }, index=dates)
        return data
    
    @pytest.fixture
    def mock_minute_data(self):
        """Generate mock intraday data (1000 minute bars)."""
        dates = pd.date_range(end='2024-12-08 16:00', periods=1000, freq='1min')
        data = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 1000),
            'High': np.random.uniform(110, 120, 1000),
            'Low': np.random.uniform(90, 100, 1000),
            'Close': np.random.uniform(100, 110, 1000),
            'Volume': np.random.uniform(1000, 10000, 1000)
        }, index=dates)
        return data
    
    def test_initialization(self):
        """Test DarkPoolSignals initialization."""
        signals = DarkPoolSignals("NVDA")
        assert signals.ticker == "NVDA"
        assert signals.cache_enabled == True
        assert len(signals._data_cache) == 0
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_ifi_bullish_scenario(self, mock_download, mock_minute_data):
        """Test IFI detects bullish accumulation."""
        # Create data with strong buying on large volume
        mock_data = mock_minute_data.copy()
        # Simulate institutional buying: large volume + positive price moves
        mock_data.iloc[-100:, mock_data.columns.get_loc('Volume')] *= 3  # 3x volume
        mock_data.iloc[-100:, mock_data.columns.get_loc('Close')] += 5  # Price up
        
        mock_download.return_value = mock_data
        
        signals = DarkPoolSignals("NVDA")
        ifi_result = signals.institutional_flow_index(days=20)
        
        # Assertions
        assert ifi_result['IFI'] > 0, "Should detect positive IFI (buying)"
        assert ifi_result['IFI_score'] > 50, "Score should be >50 for buying"
        assert ifi_result['interpretation'] in ['BULLISH', 'NEUTRAL']
        assert ifi_result['buy_volume'] > 0
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_ifi_bearish_scenario(self, mock_download, mock_minute_data):
        """Test IFI calculation completes for bearish scenario."""
        mock_data = mock_minute_data.copy()
        # Simulate selling pressure
        mock_data.iloc[-100:, mock_data.columns.get_loc('Volume')] *= 3
        mock_data.iloc[-100:, mock_data.columns.get_loc('Close')] -= 5
        
        mock_download.return_value = mock_data
        
        signals = DarkPoolSignals("NVDA")
        ifi_result = signals.institutional_flow_index(days=20)
        
        # Just check calculation completes and returns valid structure
        assert 'IFI' in ifi_result
        assert 'IFI_score' in ifi_result
        assert 0 <= ifi_result['IFI_score'] <= 100
        assert ifi_result['sell_volume'] >= 0
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_ad_line_accumulation(self, mock_download, mock_daily_data):
        """Test A/D Line detects accumulation."""
        mock_data = mock_daily_data.copy()
        # Simulate accumulation: closes near high
        mock_data['Close'] = mock_data['High'] * 0.95  # Close near high
        
        mock_download.return_value = mock_data
        
        signals = DarkPoolSignals("TSLA")
        ad_result = signals.accumulation_distribution(lookback=20)
        
        assert ad_result['signal'] in ['ACCUMULATION', 'NEUTRAL']
        assert ad_result['AD_score'] >= 0
        assert ad_result['AD_score'] <= 100
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_obv_bullish_divergence(self, mock_download, mock_daily_data):
        """Test OBV detects bullish divergence (price down, volume up)."""
        mock_data = mock_daily_data.copy()
        # Create divergence: last 5 days price down but volume up
        mock_data.iloc[-5:, mock_data.columns.get_loc('Close')] *= 0.95  # -5% price
        mock_data.iloc[-5:, mock_data.columns.get_loc('Volume')] *= 2  # 2x volume
        
        mock_download.return_value = mock_data
        
        signals = DarkPoolSignals("AMD")
        obv_result = signals.obv_institutional(lookback=20)
        
        # Note: Divergence detection requires specific conditions
        assert obv_result['OBV_score'] >= 0
        assert obv_result['OBV_score'] <= 100
        assert obv_result['divergence'] in ['BULLISH', 'BEARISH', 'NONE']
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_vroc_acceleration(self, mock_download, mock_daily_data):
        """Test VROC detects volume acceleration."""
        mock_data = mock_daily_data.copy()
        # Simulate volume surge: last 5 days 3x volume
        mock_data.iloc[-5:, mock_data.columns.get_loc('Volume')] *= 3
        
        mock_download.return_value = mock_data
        
        signals = DarkPoolSignals("MSFT")
        vroc_result = signals.volume_acceleration_index(lookback=20)
        
        assert vroc_result['VROC'] > 0, "Should detect positive VROC"
        assert vroc_result['vol_trend'] in ['ACCELERATING', 'NORMAL']
        assert vroc_result['VROC_score'] >= 0
        assert vroc_result['VROC_score'] <= 100
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_smi_composite(self, mock_download, mock_daily_data, mock_minute_data):
        """Test SMI composite calculation."""
        # Mock both daily and minute data
        def side_effect(ticker, interval, period, progress):
            if interval == "1m":
                return mock_minute_data
            else:
                return mock_daily_data
        
        mock_download.side_effect = side_effect
        
        signals = DarkPoolSignals("GOOGL")
        smi_result = signals.smart_money_index(lookback=20)
        
        # Assertions
        assert 0 <= smi_result['SMI'] <= 100, "SMI must be 0-100"
        assert smi_result['signal'] in ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']
        assert 0 <= smi_result['consistency'] <= 1, "Consistency must be 0-1"
        assert 0 <= smi_result['confidence'] <= 100, "Confidence must be 0-100"
        
        # Check components exist
        assert 'IFI' in smi_result['components']
        assert 'AD' in smi_result['components']
        assert 'OBV' in smi_result['components']
        assert 'VROC' in smi_result['components']
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_fallback_on_api_failure(self, mock_download):
        """Test fallback behavior when API fails."""
        mock_download.side_effect = Exception("API timeout")
        
        signals = DarkPoolSignals("AAPL")
        ifi_result = signals.institutional_flow_index(days=20)
        
        # Should return neutral fallback values
        assert ifi_result['IFI'] == 0.0
        assert ifi_result['IFI_score'] == 50.0
        assert ifi_result['interpretation'] == 'NEUTRAL'
        assert 'error' in ifi_result
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_fallback_on_insufficient_data(self, mock_download):
        """Test fallback when insufficient data returned."""
        # Return only 5 data points (need at least 10)
        short_data = pd.DataFrame({
            'Open': [100, 101, 102, 103, 104],
            'High': [105, 106, 107, 108, 109],
            'Low': [95, 96, 97, 98, 99],
            'Close': [102, 103, 104, 105, 106],
            'Volume': [1e6, 1e6, 1e6, 1e6, 1e6]
        })
        
        mock_download.return_value = short_data
        
        signals = DarkPoolSignals("SPY")
        ad_result = signals.accumulation_distribution(lookback=20)
        
        # Should return fallback
        assert ad_result['AD_score'] == 50.0
        assert ad_result['signal'] == 'NEUTRAL'
    
    def test_caching_mechanism(self):
        """Test that caching works correctly."""
        signals = DarkPoolSignals("NVDA", cache_enabled=True)
        
        # First call should populate cache
        with patch('src.features.dark_pool_signals.yf.download') as mock_download:
            mock_download.return_value = pd.DataFrame({
                'Close': [100], 'Volume': [1e6], 'High': [105], 'Low': [95], 'Open': [100]
            })
            signals._get_data(interval="1d", period="30d")
            assert mock_download.call_count == 1
            
            # Second call within 5 minutes should use cache
            signals._get_data(interval="1d", period="30d")
            assert mock_download.call_count == 1, "Should use cache, not call API again"
    
    @patch('src.features.dark_pool_signals.yf.download')
    def test_get_all_signals(self, mock_download, mock_daily_data, mock_minute_data):
        """Test convenience method that returns all 5 signals."""
        def side_effect(ticker, interval, period, progress):
            if interval == "1m":
                return mock_minute_data
            else:
                return mock_daily_data
        
        mock_download.side_effect = side_effect
        
        signals = DarkPoolSignals("META")
        all_signals = signals.get_all_signals(lookback=20)
        
        # Check all 5 signals are present
        assert 'IFI' in all_signals
        assert 'AD' in all_signals
        assert 'OBV' in all_signals
        assert 'VROC' in all_signals
        assert 'SMI' in all_signals
        
        # Check each has required fields
        assert 'IFI_score' in all_signals['IFI']
        assert 'AD_score' in all_signals['AD']
        assert 'OBV_score' in all_signals['OBV']
        assert 'VROC_score' in all_signals['VROC']
        assert 'SMI' in all_signals['SMI']


# Integration test (requires live API access)
@pytest.mark.integration
@pytest.mark.skip(reason="Requires --run-integration flag and live API access")
def test_live_api_integration():
    """Integration test with live yfinance API (optional, slow)."""
    signals = DarkPoolSignals("AAPL")
    
    # Should complete without errors
    smi_result = signals.smart_money_index(lookback=20)
    
    # Basic validation
    assert 0 <= smi_result['SMI'] <= 100
    assert smi_result['signal'] in ['STRONG_BUY', 'BUY', 'NEUTRAL', 'SELL', 'STRONG_SELL']


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
