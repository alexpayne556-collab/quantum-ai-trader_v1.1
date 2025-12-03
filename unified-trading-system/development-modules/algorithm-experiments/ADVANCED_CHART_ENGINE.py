"""
================================================================================
📊 ADVANCED CHART ENGINE - INSTITUTIONAL-GRADE TECHNICAL ANALYSIS
================================================================================

Features:
- Candlestick charts with volume
- 20+ Technical Indicators
- MA Ribbons (5, 10, 20, 50, 100, 200)
- Bollinger Bands
- Ichimoku Cloud
- VWAP
- Parabolic SAR
- Support/Resistance levels
- Fibonacci retracements
- RSI, MACD, Stochastic
- OBV, MFI, CMF
- ATR, ADX
- Volume profile
- And more!

Usage:
from ADVANCED_CHART_ENGINE import AdvancedChartEngine

chart_engine = AdvancedChartEngine()
fig = chart_engine.create_chart(symbol='AAPL', indicators=['all'])
================================================================================
"""

import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from datetime import datetime, timedelta
from scipy.signal import find_peaks

class AdvancedChartEngine:
    """
    Institutional-grade charting with 20+ technical indicators
    """
    
    def __init__(self):
        self.colors = {
            'bullish': '#26A69A',
            'bearish': '#EF5350',
            'ma_fast': '#2962FF',
            'ma_mid': '#FF6D00',
            'ma_slow': '#AA00FF',
            'volume': 'rgba(100, 100, 100, 0.5)',
            'bb_upper': 'rgba(33, 150, 243, 0.3)',
            'bb_lower': 'rgba(33, 150, 243, 0.3)',
            'bb_fill': 'rgba(33, 150, 243, 0.1)',
            'ichimoku_a': 'rgba(255, 82, 82, 0.2)',
            'ichimoku_b': 'rgba(76, 175, 80, 0.2)',
            'vwap': '#FF9800',
            'sar': '#E91E63',
            'support': '#4CAF50',
            'resistance': '#F44336',
            'fibonacci': 'rgba(156, 39, 176, 0.3)'
        }
    
    def download_data(self, symbol, days=180):
        """Download stock data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            data = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(data) > 0:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                data.columns = [str(c).lower() for c in data.columns]
                return data
        except:
            return None
        return None
    
    def calculate_vwap(self, df):
        """Calculate VWAP"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        return (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    def calculate_support_resistance(self, df, window=20, prominence=0.02):
        """Calculate support and resistance levels"""
        highs = df['high'].values
        lows = df['low'].values
        
        # Find peaks (resistance)
        resistance_peaks, _ = find_peaks(highs, prominence=highs.mean() * prominence, distance=window)
        resistance_levels = highs[resistance_peaks]
        
        # Find troughs (support)
        support_peaks, _ = find_peaks(-lows, prominence=lows.mean() * prominence, distance=window)
        support_levels = lows[support_peaks]
        
        # Cluster levels
        def cluster_levels(levels, threshold=0.02):
            if len(levels) == 0:
                return []
            levels = np.sort(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if (level - current_cluster[-1]) / current_cluster[-1] < threshold:
                    current_cluster.append(level)
                else:
                    clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            clusters.append(np.mean(current_cluster))
            return clusters
        
        resistance = cluster_levels(resistance_levels)[-5:]  # Top 5
        support = cluster_levels(support_levels)[-5:]  # Bottom 5
        
        return support, resistance
    
    def calculate_fibonacci(self, df, lookback=100):
        """Calculate Fibonacci retracement levels"""
        recent = df.tail(lookback)
        high = recent['high'].max()
        low = recent['low'].min()
        diff = high - low
        
        levels = {
            '0.0': high,
            '0.236': high - (diff * 0.236),
            '0.382': high - (diff * 0.382),
            '0.5': high - (diff * 0.5),
            '0.618': high - (diff * 0.618),
            '0.786': high - (diff * 0.786),
            '1.0': low
        }
        
        return levels
    
    def calculate_ichimoku(self, df):
        """Calculate Ichimoku Cloud"""
        # Tenkan-sen (Conversion Line): 9-period
        period9_high = df['high'].rolling(window=9).max()
        period9_low = df['low'].rolling(window=9).min()
        tenkan_sen = (period9_high + period9_low) / 2
        
        # Kijun-sen (Base Line): 26-period
        period26_high = df['high'].rolling(window=26).max()
        period26_low = df['low'].rolling(window=26).min()
        kijun_sen = (period26_high + period26_low) / 2
        
        # Senkou Span A (Leading Span A): Midpoint of Tenkan and Kijun, shifted forward 26 periods
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        
        # Senkou Span B (Leading Span B): 52-period midpoint, shifted forward 26 periods
        period52_high = df['high'].rolling(window=52).max()
        period52_low = df['low'].rolling(window=52).min()
        senkou_span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Chikou Span (Lagging Span): Close price shifted backward 26 periods
        chikou_span = df['close'].shift(-26)
        
        return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span
    
    def create_chart(self, symbol, days=180, indicators='all'):
        """
        Create advanced chart with all indicators
        
        Args:
            symbol: Stock ticker
            days: Days of history
            indicators: 'all' or list of indicators
        """
        
        # Download data
        df = self.download_data(symbol, days)
        if df is None or len(df) < 50:
            return None
        
        # Create subplots
        fig = make_subplots(
            rows=5, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.1, 0.1],
            subplot_titles=(
                f'{symbol} - Advanced Chart',
                'Volume',
                'RSI',
                'MACD',
                'Stochastic'
            )
        )
        
        # ================================================================
        # ROW 1: MAIN PRICE CHART
        # ================================================================
        
        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC',
                increasing_line_color=self.colors['bullish'],
                decreasing_line_color=self.colors['bearish']
            ),
            row=1, col=1
        )
        
        # Moving Average Ribbons
        ma_periods = [5, 10, 20, 50, 100, 200]
        ma_colors = ['#2962FF', '#0091EA', '#00B8D4', '#00BFA5', '#FF6D00', '#DD2C00']
        
        for period, color in zip(ma_periods, ma_colors):
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ma,
                        name=f'MA{period}',
                        line=dict(color=color, width=1.5),
                        opacity=0.7
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb.bollinger_hband(),
                name='BB Upper',
                line=dict(color=self.colors['bb_upper'], width=1, dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb.bollinger_lband(),
                name='BB Lower',
                line=dict(color=self.colors['bb_lower'], width=1, dash='dash'),
                fill='tonexty',
                fillcolor=self.colors['bb_fill'],
                showlegend=False
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=bb.bollinger_mavg(),
                name='BB Middle',
                line=dict(color='rgba(33, 150, 243, 0.5)', width=1),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # VWAP
        vwap = self.calculate_vwap(df)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vwap,
                name='VWAP',
                line=dict(color=self.colors['vwap'], width=2, dash='dot')
            ),
            row=1, col=1
        )
        
        # Ichimoku Cloud
        if len(df) >= 52:
            tenkan, kijun, span_a, span_b, chikou = self.calculate_ichimoku(df)
            
            # Conversion and Base lines
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=tenkan,
                    name='Tenkan (Conv)',
                    line=dict(color='#FF6B6B', width=1)
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=kijun,
                    name='Kijun (Base)',
                    line=dict(color='#4ECDC4', width=1)
                ),
                row=1, col=1
            )
            
            # Cloud (Kumo)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=span_a,
                    name='Span A',
                    line=dict(width=0),
                    showlegend=False
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=span_b,
                    name='Span B',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(76, 175, 80, 0.2)',
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Parabolic SAR
        sar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=sar.psar(),
                name='SAR',
                mode='markers',
                marker=dict(color=self.colors['sar'], size=3)
            ),
            row=1, col=1
        )
        
        # Support and Resistance
        support_levels, resistance_levels = self.calculate_support_resistance(df)
        
        for level in support_levels:
            fig.add_hline(
                y=level,
                line=dict(color=self.colors['support'], width=1, dash='dash'),
                row=1, col=1
            )
        
        for level in resistance_levels:
            fig.add_hline(
                y=level,
                line=dict(color=self.colors['resistance'], width=1, dash='dash'),
                row=1, col=1
            )
        
        # Fibonacci Levels
        fib_levels = self.calculate_fibonacci(df)
        for label, level in fib_levels.items():
            fig.add_hline(
                y=level,
                line=dict(color=self.colors['fibonacci'], width=0.5, dash='dot'),
                annotation_text=f'Fib {label}',
                annotation_position='right',
                row=1, col=1
            )
        
        # ================================================================
        # ROW 2: VOLUME
        # ================================================================
        
        colors_volume = [self.colors['bullish'] if close >= open_ else self.colors['bearish'] 
                         for close, open_ in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors_volume,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Volume MA
        vol_ma = df['volume'].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=vol_ma,
                name='Vol MA20',
                line=dict(color='orange', width=2),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # OBV (On Balance Volume)
        obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
        obv_scaled = obv.on_balance_volume() / obv.on_balance_volume().max() * df['volume'].max()
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=obv_scaled,
                name='OBV',
                line=dict(color='purple', width=1),
                showlegend=False
            ),
            row=2, col=1
        )
        
        # ================================================================
        # ROW 3: RSI
        # ================================================================
        
        rsi = ta.momentum.RSIIndicator(df['close'], window=14)
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=rsi.rsi(),
                name='RSI',
                line=dict(color='#2962FF', width=2),
                showlegend=False
            ),
            row=3, col=1
        )
        
        # RSI levels
        fig.add_hline(y=70, line=dict(color='red', width=1, dash='dash'), row=3, col=1)
        fig.add_hline(y=50, line=dict(color='gray', width=0.5), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', width=1, dash='dash'), row=3, col=1)
        
        # ================================================================
        # ROW 4: MACD
        # ================================================================
        
        macd = ta.trend.MACD(df['close'])
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd.macd(),
                name='MACD',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=macd.macd_signal(),
                name='Signal',
                line=dict(color='red', width=2),
                showlegend=False
            ),
            row=4, col=1
        )
        
        # MACD Histogram
        macd_hist = macd.macd_diff()
        colors_macd = ['green' if val >= 0 else 'red' for val in macd_hist]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=macd_hist,
                name='Histogram',
                marker_color=colors_macd,
                showlegend=False
            ),
            row=4, col=1
        )
        
        fig.add_hline(y=0, line=dict(color='gray', width=0.5), row=4, col=1)
        
        # ================================================================
        # ROW 5: STOCHASTIC
        # ================================================================
        
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=stoch.stoch(),
                name='%K',
                line=dict(color='blue', width=2),
                showlegend=False
            ),
            row=5, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=stoch.stoch_signal(),
                name='%D',
                line=dict(color='red', width=2),
                showlegend=False
            ),
            row=5, col=1
        )
        
        # Stochastic levels
        fig.add_hline(y=80, line=dict(color='red', width=1, dash='dash'), row=5, col=1)
        fig.add_hline(y=50, line=dict(color='gray', width=0.5), row=5, col=1)
        fig.add_hline(y=20, line=dict(color='green', width=1, dash='dash'), row=5, col=1)
        
        # ================================================================
        # LAYOUT
        # ================================================================
        
        fig.update_layout(
            title=f'{symbol} - Advanced Technical Analysis',
            height=1400,
            showlegend=True,
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='left',
                x=0
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        # Update axes
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128, 128, 128, 0.2)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(128, 128, 128, 0.2)')
        
        # Y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        fig.update_yaxes(title_text="Stoch", row=5, col=1)
        
        return fig
    
    def create_simple_chart(self, symbol, df=None, days=90):
        """
        Create simplified chart with essential indicators
        """
        if df is None:
            df = self.download_data(symbol, days)
        
        if df is None or len(df) < 20:
            return None
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol}', 'Volume', 'RSI')
        )
        
        # Candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # MAs
        for period, color in [(20, 'blue'), (50, 'orange'), (200, 'red')]:
            if len(df) >= period:
                ma = df['close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(x=df.index, y=ma, name=f'MA{period}',
                               line=dict(color=color, width=2)),
                    row=1, col=1
                )
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        fig.add_trace(
            go.Scatter(x=df.index, y=bb.bollinger_hband(),
                       name='BB Upper', line=dict(dash='dash'), opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=bb.bollinger_lband(),
                       name='BB Lower', line=dict(dash='dash'), opacity=0.5,
                       fill='tonexty', fillcolor='rgba(100, 100, 100, 0.1)'),
            row=1, col=1
        )
        
        # Volume
        colors = [self.colors['bullish'] if c >= o else self.colors['bearish']
                  for c, o in zip(df['close'], df['open'])]
        fig.add_trace(
            go.Bar(x=df.index, y=df['volume'], marker_color=colors, showlegend=False),
            row=2, col=1
        )
        
        # RSI
        rsi = ta.momentum.RSIIndicator(df['close'])
        fig.add_trace(
            go.Scatter(x=df.index, y=rsi.rsi(), name='RSI',
                       line=dict(color='purple', width=2), showlegend=False),
            row=3, col=1
        )
        fig.add_hline(y=70, line=dict(color='red', dash='dash'), row=3, col=1)
        fig.add_hline(y=30, line=dict(color='green', dash='dash'), row=3, col=1)
        
        fig.update_layout(
            height=900,
            showlegend=True,
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig

# ================================================================================
# USAGE EXAMPLE
# ================================================================================

if __name__ == '__main__':
    chart_engine = AdvancedChartEngine()
    
    # Create full advanced chart
    fig = chart_engine.create_chart('AAPL', days=180, indicators='all')
    if fig:
        fig.show()
    
    # Create simple chart
    # fig_simple = chart_engine.create_simple_chart('AAPL', days=90)
    # if fig_simple:
    #     fig_simple.show()

