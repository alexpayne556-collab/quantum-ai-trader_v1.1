"""
Interactive Chart Engine with Glowing Pattern Highlights
TradingView-quality visualization with:
- Candlestick charts with volume
- 7 EMA ribbons with dynamic color gradients
- Detected patterns with glowing highlights
- AI forecast overlays
- Interactive tooltips

Run: python chart_engine.py
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
import talib
import json
from datetime import datetime
from pattern_detector import PatternDetector

class ChartEngine:
    """
    Interactive financial chart with pattern detection overlays.
    """
    
    # Color scheme
    COLORS = {
        'BULLISH': 'rgba(0, 255, 170, 0.6)',  # Cyan/green glow
        'BEARISH': 'rgba(255, 82, 82, 0.6)',   # Red glow
        'NEUTRAL': 'rgba(255, 215, 0, 0.6)',   # Yellow glow
        'bg': '#0e1117',  # Dark background
        'grid': '#1e2130',
        'text': '#d1d5db'
    }
    
    # EMA ribbon colors (gradient from fast to slow)
    EMA_COLORS = {
        5: 'rgba(255, 0, 255, 0.8)',    # Magenta (fastest)
        8: 'rgba(0, 255, 255, 0.7)',    # Cyan
        13: 'rgba(0, 255, 127, 0.6)',   # Spring green
        21: 'rgba(127, 255, 0, 0.5)',   # Chartreuse
        34: 'rgba(255, 255, 0, 0.4)',   # Yellow
        55: 'rgba(255, 165, 0, 0.3)',   # Orange
        89: 'rgba(255, 69, 0, 0.2)'     # Red-orange (slowest)
    }
    
    def __init__(self):
        self.detector = PatternDetector()
    
    def create_candlestick_chart(self, ticker: str, period='60d', interval='1d', 
                                  show_patterns=True, show_emas=True, show_forecast=False):
        """
        Create complete interactive chart with all features.
        """
        print(f"\n🎨 Building interactive chart for {ticker}...")
        
        # Fetch data
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if len(df) < 20:
            print(f"❌ Insufficient data for {ticker}")
            return None
        
        # Extract arrays
        dates = df.index
        open_arr = self._get_array(df, 'Open')
        high_arr = self._get_array(df, 'High')
        low_arr = self._get_array(df, 'Low')
        close_arr = self._get_array(df, 'Close')
        volume_arr = self._get_array(df, 'Volume')
        
        # Create subplots (main chart + volume + RSI + MACD)
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{ticker} - Price & Patterns', 'Volume', 'RSI (9/14)', 'MACD (5-13-1)')
        )
        
        # ===== MAIN CHART: Candlesticks =====
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=open_arr,
                high=high_arr,
                low=low_arr,
                close=close_arr,
                name='OHLC',
                increasing_line_color='rgba(0, 255, 127, 0.8)',
                decreasing_line_color='rgba(255, 82, 82, 0.8)',
                increasing_fillcolor='rgba(0, 255, 127, 0.3)',
                decreasing_fillcolor='rgba(255, 82, 82, 0.3)'
            ),
            row=1, col=1
        )
        
        # ===== EMA RIBBONS =====
        if show_emas:
            for period_ema, color in self.EMA_COLORS.items():
                ema = talib.EMA(close_arr, timeperiod=period_ema)
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=ema,
                        name=f'EMA{period_ema}',
                        line=dict(color=color, width=1.5),
                        opacity=0.7,
                        hovertemplate=f'EMA{period_ema}: %{{y:.2f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
        
        # ===== PATTERN DETECTION & OVERLAYS =====
        if show_patterns:
            pattern_results = self.detector.detect_all_patterns(ticker, period=period, interval=interval)
            patterns = pattern_results.get('patterns', [])
            
            # Group patterns by confidence level for visual hierarchy
            high_conf = [p for p in patterns if p['confidence'] > 0.8]
            med_conf = [p for p in patterns if 0.6 <= p['confidence'] <= 0.8]
            low_conf = [p for p in patterns if p['confidence'] < 0.6]
            
            print(f"  📍 Adding {len(patterns)} pattern overlays...")
            print(f"     High confidence (>0.8): {len(high_conf)}")
            print(f"     Medium confidence (0.6-0.8): {len(med_conf)}")
            print(f"     Low confidence (<0.6): {len(low_conf)}")
            
            # Add pattern highlights as shapes
            for pattern in patterns:
                self._add_pattern_shape(fig, pattern, dates, close_arr)
        
        # ===== VOLUME BARS =====
        colors_volume = ['rgba(0, 255, 127, 0.5)' if close_arr[i] >= open_arr[i] 
                         else 'rgba(255, 82, 82, 0.5)' 
                         for i in range(len(close_arr))]
        
        fig.add_trace(
            go.Bar(
                x=dates,
                y=volume_arr,
                name='Volume',
                marker_color=colors_volume,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # ===== RSI (9 and 14) =====
        rsi9 = talib.RSI(close_arr, timeperiod=9)
        rsi14 = talib.RSI(close_arr, timeperiod=14)
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=rsi9,
                name='RSI 9',
                line=dict(color='rgba(255, 0, 255, 0.8)', width=1.5),
                hovertemplate='RSI9: %{y:.1f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=rsi14,
                name='RSI 14',
                line=dict(color='rgba(0, 255, 255, 0.8)', width=1.5),
                hovertemplate='RSI14: %{y:.1f}<extra></extra>'
            ),
            row=3, col=1
        )
        
        # RSI reference lines (oversold/overbought)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255, 82, 82, 0.5)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0, 255, 127, 0.5)", row=3, col=1)
        
        # ===== MACD (5-13-1) =====
        macd, macd_signal, macd_hist = talib.MACD(close_arr, fastperiod=5, slowperiod=13, signalperiod=1)
        
        # MACD histogram
        macd_colors = ['rgba(0, 255, 127, 0.6)' if val >= 0 else 'rgba(255, 82, 82, 0.6)' 
                       for val in macd_hist]
        fig.add_trace(
            go.Bar(
                x=dates, y=macd_hist,
                name='MACD Histogram',
                marker_color=macd_colors,
                showlegend=False
            ),
            row=4, col=1
        )
        
        # MACD lines
        fig.add_trace(
            go.Scatter(
                x=dates, y=macd,
                name='MACD',
                line=dict(color='rgba(0, 255, 255, 0.8)', width=1.5)
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates, y=macd_signal,
                name='Signal',
                line=dict(color='rgba(255, 165, 0, 0.8)', width=1.5)
            ),
            row=4, col=1
        )
        
        # ===== LAYOUT STYLING =====
        fig.update_layout(
            title=dict(
                text=f'<b>{ticker}</b> - Pattern Detection & AI Analysis',
                font=dict(size=24, color=self.COLORS['text'])
            ),
            template='plotly_dark',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['bg'],
            font=dict(color=self.COLORS['text']),
            xaxis_rangeslider_visible=False,
            height=1200,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        # Update all axes
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor=self.COLORS['grid'])
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor=self.COLORS['grid'])
        
        print(f"✅ Chart built successfully!")
        
        return fig
    
    def _add_pattern_shape(self, fig, pattern, dates, close_arr):
        """
        Add glowing pattern highlight as shape overlay.
        """
        start_idx = pattern['start_idx']
        end_idx = pattern['end_idx']
        
        # Ensure indices are valid
        if start_idx < 0 or end_idx >= len(dates):
            return
        
        # Determine color based on pattern type
        color = self.COLORS.get(pattern['type'], self.COLORS['NEUTRAL'])
        
        # Adjust opacity based on confidence
        confidence = pattern['confidence']
        if confidence > 0.8:
            opacity = 0.35  # Bright glow
        elif confidence > 0.6:
            opacity = 0.25  # Medium glow
        else:
            opacity = 0.15  # Dim glow
        
        # Get price range for pattern zone
        price_high = max(close_arr[start_idx:end_idx+1])
        price_low = min(close_arr[start_idx:end_idx+1])
        price_range = price_high - price_low
        
        # Add rectangle shape (pattern zone)
        fig.add_shape(
            type="rect",
            x0=dates[start_idx],
            x1=dates[end_idx],
            y0=price_low - price_range * 0.1,
            y1=price_high + price_range * 0.1,
            fillcolor=color,
            opacity=opacity,
            line=dict(color=color.replace('0.6', '0.9'), width=2),
            layer='below',
            row=1, col=1
        )
        
        # Add annotation (pattern label)
        fig.add_annotation(
            x=dates[start_idx],
            y=price_high + price_range * 0.15,
            text=f"<b>{pattern['pattern']}</b><br>{pattern['type']}<br>{confidence:.0%}",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor=color,
            ax=0,
            ay=-40,
            bgcolor=self.COLORS['bg'],
            bordercolor=color,
            borderwidth=2,
            font=dict(size=10, color=self.COLORS['text']),
            opacity=0.9,
            row=1, col=1
        )
    
    def _get_array(self, df, col):
        """Extract numpy array from DataFrame column."""
        if isinstance(df[col], pd.DataFrame):
            return df[col].iloc[:, 0].values
        return df[col].values
    
    def save_chart(self, fig, ticker: str, output_dir='frontend/charts'):
        """
        Save interactive chart as HTML.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{output_dir}/{ticker}_chart.html"
        fig.write_html(filename)
        print(f"💾 Chart saved: {filename}")
        return filename
    
    def create_multi_ticker_view(self, tickers: list, period='60d'):
        """
        Create a grid of small charts for multiple tickers (scanner view).
        """
        print(f"\n📊 Creating multi-ticker view for {len(tickers)} tickers...")
        
        # Limit to prevent overload
        tickers = tickers[:12]  # Max 12 for readability
        
        rows = (len(tickers) + 2) // 3  # 3 columns
        
        fig = make_subplots(
            rows=rows, cols=3,
            subplot_titles=tickers,
            vertical_spacing=0.08,
            horizontal_spacing=0.05
        )
        
        for idx, ticker in enumerate(tickers):
            row = idx // 3 + 1
            col = idx % 3 + 1
            
            try:
                df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
                close = self._get_array(df, 'Close')
                
                # Mini candlestick
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=close,
                        name=ticker,
                        line=dict(color='rgba(0, 255, 127, 0.8)', width=1.5),
                        fill='tozeroy',
                        fillcolor='rgba(0, 255, 127, 0.2)',
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Add EMA 21
                ema21 = talib.EMA(close, timeperiod=21)
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=ema21,
                        name=f'{ticker} EMA21',
                        line=dict(color='rgba(255, 165, 0, 0.8)', width=1),
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
            except Exception as e:
                print(f"  ⚠️ Skipped {ticker}: {e}")
        
        fig.update_layout(
            title="Multi-Ticker Scanner",
            template='plotly_dark',
            paper_bgcolor=self.COLORS['bg'],
            plot_bgcolor=self.COLORS['bg'],
            height=300 * rows,
            showlegend=False
        )
        
        print(f"✅ Multi-ticker view created!")
        return fig


if __name__ == '__main__':
    engine = ChartEngine()
    
    # Create detailed charts for the 4 test tickers
    tickers = ['MU', 'IONQ', 'APLD', 'ANNX']
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        fig = engine.create_candlestick_chart(
            ticker, 
            period='60d', 
            interval='1d',
            show_patterns=True,
            show_emas=True
        )
        
        if fig:
            # Save interactive HTML
            engine.save_chart(fig, ticker)
            print(f"✅ {ticker} chart complete!")
    
    # Create multi-ticker overview
    print(f"\n{'='*60}")
    multi_fig = engine.create_multi_ticker_view(tickers, period='60d')
    engine.save_chart(multi_fig, 'MULTI_TICKER_OVERVIEW')
    
    print(f"\n{'='*60}")
    print("🎉 All charts generated!")
    print(f"{'='*60}")
    print("📂 View charts in: frontend/charts/")
    print("🌐 Open any .html file in browser to interact")
    print(f"{'='*60}")
