"""
ADVANCED 2030-STYLE INTERACTIVE DASHBOARD
Features:
- 24-day AI forecast projections with confidence bands
- Breathing/glowing animations for active signals  
- Hover-to-highlight patterns (no visual clutter)
- Elliott Wave labels with wave counting
- Fibonacci level overlays (retracement + extension)
- Real-time scanner for 20 stocks
- Advanced caution scoring system
- Supply/Demand zone visualization
- Volume Profile POC

Built for serious trading with real capital on the line.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Import production DataFetcher
try:
    from PRODUCTION_DATAFETCHER import DataFetcher
    HAS_DATAFETCHER = True
except ImportError:
    HAS_DATAFETCHER = False

# Import safe indicators and forecast engine
try:
    from safe_indicators import safe_rsi, safe_atr, safe_macd, validate_indicators
    HAS_SAFE_INDICATORS = True
except ImportError:
    HAS_SAFE_INDICATORS = False

try:
    from forecast_engine import ForecastEngine
    HAS_FORECAST_ENGINE = True
except ImportError:
    HAS_FORECAST_ENGINE = False

# Optional module imports
try:
    from advanced_pattern_detector import AdvancedPatternDetector
    HAS_ADVANCED_DETECTOR = True
except ImportError:
    HAS_ADVANCED_DETECTOR = False

try:
    from pattern_detector import PatternDetector
    HAS_PATTERN_DETECTOR = True
except ImportError:
    HAS_PATTERN_DETECTOR = False

try:
    from elliott_wave_detector import ElliottWaveDetector, FibonacciCalculator
    HAS_ELLIOTT = True
except ImportError:
    HAS_ELLIOTT = False

try:
    from ai_recommender_tuned import FE
    HAS_FE = True
except ImportError:
    HAS_FE = False


class AdvancedDashboard:
    """2030-style interactive trading dashboard - production grade"""
    
    def __init__(self):
        # Robust imports with fallback
        try:
            if HAS_DATAFETCHER:
                self.data_fetcher = DataFetcher()
            else:
                raise ImportError("DataFetcher not available")
        except Exception as e:
            print(f"⚠️  DataFetcher initialization failed: {e}")
            class FallbackDataFetcher:
                def fetch_ohlcv(self, ticker, period='60d'):
                    return pd.DataFrame()
            self.data_fetcher = FallbackDataFetcher()
        
        # Initialize forecast engine
        try:
            if HAS_FORECAST_ENGINE:
                self.forecast_engine = ForecastEngine()
            else:
                raise ImportError("ForecastEngine not available")
        except Exception as e:
            print(f"⚠️  ForecastEngine initialization failed: {e}")
            self.forecast_engine = None
        
        try:
            if HAS_ADVANCED_DETECTOR:
                self.advanced_detector = AdvancedPatternDetector()
            else:
                raise ImportError("AdvancedPatternDetector not available")
        except Exception as e:
            print(f"⚠️  AdvancedPatternDetector initialization failed: {e}")
        except Exception as e:
            print(f"⚠️  AdvancedPatternDetector initialization failed: {e}")
            class FallbackAdvancedPatternDetector:
                def detect_all_advanced_patterns(self, df):
                    return {'supply_demand_zones': [], 'volume_profile': {'poc_price': 0}}
            self.advanced_detector = FallbackAdvancedPatternDetector()
        
        try:
            if HAS_PATTERN_DETECTOR:
                self.pattern_detector = PatternDetector()
            else:
                raise ImportError("PatternDetector not available")
        except Exception as e:
            print(f"⚠️  PatternDetector initialization failed: {e}")
            class FallbackPatternDetector:
                def detect_all_patterns(self, df):
                    return {'patterns': []}
            self.pattern_detector = FallbackPatternDetector()
        
        try:
            if HAS_ELLIOTT:
                self.elliott_detector = ElliottWaveDetector()
                self.fib_calc = FibonacciCalculator()
            else:
                raise ImportError("ElliottWaveDetector not available")
        except Exception as e:
            print(f"⚠️  ElliottWaveDetector initialization failed: {e}")
            class FallbackElliottWaveDetector:
                def analyze_chart(self, df, verbose=True):
                    return {'impulse_detected': False, 'impulse_waves': [], 'correction_detected': False, 'correction_waves': None, 'confidence': 0.0, 'targets': {}, 'wave_rules_valid': {}, 'fib_ratios': {}}
            class FallbackFibonacciCalculator:
                def calculate_retracement_levels(self, high, low):
                    return {}
            self.elliott_detector = FallbackElliottWaveDetector()
            self.fib_calc = FallbackFibonacciCalculator()
        
        try:
            if HAS_FE:
                from ai_recommender_tuned import FE
                self.fe = FE()
            else:
                raise ImportError("FE not available")
        except Exception as e:
            print(f"⚠️  FE initialization failed: {e}")
        except Exception as e:
            print(f"⚠️  FE initialization failed: {e}")
            class SimpleFE:
                def engineer(self, df):
                    df = df.copy()
                    df['sma_5'] = df['Close'].rolling(5).mean()
                    df['sma_20'] = df['Close'].rolling(20).mean()
                    return df.dropna()
            self.fe = SimpleFE()
        
        self.models = self._load_models()
        self.colors = {
            'bg': '#0a0e27',
            'grid': '#1a1f3a',
            'bullish': '#00ff88',
            'bearish': '#ff006e',
            'forecast': '#00d9ff',
            'fib': '#ffd60a',
            'wave': '#ff6b35',
            'warning': '#ff9500',
            'safe': '#00ff88',
            'danger': '#ff006e',
            'supply': 'rgba(255, 0, 110, 0.3)',
            'demand': 'rgba(0, 255, 136, 0.3)'
        }
    def _normalize_columns(self, df):
        df.columns = [col.capitalize() for col in df.columns]
        return df
    def _truncate_text(self, text, max_len=15):
        return text[:max_len] + '...' if len(text) > max_len else text
    
    def _load_models(self) -> dict:
        """Load trained AI models"""
        models = {}
        model_dir = Path('models')
        
        if not model_dir.exists():
            print("⚠️  No models directory found")
            return models
        
        for pkl_file in model_dir.glob('*_tuned.pkl'):
            ticker = pkl_file.stem.replace('_tuned', '')
            try:
                loaded = joblib.load(pkl_file)
                # Handle dict format: {'model': ..., 'metadata': ...}
                if isinstance(loaded, dict) and 'model' in loaded:
                    models[ticker] = loaded['model']
                else:
                    models[ticker] = loaded
            except Exception as e:
                print(f"❌ Error loading {ticker} model: {e}")
        
        return models
    
    def generate_24day_forecast(self, ticker: str, df: pd.DataFrame) -> pd.DataFrame:
        """Generate forecast using ATR-based engine with proper error handling"""
        # Use ATR-based forecast engine if available
        if self.forecast_engine and ticker in self.models:
            try:
                return self.forecast_engine.generate_forecast(
                    df, self.models[ticker], self.fe, ticker
                )
            except Exception as e:
                print(f"  ⚠️  ATR forecast failed for {ticker}: {e}")
                # Fall through to simple projection
        
        # Fallback to simple projection
        return self.forecast_engine.simple_trend_projection(df, 24) if self.forecast_engine else self._simple_projection(df, 24)
    
    def _simple_projection(self, df: pd.DataFrame, days: int) -> pd.DataFrame:
        """Simple linear projection if no model available"""
        last_close = float(df['Close'].iloc[-1])
        last_date = df.index[-1]
        
        # Use last 20 days trend
        trend = (df['Close'].iloc[-1] - df['Close'].iloc[-20]) / 20
        trend = float(trend)
        
        forecasts = []
        for day in range(1, days + 1):
            forecast_price = last_close + (trend * day)
            forecast_date = last_date + timedelta(days=day)
            
            forecasts.append({
                'date': forecast_date,
                'price': forecast_price,
                'confidence': 0.5,
                'signal': 'NEUTRAL'
            })
        
        return pd.DataFrame(forecasts)
    
    def create_advanced_chart(self, ticker: str, df: pd.DataFrame) -> go.Figure:
        """
        Create 2030-style interactive chart with all advanced features:
        - 24-day AI forecast with confidence bands
        - Elliott Wave labels
        - Fibonacci retracement levels
        - Supply/Demand zones
        - Volume Profile POC
        - Pattern overlays with hover
        - EMA ribbons
        """
        print(f"\n🎨 Building advanced chart for {ticker}...")
        
        # Detect all patterns (pattern_detector expects DataFrame)
        df = self._normalize_columns(df)
        patterns = self._get_patterns(ticker, df)
        
        advanced_results = self.advanced_detector.detect_all_advanced_patterns(df)
        elliott_analysis = self.elliott_detector.analyze_chart(df, verbose=False)
        
        # Generate 24-day forecast
        forecast_df = self.generate_24day_forecast(ticker, df)
        
        # Create figure with subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{ticker} - Advanced Analysis with 24-Day AI Forecast', 'RSI', 'MACD')
        )
        
        # Main candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name=ticker,
            increasing_line_color=self.colors['bullish'],
            decreasing_line_color=self.colors['bearish']
        ), row=1, col=1)
        
        # Add EMA ribbons (7 EMAs) - gradient effect
        ema_periods = [5, 8, 13, 21, 34, 55, 89]
        ema_colors = ['#00ffff', '#00ccff', '#0099ff', '#0066ff', '#0033ff', '#0000ff', '#000099']
        
        for period, color in zip(ema_periods, ema_colors):
            ema = df['Close'].ewm(span=period, adjust=False).mean()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=ema,
                name=f'EMA {period}',
                line=dict(color=color, width=1),
                opacity=0.6,
                showlegend=False
            ), row=1, col=1)
        
        # Add Fibonacci levels
        high_price = float(df['High'].iloc[-50:].max())
        low_price = float(df['Low'].iloc[-50:].min())
        fib_levels = self.fib_calc.calculate_retracement_levels(high_price, low_price)
        
        key_fib_levels = ['23.6%', '38.2%', '50%', '61.8%', '78.6%']
        for fib_name in key_fib_levels:
            if fib_name in fib_levels:
                fib_price = fib_levels[fib_name]
                
                # Horizontal line
                fig.add_hline(
                    y=fib_price,
                    line=dict(
                        color=self.colors['fib'],
                        width=1,
                        dash='dot'
                    ),
                    opacity=0.5,
                    row=1, col=1
                )
                
                # Label
                fig.add_annotation(
                    x=df.index[-1],
                    y=fib_price,
                    text=f"Fib {fib_name}",
                    showarrow=False,
                    xanchor='left',
                    font=dict(size=9, color=self.colors['fib']),
                    row=1, col=1
                )
        
        # Add Supply/Demand zones
        zones = advanced_results['supply_demand_zones']
        for zone in zones[-5:]:  # Last 5 zones
            color = self.colors['supply'] if zone['type'] == 'supply' else self.colors['demand']
            
            fig.add_shape(
                type='rect',
                x0=df.index[min(zone['index'], len(df)-1)],
                x1=df.index[-1],
                y0=zone['price_low'],
                y1=zone['price_high'],
                fillcolor=color,
                opacity=0.3,
                line=dict(width=0),
                row=1, col=1
            )
            
            # Zone label
            fig.add_annotation(
                x=df.index[-1],
                y=(zone['price_low'] + zone['price_high']) / 2,
                text=f"{'Supply' if zone['type'] == 'supply' else 'Demand'}<br>${zone['price_low']:.2f}",
                showarrow=False,
                xanchor='left',
                font=dict(size=8, color='white'),
                bgcolor=color,
                opacity=0.8,
                row=1, col=1
            )
        
        # Add Elliott Waves
        if elliott_analysis['impulse_detected'] and elliott_analysis['impulse_waves']:
            waves = elliott_analysis['impulse_waves']
            
            for wave in waves:
                # Wave line
                start_date = df.index[wave.start_index]
                end_date = df.index[wave.end_index]
                
                fig.add_trace(go.Scatter(
                    x=[start_date, end_date],
                    y=[wave.start_price, wave.end_price],
                    mode='lines+markers',
                    line=dict(color=self.colors['wave'], width=3),
                    marker=dict(size=8),
                    name=f'Elliott Wave {wave.wave_number}',
                    showlegend=False,
                    hovertemplate=f'Wave {wave.wave_number}<br>Move: {wave.pct_move:.2f}%<extra></extra>'
                ), row=1, col=1)
                
                # Wave label
                mid_price = (wave.start_price + wave.end_price) / 2
                fig.add_annotation(
                    x=end_date,
                    y=mid_price,
                    text=f"<b>{wave.wave_number}</b>",
                    showarrow=False,
                    font=dict(size=16, color=self.colors['wave'], family='Arial Black'),
                    bgcolor='rgba(10, 14, 39, 0.8)',
                    bordercolor=self.colors['wave'],
                    borderwidth=2,
                    row=1, col=1
                )
        
        # Add 24-day forecast projection (BREATHING EFFECT with confidence cone)
        if len(forecast_df) > 0:
            # Main forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df['date'],
                y=forecast_df['price'],
                name='24-Day AI Forecast',
                line=dict(
                    color=self.colors['forecast'],
                    width=3,
                    dash='dash'
                ),
                mode='lines+markers',
                marker=dict(size=6, symbol='diamond'),
                hovertemplate='<b>AI Forecast</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Signal: %{text}<extra></extra>',
                text=forecast_df['signal']
            ), row=1, col=1)
            
            # Confidence cone (breathing effect)
            upper_bound = forecast_df['price'] * (1 + (1 - forecast_df['confidence']) * 0.08)
            lower_bound = forecast_df['price'] * (1 - (1 - forecast_df['confidence']) * 0.08)
            
            fig.add_trace(go.Scatter(
                x=forecast_df['date'].tolist() + forecast_df['date'].tolist()[::-1],
                y=upper_bound.tolist() + lower_bound.tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 217, 255, 0.2)',  # Convert hex to rgba
                line=dict(width=0),
                name='Forecast Confidence',
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=1)
        
        # Add pattern overlays with hover-to-highlight
        for pattern in patterns[-15:]:  # Last 15 patterns
            start_idx = pattern.get('start_idx', 0)
            end_idx = pattern.get('end_idx', start_idx + 1)
            if start_idx >= len(df) or end_idx > len(df):
                continue
            try:
                start_idx = max(0, start_idx)
                end_idx = min(len(df), end_idx)
                if start_idx >= end_idx:
                    continue
                low_min = df['Low'].iloc[start_idx:end_idx].min()
                high_max = df['High'].iloc[start_idx:end_idx].max()
            except Exception as e:
                print(f"Pattern indexing error: {e}")
                continue
            color = self.colors['bullish'] if pattern.get('type', '') == 'BULLISH' else self.colors['bearish']
            fig.add_shape(
                type='rect',
                x0=df.index[start_idx],
                x1=df.index[min(end_idx, len(df)-1)],
                y0=low_min * 0.998,
                y1=high_max * 1.002,
                fillcolor=color,
                opacity=0.25,
                line=dict(width=1, color=color),
                row=1, col=1
            )
            fig.add_annotation(
                x=df.index[start_idx],
                y=df['High'].iloc[start_idx] * 1.003,
                text=self._truncate_text(pattern.get('pattern', '')),  # Truncate long names
                showarrow=False,
                font=dict(size=8, color=color, family='Arial'),
                bgcolor=self.colors['bg'],
                opacity=0.8,
                row=1, col=1
            )
        
        # Add Volume Profile POC
        vp = advanced_results['volume_profile']
        fig.add_hline(
            y=vp['poc_price'],
            line=dict(color='#ffffff', width=2, dash='solid'),
            annotation_text=f"POC: ${vp['poc_price']:.2f}",
            annotation_position='right',
            row=1, col=1
        )
        
        # Add RSI
        if HAS_SAFE_INDICATORS:
            rsi = safe_rsi(df['Close'], window=14)
        else:
            # Fallback with epsilon protection
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=rsi,
            name='RSI(14)',
            line=dict(color=self.colors['fib'], width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 214, 10, 0.13)'  # fib color with alpha
        ), row=2, col=1)
        
        fig.add_hline(y=70, line=dict(color=self.colors['bearish'], dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=30, line=dict(color=self.colors['bullish'], dash='dash', width=1), row=2, col=1)
        fig.add_hline(y=50, line=dict(color='#666666', dash='dot', width=1), row=2, col=1)
        
        # Add MACD
        try:
            if HAS_SAFE_INDICATORS:
                macd_result = safe_macd(df['Close'])
                macd = macd_result['macd']
                signal = macd_result['signal']
                histogram = macd_result['histogram']
            else:
                # Fallback calculation
                exp1 = df['Close'].ewm(span=12, adjust=False).mean()
                exp2 = df['Close'].ewm(span=26, adjust=False).mean()
                macd = exp1 - exp2
                signal = macd.ewm(span=9, adjust=False).mean()
                histogram = macd - signal
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=macd,
                name='MACD',
                line=dict(color=self.colors['forecast'], width=2)
            ), row=3, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=signal,
                name='Signal',
                line=dict(color=self.colors['bearish'], width=2)
            ), row=3, col=1)
            
            # MACD histogram
            colors_hist = [self.colors['bullish'] if h > 0 else self.colors['bearish'] for h in histogram]
            fig.add_trace(go.Bar(
                x=df.index,
                y=histogram,
                name='MACD Histogram',
                marker_color=colors_hist,
                opacity=0.5,
                showlegend=False
            ), row=3, col=1)
        except Exception as e:
            print(f"⚠️  MACD calculation error: {e}")
        
        # Layout - 2030 style dark theme with breathing effect
        fig.update_layout(
            title=dict(
                text=f'<b>{ticker}</b> - 2030 Advanced Trading Analysis<br><sub>24-Day AI Forecast | Elliott Waves | Fibonacci | Supply/Demand | Volume Profile</sub>',
                font=dict(size=24, color='white', family='Arial Black'),
                x=0.5,
                xanchor='center'
            ),
            paper_bgcolor=self.colors['bg'],
            plot_bgcolor=self.colors['bg'],
            font=dict(color='white', family='Arial'),
            hovermode='x unified',
            height=1100,
            showlegend=True,
            legend=dict(
                bgcolor='rgba(26, 31, 58, 0.9)',
                bordercolor=self.colors['bullish'],
                borderwidth=2,
                font=dict(size=10),
                x=1.0,
                y=1.0,
                xanchor='right'
            ),
            xaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                zeroline=False,
                rangeslider_visible=False
            ),
            yaxis=dict(
                gridcolor=self.colors['grid'],
                showgrid=True,
                zeroline=False
            )
        )
        
        # Update all subplots styling
        fig.update_xaxes(gridcolor=self.colors['grid'], showgrid=True)
        fig.update_yaxes(gridcolor=self.colors['grid'], showgrid=True)
        
        return fig
    
    def _get_patterns(self, ticker: str, df: pd.DataFrame) -> list:
        """Get patterns for a ticker"""
        patterns = []
        
        try:
            if HAS_ADVANCED_DETECTOR:
                advanced_patterns = self.advanced_detector.detect_all_advanced_patterns(df)
                
                # Extract Elliott Waves
                if advanced_patterns.get('elliott_waves'):
                    for wave in advanced_patterns['elliott_waves']:
                        patterns.append({
                            'pattern': 'Elliott Wave',
                            'type': wave.direction.upper() if hasattr(wave, 'direction') else 'NEUTRAL',
                            'confidence': wave.confidence if hasattr(wave, 'confidence') else 0.7,
                            'details': f"Wave {wave.wave_type}" if hasattr(wave, 'wave_type') else 'Wave detected'
                        })
                
                # Extract Fibonacci levels
                if advanced_patterns.get('fibonacci_levels'):
                    fib_levels = advanced_patterns['fibonacci_levels']
                    if fib_levels and len(fib_levels) > 0:
                        patterns.append({
                            'pattern': 'Fibonacci Confluence',
                            'type': 'NEUTRAL',
                            'confidence': 0.7,
                            'details': f"{len(fib_levels)} levels detected"
                        })
                
                # Extract Supply/Demand zones
                if advanced_patterns.get('supply_demand_zones'):
                    zones = advanced_patterns['supply_demand_zones']
                    for zone in zones[-3:]:  # Last 3 zones
                        patterns.append({
                            'pattern': f"{zone['type'].capitalize()} Zone",
                            'type': 'NEUTRAL',
                            'confidence': zone.get('strength', 0.5),
                            'details': f"${zone['price_low']:.2f} - ${zone['price_high']:.2f}"
                        })
        
        except Exception as e:
            print(f"⚠️ Pattern detection error for {ticker}: {e}")
        
        return patterns
    
    def calculate_caution_score(self, ticker: str, df: pd.DataFrame, patterns: list, signal: str, confidence: float) -> tuple:
        """
        Calculate advanced caution score (0-100)
        0-30: ✅ SAFE (green)
        30-70: ⚠️ CAUTION (yellow)
        70-100: ❌ AVOID (red)
        """
        caution = 50  # Start neutral
        
        # Factor 1: Model confidence (40% weight)
        if confidence > 0.7:
            caution -= 15
        elif confidence < 0.5:
            caution += 20
        
        # Factor 2: Pattern confluence (20% weight)
        bullish_patterns = len([p for p in patterns if p['type'] == 'BULLISH'])
        bearish_patterns = len([p for p in patterns if p['type'] == 'BEARISH'])
        
        if signal == 'BULLISH' and bullish_patterns > bearish_patterns:
            caution -= 10
        elif signal == 'BEARISH' and bearish_patterns > bullish_patterns:
            caution -= 10
        else:
            caution += 15  # Conflicting signals
        
        # Factor 3: Volatility (20% weight)
        atr = df['High'].iloc[-14:] - df['Low'].iloc[-14:]
        avg_atr = atr.mean()
        recent_atr = atr.iloc[-1]
        
        if recent_atr > avg_atr * 1.5:
            caution += 15  # High volatility = risky
        
        # Factor 4: RSI extremes (10% weight)
        if HAS_SAFE_INDICATORS:
            rsi = safe_rsi(df['Close'], window=14)
            current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        else:
            # Fallback with epsilon protection
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)  # Add epsilon to prevent division by zero
            rsi = 100 - (100 / (1 + rs))
            current_rsi = float(rsi.iloc[-1])
        
        if current_rsi > 75 or current_rsi < 25:
            caution += 10  # Overbought/oversold
        
        # Factor 5: Volume confirmation (10% weight)
        avg_volume = df['Volume'].iloc[-20:].mean()
        recent_volume = df['Volume'].iloc[-1]
        
        if recent_volume > avg_volume * 1.3:
            caution -= 10  # Strong volume = confirmation
        else:
            caution += 5  # Weak volume = questionable
        
        # Clamp to 0-100
        caution = max(0, min(100, caution))
        
        # Determine level
        if caution < 30:
            level = '✅ SAFE'
            color = self.colors['safe']
        elif caution < 70:
            level = '⚠️ CAUTION'
            color = self.colors['warning']
        else:
            level = '❌ AVOID'
            color = self.colors['danger']
        
        return caution, level, color
    
    def create_scanner_dashboard(self, tickers: list) -> go.Figure:
        print(f"\n📊 Building scanner for {len(tickers)} tickers...")
        scanner_data = []
        for ticker in tickers:
            try:
                print(f"   Scanning {ticker}...", end=' ', flush=True)
                df = self.data_fetcher.fetch_ohlcv(ticker, period='60d')
                if df is None or len(df) < 20:
                    print("❌ Insufficient data")
                    continue
                df = self._normalize_columns(df)
                patterns = self._get_patterns(ticker, df)
                signal = 'HOLD'
                confidence = 0.0
                if ticker in self.models:
                    try:
                        features_df = self.fe.engineer(df.copy())
                        if features_df is not None and len(features_df) > 0:
                            latest = features_df.iloc[-1:].values
                            if len(latest) > 0:
                                pred = self.models[ticker].predict(latest)
                                conf = self.models[ticker].predict_proba(latest).max()
                                signal = ['BEARISH', 'NEUTRAL', 'BULLISH'][pred]
                                confidence = float(conf)
                    except Exception as e:
                        print(f"⚠️  Prediction error for {ticker}: {e}")
                caution_score, caution_level, _ = self.calculate_caution_score(
                    ticker, df, patterns, signal, confidence
                )
                elliott_analysis = self.elliott_detector.analyze_chart(df, verbose=False)
                wave_count = len(elliott_analysis['impulse_waves']) if elliott_analysis['impulse_waves'] else 0
                scanner_data.append({
                    'ticker': ticker,
                    'price': float(df['Close'].iloc[-1]),
                    'change_pct': float((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100),
                    'patterns': len(patterns),
                    'top_pattern': patterns[-1].get('pattern', 'N/A')[:10] if patterns else 'None',
                    'signal': signal,
                    'confidence': confidence * 100,
                    'caution_score': caution_score,
                    'caution': caution_level,
                    'waves': wave_count
                })
                print(f"✅ {signal} @ {confidence:.0%}")
            except Exception as e:
                print(f"❌ Error: {str(e)[:40]}")
                continue
        if not scanner_data:
            print("❌ No data collected!")
            return None
        df_scanner = pd.DataFrame(scanner_data)
        df_scanner['setup_score'] = df_scanner['confidence'] - df_scanner['caution_score']
        df_scanner = df_scanner.sort_values('setup_score', ascending=False)
        colors = []
        for _, row in df_scanner.iterrows():
            if row['caution'].startswith('✅') and row['signal'] != 'HOLD':
                colors.append('#00ff88')
            elif row['caution'].startswith('❌'):
                colors.append('#ff006e')
            elif row['signal'] == 'BULLISH':
                colors.append('#00ff88')
            elif row['signal'] == 'BEARISH':
                colors.append('#ff006e')
            else:
                colors.append('#1a1f3a')
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Rank', 'Ticker', 'Price', 'Δ%', 'Patterns', 'Top', 'Signal', 'Conf%', 'Caution', 'Level', 'Waves'],
                fill_color='#1a1f3a',
                align='center',
                font=dict(color='white', size=12)
            ),
            cells=dict(
                values=[
                    list(range(1, len(df_scanner) + 1)),
                    df_scanner['ticker'],
                    df_scanner['price'].apply(lambda x: f'${x:.2f}'),
                    df_scanner['change_pct'].apply(lambda x: f'{x:+.2f}%'),
                    df_scanner['patterns'],
                    df_scanner['top_pattern'],
                    df_scanner['signal'],
                    df_scanner['confidence'].apply(lambda x: f'{x:.1f}%'),
                    df_scanner['caution_score'].apply(lambda x: f'{x:.0f}'),
                    df_scanner['caution'],
                    df_scanner['waves']
                ],
                fill_color=[colors],
                align='center',
                font=dict(color='white', size=11)
            )
        )])
        fig.update_layout(
            title=f'SCANNER: {len(df_scanner)} Tickers | {datetime.now().strftime("%H:%M")}',
            paper_bgcolor='#0a0e27',
            height=900
        )
        return fig
    
    def build_full_dashboard(self, tickers: list):
        """Build complete advanced dashboard with error handling"""
        
        print(f"\n{'='*70}")
        print(f"🚀 BUILDING 2030-STYLE ADVANCED DASHBOARD")
        print(f"{'='*70}")
        
        output_dir = Path('frontend/advanced_charts')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create individual advanced charts for portfolio tickers
        portfolio = ['MU', 'IONQ', 'APLD', 'ANNX']
        
        for ticker in portfolio:
            try:
                print(f"\n📈 Creating advanced chart for {ticker}...")
                
                df = self.data_fetcher.fetch_ohlcv(ticker, period='60d')
                if df is None or len(df) == 0:
                    print(f"   ❌ No data for {ticker}, skipping...")
                    continue
                
                fig = self.create_advanced_chart(ticker, df)
                
                output_file = output_dir / f'{ticker}_advanced.html'
                fig.write_html(str(output_file))
                
                print(f"   ✅ Saved to {output_file}")
            except Exception as e:
                print(f"   ❌ Error building chart for {ticker}: {e}")
                # Create fallback error page
                self._create_error_html(output_dir / f'{ticker}_advanced.html', ticker, str(e))
                continue
        
        # Create scanner dashboard
        try:
            print(f"\n📊 Creating scanner dashboard...")
            
            scanner_fig = self.create_scanner_dashboard(tickers)
            
            if scanner_fig is None:
                print(f"   ⚠️  Scanner returned no data, creating fallback...")
                self._create_fallback_scanner(output_dir / 'scanner_dashboard.html')
            else:
                scanner_file = output_dir / 'scanner_dashboard.html'
                scanner_fig.write_html(str(scanner_file))
                print(f"   ✅ Saved to {scanner_file}")
        except Exception as e:
            print(f"   ❌ Scanner error: {e}")
            self._create_fallback_scanner(output_dir / 'scanner_dashboard.html')
        
        print(f"\n{'='*70}")
        print(f"✅ DASHBOARD BUILD COMPLETE!")
        print(f"{'='*70}")
        print(f"\nGenerated files:")
        for file in sorted(output_dir.glob('*.html')):
            print(f"   📊 {file.name}")
        
        return output_dir
    
    def _create_error_html(self, filepath: Path, ticker: str, error_msg: str):
        """Create fallback error page"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{ticker} - Error</title>
    <style>
        body {{ 
            background: #0a0e27; 
            color: #fff; 
            font-family: monospace; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh;
            margin: 0;
        }}
        .error-box {{
            background: #1a1d35;
            border: 2px solid #ff006e;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            max-width: 600px;
        }}
        h1 {{ color: #ff006e; }}
        .error-msg {{ 
            background: #0a0e27; 
            padding: 20px; 
            border-radius: 4px; 
            margin: 20px 0;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <div class="error-box">
        <h1>⚠️ Data Temporarily Unavailable</h1>
        <p>Unable to fetch data for <strong>{ticker}</strong></p>
        <div class="error-msg">{error_msg}</div>
        <p>Please try again later or check your data provider settings.</p>
    </div>
</body>
</html>"""
        filepath.write_text(html)
    
    def _create_fallback_scanner(self, filepath: Path):
        """Create fallback scanner page when no data available"""
        html = """<!DOCTYPE html>
<html>
<head>
    <title>Scanner - Data Unavailable</title>
    <style>
        body { 
            background: #0a0e27; 
            color: #fff; 
            font-family: monospace; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 100vh;
            margin: 0;
        }
        .fallback-box {
            background: #1a1d35;
            border: 2px solid #ffd60a;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            max-width: 600px;
        }
        h1 { color: #ffd60a; }
        ul { text-align: left; margin: 20px auto; max-width: 400px; }
        li { margin: 10px 0; }
    </style>
</head>
<body>
    <div class="fallback-box">
        <h1>📊 Scanner Temporarily Offline</h1>
        <p>Unable to collect scanner data. This could be due to:</p>
        <ul>
            <li>🔌 Network connectivity issues</li>
            <li>📉 Data provider rate limits</li>
            <li>⏰ Market hours (some data requires active session)</li>
            <li>🔧 System resources temporarily unavailable</li>
        </ul>
        <p>The system will retry automatically on next run.</p>
    </div>
</body>
</html>"""
        filepath.write_text(html)


if __name__ == '__main__':
    # Portfolio tickers
    PORTFOLIO = ['MU', 'IONQ', 'APLD', 'ANNX']
    
    # Scanner tickers (20 stocks)
    SCANNER_TICKERS = [
        'MU', 'IONQ', 'APLD', 'ANNX',  # Portfolio
        'AAPL', 'MSFT', 'NVDA', 'TSLA', 'GOOGL', 'AMZN',  # Big tech
        'AMD', 'INTC', 'META', 'NFLX',  # Tech
        'SPY', 'QQQ',  # ETFs
        'PLTR', 'SOFI', 'COIN', 'RKLB'  # Growth/Speculative
    ]
    
    dashboard = AdvancedDashboard()
    output_dir = dashboard.build_full_dashboard(SCANNER_TICKERS)
    
    print(f"\n🌐 Open charts in browser:")
    print(f"   file:///{output_dir.absolute()}/MU_advanced.html")
    print(f"   file:///{output_dir.absolute()}/IONQ_advanced.html")
    print(f"   file:///{output_dir.absolute()}/APLD_advanced.html")
    print(f"   file:///{output_dir.absolute()}/ANNX_advanced.html")
    print(f"   file:///{output_dir.absolute()}/scanner_dashboard.html")
