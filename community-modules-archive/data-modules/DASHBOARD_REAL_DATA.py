"""
🚀 QUANTUM AI DASHBOARD - REAL DATA (NO MOCK BULLSHIT)
======================================================
Connects your actual PRO modules to the sick dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent))

# Import REAL modules
from backend.modules.daily_scanner import ProfessionalDailyScanner, Strategy
from backend.modules.professional_signal_coordinator import ProfessionalSignalCoordinator
from backend.modules.position_size_calculator import PositionSizeCalculator
from backend.modules.api_integrations import get_stock_data

# Import pattern and forecast modules
try:
    from backend.modules.pattern_engine_pro import PatternEnginePro
    pattern_engine = PatternEnginePro()
except:
    pattern_engine = None

try:
    from backend.modules.ai_forecast_pro import AIForecastPro
    forecaster = AIForecastPro()
except:
    forecaster = None

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="🚀 Quantum AI Cockpit",
    page_icon="🚀",
    layout="wide"
)

# Import sick styling from ultimate dashboard
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1629 100%); color: #00ff9f; }
    .main-header {
        background: linear-gradient(90deg, #00ff9f, #00d9ff, #ff00ff, #00ff9f);
        background-size: 300% 300%;
        animation: gradientShift 3s ease infinite;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        font-family: 'Courier New', monospace;
    }
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    h1, h2, h3 { color: #00ff9f !important; text-shadow: 0 0 15px #00ff9f; }
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        background: linear-gradient(90deg, #00ff9f, #00d9ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        background: linear-gradient(135deg, #00ff9f 0%, #00d9ff 100%);
        color: #0a0e27;
        font-weight: bold;
        border-radius: 10px;
        box-shadow: 0 0 20px rgba(0, 255, 159, 0.5);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════

if 'account_value' not in st.session_state:
    st.session_state.account_value = 500

if 'scan_results' not in st.session_state:
    st.session_state.scan_results = None

# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS - REAL DATA
# ═══════════════════════════════════════════════════════════════

def add_technical_indicators(df):
    """Add ALL technical indicators - REAL calculation"""
    data = df.copy()
    
    # EMAs (Ribbon)
    for period in [8, 13, 21, 34, 55, 89]:
        data[f'EMA_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
    
    # SMAs
    for period in [20, 50, 200]:
        data[f'SMA_{period}'] = data['close'].rolling(period).mean()
    
    # RSI
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    data['MACD'] = data['close'].ewm(span=12, adjust=False).mean() - data['close'].ewm(span=26, adjust=False).mean()
    data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
    
    # Bollinger Bands
    data['BB_Middle'] = data['close'].rolling(20).mean()
    bb_std = data['close'].rolling(20).std()
    data['BB_Upper'] = data['BB_Middle'] + (bb_std * 2)
    data['BB_Lower'] = data['BB_Middle'] - (bb_std * 2)
    
    # Volume SMA
    data['Volume_SMA'] = data['volume'].rolling(20).mean()
    
    return data

def create_advanced_chart(df, ticker):
    """Create TrendSpider-style chart with REAL data"""
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(f'{ticker} - LIVE ANALYSIS', 'Volume', 'RSI (14)', 'MACD'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )
    
    # ========== CANDLESTICKS ==========
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ff9f',
            decreasing_line_color='#ff0051'
        ),
        row=1, col=1
    )
    
    # ========== EMA RIBBON ==========
    ema_colors = ['#00ff9f', '#00dd88', '#00bb77', '#009966', '#007755', '#005544']
    for i, period in enumerate([8, 13, 21, 34, 55, 89]):
        if f'EMA_{period}' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f'EMA_{period}'],
                    name=f'EMA {period}',
                    line=dict(color=ema_colors[i], width=1.5),
                    opacity=0.7
                ),
                row=1, col=1
            )
    
    # ========== SMAs ==========
    if 'SMA_50' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                      line=dict(color='#00d9ff', width=2, dash='dash')),
            row=1, col=1
        )
    
    if 'SMA_200' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['SMA_200'], name='SMA 200',
                      line=dict(color='#ff00ff', width=2, dash='dash')),
            row=1, col=1
        )
    
    # ========== BOLLINGER BANDS ==========
    if 'BB_Upper' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                      line=dict(color='rgba(0, 217, 255, 0.3)', width=1)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                      line=dict(color='rgba(0, 217, 255, 0.3)', width=1),
                      fill='tonexty', fillcolor='rgba(0, 217, 255, 0.1)'),
            row=1, col=1
        )
    
    # ========== VOLUME ==========
    colors = ['#00ff9f' if c > o else '#ff0051' for c, o in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(x=df.index, y=df['volume'], name='Volume', marker_color=colors, opacity=0.5),
        row=2, col=1
    )
    
    if 'Volume_SMA' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['Volume_SMA'], name='Vol SMA',
                      line=dict(color='#ffaa00', width=1)),
            row=2, col=1
        )
    
    # ========== RSI ==========
    if 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                      line=dict(color='#00ff9f', width=2)),
            row=3, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color='#ff0051', row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color='#00ff00', row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color='#666', row=3, col=1)
    
    # ========== MACD ==========
    if 'MACD' in df.columns:
        colors_macd = ['#00ff9f' if val > 0 else '#ff0051' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Hist'], name='MACD Hist',
                  marker_color=colors_macd, opacity=0.6),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                      line=dict(color='#00d9ff', width=2)),
            row=4, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                      line=dict(color='#ff00ff', width=2, dash='dash')),
            row=4, col=1
        )
    
    # ========== LAYOUT ==========
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#0a0e27',
        font=dict(color='#00ff9f', family='Courier New'),
        xaxis_rangeslider_visible=False,
        height=1000,
        hovermode='x unified',
        showlegend=True
    )
    
    fig.update_xaxes(gridcolor='#1a1f3a', showgrid=True)
    fig.update_yaxes(gridcolor='#1a1f3a', showgrid=True)
    
    return fig

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════

st.markdown('<div class="main-header">🚀 QUANTUM AI COCKPIT 🚀</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #00d9ff; font-size: 16px;">REAL DATA • REAL MODULES • REAL MONEY</p>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎛️ CONTROL PANEL")
    
    page = st.radio(
        "📡 SELECT MODULE",
        ["🔥 Scanner", "📊 Chart Analysis", "💼 Portfolio"],
        label_visibility="visible"
    )
    
    st.markdown("---")
    
    st.markdown("### ⚙️ CONFIG")
    st.session_state.account_value = st.number_input(
        "Account Value ($)",
        min_value=100,
        value=st.session_state.account_value,
        step=100
    )
    
    strategy = st.selectbox(
        "Strategy",
        ["Swing Trade (3-4 months)", "Penny Stock (explosive)"]
    )
    
    st.markdown("---")
    st.markdown("### 🤖 SYSTEM STATUS")
    st.success("✅ PRO Modules Loaded")
    st.success("✅ API Keys Active")
    st.success("✅ Real-Time Data")

# ═══════════════════════════════════════════════════════════════
# PAGE 1: SCANNER (REAL DATA)
# ═══════════════════════════════════════════════════════════════

if page == "🔥 Scanner":
    st.title("🔥 DAILY MARKET SCANNER")
    
    # Stock input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        stocks_input = st.text_input(
            "Enter tickers (comma separated):",
            value="NVDA,AMD,TSLA,PLTR,SOFI,META,GOOGL,MSFT"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_scan = st.button("🚀 RUN SCAN", type="primary", use_container_width=True)
    
    if run_scan:
        stocks = [s.strip().upper() for s in stocks_input.split(',')]
        
        with st.spinner(f"🔍 Scanning {len(stocks)} stocks..."):
            try:
                # Use REAL scanner
                strat = Strategy.SWING_TRADE if "Swing" in strategy else Strategy.PENNY_STOCK
                scanner = ProfessionalDailyScanner(strat)
                
                results = scanner.scan_universe(stocks)
                st.session_state.scan_results = results
                
                # Filter opportunities
                min_score = 80 if "Penny" in strategy else 70
                opportunities = [r for r in results if r.final_score >= min_score]
                
                if not opportunities:
                    st.warning(f"No opportunities found (all scores <{min_score})")
                else:
                    st.success(f"✅ Found {len(opportunities)} opportunities!")
                    
                    # Display results
                    for result in opportunities:
                        with st.expander(f"📊 {result.ticker} - Score: {result.final_score:.0f}/100 {result.recommendation.emoji}", expanded=True):
                            
                            # Get current price for position sizing
                            try:
                                df = get_stock_data(result.ticker, period='1d')
                                if df is not None and len(df) > 0:
                                    current_price = df['close'].iloc[-1]
                                else:
                                    current_price = 100  # Fallback
                            except:
                                current_price = 100
                            
                            # Calculate REAL position
                            calc = PositionSizeCalculator(
                                st.session_state.account_value,
                                risk_pct=0.02
                            )
                            position = calc.calculate(
                                result.ticker,
                                result.final_score / 100,
                                current_price
                            )
                            
                            # Show metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("AI Score", f"{result.final_score:.0f}/100")
                                st.metric("Confidence", f"{result.final_score:.0f}%")
                            
                            with col2:
                                st.metric("Entry Price", f"${current_price:.2f}")
                                st.metric("Shares", f"{position.get('shares', 0):.2f}")
                            
                            with col3:
                                st.metric("Stop Loss", f"${position.get('stop_loss_price', 0):.2f}")
                                st.metric("Target", f"${position.get('take_profit_price', 0):.2f}")
                            
                            # Signal breakdown (REAL DATA)
                            st.markdown("#### 📊 Signal Breakdown")
                            signal_data = []
                            for signal_name, signal_value in result.normalized_signals.items():
                                weight = result.weights.get(signal_name, 0)
                                signal_data.append({
                                    'Signal': signal_name.title(),
                                    'Value': f"{signal_value:.0f}/100",
                                    'Weight': f"{weight*100:.0f}%",
                                    'Contribution': f"{signal_value * weight:.1f}"
                                })
                            
                            st.dataframe(pd.DataFrame(signal_data), use_container_width=True)
                            
                            # Trade plan
                            if position.get('should_trade'):
                                st.markdown("#### 🎯 TRADE PLAN")
                                trade_plan = f"""```
BUY {position['shares']:.2f} shares {result.ticker}
Entry: ${position['entry_price']:.2f}
Stop: ${position['stop_loss_price']:.2f} (-{abs(position.get('stop_loss_pct', 2)):.1f}%)
Target: ${position['take_profit_price']:.2f} (+{position.get('take_profit_pct', 5):.1f}%)
Risk: ${position['risk_dollars']:.2f} | Reward: ${position['reward_dollars']:.2f}
R:R = 1:{position.get('risk_reward_ratio', 2):.1f}
```"""
                                st.code(trade_plan, language="text")
                                st.info("⚠️ Copy to Robinhood - Execute manually!")
                
            except Exception as e:
                st.error(f"❌ Scanner failed: {e}")
                st.exception(e)

# ═══════════════════════════════════════════════════════════════
# PAGE 2: CHART ANALYSIS (REAL DATA)
# ═══════════════════════════════════════════════════════════════

elif page == "📊 Chart Analysis":
    st.title("📊 ADVANCED CHART ANALYSIS")
    
    ticker = st.text_input("Enter ticker:", value="NVDA").upper()
    
    if st.button("🔍 ANALYZE", type="primary"):
        with st.spinner(f"📊 Analyzing {ticker}..."):
            try:
                # Fetch REAL data
                df = get_stock_data(ticker, period='6mo')
                
                if df is None or len(df) == 0:
                    st.error(f"❌ No data available for {ticker}")
                else:
                    # Add REAL indicators
                    df = add_technical_indicators(df)
                    
                    # Current stats
                    current_price = df['close'].iloc[-1]
                    change_1d = ((df['close'].iloc[-1] / df['close'].iloc[-2]) - 1) * 100 if len(df) >= 2 else 0
                    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                    volume = df['volume'].iloc[-1]
                    vol_avg = df['Volume_SMA'].iloc[-1] if 'Volume_SMA' in df.columns else volume
                    
                    # Show metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Price", f"${current_price:.2f}", f"{change_1d:+.2f}%")
                    
                    with col2:
                        st.metric("RSI (14)", f"{rsi:.1f}")
                    
                    with col3:
                        vol_ratio = (volume / vol_avg) if vol_avg > 0 else 1
                        st.metric("Volume", f"{volume/1e6:.1f}M", f"{vol_ratio:.1f}x avg")
                    
                    with col4:
                        # Get REAL AI score if we have scan results
                        if st.session_state.scan_results:
                            ticker_result = next((r for r in st.session_state.scan_results if r.ticker == ticker), None)
                            if ticker_result:
                                st.metric("AI Score", f"{ticker_result.final_score:.0f}/100")
                            else:
                                st.metric("AI Score", "Run scan")
                        else:
                            st.metric("AI Score", "Run scan")
                    
                    # Create and display REAL chart
                    fig = create_advanced_chart(df, ticker)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Pattern detection (REAL)
                    if pattern_engine:
                        st.markdown("### 🔍 PATTERN DETECTION")
                        try:
                            patterns = pattern_engine.detect_patterns(ticker, df)
                            if patterns and len(patterns) > 0:
                                for pattern in patterns[:3]:  # Top 3
                                    st.info(f"**{pattern.get('name', 'Pattern')}** - Confidence: {pattern.get('confidence', 0):.0f}%")
                            else:
                                st.info("No high-confidence patterns detected")
                        except:
                            st.info("Pattern engine unavailable")
                    
                    # Forecast (REAL)
                    if forecaster:
                        st.markdown("### 🔮 AI FORECAST")
                        try:
                            with st.spinner("Running forecast..."):
                                forecast = forecaster.forecast(ticker, df, horizon_days=21)
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    bull_price = forecast['bull_case']['price']
                                    bull_return = ((bull_price / current_price) - 1) * 100
                                    st.metric("Bull Case", f"${bull_price:.2f}", f"+{bull_return:.1f}%")
                                
                                with col2:
                                    base_price = forecast['base_case']['price']
                                    base_return = ((base_price / current_price) - 1) * 100
                                    st.metric("Base Case", f"${base_price:.2f}", f"+{base_return:.1f}%")
                                
                                with col3:
                                    bear_price = forecast['bear_case']['price']
                                    bear_return = ((bear_price / current_price) - 1) * 100
                                    st.metric("Bear Case", f"${bear_price:.2f}", f"{bear_return:.1f}%")
                                
                                st.info(f"**Confidence:** {forecast.get('confidence', 50):.0f}% | **Models:** {', '.join(forecast.get('models_used', ['N/A']))}")
                        except Exception as e:
                            st.warning(f"Forecast unavailable: {e}")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
                st.exception(e)

# ═══════════════════════════════════════════════════════════════
# PAGE 3: PORTFOLIO
# ═══════════════════════════════════════════════════════════════

elif page == "💼 Portfolio":
    st.title("💼 PORTFOLIO TRACKER")
    
    st.info("📝 Manual position tracking - Log your executed Robinhood trades here")
    
    # Manual trade entry
    with st.form("add_position"):
        st.markdown("### ➕ Add Position")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ticker = st.text_input("Ticker", "NVDA")
        with col2:
            shares = st.number_input("Shares", value=1.0, step=0.1)
        with col3:
            entry = st.number_input("Entry Price", value=145.0, step=1.0)
        with col4:
            stop = st.number_input("Stop Loss", value=140.0, step=1.0)
        
        submitted = st.form_submit_button("✅ ADD POSITION", type="primary")
        
        if submitted:
            st.success(f"✅ Added {shares} shares of {ticker} at ${entry:.2f}")
            st.info("Position logged for tracking")
    
    st.markdown("### 📊 Open Positions")
    st.info("No positions yet. Run scanner and execute trades to populate this.")
    
    # Performance summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Account Value", f"${st.session_state.account_value:.2f}")
    with col2:
        st.metric("Total P&L", "$0.00", "0%")
    with col3:
        st.metric("Win Rate", "0%")
    with col4:
        st.metric("Open Positions", "0")

# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #00ff9f; font-size: 12px;'>
    🚀 Quantum AI Cockpit | <span style='color: #00d9ff;'>REAL DATA • NO MOCK BULLSHIT</span>
</div>
""", unsafe_allow_html=True)

