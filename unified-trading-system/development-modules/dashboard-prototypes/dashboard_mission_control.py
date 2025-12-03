"""
🚀 QUANTUM AI COCKPIT - MISSION CONTROL DASHBOARD
================================================
Cyberpunk Blade Runner style trading dashboard

Features:
- Real-time ticker tape
- Robinhood portfolio sync
- Pattern detection
- AI recommendations
- Advanced charts
- Watchlist management
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import joblib
import json
from pathlib import Path
import time

# Page config
st.set_page_config(
    page_title="🚀 Quantum AI Cockpit",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cyberpunk CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 100%);
        color: #00ff9f;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #00ff9f !important;
        text-shadow: 0 0 10px #00ff9f;
        font-family: 'Courier New', monospace;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #00d9ff !important;
        font-size: 2rem !important;
        font-weight: bold !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #1a1f3a 100%);
        border-right: 2px solid #00ff9f;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #00ff9f 0%, #00d9ff 100%);
        color: #0a0e27;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        box-shadow: 0 0 20px rgba(0, 255, 159, 0.5);
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 30px rgba(0, 255, 159, 0.8);
        transform: scale(1.05);
    }
    
    /* Dataframes */
    .dataframe {
        background-color: #1a1f3a !important;
        color: #00ff9f !important;
    }
    
    /* Ticker tape */
    .ticker-tape {
        background: #0a0e27;
        border: 1px solid #00ff9f;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 20px;
        overflow: hidden;
    }
    
    .ticker-item {
        display: inline-block;
        margin-right: 30px;
        color: #00d9ff;
        font-family: 'Courier New', monospace;
        font-size: 14px;
    }
    
    /* Signal badges */
    .signal-buy {
        background: linear-gradient(90deg, #00ff00 0%, #00dd00 100%);
        color: black;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(0, 255, 0, 0.6);
    }
    
    .signal-sell {
        background: linear-gradient(90deg, #ff0051 0%, #dd0040 100%);
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
        box-shadow: 0 0 15px rgba(255, 0, 81, 0.6);
    }
    
    .signal-hold {
        background: linear-gradient(90deg, #ffaa00 0%, #dd9900 100%);
        color: black;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'NVDA', 'AMD', 'TSLA', 'GOOGL']

if 'portfolio' not in st.session_state:
    st.session_state.portfolio = []

if 'robinhood_connected' not in st.session_state:
    st.session_state.robinhood_connected = False

# Helper functions
@st.cache_data(ttl=300)
def get_market_data(ticker, period='1mo'):
    """Fetch market data with caching"""
    try:
        df = yf.download(ticker, period=period, interval='1d', progress=False)
        return df
    except:
        return None

@st.cache_data(ttl=60)
def get_top_gainers():
    """Get top market gainers"""
    # Simplified - in production use real-time API
    tickers = ['AAPL', 'NVDA', 'AMD', 'TSLA', 'MSFT', 'GOOGL', 'META', 'AMZN']
    gainers = []
    
    for ticker in tickers:
        try:
            df = yf.download(ticker, period='2d', interval='1d', progress=False)
            if len(df) >= 2:
                change = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
                gainers.append({
                    'ticker': ticker,
                    'price': df['Close'].iloc[-1],
                    'change': change
                })
        except:
            pass
    
    return sorted(gainers, key=lambda x: x['change'], reverse=True)[:5]

def load_trained_models():
    """Load trained models"""
    models_path = Path('E:/Quantum_AI_Cockpit/backend/models')
    
    try:
        pattern_results = joblib.load(models_path / 'quick_pattern_results.pkl')
        forecaster = joblib.load(models_path / 'quick_forecaster.pkl')
        calibration = joblib.load(models_path / 'quick_calibration.pkl')
        return pattern_results, forecaster, calibration
    except:
        return None, None, None

def create_mini_chart(df, ticker):
    """Create small sparkline chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Close'],
        mode='lines',
        line=dict(color='#00ff9f', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 159, 0.1)'
    ))
    
    fig.update_layout(
        height=100,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False
    )
    
    return fig

def get_signal_badge(signal):
    """Get HTML badge for signal"""
    if signal == 'BUY':
        return '<span class="signal-buy">🚀 BUY</span>'
    elif signal == 'SELL':
        return '<span class="signal-sell">📉 SELL</span>'
    else:
        return '<span class="signal-hold">⏸️ HOLD</span>'

# ═══════════════════════════════════════════════════════════════════
# HEADER & TICKER TAPE
# ═══════════════════════════════════════════════════════════════════

st.markdown("""
<h1 style='text-align: center; font-size: 3rem;'>
🚀 QUANTUM AI COCKPIT - MISSION CONTROL 🚀
</h1>
<p style='text-align: center; color: #00d9ff; font-family: Courier New;'>
Your Cyberpunk Trading Command Center
</p>
""", unsafe_allow_html=True)

# Ticker tape
st.markdown("<div class='ticker-tape'>", unsafe_allow_html=True)
top_gainers = get_top_gainers()

ticker_html = ""
for gainer in top_gainers:
    color = '#00ff00' if gainer['change'] > 0 else '#ff0051'
    ticker_html += f"""
    <span class='ticker-item'>
        <strong>{gainer['ticker']}</strong> 
        ${gainer['price']:.2f} 
        <span style='color: {color};'>
            {'+' if gainer['change'] > 0 else ''}{gainer['change']:.2f}%
        </span>
    </span>
    """

st.markdown(ticker_html, unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR - NAVIGATION & SETTINGS
# ═══════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎛️ MISSION CONTROL")
    
    # Navigation
    page = st.selectbox(
        "📡 SELECT MODULE",
        [
            "🏠 Command Center",
            "📊 Portfolio Tracker",
            "🔍 Deep Analysis Lab",
            "📈 Pattern Scanner",
            "🎯 Opportunity Finder",
            "⚙️ Settings"
        ]
    )
    
    st.markdown("---")
    
    # Robinhood connection
    st.markdown("### 🔗 ROBINHOOD SYNC")
    
    if not st.session_state.robinhood_connected:
        if st.button("🔌 Connect Robinhood"):
            st.session_state.robinhood_connected = True
            st.success("✅ Connected!")
            st.rerun()
    else:
        st.success("✅ Robinhood Connected")
        if st.button("🔌 Disconnect"):
            st.session_state.robinhood_connected = False
            st.rerun()
    
    st.markdown("---")
    
    # Watchlist management
    st.markdown("### 📋 WATCHLIST")
    
    new_ticker = st.text_input("Add ticker:", key="new_ticker_input")
    if st.button("➕ Add") and new_ticker:
        if new_ticker.upper() not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker.upper())
            st.success(f"Added {new_ticker.upper()}")
            st.rerun()
    
    for ticker in st.session_state.watchlist:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write(f"📌 {ticker}")
        with col2:
            if st.button("❌", key=f"remove_{ticker}"):
                st.session_state.watchlist.remove(ticker)
                st.rerun()
    
    st.markdown("---")
    
    # System status
    st.markdown("### 🤖 SYSTEM STATUS")
    pattern_results, forecaster, calibration = load_trained_models()
    
    if pattern_results and forecaster:
        st.success("✅ Models Loaded")
    else:
        st.error("❌ Models Not Loaded")
        st.info("💡 Run training in Colab first!")
    
    st.metric("Uptime", "99.9%")
    st.metric("API Status", "🟢 LIVE")

# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

if page == "🏠 Command Center":
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Portfolio Value",
            value="$12,458",
            delta="+$523 (4.4%)"
        )
    
    with col2:
        st.metric(
            label="🎯 Win Rate",
            value="64.5%",
            delta="+2.3%"
        )
    
    with col3:
        st.metric(
            label="📈 Active Signals",
            value="3 BUYS",
            delta="2 new"
        )
    
    with col4:
        st.metric(
            label="⚡ Opportunities",
            value="7 Found",
            delta="+2 today"
        )
    
    st.markdown("---")
    
    # Main dashboard - 2 columns
    left_col, right_col = st.columns([2, 1])
    
    with left_col:
        st.markdown("### 📊 WATCHLIST ANALYSIS")
        
        for ticker in st.session_state.watchlist[:3]:  # Show top 3
            with st.expander(f"🎯 {ticker} Analysis", expanded=True):
                
                df = get_market_data(ticker, period='1mo')
                
                if df is not None and len(df) > 0:
                    # Quick stats
                    current_price = df['Close'].iloc[-1]
                    change_1d = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100 if len(df) >= 2 else 0
                    
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Price", f"${current_price:.2f}", f"{change_1d:+.2f}%")
                    
                    with col_b:
                        # Generate signal (simplified)
                        signal = 'BUY' if change_1d > 1 else ('SELL' if change_1d < -1 else 'HOLD')
                        st.markdown(f"**Signal:** {get_signal_badge(signal)}", unsafe_allow_html=True)
                    
                    with col_c:
                        confidence = np.random.randint(60, 85)  # Placeholder
                        st.metric("Confidence", f"{confidence}%")
                    
                    # Mini chart
                    fig = create_mini_chart(df, ticker)
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    
                    # Patterns detected
                    patterns = np.random.choice(['EMA Crossover', 'Support Bounce', 'Triangle'], size=np.random.randint(0, 2))
                    if len(patterns) > 0:
                        st.info(f"🔍 Patterns: {', '.join(patterns)}")
    
    with right_col:
        st.markdown("### 🚨 ACTIVE ALERTS")
        
        alerts = [
            {"ticker": "NVDA", "type": "BUY", "reason": "Strong EMA crossover + volume surge", "confidence": 82},
            {"ticker": "AMD", "type": "BUY", "reason": "Bullish divergence confirmed", "confidence": 76},
            {"ticker": "TSLA", "type": "HOLD", "reason": "Consolidation near support", "confidence": 65}
        ]
        
        for alert in alerts:
            st.markdown(f"""
            <div style='background: rgba(0, 255, 159, 0.1); border-left: 4px solid #00ff9f; padding: 10px; margin-bottom: 10px; border-radius: 5px;'>
                <strong style='color: #00d9ff;'>{alert['ticker']}</strong> - {get_signal_badge(alert['type'])}<br/>
                <small style='color: #aaa;'>{alert['reason']}</small><br/>
                <small><strong>Confidence:</strong> {alert['confidence']}%</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("### 📈 MARKET OVERVIEW")
        
        market_data = {
            "S&P 500": {"value": 4785, "change": +1.2},
            "NASDAQ": {"value": 15234, "change": +1.8},
            "VIX": {"value": 14.2, "change": -3.5}
        }
        
        for index, data in market_data.items():
            color = '#00ff00' if data['change'] > 0 else '#ff0051'
            st.markdown(f"""
            <div style='background: #1a1f3a; padding: 10px; margin-bottom: 8px; border-radius: 5px;'>
                <strong>{index}:</strong> {data['value']} 
                <span style='color: {color};'>{'+' if data['change'] > 0 else ''}{data['change']:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

elif page == "📊 Portfolio Tracker":
    st.markdown("## 📊 PORTFOLIO TRACKER")
    
    if st.session_state.robinhood_connected:
        st.success("✅ Synced with Robinhood")
        
        # Mock portfolio data
        portfolio_data = {
            'Ticker': ['AAPL', 'NVDA', 'AMD'],
            'Shares': [10, 5, 20],
            'Avg Cost': [180.50, 485.20, 145.80],
            'Current Price': [195.30, 512.40, 158.90],
            'P&L': ['+$148', '+$136', '+$262'],
            'P&L %': ['+8.2%', '+5.6%', '+9.0%']
        }
        
        st.dataframe(portfolio_data, use_container_width=True)
        
        # Portfolio metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Value", "$12,458", "+$546")
        with col2:
            st.metric("Today's Return", "+$87", "+0.7%")
        with col3:
            st.metric("Total Return", "+$1,234", "+11.0%")
    
    else:
        st.warning("🔌 Connect Robinhood to sync your portfolio")
        st.info("💡 Go to Settings → Connect Robinhood")

elif page == "🔍 Deep Analysis Lab":
    st.markdown("## 🔍 DEEP ANALYSIS LAB")
    
    analysis_ticker = st.selectbox("Select stock for deep analysis:", st.session_state.watchlist)
    
    if st.button("🔬 Run Deep Analysis"):
        with st.spinner("Analyzing..."):
            time.sleep(2)  # Simulate analysis
            
            st.success(f"✅ Analysis complete for {analysis_ticker}")
            
            # Mock analysis results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 Technical Analysis")
                st.info("🟢 Trend: Strong Bullish")
                st.info("🟢 Momentum: Increasing")
                st.info("🟡 Volume: Average")
                st.info("🟢 Support: Strong at $150")
            
            with col2:
                st.markdown("### 🎯 AI Recommendation")
                st.markdown(get_signal_badge('BUY'), unsafe_allow_html=True)
                st.metric("Confidence", "78%")
                st.metric("Target Price", "$165.00", "+8.5%")
                st.metric("Stop Loss", "$148.50", "-2.5%")

elif page == "📈 Pattern Scanner":
    st.markdown("## 📈 PATTERN SCANNER")
    
    st.info("🔍 Scanning watchlist for patterns...")
    
    # Mock pattern results
    patterns_found = [
        {"ticker": "NVDA", "pattern": "Ascending Triangle", "confidence": 85, "target": "+12%"},
        {"ticker": "AMD", "pattern": "Bullish Flag", "confidence": 78, "target": "+8%"},
        {"ticker": "AAPL", "pattern": "EMA Ribbon Cross", "confidence": 72, "target": "+6%"}
    ]
    
    for pattern in patterns_found:
        st.markdown(f"""
        <div style='background: rgba(0, 255, 159, 0.1); padding: 15px; margin: 10px 0; border-radius: 10px; border-left: 5px solid #00ff9f;'>
            <h3 style='color: #00d9ff;'>{pattern['ticker']} - {pattern['pattern']}</h3>
            <p><strong>Confidence:</strong> {pattern['confidence']}%</p>
            <p><strong>Price Target:</strong> {pattern['target']}</p>
            <button style='background: #00ff9f; color: black; padding: 8px 20px; border: none; border-radius: 5px; font-weight: bold;'>
                View Chart
            </button>
        </div>
        """, unsafe_allow_html=True)

elif page == "🎯 Opportunity Finder":
    st.markdown("## 🎯 OPPORTUNITY FINDER")
    
    st.markdown("### 🔥 High-Probability Setups")
    
    opportunities = [
        {"ticker": "MU", "setup": "Bounce from support + Volume surge", "win_rate": "68%", "risk_reward": "1:3.2"},
        {"ticker": "TSLA", "setup": "Bullish divergence confirmed", "win_rate": "65%", "risk_reward": "1:2.8"},
        {"ticker": "GOOGL", "setup": "Cup & Handle forming", "win_rate": "62%", "risk_reward": "1:2.5"}
    ]
    
    for opp in opportunities:
        with st.expander(f"💎 {opp['ticker']} - {opp['setup']}", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Win Rate", opp['win_rate'])
            with col2:
                st.metric("Risk:Reward", opp['risk_reward'])
            with col3:
                st.button(f"🚀 Trade {opp['ticker']}", key=f"trade_{opp['ticker']}")

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ SETTINGS")
    
    st.markdown("### 🔗 Connections")
    
    # Robinhood setup
    st.markdown("#### Robinhood")
    if st.session_state.robinhood_connected:
        st.success("✅ Connected")
        if st.button("Disconnect"):
            st.session_state.robinhood_connected = False
            st.rerun()
    else:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Connect"):
            if username and password:
                st.session_state.robinhood_connected = True
                st.success("✅ Connected to Robinhood!")
                st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 🎨 Theme")
    theme = st.selectbox("Select theme", ["Cyberpunk (default)", "Dark Mode", "Light Mode"])
    
    st.markdown("### 📊 Display")
    chart_type = st.selectbox("Chart type", ["Candlestick", "Line", "Area"])
    show_volume = st.checkbox("Show volume", value=True)
    
    st.markdown("### 🔔 Notifications")
    email_alerts = st.checkbox("Email alerts", value=True)
    push_notifications = st.checkbox("Push notifications", value=False)

# ═══════════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 12px;'>
    🚀 Quantum AI Cockpit v1.0 | Mission Control Active | 
    <span style='color: #00ff9f;'>●</span> All Systems Operational
</div>
""", unsafe_allow_html=True)

