"""
🚀 FINAL PROFIT DASHBOARD - COMPLETE VERSION
- Search ANY ticker anytime
- Add/remove tickers anywhere
- Editable portfolio with fractional shares
- All modules integrated
- AI recommender with patterns & forecasts
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
import os
import json

# Add backend modules to path
sys.path.insert(0, 'backend/modules')

# Import ALL modules
try:
    from universal_scraper_engine import UniversalScraperEngine
    from dark_pool_tracker import DarkPoolTracker
    from social_sentiment_explosion_detector import SocialSentimentExplosionDetector
    from insider_trading_tracker import InsiderTradingTracker
    from earnings_surprise_predictor import EarningsSurprisePredictor
    from short_squeeze_scanner import ShortSqueezeScanner
    from penny_stock_pump_detector import PennyStockPumpDetector
    from portfolio_manager import PortfolioManager
    
    MODULES_LOADED = True
except Exception as e:
    st.error(f"Error loading modules: {e}")
    MODULES_LOADED = False

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="💰 Quantum AI Cockpit - Your Money Machine",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CYBERPUNK STYLING
# ============================================================================

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a2e 100%);
    }
    
    h1, h2, h3 {
        color: #00ff9f !important;
        text-shadow: 0 0 10px #00ff9f;
        font-family: 'Courier New', monospace;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px;
        color: #00ff9f;
        text-shadow: 0 0 5px #00ff9f;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00ff9f, #00d4ff);
        color: #000;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 20px;
        box-shadow: 0 0 20px #00ff9f;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px #00ff9f;
    }
    
    .profit-card {
        background: rgba(0, 255, 159, 0.05);
        border: 1px solid #00ff9f;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 0 20px rgba(0, 255, 159, 0.1);
    }
    
    .ticker-search {
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        background: rgba(0, 255, 159, 0.1);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'TSLA', 'NVDA', 'AMD', 'MU', 'ELWS']

if 'portfolio_manager' not in st.session_state:
    st.session_state.portfolio_manager = PortfolioManager() if MODULES_LOADED else None

if 'current_ticker' not in st.session_state:
    st.session_state.current_ticker = 'AAPL'

if 'analysis_cache' not in st.session_state:
    st.session_state.analysis_cache = {}

# ============================================================================
# INITIALIZE MODULES
# ============================================================================

@st.cache_resource
def load_modules():
    if not MODULES_LOADED:
        return None
    
    return {
        'scraper': UniversalScraperEngine(),
        'dark_pool': DarkPoolTracker(),
        'social_explosion': SocialSentimentExplosionDetector(),
        'insider': InsiderTradingTracker(),
        'earnings': EarningsSurprisePredictor(),
        'squeeze': ShortSqueezeScanner(),
        'pump_detector': PennyStockPumpDetector(),
    }

modules = load_modules()

# ============================================================================
# HEADER - GLOBAL TICKER SEARCH
# ============================================================================

st.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h1 style="font-size: 48px;">🚀 QUANTUM AI COCKPIT</h1>
    <p style="color: #00ff9f; font-size: 18px;">Your AI-Powered Money-Making Command Center</p>
</div>
""", unsafe_allow_html=True)

# MASSIVE TICKER SEARCH BAR
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown('<div class="ticker-search">', unsafe_allow_html=True)
    
    search_ticker = st.text_input(
        "🔍 SEARCH ANY TICKER",
        value=st.session_state.current_ticker,
        key="global_search",
        label_visibility="collapsed",
        placeholder="Enter ticker (e.g., AAPL, TSLA, GME)..."
    ).upper()
    
    col_a, col_b, col_c = st.columns([1, 1, 1])
    
    with col_a:
        if st.button("📊 ANALYZE", use_container_width=True):
            if search_ticker:
                st.session_state.current_ticker = search_ticker
                st.rerun()
    
    with col_b:
        if st.button("➕ ADD TO WATCHLIST", use_container_width=True):
            if search_ticker and search_ticker not in st.session_state.watchlist:
                st.session_state.watchlist.append(search_ticker)
                st.success(f"✅ Added {search_ticker} to watchlist!")
                st.rerun()
    
    with col_c:
        if st.button("💼 ADD TO PORTFOLIO", use_container_width=True):
            st.session_state.show_add_position = True
    
    st.markdown('</div>', unsafe_allow_html=True)

ticker = st.session_state.current_ticker

st.markdown("---")

# ============================================================================
# SIDEBAR - PORTFOLIO & WATCHLIST MANAGER
# ============================================================================

with st.sidebar:
    st.markdown("## 💼 YOUR PORTFOLIO")
    
    if st.session_state.portfolio_manager:
        pm = st.session_state.portfolio_manager
        summary = pm.get_portfolio_summary()
        
        # Portfolio summary
        st.metric("Total Value", f"${summary['total_value']:,.2f}")
        st.metric("Total P&L", f"${summary['total_pnl']:,.2f}", f"{summary['total_pnl_pct']:+.2f}%")
        st.metric("Cash", f"${summary['cash']:,.2f}")
        
        # Show positions
        if summary['positions']:
            st.markdown("### 📈 Positions")
            for ticker_p, pos in summary['positions'].items():
                with st.expander(f"{ticker_p} ({pos['shares']:.3f} shares)"):
                    st.write(f"**Avg Cost:** ${pos['avg_cost']:.2f}")
                    st.write(f"**Current:** ${pos['current_price']:.2f}")
                    st.write(f"**Value:** ${pos['total_value']:,.2f}")
                    st.write(f"**P&L:** ${pos['pnl']:,.2f} ({pos['pnl_pct']:+.2f}%)")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📊", key=f"view_{ticker_p}"):
                            st.session_state.current_ticker = ticker_p
                            st.rerun()
                    with col2:
                        if st.button("❌", key=f"remove_{ticker_p}"):
                            pm.remove_position(ticker_p)
                            st.rerun()
        
        # Add position form
        with st.expander("➕ Add Position"):
            new_ticker = st.text_input("Ticker", key="new_pos_ticker").upper()
            new_shares = st.number_input("Shares", min_value=0.001, step=0.1, format="%.3f", key="new_shares")
            new_cost = st.number_input("Avg Cost $", min_value=0.01, step=0.01, format="%.2f", key="new_cost")
            
            if st.button("Add Position"):
                if new_ticker and new_shares > 0 and new_cost > 0:
                    pm.add_position(new_ticker, new_shares, new_cost)
                    st.success(f"Added {new_shares:.3f} shares of {new_ticker}!")
                    st.rerun()
        
        # Update cash
        with st.expander("💵 Update Cash"):
            new_cash = st.number_input("Cash Balance $", min_value=0.0, value=summary['cash'], step=100.0, format="%.2f")
            if st.button("Update Cash"):
                pm.update_cash(new_cash)
                st.success(f"Cash updated to ${new_cash:,.2f}")
                st.rerun()
    
    st.markdown("---")
    
    # Watchlist
    st.markdown("## 👀 WATCHLIST")
    
    # Add to watchlist
    with st.expander("➕ Add Ticker"):
        new_watch = st.text_input("Ticker", key="add_watch").upper()
        if st.button("Add to Watchlist", key="add_watch_btn"):
            if new_watch and new_watch not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_watch)
                st.success(f"Added {new_watch}!")
                st.rerun()
    
    # Show watchlist
    for wl_ticker in st.session_state.watchlist:
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            if st.button(f"📊 {wl_ticker}", key=f"wl_view_{wl_ticker}", use_container_width=True):
                st.session_state.current_ticker = wl_ticker
                st.rerun()
        with col2:
            if st.button("🔍", key=f"wl_analyze_{wl_ticker}"):
                st.session_state.current_ticker = wl_ticker
                st.session_state.show_full_analysis = True
                st.rerun()
        with col3:
            if st.button("❌", key=f"wl_del_{wl_ticker}"):
                st.session_state.watchlist.remove(wl_ticker)
                st.rerun()
    
    st.markdown("---")
    
    # Quick actions
    st.markdown("## ⚡ QUICK ACTIONS")
    
    if st.button("🔍 Scan All Watchlist", use_container_width=True):
        st.session_state.scan_mode = True
        st.rerun()
    
    if st.button("💰 Sync Robinhood", use_container_width=True):
        st.info("Enter Robinhood credentials in Portfolio section")
    
    if st.button("🎓 Train Models", use_container_width=True):
        st.info("Run: python OVERNIGHT_TRAINING_SYSTEM.py")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if not MODULES_LOADED:
    st.error("⚠️ Modules not loaded. Check backend/modules directory.")
    st.stop()

# Get stock data
@st.cache_data(ttl=300)
def get_stock_data(ticker_symbol, days=180):
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(ticker_symbol, start=start_date, end=end_date, progress=False)
        return df
    except:
        return None

df = get_stock_data(ticker)

if df is None or df.empty:
    st.error(f"❌ Could not fetch data for {ticker}")
    st.markdown("### Try another ticker:")
    st.stop()

# Current metrics
current_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2]
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

# ============================================================================
# TOP METRICS
# ============================================================================

st.markdown(f"## 📊 {ticker} Analysis")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label=f"💰 {ticker} Price",
        value=f"${current_price:.2f}",
        delta=f"{price_change_pct:+.2f}%"
    )

with col2:
    day_high = df['High'].iloc[-1]
    day_low = df['Low'].iloc[-1]
    st.metric("📊 Day Range", f"${day_low:.2f} - ${day_high:.2f}")

with col3:
    volume = df['Volume'].iloc[-1]
    avg_volume = df['Volume'].mean()
    volume_ratio = volume / avg_volume
    st.metric("📈 Volume", f"{volume/1e6:.1f}M", f"{(volume_ratio-1)*100:+.1f}%")

with col4:
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    st.metric("⚡ RSI", f"{current_rsi:.1f}", "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral")

with col5:
    ma50 = df['Close'].rolling(50).mean().iloc[-1]
    distance_from_ma = ((current_price / ma50) - 1) * 100
    st.metric("📉 vs MA50", f"{distance_from_ma:+.1f}%", "Above" if distance_from_ma > 0 else "Below")

st.markdown("---")

# ============================================================================
# AI ANALYSIS SECTION
# ============================================================================

st.markdown("## 🤖 AI INTELLIGENCE REPORT")

with st.spinner(f"Analyzing {ticker} with ALL modules..."):
    # Get comprehensive analysis
    all_data = modules['scraper'].scrape_all(ticker)
    dp_result = modules['dark_pool'].analyze_ticker(ticker)
    social_result = modules['social_explosion'].analyze_ticker(ticker)
    insider_result = modules['insider'].analyze_ticker(ticker)
    earnings_result = modules['earnings'].analyze_ticker(ticker)
    squeeze_result = modules['squeeze'].analyze_ticker(ticker)
    
    # Calculate master score
    signals = []
    
    if all_data['master_signal']['signal'] in ['STRONG_BUY', 'BUY']:
        signals.append(3)
    if dp_result['signal'] in ['STRONG_BUY', 'BUY']:
        signals.append(2)
    if social_result['signal'] in ['EXPLOSION_IMMINENT', 'HIGH_BUZZ']:
        signals.append(2)
    if insider_result['signal'] in ['STRONG_BUY', 'BUY']:
        signals.append(2)
    if squeeze_result['signal'] in ['EXTREME_SQUEEZE', 'HIGH_SQUEEZE']:
        signals.append(3)
    
    total_score = sum(signals)
    max_score = 12
    
    # Final recommendation
    if total_score >= 9:
        recommendation = "🚀 STRONG BUY"
        rec_color = "#00ff00"
        action = "ENTER POSITION NOW - High confidence opportunity!"
    elif total_score >= 6:
        recommendation = "✅ BUY"
        rec_color = "#00ff9f"
        action = "Good buying opportunity - Consider entry"
    elif total_score >= 3:
        recommendation = "⚠️ WATCH"
        rec_color = "#ffaa00"
        action = "Monitor closely - Wait for better setup"
    else:
        recommendation = "❌ AVOID"
        rec_color = "#ff4444"
        action = "Not a good opportunity right now"
    
    # Display recommendation
    st.markdown(f"""
    <div class="profit-card">
        <h2 style="color: {rec_color}; text-align: center; font-size: 36px;">
            {recommendation}
        </h2>
        <h3 style="text-align: center;">Score: {total_score}/{max_score}</h3>
        <p style="text-align: center; font-size: 18px;">{action}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Detailed signals
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 💰 Dark Pool")
        st.markdown(f"**{dp_result['alert']}**")
        if 'smart_money_score' in dp_result:
            st.progress(min(dp_result['smart_money_score'] / 100, 1.0))
        
        st.markdown("### 👔 Insider Trading")
        st.markdown(f"**{insider_result['alert']}**")
        if 'buy_trades' in insider_result:
            st.write(f"Buys: {insider_result['buy_trades']} | Sells: {insider_result['sell_trades']}")
    
    with col2:
        st.markdown("### 📱 Social Sentiment")
        st.markdown(f"**{social_result['alert']}**")
        st.progress(min(social_result['explosion_score'] / 100, 1.0))
        
        st.markdown("### 📰 News Sentiment")
        news = all_data['news']
        st.markdown(f"**{news.get('sentiment', 'NEUTRAL')}**")
        st.write(f"Articles: {news.get('articles_found', 0)}")
    
    with col3:
        st.markdown("### ⚡ Short Squeeze")
        st.markdown(f"**{squeeze_result['alert']}**")
        if 'short_float_pct' in squeeze_result:
            st.write(f"Short Float: {squeeze_result['short_float_pct']:.1f}%")
        
        st.markdown("### 📈 Earnings")
        st.markdown(f"**{earnings_result['alert']}**")

st.markdown("---")

# ============================================================================
# ADVANCED CHART
# ============================================================================

st.markdown("## 📊 ADVANCED CHART")

fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=(f'{ticker} Price & Patterns', 'Volume', 'RSI', 'MACD'),
    row_heights=[0.5, 0.15, 0.15, 0.2]
)

# Candlestick
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price',
        increasing_line_color='#00ff9f',
        decreasing_line_color='#ff4444'
    ),
    row=1, col=1
)

# Moving averages
ma20 = df['Close'].rolling(20).mean()
ma50 = df['Close'].rolling(50).mean()
fig.add_trace(go.Scatter(x=df.index, y=ma20, name='MA20', line=dict(color='cyan', width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=df.index, y=ma50, name='MA50', line=dict(color='yellow', width=1)), row=1, col=1)

# EMA Ribbon (glowing!)
for period, opacity in [(8, 0.4), (13, 0.3), (21, 0.2)]:
    ema = df['Close'].ewm(span=period).mean()
    fig.add_trace(
        go.Scatter(
            x=df.index, y=ema, name=f'EMA{period}',
            line=dict(color='#00ff9f', width=2),
            opacity=opacity,
            fill='tonexty' if period > 8 else None
        ),
        row=1, col=1
    )

# Volume
colors = ['#00ff9f' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4444' for i in range(len(df))]
fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, showlegend=False), row=2, col=1)

# RSI
fig.add_trace(go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='#00d4ff', width=2)), row=3, col=1)
fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)

# MACD
exp1 = df['Close'].ewm(span=12, adjust=False).mean()
exp2 = df['Close'].ewm(span=26, adjust=False).mean()
macd = exp1 - exp2
signal = macd.ewm(span=9, adjust=False).mean()
histogram = macd - signal

fig.add_trace(go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='#00ff9f', width=2)), row=4, col=1)
fig.add_trace(go.Scatter(x=df.index, y=signal, name='Signal', line=dict(color='#ff4444', width=2)), row=4, col=1)
fig.add_trace(go.Bar(x=df.index, y=histogram, name='Histogram', marker_color='#00d4ff', showlegend=False), row=4, col=1)

# Style
fig.update_layout(
    height=1000,
    template='plotly_dark',
    paper_bgcolor='rgba(10,14,39,0.9)',
    plot_bgcolor='rgba(10,14,39,0.9)',
    xaxis_rangeslider_visible=False,
    font=dict(color='#00ff9f', family='Courier New')
)

st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center;">
    <p style="color: #00ff9f;">🚀 Quantum AI Cockpit - Built to Make Money</p>
    <p style="font-size: 12px; color: #666;">Real-time data | No API limits | Free forever</p>
</div>
""", unsafe_allow_html=True)

