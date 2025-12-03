# ============================================================================
# ULTIMATE PRODUCTION DASHBOARD - YOUR REAL DATA, NO MOCK SHIT
# ============================================================================
# Shows YOUR actual signals, YOUR actual trades, YOUR actual P&L
# No placeholders. No mock data. REAL ONLY.

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Quantum AI Cockpit",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .big-metric {
        font-size: 48px;
        font-weight: bold;
    }
    .profit {
        color: #00ff00;
    }
    .loss {
        color: #ff0000;
    }
    .pending {
        color: #ffaa00;
    }
    .stMetric {
        background: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD YOUR REAL DATA
# ============================================================================

@st.cache_data(ttl=60)
def load_real_data():
    """Load all YOUR real data - no mocking"""
    data = {}
    project_root = Path(__file__).parent
    
    # Your validated signals
    try:
        signals_file = project_root / 'data' / 'validated_signals.json'
        if not signals_file.exists():
            signals_file = project_root / 'data' / 'daily_signals.json'
        if signals_file.exists():
            with open(signals_file) as f:
                data['signals'] = json.load(f)
        else:
            data['signals'] = []
    except:
        data['signals'] = []
    
    # Your paper trades
    try:
        trades_file = project_root / 'logs' / 'paper_trades.txt'
        if trades_file.exists():
            with open(trades_file) as f:
                data['trades'] = f.readlines()
        else:
            data['trades'] = []
    except:
        data['trades'] = []
    
    # Your production config
    try:
        config_file = project_root / 'optimized_configs' / 'PRODUCTION_CONFIG.json'
        if config_file.exists():
            with open(config_file) as f:
                data['config'] = json.load(f)
        else:
            data['config'] = {}
    except:
        data['config'] = {}
    
    # Your portfolio tracker
    try:
        portfolio_file = project_root / 'portfolio_tracker.csv'
        if portfolio_file.exists():
            data['portfolio'] = pd.read_csv(portfolio_file)
        else:
            data['portfolio'] = pd.DataFrame()
    except:
        data['portfolio'] = pd.DataFrame()
    
    # Your backtest results
    try:
        backtest_file = project_root / 'backtest_results.csv'
        if backtest_file.exists():
            data['backtest'] = pd.read_csv(backtest_file)
        else:
            data['backtest'] = pd.DataFrame()
    except:
        data['backtest'] = pd.DataFrame()
    
    return data

data = load_real_data()

# ============================================================================
# SIDEBAR - SYSTEM STATUS
# ============================================================================

st.sidebar.title("🎯 Quantum AI Cockpit")
st.sidebar.markdown("---")

# Calculate real metrics
if len(data['portfolio']) > 0:
    portfolio = data['portfolio']
    total_pnl = 0
    wins = 0
    total_active = 0
    
    for _, row in portfolio.iterrows():
        if row.get('Current', 0) > 0:
            total_active += 1
            entry = row.get('Entry', 0)
            current = row.get('Current', 0)
            shares = row.get('Shares', 0)
            pnl = (current - entry) * shares
            total_pnl += pnl
            if pnl > 0:
                wins += 1
    
    win_rate = (wins / total_active * 100) if total_active > 0 else 0
    account_value = 437.0 + total_pnl
    today_pnl_pct = (total_pnl / 437.0) * 100 if total_pnl != 0 else 0
else:
    account_value = 437.0
    today_pnl_pct = 0
    win_rate = 0

st.sidebar.metric("System Status", "🟢 LIVE")
st.sidebar.metric("Account Value", f"${account_value:.2f}")
st.sidebar.metric("Today's P&L", f"{today_pnl_pct:+.1f}%", f"{'+' if today_pnl_pct > 0 else ''}${total_pnl:.2f}" if len(data['portfolio']) > 0 else None)
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Settings")
st.sidebar.write(f"Risk per Trade: 2%")
st.sidebar.write(f"Max Position: 10%")
st.sidebar.write(f"Win Rate: {win_rate:.0f}%")

# Refresh button
if st.sidebar.button("🔄 Refresh Data"):
    st.cache_data.clear()
    st.rerun()

# ============================================================================
# MAIN DASHBOARD
# ============================================================================

st.title("🚀 Quantum AI Trading Cockpit")
st.markdown("### Your Real Trading Data - Live")

# ============================================================================
# TOP METRICS
# ============================================================================

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    signal_count = len(data['signals'])
    st.metric(
        "Active Signals",
        signal_count,
        f"+{signal_count - 20}" if signal_count > 20 else None
    )

with col2:
    if len(data['portfolio']) > 0:
        st.metric("Win Rate", f"{win_rate:.0f}%")
    else:
        st.metric("Win Rate", "N/A")

with col3:
    if len(data['portfolio']) > 0:
        st.metric("Total P&L", f"${total_pnl:.2f}",
            f"{(total_pnl/437)*100:.1f}%" if total_pnl != 0 else None)
    else:
        st.metric("Total P&L", "$0.00")

with col4:
    st.metric("Paper Trades", len(data['trades']))

with col5:
    if len(data['backtest']) > 0:
        expectancy = data['backtest']['pnl_pct'].mean()
        st.metric("Expectancy", f"{expectancy:.1%}")
    else:
        st.metric("Expectancy", "N/A")

st.markdown("---")

# ============================================================================
# ACTIVE SIGNALS TABLE
# ============================================================================

st.subheader("📊 Active Trading Signals")

if len(data['signals']) > 0:
    df_signals = pd.DataFrame(data['signals'])
    
    # Display columns
    display_cols = ['ticker', 'entry_price', 'target_price', 'stop_loss', 
                   'expected_return', 'confidence', 'strategy']
    
    # Filter to only columns that exist
    display_cols = [col for col in display_cols if col in df_signals.columns]
    
    # Format the dataframe
    df_display = df_signals[display_cols].copy()
    
    if 'expected_return' in df_display.columns:
        df_display['expected_return'] = df_display['expected_return'].apply(lambda x: f"{x*100:.1f}%" if isinstance(x, (int, float)) else str(x))
    if 'confidence' in df_display.columns:
        df_display['confidence'] = df_display['confidence'].apply(lambda x: f"{x}%" if isinstance(x, (int, float)) else str(x))
    if 'entry_price' in df_display.columns:
        df_display['entry_price'] = df_display['entry_price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else str(x))
    if 'target_price' in df_display.columns:
        df_display['target_price'] = df_display['target_price'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else str(x))
    if 'stop_loss' in df_display.columns:
        df_display['stop_loss'] = df_display['stop_loss'].apply(lambda x: f"${x:.2f}" if isinstance(x, (int, float)) else str(x))
    
    st.dataframe(
        df_display,
        use_container_width=True,
        height=400
    )
else:
    st.warning("No active signals. Run daily scan.")

# ============================================================================
# PORTFOLIO TRACKER
# ============================================================================

st.markdown("---")
st.subheader("💼 Portfolio Tracker")

if len(data['portfolio']) > 0:
    portfolio = data['portfolio'].copy()
    
    # Add action alerts
    def get_alert(row):
        current = row.get('Current', 0)
        if current == 0:
            return "⏳ Pending"
        elif current >= row.get('Target', 0):
            return "🎯 SELL NOW (Target Hit)"
        elif current <= row.get('Stop', 0):
            return "🛑 SELL NOW (Stop Loss)"
        elif current > row.get('Entry', 0):
            return "✅ Winning"
        else:
            return "⚠️ Losing"
    
    portfolio['Alert'] = portfolio.apply(get_alert, axis=1)
    
    # Display
    st.dataframe(portfolio, use_container_width=True, height=400)
    
    # Sell alerts
    sell_alerts = portfolio[portfolio['Alert'].str.contains('SELL NOW', na=False)]
    if len(sell_alerts) > 0:
        st.error(f"🚨 IMMEDIATE ACTION REQUIRED: {len(sell_alerts)} positions to SELL NOW!")
        st.dataframe(sell_alerts[['Ticker', 'Entry', 'Current', 'Target', 'Stop', 'Alert']])
else:
    st.info("No positions tracked yet. Update portfolio_tracker.csv with current prices.")

# ============================================================================
# PERFORMANCE CHART
# ============================================================================

st.markdown("---")
st.subheader("📈 Performance Over Time")

if len(data['backtest']) > 0:
    backtest = data['backtest'].copy()
    if 'pnl_pct' in backtest.columns:
        backtest['cumulative_return'] = (1 + backtest['pnl_pct']).cumprod() - 1
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=backtest.index,
            y=backtest['cumulative_return'] * 100,
            mode='lines',
            name='Cumulative Return',
            line=dict(color='#00ff00', width=2)
        ))
        fig.update_layout(
            title="Account Growth",
            xaxis_title="Trade Number",
            yaxis_title="Return (%)",
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Backtest data missing pnl_pct column")
else:
    st.info("Run backtest to see performance chart")

# ============================================================================
# STRATEGY BREAKDOWN
# ============================================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Strategy Performance")
    
    if len(data['signals']) > 0:
        df_signals = pd.DataFrame(data['signals'])
        if 'strategy' in df_signals.columns:
            strategy_counts = df_signals['strategy'].value_counts()
            
            fig = px.pie(
                values=strategy_counts.values,
                names=strategy_counts.index,
                title="Signals by Strategy"
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No strategy column in signals")

with col2:
    st.subheader("💰 Expected Returns")
    
    if len(data['signals']) > 0:
        df_signals = pd.DataFrame(data['signals'])
        if 'expected_return' in df_signals.columns:
            top_returns = df_signals.nlargest(5, 'expected_return')[['ticker', 'expected_return']]
            
            fig = px.bar(
                top_returns,
                x='ticker',
                y='expected_return',
                title="Top 5 Expected Returns"
            )
            fig.update_layout(template="plotly_dark", yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No expected_return column in signals")

# ============================================================================
# AGGREGATED SIGNALS SECTION (NEW)
# ============================================================================

st.markdown("---")
st.subheader("🌐 AI Tools Aggregation")

# Load aggregated signals
try:
    agg_file = project_root / 'data' / 'aggregated_signals.json'
    if agg_file.exists():
        with open(agg_file) as f:
            agg_signals = json.load(f)
        
        st.write(f"**Scraped from:** Finviz, Yahoo, StockTwits, Reddit, TradingView")
        st.write(f"**Total Mentions:** {len(agg_signals)}")
        st.write(f"**Validated by YOUR Algorithm:** {len([s for s in agg_signals if s.get('validated')])} signals")
        
        if agg_signals:
            df_agg = pd.DataFrame(agg_signals)
            display_cols = ['ticker', 'entry_price', 'target_price', 'expected_return', 'rsi']
            display_cols = [col for col in display_cols if col in df_agg.columns]
            st.dataframe(
                df_agg[display_cols],
                use_container_width=True
            )
    else:
        st.warning("No aggregated signals yet. Run signal_aggregator.py")
except Exception as e:
    st.warning(f"Error loading aggregated signals: {e}")

# ============================================================================
# RECENT ACTIVITY LOG
# ============================================================================

st.markdown("---")
st.subheader("📝 Recent Activity")

if len(data['trades']) > 0:
    st.text_area(
        "Paper Trades Log",
        '\n'.join(data['trades'][-20:]),
        height=200
    )
else:
    st.info("No trades logged yet")

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================

with st.expander("⚙️ System Configuration"):
    if data['config']:
        st.json(data['config'])
    else:
        st.warning("No configuration loaded")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
col1, col2, col3 = st.columns(3)
col1.write(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
col2.write(f"Signals: {len(data['signals'])}")
col3.write(f"Status: 🟢 LIVE")

