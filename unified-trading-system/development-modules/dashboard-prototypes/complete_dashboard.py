"""
COMPLETE TRADING DASHBOARD - Like Intellectia AI
Shows: Forecast charts, signals, portfolio, trade plans
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

# Import your modules
from backend.modules.daily_scanner import ProfessionalDailyScanner, Strategy
from backend.modules.position_size_calculator import PositionSizeCalculator
from backend.modules.ai_forecast_pro import AIForecastPro
from backend.modules.pattern_engine_pro import PatternEnginePro
from backend.modules.api_integrations import get_stock_data
import yfinance as yf

st.set_page_config(page_title="Quantum AI Cockpit", layout="wide", page_icon="🚀")

# Sidebar
st.sidebar.title("🚀 Quantum AI")
page = st.sidebar.radio("", ["📊 Scanner", "🔮 Forecast", "💼 Portfolio"])

account_value = st.sidebar.number_input("Account Value ($)", value=500, step=100)

#================================================================================
# PAGE 1: SCANNER
#================================================================================
if page == "📊 Scanner":
    st.title("🔥 Daily Market Scanner")
    
    stocks = st.multiselect(
        "Select stocks:",
        ['NVDA', 'AMD', 'TSLA', 'PLTR', 'SOFI', 'COIN', 'META', 'GOOGL'],
        default=['NVDA', 'AMD', 'TSLA']
    )
    
    if st.button("🔥 RUN SCAN", type="primary"):
        with st.spinner("Scanning..."):
            scanner = ProfessionalDailyScanner(Strategy.SWING_TRADE)
            results = scanner.scan_universe(stocks)
            
            # Filter
            opportunities = [r for r in results if r.final_score >= 70]
            
            if not opportunities:
                st.warning("No opportunities (all scores <70)")
            else:
                st.success(f"Found {len(opportunities)} opportunities!")
                
                calc = PositionSizeCalculator(account_value, risk_pct=0.02)
                
                for result in opportunities:
                    with st.expander(f"📊 {result.ticker} - Score: {result.final_score:.0f}/100"):
                        # Get current price
                        try:
                            stock = yf.Ticker(result.ticker)
                            current_price = stock.info.get('regularMarketPrice', 100)
                        except:
                            current_price = 100
                        
                        # Calculate position
                        position = calc.calculate(
                            result.ticker,
                            result.final_score / 100,
                            current_price
                        )
                        
                        if position.get('should_trade'):
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Entry", f"${position['entry_price']:.2f}")
                                st.metric("Shares", f"{position['shares']:.2f}")
                            
                            with col2:
                                st.metric("Stop", f"${position['stop_loss_price']:.2f}")
                                st.metric("Risk", f"${position['risk_dollars']:.2f}")
                            
                            with col3:
                                st.metric("Target", f"${position['take_profit_price']:.2f}")
                                st.metric("Reward", f"${position['reward_dollars']:.2f}")
                            
                            # Trade plan
                            trade = f"""BUY {position['shares']:.2f} {result.ticker}
Entry: ${position['entry_price']:.2f}
Stop: ${position['stop_loss_price']:.2f}
Target: ${position['take_profit_price']:.2f}"""
                            
                            st.code(trade)
                            st.info("⚠️ Copy to Robinhood - Execute manually!")

#================================================================================
# PAGE 2: FORECAST CHART
#================================================================================
elif page == "🔮 Forecast":
    st.title("📈 Stock Forecast Analysis")
    
    ticker = st.text_input("Enter ticker:", "NVDA").upper()
    
    if ticker:
        with st.spinner(f"Analyzing {ticker}..."):
            # Get data
            df = get_stock_data(ticker, period='6mo')
            
            if df is None or df.empty:
                st.error(f"No data available for {ticker}")
            else:
                # Get forecast
                try:
                    forecaster = AIForecastPro()
                    forecast = forecaster.forecast(ticker, df, horizon_days=5)
                    
                    # Create chart
                    fig = make_subplots(
                        rows=2, cols=1,
                        row_heights=[0.7, 0.3],
                        subplot_titles=(f"{ticker} Price Chart", "Volume"),
                        vertical_spacing=0.1
                    )
                    
                    # Candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=df.index,
                            open=df['open'],
                            high=df['high'],
                            low=df['low'],
                            close=df['close'],
                            name="Price"
                        ),
                        row=1, col=1
                    )
                    
                    # Forecast lines
                    last_date = df.index[-1]
                    current_price = df['close'].iloc[-1]
                    
                    # Bull case
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, last_date],
                            y=[current_price, forecast['bull_case']['price']],
                            mode='lines+markers',
                            name=f"Bull: ${forecast['bull_case']['price']:.2f} (+{forecast['bull_case']['return_pct']:.1f}%)",
                            line=dict(color='green', width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # Base case
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, last_date],
                            y=[current_price, forecast['base_case']['price']],
                            mode='lines+markers',
                            name=f"Base: ${forecast['base_case']['price']:.2f} (+{forecast['base_case']['return_pct']:.1f}%)",
                            line=dict(color='blue', width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # Bear case
                    fig.add_trace(
                        go.Scatter(
                            x=[last_date, last_date],
                            y=[current_price, forecast['bear_case']['price']],
                            mode='lines+markers',
                            name=f"Bear: ${forecast['bear_case']['price']:.2f} ({forecast['bear_case']['return_pct']:.1f}%)",
                            line=dict(color='red', width=2, dash='dash')
                        ),
                        row=1, col=1
                    )
                    
                    # Volume
                    fig.add_trace(
                        go.Bar(x=df.index, y=df['volume'], name="Volume", marker_color='lightblue'),
                        row=2, col=1
                    )
                    
                    fig.update_layout(
                        height=700,
                        xaxis_rangeslider_visible=False,
                        template="plotly_dark",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Forecast details
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current", f"${current_price:.2f}")
                    
                    with col2:
                        st.metric(
                            "Bull Case", 
                            f"${forecast['bull_case']['price']:.2f}",
                            f"+{forecast['bull_case']['return_pct']:.1f}%",
                            delta_color="normal"
                        )
                    
                    with col3:
                        st.metric(
                            "Base Case",
                            f"${forecast['base_case']['price']:.2f}",
                            f"+{forecast['base_case']['return_pct']:.1f}%",
                            delta_color="normal"
                        )
                    
                    with col4:
                        st.metric(
                            "Bear Case",
                            f"${forecast['bear_case']['price']:.2f}",
                            f"{forecast['bear_case']['return_pct']:.1f}%",
                            delta_color="inverse"
                        )
                    
                    # Additional info
                    st.markdown("### 📊 Forecast Details")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Confidence:** {forecast['confidence']:.0f}%")
                        st.write(f"**Models:** {', '.join(forecast.get('models_used', []))}")
                    
                    with col2:
                        st.write(f"**Volatility:** {forecast.get('volatility_pct', 0):.1f}%")
                        st.write(f"**Horizon:** {forecast['horizon_days']} days")
                    
                    # Calculate position
                    st.markdown("### 💼 Trade Plan")
                    calc = PositionSizeCalculator(account_value, risk_pct=0.02)
                    position = calc.calculate(ticker, forecast['confidence']/100, current_price)
                    
                    if position.get('should_trade'):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Position Size", f"${position['position_size_dollars']:.2f}")
                            st.metric("Shares", f"{position['shares']:.2f}")
                        
                        with col2:
                            st.metric("Stop Loss", f"${position['stop_loss_price']:.2f}")
                            st.metric("Risk", f"${position['risk_dollars']:.2f}")
                        
                        with col3:
                            st.metric("Target", f"${position['take_profit_price']:.2f}")
                            st.metric("Reward", f"${position['reward_dollars']:.2f}")
                        
                        trade_plan = f"""BUY {position['shares']:.2f} shares {ticker}
Entry: ${position['entry_price']:.2f}
Stop: ${position['stop_loss_price']:.2f}
Target: ${position['take_profit_price']:.2f}
Risk: ${position['risk_dollars']:.2f} | Reward: ${position['reward_dollars']:.2f}"""
                        
                        st.code(trade_plan)
                        st.success("✅ Copy this trade plan to Robinhood!")
                    
                except Exception as e:
                    st.error(f"Forecast failed: {e}")
                    st.exception(e)

#================================================================================
# PAGE 3: PORTFOLIO
#================================================================================
elif page == "💼 Portfolio":
    st.title("💼 Portfolio Management")
    
    st.info("📝 Manual position tracking - Add your executed trades here")
    
    # Manual trade entry
    with st.form("add_trade"):
        st.subheader("Add Position")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            trade_ticker = st.text_input("Ticker", "NVDA")
        with col2:
            trade_shares = st.number_input("Shares", value=1.0, step=0.1)
        with col3:
            trade_entry = st.number_input("Entry Price", value=145.0, step=1.0)
        with col4:
            trade_stop = st.number_input("Stop Loss", value=133.0, step=1.0)
        
        if st.form_submit_button("Add Position"):
            st.success(f"Added {trade_shares} shares of {trade_ticker} at ${trade_entry}")
    
    # Placeholder for saved positions
    st.markdown("### 📊 Open Positions")
    st.info("No open positions. Run scanner and execute trades to populate this.")
    
    # Performance metrics
    st.markdown("### 📈 Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Account Value", f"${account_value:.2f}")
    with col2:
        st.metric("Total P&L", "$0.00", "0%")
    with col3:
        st.metric("Win Rate", "0%")
    with col4:
        st.metric("Open Positions", "0")

