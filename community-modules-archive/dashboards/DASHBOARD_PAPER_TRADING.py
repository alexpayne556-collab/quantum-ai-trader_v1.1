"""
================================================================================
📊 QUANTUM AI COCKPIT - RANKING MODEL + PAPER TRADING DASHBOARD
================================================================================

Features:
1. Top 10 Daily Scanner (Ranking Model)
2. Universal Ticker Lookup (Test ANY stock!)
3. Paper Trading Portfolio
4. Performance Analytics
5. Prediction vs Actual Tracking
6. Model Health Monitoring

Usage:
streamlit run DASHBOARD_PAPER_TRADING.py
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import ta
import os

# ================================================================================
# CONFIGURATION
# ================================================================================

st.set_page_config(
    page_title="Quantum AI Cockpit",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_DIR = '/content/drive/MyDrive/QuantumAI/models_ranking'
PAPER_TRADES_DIR = '/content/drive/MyDrive/QuantumAI/paper_trades'
UNIVERSE = [
    'GME', 'AMC', 'BBBY', 'SHOP', 'PLTR', 'NIO', 'RIVN', 'LCID',
    'MARA', 'RIOT', 'COIN', 'MSTR', 'HUT', 'BTBT',
    'TSLA', 'NVDA', 'AMD', 'SNAP', 'HOOD', 'UPST', 'AFRM',
    'SNDL', 'CLOV', 'SOFI', 'BB', 'TLRY',
    'RBLX', 'ABNB', 'DASH', 'SNOW', 'DKNG',
    'SPCE', 'FUBO', 'WKHS', 'SKLZ', 'OPEN',
    'PYPL', 'ROKU', 'UBER', 'LYFT',
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX', 'DIS',
    'ADBE', 'CRM', 'NOW',
    'JPM', 'BAC', 'WFC', 'V', 'MA',
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK',
    'BABA', 'PINS', 'TWLO', 'CRWD', 'ZM', 'DOCU'
]

# Create directories
os.makedirs(PAPER_TRADES_DIR, exist_ok=True)

# ================================================================================
# LOAD MODELS (Cached)
# ================================================================================

@st.cache_resource
def load_models():
    """Load trained ranking models"""
    try:
        models = {
            'lgbm': joblib.load(f'{MODEL_DIR}/lgbm_ranking.pkl'),
            'xgb': joblib.load(f'{MODEL_DIR}/xgb_ranking.pkl'),
            'rf': joblib.load(f'{MODEL_DIR}/rf_ranking.pkl'),
            'mlp': joblib.load(f'{MODEL_DIR}/mlp_ranking.pkl'),
        }
        scaler = joblib.load(f'{MODEL_DIR}/scaler.pkl')
        
        with open(f'{MODEL_DIR}/metadata.json', 'r') as f:
            metadata = json.load(f)
        
        return models, scaler, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

# ================================================================================
# FEATURE ENGINEERING
# ================================================================================

def engineer_features(df, spy_data, feature_names):
    """Engineer features for prediction"""
    
    # Price-based features
    df['ma_10'] = df['close'].rolling(10, min_periods=5).mean()
    df['ma_20'] = df['close'].rolling(20, min_periods=10).mean()
    df['ma_50'] = df['close'].rolling(50, min_periods=25).mean()
    
    df['dist_ma10'] = (df['close'] - df['ma_10']) / df['close']
    df['dist_ma20'] = (df['close'] - df['ma_20']) / df['close']
    df['dist_ma50'] = (df['close'] - df['ma_50']) / df['close']
    
    # Volatility
    df['atr_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    df['atr_ratio'] = df['atr_14'] / df['close']
    
    bb = ta.volatility.BollingerBands(df['close'], window=20)
    df['bb_width'] = (bb.bollinger_hband() - bb.bollinger_lband()) / df['close']
    df['bb_position'] = (df['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    
    df['volatility_10d'] = df['close'].pct_change().rolling(10, min_periods=5).std()
    df['volatility_20d'] = df['close'].pct_change().rolling(20, min_periods=10).std()
    
    # Momentum
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    
    macd_indicator = ta.trend.MACD(df['close'])
    df['macd'] = macd_indicator.macd_diff()
    
    df['roc_10'] = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10)) * 100
    df['roc_20'] = ((df['close'] - df['close'].shift(20)) / df['close'].shift(20)) * 100
    
    df['momentum_5d'] = df['close'].pct_change(5)
    df['momentum_10d'] = df['close'].pct_change(10)
    df['momentum_20d'] = df['close'].pct_change(20)
    
    # Volume
    df['volume_ma_20'] = df['volume'].rolling(20, min_periods=10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_ma_20'] + 1)
    
    obv = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume'])
    df['obv'] = obv.on_balance_volume()
    df['obv_change'] = df['obv'].pct_change(20)
    
    df['volume_price_trend'] = ((df['close'] - df['close'].shift(1)) / df['close'].shift(1)) * df['volume']
    df['vpt_20d'] = df['volume_price_trend'].rolling(20, min_periods=10).sum()
    
    # Market regime
    spy_aligned = spy_data.reindex(df.index, method='ffill').fillna(method='bfill')
    
    spy_ma_50 = spy_aligned['close'].rolling(50, min_periods=25).mean()
    spy_ma_100 = spy_aligned['close'].rolling(100, min_periods=50).mean()
    
    df['market_trend'] = np.where(
        (spy_aligned['close'] > spy_ma_100) & (spy_ma_50 > spy_ma_100),
        2, np.where((spy_aligned['close'] < spy_ma_100) & (spy_ma_50 < spy_ma_100), 0, 1)
    )
    
    stock_return_20d = df['close'].pct_change(20)
    spy_return_20d = spy_aligned['close'].pct_change(20)
    df['relative_performance'] = stock_return_20d - spy_return_20d
    
    # Price patterns
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_open_ratio'] = (df['close'] - df['open']) / df['open']
    
    # Clean
    df = df.replace([np.inf, -np.inf], np.nan)
    for col in df.select_dtypes(include=[np.number]).columns:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            df[col] = df[col].clip(mean - 100*std, mean + 100*std)
    
    df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Extract latest features
    latest = df.iloc[-1]
    features = []
    for feat in feature_names:
        features.append(latest[feat] if feat in latest.index else 0)
    
    return features

# ================================================================================
# PREDICTION FUNCTIONS
# ================================================================================

@st.cache_data(ttl=3600)  # Cache for 1 hour
def download_stock_data(symbol, days=200):
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

def predict_stock(symbol, models, scaler, metadata, spy_data):
    """Predict return for a single stock"""
    
    # Download data
    df = download_stock_data(symbol)
    if df is None or len(df) < 100:
        return None
    
    # Engineer features
    try:
        features = engineer_features(df, spy_data, metadata['features'])
        features_array = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features_array)
        
        # Predict with each model
        pred_lgbm = models['lgbm'].predict(features_array)[0]
        pred_xgb = models['xgb'].predict(features_array)[0]
        pred_rf = models['rf'].predict(features_array)[0]
        pred_mlp = models['mlp'].predict(features_scaled)[0]
        
        # Ensemble
        pred_ensemble = np.mean([pred_lgbm, pred_xgb, pred_rf, pred_mlp])
        
        # Model agreement (lower = better agreement)
        preds = [pred_lgbm, pred_xgb, pred_rf, pred_mlp]
        agreement = np.std(preds) / (abs(np.mean(preds)) + 1e-10)
        
        return {
            'symbol': symbol,
            'current_price': df['close'].iloc[-1],
            'predicted_return': pred_ensemble,
            'lgbm': pred_lgbm,
            'xgb': pred_xgb,
            'rf': pred_rf,
            'mlp': pred_mlp,
            'agreement_score': agreement,
            'confidence': 'High' if agreement < 0.3 else ('Medium' if agreement < 0.5 else 'Low')
        }
    except Exception as e:
        st.error(f"Error predicting {symbol}: {e}")
        return None

# ================================================================================
# PAPER TRADING FUNCTIONS
# ================================================================================

def load_paper_trades():
    """Load all paper trades"""
    trades_file = f'{PAPER_TRADES_DIR}/paper_portfolio.csv'
    if os.path.exists(trades_file):
        return pd.read_csv(trades_file)
    return pd.DataFrame(columns=['symbol', 'entry_date', 'entry_price', 'predicted_return', 
                                   'exit_date', 'exit_price', 'actual_return', 'success', 'status'])

def save_paper_trade(symbol, entry_price, predicted_return):
    """Save a new paper trade"""
    trades = load_paper_trades()
    
    new_trade = pd.DataFrame([{
        'symbol': symbol,
        'entry_date': datetime.now().strftime('%Y-%m-%d'),
        'entry_price': entry_price,
        'predicted_return': predicted_return,
        'exit_date': (datetime.now() + timedelta(days=5)).strftime('%Y-%m-%d'),
        'exit_price': np.nan,
        'actual_return': np.nan,
        'success': np.nan,
        'status': 'open'
    }])
    
    trades = pd.concat([trades, new_trade], ignore_index=True)
    trades.to_csv(f'{PAPER_TRADES_DIR}/paper_portfolio.csv', index=False)
    return True

def update_paper_trade_results():
    """Update paper trades with actual results"""
    trades = load_paper_trades()
    open_trades = trades[trades['status'] == 'open'].copy()
    
    updated = False
    for idx, trade in open_trades.iterrows():
        exit_date = pd.to_datetime(trade['exit_date'])
        if datetime.now().date() >= exit_date.date():
            # Get current price
            df = download_stock_data(trade['symbol'], days=10)
            if df is not None and len(df) > 0:
                current_price = df['close'].iloc[-1]
                actual_return = (current_price - trade['entry_price']) / trade['entry_price']
                
                trades.at[idx, 'exit_price'] = current_price
                trades.at[idx, 'actual_return'] = actual_return
                trades.at[idx, 'success'] = 1 if actual_return >= 0.02 else 0
                trades.at[idx, 'status'] = 'closed'
                updated = True
    
    if updated:
        trades.to_csv(f'{PAPER_TRADES_DIR}/paper_portfolio.csv', index=False)
    
    return trades

# ================================================================================
# DASHBOARD UI
# ================================================================================

def main():
    
    # Load models
    models, scaler, metadata = load_models()
    
    if models is None:
        st.error("❌ Models not loaded! Please train models first.")
        st.stop()
    
    # Load SPY data (for features)
    spy_data = download_stock_data('SPY', days=200)
    
    # Sidebar
    st.sidebar.title("🚀 Quantum AI Cockpit")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["📊 Top 10 Scanner", "🔍 Ticker Lookup", "📈 Paper Portfolio", "📉 Performance Analytics"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Model Info:**")
    st.sidebar.markdown(f"Trained: {metadata['trained_date']}")
    st.sidebar.markdown(f"Success Rate: {metadata['performance']['success_rate_top10']:.1%}")
    st.sidebar.markdown(f"Avg Return: {metadata['performance']['avg_return_top10']:.2%}")
    
    # ========================================================================
    # PAGE 1: TOP 10 SCANNER
    # ========================================================================
    
    if page == "📊 Top 10 Scanner":
        st.title("📊 Top 10 Stock Scanner")
        st.markdown("**Ranking Model - Institutional Approach**")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Model Success Rate", f"{metadata['performance']['success_rate_top10']:.1%}", "Historical")
        col2.metric("Avg Return (Top 10)", f"{metadata['performance']['avg_return_top10']:.2%}", "5-day")
        col3.metric("Ranking Correlation", f"{metadata['performance']['spearman_correlation']:.3f}", "Spearman")
        
        st.markdown("---")
        
        if st.button("🔄 Scan Universe (66 Stocks)", type="primary"):
            with st.spinner("Scanning all stocks..."):
                progress_bar = st.progress(0)
                predictions = []
                
                for i, symbol in enumerate(UNIVERSE):
                    pred = predict_stock(symbol, models, scaler, metadata, spy_data)
                    if pred:
                        predictions.append(pred)
                    progress_bar.progress((i + 1) / len(UNIVERSE))
                
                if predictions:
                    df_pred = pd.DataFrame(predictions)
                    df_pred = df_pred.sort_values('predicted_return', ascending=False)
                    
                    st.success(f"✅ Scanned {len(df_pred)} stocks successfully!")
                    
                    # Top 10
                    st.subheader("🎯 Top 10 Stocks to Buy")
                    
                    top_10 = df_pred.head(10)
                    
                    for idx, row in top_10.iterrows():
                        rank = list(top_10.index).index(idx) + 1
                        
                        col1, col2, col3, col4, col5 = st.columns([1, 2, 2, 2, 2])
                        
                        col1.markdown(f"**#{rank}**")
                        col2.markdown(f"**{row['symbol']}**")
                        col3.markdown(f"${row['current_price']:.2f}")
                        
                        # Color code prediction
                        pred_color = "green" if row['predicted_return'] > 0.05 else ("orange" if row['predicted_return'] > 0.02 else "red")
                        col4.markdown(f":{pred_color}[{row['predicted_return']:+.2%}]")
                        
                        # Confidence
                        conf_emoji = "🟢" if row['confidence'] == 'High' else ("🟡" if row['confidence'] == 'Medium' else "🔴")
                        col5.markdown(f"{conf_emoji} {row['confidence']}")
                    
                    # Add all top 10 to paper portfolio
                    if st.button("📋 Add Top 10 to Paper Portfolio"):
                        for idx, row in top_10.iterrows():
                            save_paper_trade(row['symbol'], row['current_price'], row['predicted_return'])
                        st.success("✅ Added top 10 to paper portfolio!")
                        st.balloons()
                    
                    # Show full ranking
                    with st.expander("📊 View Full Rankings (All 66)"):
                        st.dataframe(df_pred[['symbol', 'current_price', 'predicted_return', 'confidence']].style.format({
                            'current_price': '${:.2f}',
                            'predicted_return': '{:+.2%}'
                        }))
    
    # ========================================================================
    # PAGE 2: TICKER LOOKUP
    # ========================================================================
    
    elif page == "🔍 Ticker Lookup":
        st.title("🔍 Universal Ticker Lookup")
        st.markdown("**Test ANY stock - instant prediction**")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            ticker = st.text_input("Enter Stock Ticker", value="AAPL", max_chars=10).upper()
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("🎯 Predict", type="primary")
        
        if predict_btn and ticker:
            with st.spinner(f"Analyzing {ticker}..."):
                pred = predict_stock(ticker, models, scaler, metadata, spy_data)
                
                if pred:
                    st.markdown("---")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Current Price", f"${pred['current_price']:.2f}")
                    col2.metric("Predicted Return", f"{pred['predicted_return']:+.2%}", "(5-day)")
                    col3.metric("Confidence", pred['confidence'])
                    col4.metric("Agreement Score", f"{pred['agreement_score']:.3f}", "Lower=Better")
                    
                    # Individual model predictions
                    st.subheader("📊 Model Predictions")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("LightGBM", f"{pred['lgbm']:+.2%}")
                    col2.metric("XGBoost", f"{pred['xgb']:+.2%}")
                    col3.metric("Random Forest", f"{pred['rf']:+.2%}")
                    col4.metric("Neural Net", f"{pred['mlp']:+.2%}")
                    
                    # Add to paper portfolio
                    st.markdown("---")
                    if st.button(f"📋 Add {ticker} to Paper Portfolio"):
                        save_paper_trade(ticker, pred['current_price'], pred['predicted_return'])
                        st.success(f"✅ Added {ticker} to paper portfolio!")
                        st.balloons()
                else:
                    st.error(f"❌ Could not analyze {ticker}. Check ticker symbol or data availability.")
    
    # ========================================================================
    # PAGE 3: PAPER PORTFOLIO
    # ========================================================================
    
    elif page == "📈 Paper Portfolio":
        st.title("📈 Paper Trading Portfolio")
        st.markdown("**Track predictions vs actual results**")
        
        # Update results
        if st.button("🔄 Update Results"):
            with st.spinner("Updating paper trade results..."):
                trades = update_paper_trade_results()
            st.success("✅ Results updated!")
        
        trades = load_paper_trades()
        
        if len(trades) == 0:
            st.info("📝 No paper trades yet. Add some predictions!")
        else:
            # Summary metrics
            closed_trades = trades[trades['status'] == 'closed']
            
            if len(closed_trades) > 0:
                success_rate = closed_trades['success'].mean()
                avg_return = closed_trades['actual_return'].mean()
                total_return = closed_trades['actual_return'].sum()
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Total Trades", len(trades))
                col2.metric("Closed Trades", len(closed_trades))
                col3.metric("Success Rate", f"{success_rate:.1%}", f"Target: 80%")
                col4.metric("Avg Return", f"{avg_return:+.2%}")
                
                # Comparison to model expectations
                st.markdown("---")
                st.subheader("🎯 Model Validation")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Historical Success Rate",
                        f"{metadata['performance']['success_rate_top10']:.1%}",
                        "Training Data"
                    )
                
                with col2:
                    delta = success_rate - metadata['performance']['success_rate_top10']
                    st.metric(
                        "Current Success Rate",
                        f"{success_rate:.1%}",
                        f"{delta:+.1%} vs Historical",
                        delta_color="normal"
                    )
                
                # Status indicator
                if success_rate >= 0.75:
                    st.success("✅ Model performing EXCELLENT! (≥75%)")
                elif success_rate >= 0.60:
                    st.warning("⚠️ Model performing OK (60-75%)")
                else:
                    st.error("❌ Model underperforming (<60%)")
            
            # Show trades
            st.markdown("---")
            st.subheader("📊 All Paper Trades")
            
            # Tabs for open vs closed
            tab1, tab2 = st.tabs(["🟢 Open Trades", "🔵 Closed Trades"])
            
            with tab1:
                open_trades = trades[trades['status'] == 'open']
                if len(open_trades) > 0:
                    st.dataframe(open_trades[['symbol', 'entry_date', 'entry_price', 'predicted_return', 'exit_date']].style.format({
                        'entry_price': '${:.2f}',
                        'predicted_return': '{:+.2%}'
                    }), use_container_width=True)
                else:
                    st.info("No open trades")
            
            with tab2:
                if len(closed_trades) > 0:
                    # Add success emoji
                    display_df = closed_trades[['symbol', 'entry_date', 'entry_price', 'predicted_return', 
                                                  'exit_price', 'actual_return', 'success']].copy()
                    display_df['result'] = display_df['success'].apply(lambda x: '✅' if x == 1 else '❌')
                    
                    st.dataframe(display_df.style.format({
                        'entry_price': '${:.2f}',
                        'exit_price': '${:.2f}',
                        'predicted_return': '{:+.2%}',
                        'actual_return': '{:+.2%}'
                    }), use_container_width=True)
                    
                    # Chart
                    fig = px.scatter(
                        closed_trades,
                        x='predicted_return',
                        y='actual_return',
                        color='success',
                        hover_data=['symbol'],
                        title="Predicted vs Actual Returns",
                        labels={'predicted_return': 'Predicted Return', 'actual_return': 'Actual Return'}
                    )
                    fig.add_shape(type='line', x0=-0.1, y0=-0.1, x1=0.2, y1=0.2, 
                                  line=dict(color='gray', dash='dash'))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No closed trades yet")
    
    # ========================================================================
    # PAGE 4: PERFORMANCE ANALYTICS
    # ========================================================================
    
    elif page == "📉 Performance Analytics":
        st.title("📉 Performance Analytics")
        st.markdown("**Model health & accuracy tracking**")
        
        trades = load_paper_trades()
        closed_trades = trades[trades['status'] == 'closed']
        
        if len(closed_trades) < 5:
            st.info("📊 Need at least 5 closed trades for analytics. Keep paper trading!")
        else:
            # Time series of success rate
            closed_trades['entry_date'] = pd.to_datetime(closed_trades['entry_date'])
            closed_trades = closed_trades.sort_values('entry_date')
            closed_trades['cumulative_success_rate'] = closed_trades['success'].expanding().mean()
            
            # Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=closed_trades['entry_date'],
                y=closed_trades['cumulative_success_rate'] * 100,
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='blue', width=3)
            ))
            fig.add_hline(y=80, line_dash="dash", line_color="green", 
                          annotation_text="Historical: 80%")
            fig.update_layout(
                title="Success Rate Over Time",
                xaxis_title="Date",
                yaxis_title="Success Rate (%)",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Return distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(
                    closed_trades,
                    x='actual_return',
                    nbins=20,
                    title="Return Distribution",
                    labels={'actual_return': 'Actual Return'}
                )
                fig.add_vline(x=0.02, line_dash="dash", line_color="green",
                              annotation_text="2% Target")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Success by stock
                stock_performance = closed_trades.groupby('symbol').agg({
                    'success': 'mean',
                    'actual_return': 'mean'
                }).reset_index()
                stock_performance = stock_performance.sort_values('success', ascending=False).head(10)
                
                fig = px.bar(
                    stock_performance,
                    x='symbol',
                    y='success',
                    title="Top 10 Stocks by Success Rate",
                    labels={'success': 'Success Rate', 'symbol': 'Stock'}
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == '__main__':
    main()

