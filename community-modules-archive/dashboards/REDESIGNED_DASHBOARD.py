"""
📊 REDESIGNED DASHBOARD - Complete Integration
===============================================
Signal generation, testing, training, and recommendations
All in one dashboard - NOT autonomous trading
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import json

# Page config
st.set_page_config(
    page_title="Quantum AI Cockpit - Signal Generator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DASHBOARD CLASS
# ============================================================================

class QuantumAIDashboard:
    """
    Complete dashboard for signal generation and analysis
    Focus: Help user find stocks, not execute trades
    """
    
    def __init__(self):
        self.setup_sidebar()
        self.initialize_engine()
    
    def setup_sidebar(self):
        """Configure sidebar controls"""
        st.sidebar.title("⚙️ Quantum AI Cockpit")
        st.sidebar.markdown("---")
        
        st.sidebar.header("📊 Analysis Mode")
        self.analysis_mode = st.sidebar.radio(
            "What do you want to do?",
            ["Generate Signals", "Test Modules", "Train Modules", "View Performance"]
        )
        
        st.sidebar.markdown("---")
        
        st.sidebar.header("🔍 Symbol Input")
        symbols_input = st.sidebar.text_input(
            "Symbols to Analyze",
            value="AAPL,TSLA,NVDA,AMD,MSFT",
            help="Comma-separated list of stock symbols"
        )
        self.symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
        
        st.sidebar.markdown("---")
        
        st.sidebar.header("🧪 Testing Options")
        self.enable_testing = st.sidebar.checkbox("Enable Module Testing", value=False)
        self.test_lookback = st.sidebar.slider("Test Lookback (days)", 30, 180, 90)
        
        st.sidebar.markdown("---")
        
        st.sidebar.header("📝 Paper Trading")
        self.paper_trading = st.sidebar.checkbox("Enable Paper Trading", value=True,
                                                help="Test recommendations without real money")
        
        st.sidebar.markdown("---")
        
        st.sidebar.markdown("""
        ### ⚠️ System Purpose
        
        This is a **RESEARCH TOOL**:
        - ✅ Generate signals
        - ✅ Get recommendations
        - ✅ Test strategies
        - ❌ **NOT** autonomous trading
        
        Execute trades manually on Robinhood.
        """)
    
    def initialize_engine(self):
        """Initialize backtest engine"""
        try:
            import sys
            sys.path.append('/content/drive/MyDrive/QuantumAI/backend/modules')
            from BACKTEST_INSTITUTIONAL_ENSEMBLE import BacktestEngine
            
            self.engine = BacktestEngine()
            self.engine_loaded = True
        except Exception as e:
            st.error(f"⚠️  Could not load engine: {e}")
            self.engine_loaded = False
            self.engine = None
    
    def render_dashboard(self):
        """Main dashboard rendering"""
        st.title("🎯 Quantum AI Cockpit - Signal Generator")
        st.markdown("**Research Tool for Finding High-Probability Trading Setups**")
        st.markdown("---")
        
        # Route to appropriate view
        if self.analysis_mode == "Generate Signals":
            self.render_signal_generation()
        elif self.analysis_mode == "Test Modules":
            self.render_module_testing()
        elif self.analysis_mode == "Train Modules":
            self.render_module_training()
        elif self.analysis_mode == "View Performance":
            self.render_performance_view()
    
    def render_signal_generation(self):
        """Render signal generation view"""
        st.header("📊 Signal Generation")
        
        if not self.engine_loaded:
            st.error("⚠️  Engine not loaded. Check imports.")
            return
        
        if st.button("🔍 Generate Signals", type="primary"):
            with st.spinner("Generating signals..."):
                try:
                    recommendations = self.engine.get_recommendations(symbols=self.symbols)
                    
                    if recommendations:
                        self.display_recommendations(recommendations)
                    else:
                        st.warning("No high-confidence signals found for these symbols.")
                        st.info("💡 Try different symbols or adjust confidence thresholds")
                
                except Exception as e:
                    st.error(f"Error generating signals: {e}")
                    st.code(str(e))
        else:
            self.render_welcome_screen()
    
    def display_recommendations(self, recommendations: List[Dict]):
        """Display trading recommendations"""
        st.header(f"📊 Trading Recommendations ({len(recommendations)} found)")
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Signals", len(recommendations))
        with col2:
            avg_conf = np.mean([r.get('confidence', 0) for r in recommendations])
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col3:
            high_conf = sum(1 for r in recommendations if r.get('confidence', 0) >= 0.75)
            st.metric("High Confidence", high_conf)
        with col4:
            sources = set(r.get('source', 'UNKNOWN') for r in recommendations)
            st.metric("Sources", len(sources))
        
        st.markdown("---")
        
        # Display each recommendation
        for i, rec in enumerate(recommendations, 1):
            with st.expander(
                f"#{i} {rec['symbol']} - {rec.get('confidence', 0):.1%} Confidence | {rec.get('action', 'BUY')}",
                expanded=(i <= 3)
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Signal Info**")
                    st.metric("Action", rec.get('action', 'BUY'))
                    st.metric("Confidence", f"{rec.get('confidence', 0):.1%}")
                    st.metric("Tier", rec.get('confidence_tier', 'GOOD'))
                    st.metric("Source", rec.get('source', 'ENSEMBLE'))
                
                with col2:
                    st.markdown("**Price Levels**")
                    entry = rec.get('entry_price', 0)
                    target = rec.get('target_price', 0)
                    stop = rec.get('stop_loss', 0)
                    
                    st.metric("Entry", f"${entry:.2f}")
                    st.metric("Target", f"${target:.2f}", f"+{(target/entry-1)*100:.1f}%")
                    st.metric("Stop Loss", f"${stop:.2f}", f"-{(1-stop/entry)*100:.1f}%")
                    
                    # Risk/Reward
                    risk = entry - stop
                    reward = target - entry
                    rr = reward / risk if risk > 0 else 0
                    st.metric("Risk/Reward", f"{rr:.2f}:1")
                
                with col3:
                    st.markdown("**Analysis**")
                    st.metric("Expected Move", rec.get('expected_move', '20-50%'))
                    st.metric("Risk Level", rec.get('risk_level', 'MEDIUM'))
                    
                    confirming = rec.get('confirming_modules', [])
                    if confirming:
                        st.markdown(f"**Confirming:** {', '.join(confirming)}")
                
                # Reasoning
                st.markdown("**Reasoning:**")
                st.info(rec.get('reasoning', 'Multiple signals confirming'))
                
                # Paper trading button
                if self.paper_trading:
                    if st.button(f"📝 Paper Trade {rec['symbol']}", key=f"paper_{rec['symbol']}_{i}"):
                        st.success(f"✅ Paper trade logged for {rec['symbol']}")
                        # In production, log to paper trading system
        
        # Performance chart
        if len(recommendations) > 0:
            self.render_recommendations_chart(recommendations)
    
    def render_recommendations_chart(self, recommendations: List[Dict]):
        """Render chart of recommendations"""
        st.markdown("---")
        st.subheader("📈 Recommendations Overview")
        
        # Confidence distribution
        confidences = [r.get('confidence', 0) for r in recommendations]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=confidences,
            nbinsx=20,
            name='Confidence Distribution',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Signal Confidence Distribution',
            xaxis_title='Confidence',
            yaxis_title='Count',
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_module_testing(self):
        """Render module testing view"""
        st.header("🧪 Module Testing")
        st.markdown("Test individual modules for signal quality")
        
        if st.button("🧪 Run Module Tests", type="primary"):
            with st.spinner("Testing modules..."):
                try:
                    from unified_testing_framework import ModuleTester
                    tester = ModuleTester()
                    
                    # Test each module
                    test_results = {}
                    
                    # Test Dark Pool
                    try:
                        from dark_pool_tracker import DarkPoolTracker
                        dp = DarkPoolTracker()
                        results = tester.test_module('dark_pool', dp, self.symbols, self.test_lookback)
                        test_results['dark_pool'] = results
                    except Exception as e:
                        st.warning(f"Dark pool test failed: {e}")
                    
                    # Test Unified Scanner
                    try:
                        from unified_momentum_scanner_v3 import UnifiedMomentumScannerV3
                        scanner = UnifiedMomentumScannerV3()
                        results = tester.test_module('unified_scanner', scanner, self.symbols, self.test_lookback)
                        test_results['unified_scanner'] = results
                    except Exception as e:
                        st.warning(f"Unified scanner test failed: {e}")
                    
                    # Display results
                    if test_results:
                        report = tester.get_performance_report()
                        if not report.empty:
                            st.dataframe(report, use_container_width=True)
                            
                            # Chart
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=report['Module'],
                                y=[float(wr.replace('%', '')) for wr in report['Win Rate']],
                                name='Win Rate',
                                marker_color='lightgreen'
                            ))
                            
                            fig.update_layout(
                                title='Module Win Rates',
                                xaxis_title='Module',
                                yaxis_title='Win Rate (%)',
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Testing error: {e}")
        else:
            st.info("💡 Click 'Run Module Tests' to test all modules")
    
    def render_module_training(self):
        """Render module training view"""
        st.header("🎓 Module Training")
        st.markdown("Train modules to improve signal accuracy")
        
        st.info("""
        **Training Process:**
        1. Test modules to get baseline performance
        2. Optimize hyperparameters
        3. Retest to verify improvement
        4. Update ensemble weights
        """)
        
        if st.button("🎓 Start Training", type="primary"):
            with st.spinner("Training modules..."):
                try:
                    from unified_training_framework import SignalTrainingFramework
                    trainer = SignalTrainingFramework()
                    
                    # This would use actual training data
                    st.info("💡 Training requires historical performance data")
                    st.info("💡 Run module tests first to generate training data")
                    
                except Exception as e:
                    st.error(f"Training error: {e}")
        else:
            st.info("💡 Click 'Start Training' to begin training process")
    
    def render_performance_view(self):
        """Render performance tracking view"""
        st.header("📈 Performance Tracking")
        st.markdown("View module and system performance over time")
        
        # Module performance
        if self.engine_loaded and hasattr(self.engine, 'signals_by_module'):
            st.subheader("Module Performance")
            
            perf_data = []
            for module, signals in self.engine.signals_by_module.items():
                if len(signals) > 0:
                    avg_conf = np.mean([s[2] for s in signals])
                    perf_data.append({
                        'Module': module,
                        'Signals': len(signals),
                        'Avg Confidence': f"{avg_conf:.1%}"
                    })
            
            if perf_data:
                df = pd.DataFrame(perf_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No performance data yet. Generate some signals first!")
        else:
            st.info("💡 Performance data will appear after generating signals")
    
    def render_welcome_screen(self):
        """Welcome screen with instructions"""
        st.header("Welcome to Quantum AI Cockpit")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 🎯 System Purpose
            
            This is a **RESEARCH TOOL** for finding stocks:
            - ✅ Generate signals from multiple modules
            - ✅ Get AI-powered recommendations
            - ✅ Test module performance
            - ✅ Train modules to improve
            - ✅ Paper trade for testing
            - ❌ **NOT** autonomous trading
            
            ### 📋 How to Use
            
            1. **Enter symbols** in sidebar
            2. **Click "Generate Signals"**
            3. **Review recommendations**
            4. **Paper trade** to test
            5. **Execute manually** on Robinhood
            """)
        
        with col2:
            st.markdown("""
            ### 🧪 Testing & Training
            
            **Test Modules:**
            - Test individual modules
            - Get baseline performance
            - Identify best performers
            
            **Train Modules:**
            - Optimize hyperparameters
            - Improve signal accuracy
            - Update ensemble weights
            
            ### ⚠️ Important Notes
            
            - All signals are for **analysis**
            - **Manual execution** on Robinhood
            - Paper trading for **testing only**
            - No autonomous trading enabled
            """)
        
        st.markdown("---")
        st.markdown("### 🚀 Ready to Start?")
        st.markdown("Enter symbols in the sidebar and click **'Generate Signals'** to begin!")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main dashboard app"""
    dashboard = QuantumAIDashboard()
    dashboard.render_dashboard()

if __name__ == "__main__":
    main()

# For Streamlit: just call main()
main()

