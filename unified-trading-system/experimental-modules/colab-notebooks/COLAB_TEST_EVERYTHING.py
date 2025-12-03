"""
COMPLETE SYSTEM TEST - Run in Google Colab
Tests every module and shows visuals before finalizing
"""

# ============================================================================
# CELL 1: SETUP & INSTALL
# ============================================================================

print("Installing dependencies...")

# Install all required packages
import subprocess
import sys

packages = [
    'yfinance',
    'beautifulsoup4', 
    'requests',
    'pandas',
    'numpy',
    'plotly',
    'robin-stocks'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("Dependencies installed successfully!\n")

# ============================================================================
# CELL 2: MOUNT DRIVE & IMPORT MODULES
# ============================================================================

from google.colab import drive
drive.mount('/content/drive')

import sys
sys.path.insert(0, '/content/drive/MyDrive/Quantum_AI_Cockpit/backend/modules')

print("✅ Drive mounted!\n")

# ============================================================================
# CELL 3: TEST UNIVERSAL SCRAPER
# ============================================================================

print("="*80)
print("TEST 1: UNIVERSAL SCRAPER ENGINE")
print("="*80 + "\n")

try:
    from universal_scraper_engine import UniversalScraperEngine
    
    scraper = UniversalScraperEngine()
    
    # Test on AAPL
    print("Testing scraper on AAPL...\n")
    result = scraper.scrape_all('AAPL')
    
    print(f"✅ SCRAPER WORKS!")
    print(f"\nMaster Signal: {result['master_signal']['signal']}")
    print(f"Confidence: {result['master_signal']['confidence']:.0%}")
    print(f"Score: {result['master_signal']['score']}/{result['master_signal']['max_score']}")
    
    print("\n📊 Data Sources:")
    print(f"  - Dark Pool: {'✅' if 'error' not in result['dark_pool'] else '❌'}")
    print(f"  - Insider Trading: {'✅' if 'error' not in result['insider_trading'] else '❌'}")
    print(f"  - Short Interest: {'✅' if 'error' not in result['short_interest'] else '❌'}")
    print(f"  - Reddit: {result['reddit_sentiment'].get('mentions', 0)} mentions")
    print(f"  - StockTwits: {result['stocktwits_sentiment'].get('bullish_pct', 0):.0f}% bullish")
    print(f"  - News: {result['news'].get('articles_found', 0)} articles")
    
    SCRAPER_OK = True
    
except Exception as e:
    print(f"❌ SCRAPER ERROR: {e}")
    SCRAPER_OK = False

print("\n" + "="*80 + "\n")

# ============================================================================
# CELL 4: TEST DARK POOL TRACKER
# ============================================================================

print("="*80)
print("TEST 2: DARK POOL TRACKER")
print("="*80 + "\n")

try:
    from dark_pool_tracker import DarkPoolTracker
    
    tracker = DarkPoolTracker()
    
    # Test on multiple tickers
    test_tickers = ['AAPL', 'TSLA', 'NVDA']
    
    for ticker in test_tickers:
        result = tracker.analyze_ticker(ticker)
        print(f"\n{ticker}:")
        print(f"  Signal: {result['signal']}")
        print(f"  Confidence: {result['confidence']:.0%}")
        print(f"  Alert: {result['alert']}")
        if 'smart_money_score' in result:
            print(f"  Smart Money Score: {result['smart_money_score']:.0f}/100")
    
    print("\n✅ DARK POOL TRACKER WORKS!")
    DARK_POOL_OK = True
    
except Exception as e:
    print(f"❌ DARK POOL ERROR: {e}")
    DARK_POOL_OK = False

print("\n" + "="*80 + "\n")

# ============================================================================
# CELL 5: TEST SOCIAL SENTIMENT EXPLOSION DETECTOR
# ============================================================================

print("="*80)
print("TEST 3: SOCIAL SENTIMENT EXPLOSION DETECTOR")
print("="*80 + "\n")

try:
    from social_sentiment_explosion_detector import SocialSentimentExplosionDetector
    
    detector = SocialSentimentExplosionDetector()
    
    # Test on meme stocks
    test_tickers = ['GME', 'AMC', 'TSLA']
    
    for ticker in test_tickers:
        result = detector.analyze_ticker(ticker)
        print(f"\n{ticker}:")
        print(f"  Signal: {result['signal']}")
        print(f"  Explosion Score: {result['explosion_score']}/100")
        print(f"  Alert: {result['alert']}")
        print(f"  Reddit Mentions: {result['reddit_mentions']}")
    
    print("\n✅ SOCIAL EXPLOSION DETECTOR WORKS!")
    SOCIAL_OK = True
    
except Exception as e:
    print(f"❌ SOCIAL DETECTOR ERROR: {e}")
    SOCIAL_OK = False

print("\n" + "="*80 + "\n")

# ============================================================================
# CELL 6: TEST ALL MONEY-MAKER MODULES
# ============================================================================

print("="*80)
print("TEST 4: ALL MONEY-MAKER MODULES")
print("="*80 + "\n")

modules_status = {}

# Insider Trading Tracker
try:
    from insider_trading_tracker import InsiderTradingTracker
    tracker = InsiderTradingTracker()
    result = tracker.analyze_ticker('AAPL')
    print(f"✅ Insider Trading Tracker: {result['signal']}")
    modules_status['insider'] = True
except Exception as e:
    print(f"❌ Insider Trading Tracker: {e}")
    modules_status['insider'] = False

# Earnings Surprise Predictor
try:
    from earnings_surprise_predictor import EarningsSurprisePredictor
    predictor = EarningsSurprisePredictor()
    result = predictor.analyze_ticker('AAPL')
    print(f"✅ Earnings Surprise Predictor: {result['signal']}")
    modules_status['earnings'] = True
except Exception as e:
    print(f"❌ Earnings Surprise Predictor: {e}")
    modules_status['earnings'] = False

# Short Squeeze Scanner
try:
    from short_squeeze_scanner import ShortSqueezeScanner
    scanner = ShortSqueezeScanner()
    result = scanner.analyze_ticker('GME')
    print(f"✅ Short Squeeze Scanner: {result['signal']}")
    modules_status['squeeze'] = True
except Exception as e:
    print(f"❌ Short Squeeze Scanner: {e}")
    modules_status['squeeze'] = False

# Penny Stock Pump Detector
try:
    from penny_stock_pump_detector import PennyStockPumpDetector
    detector = PennyStockPumpDetector()
    result = detector.analyze_ticker('AMC')
    print(f"✅ Penny Stock Pump Detector: {result['signal']}")
    modules_status['pump'] = True
except Exception as e:
    print(f"❌ Penny Stock Pump Detector: {e}")
    modules_status['pump'] = False

print("\n" + "="*80 + "\n")

# ============================================================================
# CELL 7: TEST PORTFOLIO MANAGER
# ============================================================================

print("="*80)
print("TEST 5: PORTFOLIO MANAGER")
print("="*80 + "\n")

try:
    from portfolio_manager import PortfolioManager
    
    pm = PortfolioManager('test_portfolio.json')
    
    # Add test positions
    print("Adding test positions...")
    pm.add_position('AAPL', 10.5, 150.00)
    pm.add_position('TSLA', 5.25, 200.00)
    pm.add_position('NVDA', 3.75, 400.00)
    pm.update_cash(5000.00)
    
    # Get summary
    summary = pm.get_portfolio_summary()
    
    print(f"\n✅ PORTFOLIO MANAGER WORKS!")
    print(f"\nPortfolio Summary:")
    print(f"  Cash: ${summary['cash']:,.2f}")
    print(f"  Positions: {summary['positions_count']}")
    print(f"  Total Value: ${summary['total_value']:,.2f}")
    print(f"  Total P&L: ${summary['total_pnl']:,.2f} ({summary['total_pnl_pct']:+.2f}%)")
    
    print(f"\nPositions:")
    for ticker, pos in summary['positions'].items():
        print(f"  {ticker}: {pos['shares']:.3f} shares @ ${pos['avg_cost']:.2f}")
        print(f"    Current: ${pos['current_price']:.2f} = ${pos['total_value']:,.2f}")
        print(f"    P&L: ${pos['pnl']:,.2f} ({pos['pnl_pct']:+.2f}%)")
    
    PORTFOLIO_OK = True
    
except Exception as e:
    print(f"❌ PORTFOLIO MANAGER ERROR: {e}")
    PORTFOLIO_OK = False

print("\n" + "="*80 + "\n")

# ============================================================================
# CELL 8: TEST CHART GENERATION (VISUALS!)
# ============================================================================

print("="*80)
print("TEST 6: CHART GENERATION & VISUALS")
print("="*80 + "\n")

import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

try:
    # Fetch data for MU
    ticker = 'MU'
    print(f"Fetching data for {ticker}...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    print(f"✅ Got {len(df)} days of data\n")
    
    # Create chart
    print("Creating advanced chart...")
    
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
    
    # EMA Ribbon (glowing effect!)
    for period, color, opacity in [(8, '#00ff9f', 0.4), (13, '#00ff9f', 0.3), (21, '#00ff9f', 0.2)]:
        ema = df['Close'].ewm(span=period).mean()
        fig.add_trace(
            go.Scatter(
                x=df.index, 
                y=ema, 
                name=f'EMA{period}', 
                line=dict(color=color, width=2),
                opacity=opacity,
                fill='tonexty' if period > 8 else None
            ), 
            row=1, col=1
        )
    
    # Volume
    colors = ['#00ff9f' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff4444' for i in range(len(df))]
    fig.add_trace(
        go.Bar(x=df.index, y=df['Volume'], name='Volume', marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
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
    
    # Style - Cyberpunk theme!
    fig.update_layout(
        height=1000,
        template='plotly_dark',
        paper_bgcolor='rgba(10,14,39,0.9)',
        plot_bgcolor='rgba(10,14,39,0.9)',
        xaxis_rangeslider_visible=False,
        font=dict(color='#00ff9f', family='Courier New'),
        title=dict(
            text=f'🚀 {ticker} - QUANTUM AI COCKPIT ANALYSIS',
            font=dict(size=24, color='#00ff9f')
        )
    )
    
    # Show the chart!
    fig.show()
    
    print("\n✅ CHART GENERATED SUCCESSFULLY!")
    print("👆 Check the chart above - this is what you'll see in the dashboard!")
    
    CHART_OK = True
    
except Exception as e:
    print(f"❌ CHART ERROR: {e}")
    CHART_OK = False

print("\n" + "="*80 + "\n")

# ============================================================================
# CELL 9: TEST COMPLETE TICKER ANALYSIS
# ============================================================================

print("="*80)
print("TEST 7: COMPLETE TICKER ANALYSIS (ALL-IN-ONE)")
print("="*80 + "\n")

def analyze_ticker_complete(ticker):
    """
    Complete analysis combining:
    - Scraped data (dark pool, insider, social)
    - Pattern detection
    - AI signals
    - Chart
    """
    print(f"🔍 ANALYZING {ticker}...\n")
    
    # 1. Scrape all data
    print("1️⃣ Scraping all data sources...")
    scraper = UniversalScraperEngine()
    scraped_data = scraper.scrape_all(ticker)
    
    master_signal = scraped_data['master_signal']
    print(f"   Master Signal: {master_signal['signal']}")
    print(f"   Confidence: {master_signal['confidence']:.0%}")
    print(f"   Score: {master_signal['score']}/{master_signal['max_score']}\n")
    
    # 2. Dark pool analysis
    print("2️⃣ Dark pool analysis...")
    dp_tracker = DarkPoolTracker()
    dp_result = dp_tracker.analyze_ticker(ticker)
    print(f"   {dp_result['alert']}\n")
    
    # 3. Social sentiment
    print("3️⃣ Social sentiment...")
    social_detector = SocialSentimentExplosionDetector()
    social_result = social_detector.analyze_ticker(ticker)
    print(f"   {social_result['alert']}")
    print(f"   Explosion Score: {social_result['explosion_score']}/100\n")
    
    # 4. Final recommendation
    print("="*70)
    print("🎯 FINAL RECOMMENDATION")
    print("="*70)
    
    # Calculate overall score
    signals = []
    if master_signal['signal'] in ['STRONG_BUY', 'BUY']:
        signals.append(3)
    if dp_result['signal'] in ['STRONG_BUY', 'BUY']:
        signals.append(2)
    if social_result['signal'] in ['EXPLOSION_IMMINENT', 'HIGH_BUZZ']:
        signals.append(2)
    
    total_score = sum(signals)
    
    if total_score >= 6:
        recommendation = "🚀 STRONG BUY"
        action = "Buy now with high confidence!"
        color = "🟢"
    elif total_score >= 4:
        recommendation = "✅ BUY"
        action = "Good opportunity - consider buying"
        color = "🟢"
    elif total_score >= 2:
        recommendation = "⚠️ WATCH"
        action = "Add to watchlist - wait for better entry"
        color = "🟡"
    else:
        recommendation = "❌ AVOID/HOLD"
        action = "Not a good opportunity right now"
        color = "🔴"
    
    print(f"\n{color} RECOMMENDATION: {recommendation}")
    print(f"   Action: {action}")
    print(f"   Overall Score: {total_score}/10\n")
    
    print("="*70 + "\n")
    
    return {
        'ticker': ticker,
        'recommendation': recommendation,
        'score': total_score,
        'scraped': scraped_data,
        'dark_pool': dp_result,
        'social': social_result
    }

# Test on MU and ELWS
try:
    print("\n" + "="*80)
    print("TESTING ON YOUR TICKERS: MU & ELWS")
    print("="*80 + "\n")
    
    mu_analysis = analyze_ticker_complete('MU')
    print("\n")
    elws_analysis = analyze_ticker_complete('ELWS')
    
    print("\n✅ COMPLETE ANALYSIS WORKS!")
    ANALYSIS_OK = True
    
except Exception as e:
    print(f"❌ ANALYSIS ERROR: {e}")
    ANALYSIS_OK = False

print("\n" + "="*80 + "\n")

# ============================================================================
# CELL 10: FINAL TEST REPORT
# ============================================================================

print("\n" + "="*80)
print("📊 FINAL TEST REPORT")
print("="*80 + "\n")

test_results = {
    'Universal Scraper': SCRAPER_OK if 'SCRAPER_OK' in locals() else False,
    'Dark Pool Tracker': DARK_POOL_OK if 'DARK_POOL_OK' in locals() else False,
    'Social Explosion Detector': SOCIAL_OK if 'SOCIAL_OK' in locals() else False,
    'Insider Trading': modules_status.get('insider', False) if 'modules_status' in locals() else False,
    'Earnings Predictor': modules_status.get('earnings', False) if 'modules_status' in locals() else False,
    'Short Squeeze Scanner': modules_status.get('squeeze', False) if 'modules_status' in locals() else False,
    'Pump Detector': modules_status.get('pump', False) if 'modules_status' in locals() else False,
    'Portfolio Manager': PORTFOLIO_OK if 'PORTFOLIO_OK' in locals() else False,
    'Chart Generation': CHART_OK if 'CHART_OK' in locals() else False,
    'Complete Analysis': ANALYSIS_OK if 'ANALYSIS_OK' in locals() else False,
}

passed = sum(test_results.values())
total = len(test_results)

print(f"Tests Passed: {passed}/{total}\n")

for test_name, passed in test_results.items():
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} - {test_name}")

print("\n" + "="*80)

if passed == total:
    print("🎉 ALL TESTS PASSED! SYSTEM IS READY!")
    print("\nNext step: Run ULTIMATE_PROFIT_DASHBOARD.py")
elif passed >= total * 0.8:
    print("⚠️ MOST TESTS PASSED - System mostly ready")
    print(f"\nFix {total - passed} failing modules, then launch!")
else:
    print("❌ MULTIPLE TESTS FAILED - Need debugging")
    print("\nCheck errors above and fix before launching")

print("="*80 + "\n")

# ============================================================================
# SAVE TEST RESULTS
# ============================================================================

import json
from datetime import datetime

test_report = {
    'timestamp': datetime.now().isoformat(),
    'tests': test_results,
    'passed': passed,
    'total': total,
    'success_rate': (passed / total) * 100,
    'ready_to_launch': passed >= total * 0.8
}

with open('/content/drive/MyDrive/Quantum_AI_Cockpit/test_results.json', 'w') as f:
    json.dump(test_report, f, indent=2)

print("📄 Test results saved to: test_results.json")
print("\n🚀 You can now review the visuals and decide if you want to proceed!")

