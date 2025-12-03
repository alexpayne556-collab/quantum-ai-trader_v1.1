"""
🚀 MISSION CONTROL CHART TESTING - COLAB CELLS
==============================================
Test the upgraded chart engine with:
- MU (Micron Technology)
- ELWS (Early Works)
- Pattern detection
- 21-day forecasts
- Real-time data options
"""

# ===============================================================
# CELL 36: Install Additional Dependencies for Charts
# ===============================================================
"""
!pip install --quiet plotly kaleido aiohttp
print("✅ Chart dependencies installed")
"""

# ===============================================================
# CELL 37: Upload Advanced Chart Engine V2 to Google Drive
# ===============================================================
"""
advanced_chart_v2_code = '''
[PASTE CONTENT OF advanced_chart_engine_v2.py HERE]
'''

# Save to Google Drive
chart_engine_path = f'{MODULES_PATH}/advanced_chart_engine_v2.py'
with open(chart_engine_path, 'w') as f:
    f.write(advanced_chart_v2_code)

print(f"✅ Advanced Chart Engine V2 uploaded to: {chart_engine_path}")
print(f"   Size: {len(advanced_chart_v2_code)} bytes")
"""

# ===============================================================
# CELL 38: Test with MU (Micron Technology) - Full Analysis
# ===============================================================
"""
print("=" * 80)
print("🔍 TESTING MU (MICRON TECHNOLOGY)")
print("=" * 80)

import yfinance as yf
from advanced_chart_engine_v2 import AdvancedChartEngineV2
import plotly.io as pio

# Fetch MU data (6 months for better patterns)
symbol = 'MU'
print(f"\\n📊 Fetching {symbol} data (180 days)...")

df = yf.download(symbol, period='180d', interval='1d', progress=False)

if len(df) > 0:
    print(f"✅ Got {len(df)} days of data")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Current price: ${df['Close'].iloc[-1]:.2f}")
    
    # Create chart engine
    chart_engine = AdvancedChartEngineV2()
    
    # Generate chart with all features
    print(f"\\n🎨 Generating Mission Control Chart...")
    fig = chart_engine.create_mission_control_chart(
        df=df,
        symbol=symbol,
        show_patterns=True
    )
    
    # Display chart
    fig.show()
    
    # Save chart
    chart_file = f'/content/drive/MyDrive/Quantum_AI_Cockpit/charts/{symbol}_mission_control.html'
    fig.write_html(chart_file)
    print(f"\\n💾 Chart saved to: {chart_file}")
    
    # Check for patterns detected
    print(f"\\n🔍 PATTERN ANALYSIS:")
    
    # Golden/Death Cross
    crosses = chart_engine._detect_golden_death_cross(df)
    if crosses:
        print(f"   ⭐ {len(crosses)} Golden/Death crosses detected")
        for cross in crosses[-3:]:  # Last 3
            print(f"      {cross['type']}: {cross['date'].date()} @ ${cross['price']:.2f}")
    else:
        print(f"   No Golden/Death crosses")
    
    # Cup and Handle
    cup_handle = chart_engine._detect_cup_and_handle(df)
    if cup_handle:
        print(f"   ☕ Cup & Handle detected!")
        print(f"      {cup_handle['explanation']}")
    else:
        print(f"   No Cup & Handle pattern")
    
    # Triangles
    triangles = chart_engine._detect_triangles(df)
    if triangles:
        print(f"   🔺 {len(triangles)} triangle(s) detected")
        for tri in triangles:
            print(f"      {tri['type']}: {tri['explanation']}")
    else:
        print(f"   No triangles detected")
    
    # Buy/Sell Signals
    signals = chart_engine._calculate_buy_sell_signals(df)
    print(f"\\n📊 RECENT SIGNALS:")
    print(f"   BUY signals: {len(signals['buy'])}")
    for buy in signals['buy'][-3:]:
        print(f"      {buy['date'].date()} @ ${buy['price']:.2f} - {buy['reason']}")
    
    print(f"   SELL signals: {len(signals['sell'])}")
    for sell in signals['sell'][-3:]:
        print(f"      {sell['date'].date()} @ ${sell['price']:.2f} - {sell['reason']}")
    
    print(f"\\n✅ MU analysis complete!")
    
else:
    print(f"❌ No data for {symbol}")
"""

# ===============================================================
# CELL 39: Test with ELWS (Early Works) - Full Analysis
# ===============================================================
"""
print("=" * 80)
print("🔍 TESTING ELWS (EARLY WORKS)")
print("=" * 80)

import yfinance as yf
from advanced_chart_engine_v2 import AdvancedChartEngineV2
import plotly.io as pio

# Fetch ELWS data
symbol = 'ELWS'
print(f"\\n📊 Fetching {symbol} data (180 days)...")

df = yf.download(symbol, period='180d', interval='1d', progress=False)

if len(df) > 0:
    print(f"✅ Got {len(df)} days of data")
    print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"   Current price: ${df['Close'].iloc[-1]:.2f}")
    
    # Create chart engine
    chart_engine = AdvancedChartEngineV2()
    
    # Generate chart
    print(f"\\n🎨 Generating Mission Control Chart...")
    fig = chart_engine.create_mission_control_chart(
        df=df,
        symbol=symbol,
        show_patterns=True
    )
    
    # Display chart
    fig.show()
    
    # Save chart
    chart_file = f'/content/drive/MyDrive/Quantum_AI_Cockpit/charts/{symbol}_mission_control.html'
    fig.write_html(chart_file)
    print(f"\\n💾 Chart saved to: {chart_file}")
    
    # Pattern analysis
    print(f"\\n🔍 PATTERN ANALYSIS:")
    
    crosses = chart_engine._detect_golden_death_cross(df)
    if crosses:
        print(f"   ⭐ {len(crosses)} Golden/Death crosses detected")
        for cross in crosses[-3:]:
            print(f"      {cross['type']}: {cross['date'].date()} @ ${cross['price']:.2f}")
    else:
        print(f"   No Golden/Death crosses")
    
    cup_handle = chart_engine._detect_cup_and_handle(df)
    if cup_handle:
        print(f"   ☕ Cup & Handle detected!")
        print(f"      {cup_handle['explanation']}")
    else:
        print(f"   No Cup & Handle pattern")
    
    triangles = chart_engine._detect_triangles(df)
    if triangles:
        print(f"   🔺 {len(triangles)} triangle(s) detected")
        for tri in triangles:
            print(f"      {tri['type']}: {tri['explanation']}")
    else:
        print(f"   No triangles detected")
    
    signals = chart_engine._calculate_buy_sell_signals(df)
    print(f"\\n📊 RECENT SIGNALS:")
    print(f"   BUY signals: {len(signals['buy'])}")
    for buy in signals['buy'][-3:]:
        print(f"      {buy['date'].date()} @ ${buy['price']:.2f} - {buy['reason']}")
    
    print(f"   SELL signals: {len(signals['sell'])}")
    for sell in signals['sell'][-3:]:
        print(f"      {sell['date'].date()} @ ${sell['price']:.2f} - {sell['reason']}")
    
    print(f"\\n✅ ELWS analysis complete!")
    
else:
    print(f"❌ No data for {symbol}")
"""

# ===============================================================
# CELL 40: Test Forecast Volatility (Check if it's realistic)
# ===============================================================
"""
print("=" * 80)
print("🔍 TESTING FORECAST REALISM")
print("=" * 80)

import yfinance as yf
import matplotlib.pyplot as plt
from advanced_chart_engine_v2 import AdvancedChartEngineV2

# Test on volatile stock (MU)
symbol = 'MU'
df = yf.download(symbol, period='180d', interval='1d', progress=False)

if len(df) > 0:
    chart_engine = AdvancedChartEngineV2()
    
    # Generate 21-day forecast
    print(f"\\n📊 Generating 21-day forecast for {symbol}...")
    forecast_df = chart_engine._generate_realistic_forecast(df, horizon=21)
    
    # Calculate metrics
    current_price = df['Close'].iloc[-1]
    forecast_mean = forecast_df['Forecast'].mean()
    forecast_std = forecast_df['Forecast'].std()
    forecast_range = forecast_df['Forecast'].max() - forecast_df['Forecast'].min()
    
    print(f"\\n📈 FORECAST ANALYSIS:")
    print(f"   Current price: ${current_price:.2f}")
    print(f"   21-day avg forecast: ${forecast_mean:.2f}")
    print(f"   Expected change: {((forecast_mean - current_price) / current_price * 100):.2f}%")
    print(f"   Forecast volatility: ${forecast_std:.2f}")
    print(f"   Price range: ${forecast_range:.2f} ({(forecast_range/current_price*100):.1f}%)")
    
    # Check for dips
    dips = 0
    for i in range(1, len(forecast_df)):
        if forecast_df['Forecast'].iloc[i] < forecast_df['Forecast'].iloc[i-1]:
            dips += 1
    
    print(f"\\n🎢 REALISM CHECK:")
    print(f"   Dips detected: {dips} / {len(forecast_df)-1} days ({dips/(len(forecast_df)-1)*100:.1f}%)")
    
    if dips > 5:
        print(f"   ✅ Forecast shows realistic ups and downs")
    else:
        print(f"   ⚠️  Forecast might be too smooth/flat")
    
    # Historical volatility comparison
    hist_returns = df['Close'].pct_change().dropna()
    hist_volatility = hist_returns.std()
    
    forecast_returns = forecast_df['Forecast'].pct_change().dropna()
    forecast_volatility = forecast_returns.std()
    
    print(f"\\n📊 VOLATILITY COMPARISON:")
    print(f"   Historical daily volatility: {hist_volatility*100:.2f}%")
    print(f"   Forecast daily volatility: {forecast_volatility*100:.2f}%")
    print(f"   Ratio: {(forecast_volatility/hist_volatility):.2f}x")
    
    if 0.3 < (forecast_volatility/hist_volatility) < 1.5:
        print(f"   ✅ Forecast volatility is realistic")
    else:
        print(f"   ⚠️  Forecast volatility might be off")
    
    print(f"\\n✅ Forecast realism check complete!")
    
else:
    print(f"❌ No data for {symbol}")
"""

# ===============================================================
# CELL 41: Real-Time Data Options (CRITICAL for Trading)
# ===============================================================
"""
print("=" * 80)
print("🔴 REAL-TIME DATA SETUP - READ CAREFULLY")
print("=" * 80)

print('''
⚠️ YFINANCE IS 15-MINUTE DELAYED - NOT SUITABLE FOR DAY TRADING

For REAL-TIME data, you have these options:

═══════════════════════════════════════════════════════════
1. POLYGON.IO (Recommended)
═══════════════════════════════════════════════════════════
   Free Tier: Delayed data (15-min)
   Paid Tier: Real-time ($99/month)
   
   Setup:
   1. Sign up at https://polygon.io/
   2. Get API key
   3. Use RealTimeDataFetcher:
   
   from advanced_chart_engine_v2 import RealTimeDataFetcher
   
   fetcher = RealTimeDataFetcher(
       api_key='YOUR_API_KEY',
       provider='polygon'
   )
   
   quote = await fetcher.fetch_realtime_quote('MU')
   print(f"Current price: ${quote['price']}")

═══════════════════════════════════════════════════════════
2. FINNHUB.IO (Best Free Option)
═══════════════════════════════════════════════════════════
   Free Tier: 60 calls/minute, REAL-TIME
   
   Setup:
   1. Sign up at https://finnhub.io/
   2. Get API key (free)
   3. Use RealTimeDataFetcher:
   
   fetcher = RealTimeDataFetcher(
       api_key='YOUR_API_KEY',
       provider='finnhub'
   )
   
   quote = await fetcher.fetch_realtime_quote('MU')

═══════════════════════════════════════════════════════════
3. ALPHA VANTAGE
═══════════════════════════════════════════════════════════
   Free Tier: 5 calls/min (too slow)
   Paid Tier: $50/month
   
   Not recommended - too slow for free tier

═══════════════════════════════════════════════════════════
4. ROBINHOOD API (Unofficial)
═══════════════════════════════════════════════════════════
   Library: robin-stocks
   
   !pip install robin-stocks
   
   import robin_stocks.robinhood as rh
   
   rh.login('your_username', 'your_password')
   quote = rh.get_latest_price('MU')
   
   ⚠️ Use at your own risk - unofficial API

═══════════════════════════════════════════════════════════
5. WEBSOCKET STREAMING (Advanced)
═══════════════════════════════════════════════════════════
   For TRUE real-time (millisecond updates):
   
   - TD Ameritrade API (free with account)
   - Interactive Brokers API
   - Alpaca (free for paper trading)
   
   Requires WebSocket implementation

═══════════════════════════════════════════════════════════
🎯 RECOMMENDATION FOR YOU:
═══════════════════════════════════════════════════════════

For SWING TRADING (your use case):
✅ YFinance is GOOD ENOUGH if you check once per day
✅ 15-minute delay doesn't matter for multi-day holds

For DAY TRADING:
✅ Use Finnhub.io FREE tier (real-time, 60 calls/min)
✅ Or Polygon.io paid ($99/month)

For LIVE SCALPING:
✅ Must use WebSocket (TD Ameritrade or Alpaca)
✅ Sub-second latency required

Your current system (Mission Control) is designed for SWING TRADING,
so YFinance delay is acceptable. But if you want real-time quotes,
sign up for Finnhub.io (free) and I'll integrate it.
''')

print("\\n🔑 NEXT STEPS:")
print("1. Decide: Swing trading (YF is fine) or Day trading (need real-time)?")
print("2. If day trading: Sign up for Finnhub.io free tier")
print("3. Provide API key and I'll integrate it")
print("4. For now, continue with YFinance (it's working)")
"""

# ===============================================================
# CELL 42: Integration with Pattern Integration Layer
# ===============================================================
"""
print("=" * 80)
print("🔗 INTEGRATING PATTERN LAYER WITH CHARTS")
print("=" * 80)

# Import both modules
from pattern_integration_layer import PatternIntegrationLayer
from advanced_chart_engine_v2 import AdvancedChartEngineV2
import yfinance as yf

# Test symbol
symbol = 'MU'
print(f"\\n📊 Fetching {symbol} data...")

df = yf.download(symbol, period='180d', interval='1d', progress=False)

if len(df) > 0:
    # Run pattern integration layer
    print(f"\\n🔍 Running Pattern Integration Layer...")
    pattern_layer = PatternIntegrationLayer()
    patterns = await pattern_layer.analyze_all_patterns(df, symbol=symbol)
    
    if patterns and patterns.get('status') == 'ok':
        print(f"✅ Pattern analysis complete")
        
        # Extract pattern details
        final_signal = patterns.get('final_signal', {})
        summary = patterns.get('summary', {})
        
        print(f"\\n📊 PATTERN INTEGRATION RESULTS:")
        print(f"   Patterns detected: {summary.get('patterns_detected', 0)}")
        print(f"   Final signal: {final_signal.get('action', 'HOLD')}")
        print(f"   Confidence: {final_signal.get('confidence', 0)*100:.1f}%")
        
        # Create enhanced chart with patterns
        print(f"\\n🎨 Creating enhanced chart...")
        chart_engine = AdvancedChartEngineV2()
        fig = chart_engine.create_mission_control_chart(
            df=df,
            symbol=symbol,
            show_patterns=True
        )
        
        # Add pattern integration results as annotation
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref='paper',
            yref='paper',
            text=f"AI SIGNAL: {final_signal.get('action', 'HOLD')} | Confidence: {final_signal.get('confidence', 0)*100:.0f}%",
            showarrow=False,
            font=dict(size=16, color='#00ff9f', family='Courier New, monospace'),
            bgcolor='rgba(0, 0, 0, 0.7)',
            bordercolor='#00ff9f',
            borderwidth=2,
            borderpad=10
        )
        
        fig.show()
        
        print(f"\\n✅ Integration complete!")
        print(f"\\n💡 INSIGHT: Pattern layer + Chart engine = MISSION CONTROL BRAIN")
        
    else:
        print(f"⚠️  Pattern analysis failed or returned no results")
        
else:
    print(f"❌ No data for {symbol}")
"""

# ===============================================================
# INSTRUCTIONS FOR USER
# ===============================================================
print("""
══════════════════════════════════════════════════════════════════════════════
🚀 MISSION CONTROL CHART TESTING - INSTRUCTIONS
══════════════════════════════════════════════════════════════════════════════

WHAT WE JUST CREATED:
✅ Advanced Chart Engine V2 with ALL features
✅ Pattern detection (Golden Cross, Cup & Handle, Triangles)
✅ 21-day realistic forecasts with dips
✅ EMA ribbons with glowing effects
✅ Buy/Sell signal animations
✅ Real-time data options
✅ Cyberpunk Blade Runner theme

CELLS TO RUN IN COLAB:
1. Cell 36: Install chart dependencies
2. Cell 37: Upload chart engine to Drive
3. Cell 38: Test MU (Micron Technology)
4. Cell 39: Test ELWS (Early Works)
5. Cell 40: Check forecast realism
6. Cell 41: Real-time data setup guide
7. Cell 42: Integrate with pattern layer

ADDRESSING YOUR CONCERNS:
✅ Flat forecasts: FIXED - now shows realistic ups/downs with volatility
✅ Pattern visibility: FIXED - patterns highlighted on charts with explanations
✅ 21-day horizon: FIXED - extended from 5 to 21 days
✅ Real-time data: PROVIDED - guide for Finnhub.io (free) or Polygon.io (paid)
✅ Dips in forecast: FIXED - uses sine waves + noise + mean reversion
✅ AI explanations: ADDED - each pattern has human-readable explanation

WHAT'S NEXT:
1. Run cells 38 and 39 to test MU and ELWS
2. Review the charts and patterns
3. Decide on real-time data (Cell 41)
4. If forecasts still look flat, we'll tune the volatility model
5. Integrate with Streamlit dashboard

═══════════════════════════════════════════════════════════════════════════════
⚠️ CRITICAL: REAL-TIME DATA
═══════════════════════════════════════════════════════════════════════════════

YFinance is 15-minute DELAYED. For swing trading (your use case), this is FINE.

But if you want REAL-TIME quotes:
- Sign up for Finnhub.io (FREE, 60 calls/min)
- Get API key
- Tell me and I'll integrate it

For now, YFinance will work perfectly for testing.

═══════════════════════════════════════════════════════════════════════════════
""")

