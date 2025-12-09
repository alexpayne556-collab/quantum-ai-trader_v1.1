"""
Master Training Watchlist - 75+ Tickers
========================================
Comprehensive ticker universe for training all modules.

Designed for:
- Your portfolio tracking (KDK, HOOD, BA, WMT)
- High-momentum stocks across all sectors
- Small/mid-cap winners with growth potential
- Sector leaders and ETFs for regime detection

Updated: December 8, 2024
"""

# ============================================================================
# YOUR CURRENT PORTFOLIO (Priority tracking)
# ============================================================================
YOUR_PORTFOLIO = [
    'KDK',   # Kodiak Gas Services - Energy infrastructure
    'HOOD',  # Robinhood - Fintech/Brokerage
    'BA',    # Boeing - Aerospace/Defense
    'WMT',   # Walmart - Retail/Consumer staples
]

# Recently sold (track for re-entry opportunities)
RECENTLY_SOLD = [
    'AAPL',  # Apple - Tech
    'YYAI',  # YYAI - AI/Tech (small cap)
    'SERV',  # SERV - Services (small cap)
]

# ============================================================================
# SECTOR LEADERS (Large Cap - High Liquidity)
# ============================================================================

# Technology (20 tickers)
TECH_LEADERS = [
    'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'AMD', 'AVGO', 'ORCL',
    'CRM', 'ADBE', 'NOW', 'INTU', 'SNOW', 'PLTR', 'CRWD', 'NET',
    'DDOG', 'ZS', 'PANW', 'FTNT'
]

# Finance (10 tickers)
FINANCE_LEADERS = [
    'JPM', 'BAC', 'GS', 'MS', 'C', 'WFC', 'SCHW', 'BLK', 'AXP', 'V'
]

# Healthcare/Biotech (8 tickers)
HEALTHCARE_LEADERS = [
    'UNH', 'JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'GILD'
]

# Consumer Discretionary (8 tickers)
CONSUMER_DISCRETIONARY = [
    'AMZN', 'TSLA', 'HD', 'NKE', 'SBUX', 'TGT', 'LOW', 'MCD'
]

# Energy (6 tickers)
ENERGY_LEADERS = [
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY'
]

# Industrials/Aerospace (5 tickers)
INDUSTRIALS = [
    'CAT', 'DE', 'GE', 'LMT', 'RTX'
]

# ============================================================================
# SMALL/MID-CAP GROWTH STOCKS (High Potential)
# ============================================================================

# AI/Tech Small Caps (8 tickers)
AI_SMALL_CAPS = [
    'SMCI',  # Super Micro Computer - AI infrastructure
    'IONQ',  # IonQ - Quantum computing
    'BBAI',  # BigBear.ai - AI defense
    'SOUN',  # SoundHound AI - Voice AI
    'AI',    # C3.ai - Enterprise AI
    'RXRX',  # Recursion Pharma - AI drug discovery
    'MDAI',  # Spectral AI - Medical AI
    'PSTG',  # Pure Storage - Data infrastructure
]

# Fintech/Crypto Small Caps (5 tickers)
FINTECH_SMALL_CAPS = [
    'SOFI',  # SoFi - Digital banking
    'AFRM',  # Affirm - BNPL
    'COIN',  # Coinbase - Crypto exchange
    'NU',    # Nu Holdings - Brazilian fintech
    'UPST',  # Upstart - AI lending
]

# Clean Energy/EV Small Caps (5 tickers)
CLEANTECH_SMALL_CAPS = [
    'RIVN',  # Rivian - EV trucks
    'LCID',  # Lucid - EV luxury
    'CHPT',  # ChargePoint - EV charging
    'ENPH',  # Enphase - Solar inverters
    'RUN',   # Sunrun - Residential solar
]

# Biotech Small Caps (5 tickers)
BIOTECH_SMALL_CAPS = [
    'MRNA',  # Moderna - mRNA vaccines
    'CRSP',  # CRISPR Therapeutics - Gene editing
    'NTLA',  # Intellia - CRISPR
    'BEAM',  # Beam Therapeutics - Base editing
    'VRTX',  # Vertex Pharma - CF/sickle cell
]

# ============================================================================
# ETFs (Market Regime Detection)
# ============================================================================
ETFS = [
    'SPY',   # S&P 500
    'QQQ',   # Nasdaq 100
    'IWM',   # Russell 2000 (small cap)
    'DIA',   # Dow Jones
    'XLF',   # Financials sector
    'XLE',   # Energy sector
    'XLK',   # Technology sector
    'XLV',   # Healthcare sector
    'GLD',   # Gold (risk-off)
    'TLT',   # 20Y Treasury (bonds)
]

# ============================================================================
# MASTER WATCHLIST (All Combined)
# ============================================================================

def get_master_watchlist():
    """Get complete 75+ ticker watchlist."""
    return (
        YOUR_PORTFOLIO +
        TECH_LEADERS +
        FINANCE_LEADERS +
        HEALTHCARE_LEADERS +
        CONSUMER_DISCRETIONARY +
        ENERGY_LEADERS +
        INDUSTRIALS +
        AI_SMALL_CAPS +
        FINTECH_SMALL_CAPS +
        CLEANTECH_SMALL_CAPS +
        BIOTECH_SMALL_CAPS +
        ETFS
    )

def get_priority_watchlist():
    """Get priority tickers (your portfolio + high-conviction plays)."""
    return YOUR_PORTFOLIO + RECENTLY_SOLD + [
        'NVDA', 'AMD', 'SMCI',  # AI infrastructure
        'PLTR', 'CRWD', 'NET',  # Cybersecurity/AI
        'COIN', 'SOFI', 'HOOD',  # Fintech
        'TSLA', 'RIVN',  # EV
        'SPY', 'QQQ',  # Market benchmarks
    ]

def get_training_tiers():
    """
    Get training tiers by priority.
    
    Tier 1: Your portfolio + high-liquidity leaders (train daily)
    Tier 2: Sector leaders (train weekly)
    Tier 3: Small/mid-caps (train bi-weekly for emerging patterns)
    """
    return {
        'tier1_daily': YOUR_PORTFOLIO + TECH_LEADERS[:10] + FINANCE_LEADERS[:5] + ETFS[:5],
        'tier2_weekly': TECH_LEADERS[10:] + FINANCE_LEADERS[5:] + HEALTHCARE_LEADERS + CONSUMER_DISCRETIONARY,
        'tier3_biweekly': AI_SMALL_CAPS + FINTECH_SMALL_CAPS + CLEANTECH_SMALL_CAPS + BIOTECH_SMALL_CAPS + ETFS[5:]
    }

def get_sector_breakdown():
    """Get tickers organized by sector for sector rotation strategy."""
    return {
        'portfolio': YOUR_PORTFOLIO,
        'technology': TECH_LEADERS,
        'finance': FINANCE_LEADERS,
        'healthcare': HEALTHCARE_LEADERS,
        'consumer': CONSUMER_DISCRETIONARY,
        'energy': ENERGY_LEADERS,
        'industrials': INDUSTRIALS,
        'ai_smallcap': AI_SMALL_CAPS,
        'fintech_smallcap': FINTECH_SMALL_CAPS,
        'cleantech_smallcap': CLEANTECH_SMALL_CAPS,
        'biotech_smallcap': BIOTECH_SMALL_CAPS,
        'etfs': ETFS
    }

def validate_tickers():
    """Validate ticker list and remove duplicates."""
    master = get_master_watchlist()
    unique = list(dict.fromkeys(master))  # Preserve order, remove dupes
    
    print(f"Total tickers: {len(master)}")
    print(f"Unique tickers: {len(unique)}")
    print(f"Duplicates removed: {len(master) - len(unique)}")
    
    return unique

# ============================================================================
# RESEARCH RATIONALE
# ============================================================================

RESEARCH_NOTES = {
    'KDK': 'Natural gas compression - energy infrastructure play, high FCF',
    'HOOD': 'Retail trading platform - benefits from market volatility, growing crypto',
    'BA': 'Aerospace recovery - 737 MAX production ramp, China reopening',
    'WMT': 'Defensive consumer staples - inflation hedge, strong e-commerce growth',
    
    'NVDA': 'AI GPU leader - 80% market share, data center growth',
    'AMD': 'CPU/GPU challenger - server market share gains, AI momentum',
    'SMCI': 'AI infrastructure - direct lever on AI capex, high-beta NVDA proxy',
    
    'PLTR': 'AI/analytics platform - government contracts, commercial expansion',
    'CRWD': 'Cybersecurity leader - AI-powered, 90%+ gross margins',
    'NET': 'Edge computing/security - cloudflare network, DDoS protection',
    
    'COIN': 'Crypto exchange - Bitcoin correlation, institutional adoption',
    'SOFI': 'Digital bank - GAAP profitable Q3 2024, student loan refinancing',
    
    'TSLA': 'EV leader - FSD progress, Cybertruck ramp, energy storage',
    'RIVN': 'EV trucks - R1T/R1S production scaling, Amazon partnership',
    
    'SPY': 'S&P 500 benchmark - broad market exposure',
    'QQQ': 'Tech benchmark - 50% NASDAQ 100 is FAANG+NVDA',
    'IWM': 'Small cap - rate cut beneficiary, domestic economy play'
}

# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MASTER TRAINING WATCHLIST")
    print("=" * 80)
    
    master = validate_tickers()
    
    print(f"\n📊 Watchlist Breakdown:")
    print(f"   Your Portfolio: {len(YOUR_PORTFOLIO)} tickers")
    print(f"   Recently Sold: {len(RECENTLY_SOLD)} tickers")
    print(f"   Tech Leaders: {len(TECH_LEADERS)} tickers")
    print(f"   Finance: {len(FINANCE_LEADERS)} tickers")
    print(f"   Healthcare: {len(HEALTHCARE_LEADERS)} tickers")
    print(f"   Consumer: {len(CONSUMER_DISCRETIONARY)} tickers")
    print(f"   Energy: {len(ENERGY_LEADERS)} tickers")
    print(f"   Industrials: {len(INDUSTRIALS)} tickers")
    print(f"   AI Small Caps: {len(AI_SMALL_CAPS)} tickers")
    print(f"   Fintech Small Caps: {len(FINTECH_SMALL_CAPS)} tickers")
    print(f"   Clean Tech: {len(CLEANTECH_SMALL_CAPS)} tickers")
    print(f"   Biotech: {len(BIOTECH_SMALL_CAPS)} tickers")
    print(f"   ETFs: {len(ETFS)} tickers")
    print(f"   ─────────────────────────")
    print(f"   TOTAL: {len(master)} tickers")
    
    print(f"\n🎯 Priority Watchlist (High Conviction):")
    priority = get_priority_watchlist()
    for i, ticker in enumerate(priority, 1):
        note = RESEARCH_NOTES.get(ticker, 'N/A')
        print(f"   {i:2d}. {ticker:<6} - {note}")
    
    print(f"\n📈 Training Tiers:")
    tiers = get_training_tiers()
    print(f"   Tier 1 (Daily):    {len(tiers['tier1_daily'])} tickers")
    print(f"   Tier 2 (Weekly):   {len(tiers['tier2_weekly'])} tickers")
    print(f"   Tier 3 (Bi-weekly): {len(tiers['tier3_biweekly'])} tickers")
    
    print(f"\n💾 Saving to watchlist.txt...")
    with open('watchlist.txt', 'w') as f:
        for ticker in master:
            f.write(f"{ticker}\n")
    
    print(f"✅ Saved {len(master)} tickers to watchlist.txt")
    
    print(f"\n🔬 Research Notes Available for:")
    for ticker in sorted(RESEARCH_NOTES.keys()):
        print(f"   - {ticker}")
