"""
LEGENDARY TICKERS - Complete 90+ Ticker Universe
Combines Alpha 76 + User Additions + Perplexity Hot Picks
"""

# Alpha 76 - Small-cap momentum tickers ($50M - $5B market cap)
ALPHA_76 = [
    # AI/ML Infrastructure
    'IONQ', 'RGTI', 'QUBT', 'ARQQ', 'QBTS',
    
    # Quantum Computing
    'IONQ', 'RGTI', 'QUBT',
    
    # Energy/Utilities
    'OKLO', 'SMR', 'VST', 'CEG',
    
    # Fintech
    'HOOD', 'SOFI', 'AFRM', 'UPST',
    
    # Cloud/SaaS
    'SNOW', 'NOW', 'DDOG', 'NET', 'CRWD', 'S', 'ZS',
    
    # Semiconductors
    'NVDA', 'AMD', 'AVGO', 'MRVL', 'QCOM', 'INTC', 'ARM', 'ASML',
    
    # EV/Transportation
    'TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI',
    
    # Space/Defense
    'LUNR', 'RKLB', 'ASTS', 'PL',
    
    # Biotech/Healthcare
    'RXRX', 'SDGR', 'BEAM', 'CRSP', 'NTLA', 'EDIT', 'VERV',
    
    # Crypto/Web3
    'COIN', 'MSTR', 'RIOT', 'MARA', 'CLSK', 'CIFR',
    
    # Cybersecurity
    'PANW', 'FTNT', 'OKTA', 'ZS', 'CRWD',
    
    # AI Hardware
    'SMCI', 'DELL', 'HPE',
    
    # Data Centers
    'EQIX', 'DLR',
    
    # High-Growth Tech
    'PLTR', 'U', 'PATH', 'DOCN',
    
    # Mega-cap for regime analysis
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX',
    
    # Indices
    'SPY', 'QQQ', 'IWM', 'DIA'
]

# User's Additions (13 tickers)
USER_ADDITIONS = [
    'SERV',  # Serve Robotics
    'APLD',  # Applied Digital (data centers/crypto mining)
    'MRVL',  # Marvell (semiconductors)
    'OKLO',  # Nuclear energy (duplicate from Alpha 76, kept for emphasis)
    'SMR',   # NuScale (nuclear, duplicate, kept for emphasis)
    'HOOD',  # Robinhood (duplicate, kept for emphasis)
    'LUNR',  # Intuitive Machines (space, duplicate, kept for emphasis)
    'SNOW',  # Snowflake (duplicate, kept for emphasis)
    'NOW',   # ServiceNow (duplicate, kept for emphasis)
    'BA',    # Boeing (aerospace/defense)
    'LYFT',  # Lyft (rideshare)
    'RIVN',  # Rivian (EV, duplicate, kept for emphasis)
    'RXRX',  # Recursion (biotech AI, duplicate, kept for emphasis)
]

# Perplexity Hot Picks (High-conviction momentum plays)
PERPLEXITY_HOT_PICKS = [
    'DGNX',  # DiGiPath (+21% day, +293% revenue YoY, cannabis testing)
    'ELWS',  # Earlyworks (blockchain/Web3 infrastructure)
    'PALI',  # Palisade Bio (biotech catalyst pending)
]

# Combine and deduplicate
def get_legendary_tickers() -> list:
    """
    Returns deduplicated list of all legendary tickers.
    Total: ~92 tickers (after deduplication)
    """
    all_tickers = list(set(ALPHA_76 + USER_ADDITIONS + PERPLEXITY_HOT_PICKS))
    return sorted(all_tickers)


# Ticker Categorization (for sector-specific training)
TICKER_CATEGORIES = {
    'quantum_computing': ['IONQ', 'RGTI', 'QUBT', 'ARQQ', 'QBTS'],
    'nuclear_energy': ['OKLO', 'SMR', 'VST', 'CEG'],
    'fintech': ['HOOD', 'SOFI', 'AFRM', 'UPST'],
    'cloud_saas': ['SNOW', 'NOW', 'DDOG', 'NET', 'CRWD', 'S', 'ZS'],
    'semiconductors': ['NVDA', 'AMD', 'AVGO', 'MRVL', 'QCOM', 'INTC', 'ARM', 'ASML'],
    'ev_transport': ['TSLA', 'RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'LYFT'],
    'space_defense': ['LUNR', 'RKLB', 'ASTS', 'PL', 'BA'],
    'biotech_health': ['RXRX', 'SDGR', 'BEAM', 'CRSP', 'NTLA', 'EDIT', 'VERV', 'PALI'],
    'crypto_web3': ['COIN', 'MSTR', 'RIOT', 'MARA', 'CLSK', 'CIFR', 'ELWS'],
    'cybersecurity': ['PANW', 'FTNT', 'OKTA', 'ZS', 'CRWD'],
    'ai_hardware': ['SMCI', 'DELL', 'HPE', 'APLD'],
    'data_centers': ['EQIX', 'DLR'],
    'high_growth_tech': ['PLTR', 'U', 'PATH', 'DOCN', 'DGNX'],
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NFLX'],
    'indices': ['SPY', 'QQQ', 'IWM', 'DIA'],
    'robotics_automation': ['SERV'],
}


def get_tickers_by_category(category: str) -> list:
    """Get tickers in a specific category"""
    return TICKER_CATEGORIES.get(category, [])


def get_all_categories() -> list:
    """Get list of all categories"""
    return list(TICKER_CATEGORIES.keys())


# Market Cap Tiers (for regime-specific training)
MARKET_CAP_TIERS = {
    'mega_cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],  # $500B+
    'large_cap': ['AMD', 'AVGO', 'NFLX', 'SNOW', 'PANW', 'CRWD', 'NOW'],    # $100B-$500B
    'mid_cap': ['MRVL', 'HOOD', 'SOFI', 'RIVN', 'COIN', 'PLTR'],            # $10B-$100B
    'small_cap': ['IONQ', 'RGTI', 'OKLO', 'LUNR', 'RXRX', 'DGNX', 'ELWS'],  # $50M-$10B
}


# Volatility Tiers (for risk management)
VOLATILITY_TIERS = {
    'low_vol': ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL'],           # <20% annual vol
    'medium_vol': ['NVDA', 'AMD', 'SNOW', 'PLTR', 'HOOD'],        # 20-40% annual vol
    'high_vol': ['TSLA', 'IONQ', 'RGTI', 'RIVN', 'COIN'],         # 40-60% annual vol
    'extreme_vol': ['DGNX', 'ELWS', 'PALI', 'QUBT', 'MSTR'],      # 60%+ annual vol
}


if __name__ == '__main__':
    tickers = get_legendary_tickers()
    print("="*80)
    print("🎯 LEGENDARY TICKERS UNIVERSE")
    print("="*80)
    print(f"\nTotal Tickers: {len(tickers)}")
    print(f"\nBreakdown:")
    print(f"  Alpha 76: {len(set(ALPHA_76))} tickers")
    print(f"  User Additions: {len(set(USER_ADDITIONS))} tickers")
    print(f"  Perplexity Hot Picks: {len(PERPLEXITY_HOT_PICKS)} tickers")
    print(f"  After Deduplication: {len(tickers)} tickers")
    
    print(f"\nCategories:")
    for category, ticker_list in TICKER_CATEGORIES.items():
        print(f"  {category}: {len(ticker_list)} tickers")
    
    print(f"\nMarket Cap Distribution:")
    for tier, ticker_list in MARKET_CAP_TIERS.items():
        print(f"  {tier}: {len(ticker_list)} tickers")
    
    print(f"\nVolatility Distribution:")
    for tier, ticker_list in VOLATILITY_TIERS.items():
        print(f"  {tier}: {len(ticker_list)} tickers")
    
    print(f"\nFull List:")
    for i, ticker in enumerate(tickers, 1):
        print(f"  {i:2}. {ticker}", end="   ")
        if i % 6 == 0:
            print()  # New line every 6 tickers
    print("\n" + "="*80)
