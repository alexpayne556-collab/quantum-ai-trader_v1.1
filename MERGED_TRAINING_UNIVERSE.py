"""
MERGED TRAINING UNIVERSE: MASTER + ALPHA 76
============================================
Combines Master Training Watchlist (94 tickers) with Alpha 76 (76 tickers)
for comprehensive coverage of portfolio, sector leaders, and high-velocity small-caps.

Total: 155 unique tickers
- Portfolio tracking: KDK, HOOD, BA, WMT
- Recently sold: AAPL, YYAI, SERV
- Sector leaders: Tech, Finance, Healthcare, Consumer, Energy, Industrials
- Small/mid-cap growth: 40 high-conviction plays
- Regime indicators: ETFs (SPY, QQQ, IWM, etc.)

Updated: December 8, 2024
"""

from MASTER_TRAINING_WATCHLIST import get_master_watchlist, get_sector_breakdown as get_master_sectors
from ALPHA_76_WATCHLIST import get_alpha_76_tickers, get_alpha_76_by_sector

def get_merged_universe():
    """Get merged universe of Master + Alpha 76 (deduplicated)."""
    master = set(get_master_watchlist())
    alpha76 = set(get_alpha_76_tickers())
    
    merged = master.union(alpha76)
    overlap = master.intersection(alpha76)
    
    return {
        'all_tickers': sorted(list(merged)),
        'total': len(merged),
        'from_master': len(master),
        'from_alpha76': len(alpha76),
        'overlap': sorted(list(overlap)),
        'overlap_count': len(overlap)
    }

def get_priority_tiers():
    """
    Get 3-tier training priority system.
    
    Tier 1 (Daily): Portfolio + High-conviction + Regime indicators
    Tier 2 (Weekly): Sector leaders + Mid-cap growth
    Tier 3 (Bi-weekly): Stable large-caps + ETFs
    """
    # Your portfolio (highest priority)
    portfolio = ['KDK', 'HOOD', 'BA', 'WMT']
    
    # Recently sold (re-entry monitoring)
    recently_sold = ['AAPL', 'YYAI', 'SERV']
    
    # High-conviction small-caps from Alpha 76
    small_cap_leaders = [
        # Space
        'RKLB', 'ASTS', 'LUNR', 'JOBY',
        # Biotech
        'VKTX', 'NTLA', 'BEAM', 'CYTK', 'AKRO',
        # AI/Robotics
        'IONQ', 'SYM', 'SOUN', 'AMBA',
        # Fintech
        'SOFI', 'UPST', 'COIN',
        # Software
        'APP', 'DUOL', 'PATH', 'S',
        # Green Energy
        'FLNC', 'NXT', 'BE'
    ]
    
    # Regime indicators (ETFs + mega-caps)
    regime_indicators = [
        'SPY', 'QQQ', 'IWM', 'TLT', 'GLD',
        'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN', 'TSLA',
        'JPM', 'XOM', 'UNH'
    ]
    
    # Tier 1: Daily training (30 tickers)
    tier1 = portfolio + recently_sold + small_cap_leaders + regime_indicators
    tier1 = list(dict.fromkeys(tier1))  # Remove duplicates, preserve order
    
    # Get all tickers
    all_tickers = get_merged_universe()['all_tickers']
    
    # Tier 2: Weekly training (60 tickers)
    # Sector leaders + remaining small-caps
    tier2_candidates = [t for t in all_tickers if t not in tier1]
    
    # Prioritize by market cap proxy (get first 60)
    tech_leaders = ['AMD', 'AVGO', 'ORCL', 'CRM', 'ADBE', 'NOW', 'INTU', 'SNOW', 'PLTR', 'CRWD', 'NET', 'DDOG', 'ZS', 'PANW', 'FTNT']
    finance_leaders = ['BAC', 'GS', 'MS', 'C', 'WFC', 'SCHW', 'BLK', 'AXP', 'V']
    healthcare_leaders = ['JNJ', 'LLY', 'ABBV', 'MRK', 'PFE', 'TMO', 'GILD']
    consumer_leaders = ['HD', 'NKE', 'SBUX', 'TGT', 'LOW', 'MCD', 'WMT']
    energy_leaders = ['CVX', 'COP', 'SLB', 'EOG', 'OXY']
    industrials_leaders = ['CAT', 'DE', 'GE', 'LMT', 'RTX']
    
    # Alpha 76 remaining
    alpha76_remaining = [
        # Space
        'ACHR', 'SPIR', 'PL', 'RDW', 'BKSY', 'LLAP',
        # AI/Robotics
        'RGTI', 'QUBT', 'SERV', 'BBAI', 'REKR', 'LAZR', 'INVZ', 'OUST', 'CRNC', 'AEVA',
        # Biotech
        'KOD', 'AKYA', 'HALO', 'VRDN', 'URGN', 'LQDA', 'PVLA', 'OKYO', 'IOBT', 'SPRO',
        # Green Energy
        'STEM', 'AMSC', 'ARRY', 'SHLS', 'ENOV', 'QS', 'PLUG', 'FCEL', 'BLDP',
        # Fintech
        'AFRM', 'MQ', 'ALKT', 'MARA', 'RIOT', 'CLSK',
        # Software
        'ONON', 'CELH', 'ELF', 'AMPL', 'ESTC', 'DOCN', 'VRNS'
    ]
    
    tier2 = (tech_leaders + finance_leaders + healthcare_leaders + 
             consumer_leaders + energy_leaders + industrials_leaders + 
             alpha76_remaining)
    tier2 = [t for t in tier2 if t in all_tickers and t not in tier1]
    tier2 = tier2[:60]  # Limit to 60
    
    # Tier 3: Bi-weekly training (remaining tickers)
    tier3 = [t for t in all_tickers if t not in tier1 and t not in tier2]
    
    return {
        'tier1_daily': tier1,
        'tier1_count': len(tier1),
        'tier2_weekly': tier2,
        'tier2_count': len(tier2),
        'tier3_biweekly': tier3,
        'tier3_count': len(tier3)
    }

def get_sector_summary():
    """Get sector breakdown for merged universe."""
    merged = get_merged_universe()
    all_tickers = set(merged['all_tickers'])
    
    # Master sectors
    master_sectors = get_master_sectors()
    
    # Alpha 76 sectors
    alpha76_sectors = get_alpha_76_by_sector()
    
    summary = {
        'total_tickers': merged['total'],
        'from_master': merged['from_master'],
        'from_alpha76': merged['from_alpha76'],
        'overlap': merged['overlap_count'],
        'sectors': {
            'Your Portfolio': 4,
            'Recently Sold': 3,
            'Tech Leaders': len([t for t in master_sectors.get('TECH_LEADERS', []) if t in all_tickers]),
            'Finance': len([t for t in master_sectors.get('FINANCE', []) if t in all_tickers]),
            'Healthcare': len([t for t in master_sectors.get('HEALTHCARE', []) if t in all_tickers]),
            'Consumer': len([t for t in master_sectors.get('CONSUMER', []) if t in all_tickers]),
            'Energy': len([t for t in master_sectors.get('ENERGY', []) if t in all_tickers]),
            'Industrials': len([t for t in master_sectors.get('INDUSTRIALS', []) if t in all_tickers]),
            'Autonomous/AI': len(alpha76_sectors.get('autonomous_ai_hardware', {})),
            'Space Economy': len(alpha76_sectors.get('space_economy', {})),
            'Biotech': len(alpha76_sectors.get('biotech_high_beta', {})),
            'Green Energy': len(alpha76_sectors.get('green_energy_grid', {})),
            'Fintech': len(alpha76_sectors.get('fintech_crypto', {})),
            'Software': len(alpha76_sectors.get('consumer_software', {})),
            'ETFs': 10
        }
    }
    
    return summary

if __name__ == "__main__":
    print("=" * 80)
    print("MERGED TRAINING UNIVERSE: MASTER + ALPHA 76")
    print("=" * 80)
    
    # Get merged universe
    merged = get_merged_universe()
    
    print(f"\n📊 UNIVERSE STATISTICS:")
    print(f"   Total Unique Tickers: {merged['total']}")
    print(f"   From Master Watchlist: {merged['from_master']}")
    print(f"   From Alpha 76: {merged['from_alpha76']}")
    print(f"   Overlap: {merged['overlap_count']} tickers")
    
    if merged['overlap']:
        print(f"\n🔗 OVERLAPPING TICKERS:")
        for ticker in merged['overlap']:
            print(f"   - {ticker}")
    
    # Get sector summary
    summary = get_sector_summary()
    
    print(f"\n🎯 SECTOR BREAKDOWN:")
    for sector, count in summary['sectors'].items():
        print(f"   {sector}: {count} tickers")
    
    # Get priority tiers
    tiers = get_priority_tiers()
    
    print(f"\n⚡ TRAINING PRIORITY TIERS:")
    print(f"\n   TIER 1 (Daily Training): {tiers['tier1_count']} tickers")
    print(f"   - Portfolio: KDK, HOOD, BA, WMT")
    print(f"   - Recently Sold: AAPL, YYAI, SERV")
    print(f"   - Small-cap Leaders: RKLB, IONQ, VKTX, SOFI, APP, etc.")
    print(f"   - Regime Indicators: SPY, QQQ, NVDA, MSFT, etc.")
    
    print(f"\n   TIER 2 (Weekly Training): {tiers['tier2_count']} tickers")
    print(f"   - Sector Leaders: AMD, JPM, LLY, CVX, CAT, etc.")
    print(f"   - Remaining Small-caps: RGTI, AKRO, STEM, AFRM, etc.")
    
    print(f"\n   TIER 3 (Bi-weekly Training): {tiers['tier3_count']} tickers")
    print(f"   - Stable Large-caps + Remaining ETFs")
    
    # Save to files
    print(f"\n💾 SAVING FILES:")
    
    # Save merged watchlist
    with open('merged_watchlist.txt', 'w') as f:
        for ticker in merged['all_tickers']:
            f.write(f"{ticker}\n")
    print(f"   ✅ merged_watchlist.txt ({merged['total']} tickers)")
    
    # Save tier 1 (daily training)
    with open('tier1_daily.txt', 'w') as f:
        for ticker in tiers['tier1_daily']:
            f.write(f"{ticker}\n")
    print(f"   ✅ tier1_daily.txt ({tiers['tier1_count']} tickers)")
    
    # Save tier 2 (weekly training)
    with open('tier2_weekly.txt', 'w') as f:
        for ticker in tiers['tier2_weekly']:
            f.write(f"{ticker}\n")
    print(f"   ✅ tier2_weekly.txt ({tiers['tier2_count']} tickers)")
    
    # Save tier 3 (bi-weekly training)
    with open('tier3_biweekly.txt', 'w') as f:
        for ticker in tiers['tier3_biweekly']:
            f.write(f"{ticker}\n")
    print(f"   ✅ tier3_biweekly.txt ({tiers['tier3_count']} tickers)")
    
    print(f"\n" + "=" * 80)
    print("✅ MERGED UNIVERSE READY")
    print("=" * 80)
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"   1. Train Tier 1 (30 tickers) - Daily retraining for portfolio + high-conviction")
    print(f"   2. Train Tier 2 (60 tickers) - Weekly retraining for sector leaders")
    print(f"   3. Train Tier 3 (65 tickers) - Bi-weekly retraining for stable positions")
    print(f"   4. Run alpha_76_pipeline.py to filter by institutional activity")
    print(f"   5. Validate early detection on H2 2024 events (NVDA, RKLB, VKTX)")
