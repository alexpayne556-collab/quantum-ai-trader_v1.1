"""
WATCHLIST 2026 - 20 High-Growth Small/Mid Cap Tickers
======================================================
"""

WATCHLIST = [
    # SPACE (5)
    "RKLB",   # Rocket Lab - $12B - Neutron rocket
    "ASTS",   # AST SpaceMobile - $7B - Satellite constellation
    "LUNR",   # Intuitive Machines - $3B - NASA lunar
    "BKSY",   # BlackSky - $400M - Defense satellite imagery
    "SATL",   # Satellogic - $200M - Earth observation
    
    # QUANTUM (3)
    "IONQ",   # IonQ - $9B - Enterprise quantum
    "RGTI",   # Rigetti - $3B - AWS integration
    "QBTS",   # D-Wave - $2B - First revenues
    
    # NUCLEAR (3)
    "OKLO",   # Oklo - $3B - SMR + data centers
    "SMR",    # NuScale - $3B - NRC approved
    "LEU",    # Centrus - $1.5B - HALEU fuel
    
    # AI/TECH (4)
    "SOUN",   # SoundHound - $5B - Voice AI
    "BBAI",   # BigBear.ai - $1B - Defense AI
    "GRRR",   # Gorilla Tech - $200M - AI security
    "DNA",    # Ginkgo Bioworks - $400M - Synth bio
    
    # SEMIS (3)
    "MU",     # Micron - $100B - HBM memory
    "WOLF",   # Wolfspeed - $2B - SiC chips
    "AEHR",   # Aehr Test - $700M - Chip testing
    
    # OTHER (2)
    "KDK",    # Kodiak Gas - $5B - Energy infra
    "KULR",   # KULR Tech - $300M - Battery/thermal
]

# Risk categories
EXTREME_RISK = ["ASTS", "BKSY", "SATL", "RGTI", "QBTS", "SOUN", "GRRR", "DNA", "KULR"]
HIGH_RISK = ["RKLB", "LUNR", "IONQ", "OKLO", "SMR", "BBAI", "WOLF", "AEHR"]
MEDIUM_RISK = ["LEU", "MU", "KDK"]

# Sectors
SECTORS = {
    "SPACE": ["RKLB", "ASTS", "LUNR", "BKSY", "SATL"],
    "QUANTUM": ["IONQ", "RGTI", "QBTS"],
    "NUCLEAR": ["OKLO", "SMR", "LEU"],
    "AI": ["SOUN", "BBAI", "GRRR", "DNA"],
    "SEMIS": ["MU", "WOLF", "AEHR"],
    "OTHER": ["KDK", "KULR"],
}

if __name__ == "__main__":
    print(f"Watchlist: {len(WATCHLIST)} tickers")
    print(f"Extreme Risk: {len(EXTREME_RISK)}")
    print(f"High Risk: {len(HIGH_RISK)}")
    print(f"Medium Risk: {len(MEDIUM_RISK)}")
