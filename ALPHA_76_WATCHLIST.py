"""
ALPHA 76 HIGH-VELOCITY WATCHLIST
=================================
Small-to-mid-cap stocks with high beta, catalyst density, and exposure
to emerging H1 2026 themes: Autonomous/AI, Space, Biotech, Clean Energy.

Strategy:
- Filter via microstructure proxies (institutional activity >50th percentile)
- Monitor drift detection for regime changes (trending → mean reversion)
- Track ARK Invest (ARKK) flows as sector sentiment indicator
- Position size based on volatility regime

Updated: December 8, 2024
Research: Cross-referenced with institutional holdings, clinical trials, contract wins
"""

from typing import List, Dict
import pandas as pd

# ============================================================================
# ALPHA 76 CATEGORIZED BY SECTOR
# ============================================================================

# 1. AUTONOMOUS, ROBOTICS & AI HARDWARE (15 tickers)
# The "Kodiak Cluster" - Physical AI applications
AUTONOMOUS_AI_HARDWARE = {
    'KDK': {
        'name': 'Kodiak Gas Services',
        'thesis': 'Natural gas compression infrastructure, NOT robotics (different Kodiak)',
        'catalysts': 'Energy demand from AI data centers, nat gas shortage',
        'risk': 'Commodity price exposure, NOT actual AI play',
        'institutional': 'Blackstone backed, 70% institutional ownership'
    },
    'SYM': {
        'name': 'Symbotic',
        'thesis': 'Warehouse automation (Walmart partnership), robotics + AI software',
        'catalysts': 'Q4 2024 earnings (Dec), new customer wins, Walmart expansion',
        'risk': 'Customer concentration (Walmart 80%+), deployment delays',
        'institutional': 'SoftBank Vision Fund, Walmart stake'
    },
    'SERV': {
        'name': 'Serve Robotics',
        'thesis': 'Last-mile delivery robots, Nvidia backing (20% stake)',
        'catalysts': 'Expansion to new cities Q1 2025, Uber Eats partnership scale',
        'risk': 'Regulatory hurdles, small revenue base ($5M TTM)',
        'institutional': 'Nvidia 20%, Uber strategic investor'
    },
    'AMBA': {
        'name': 'Ambarella',
        'thesis': 'Edge AI vision chips for automotive/security, pivot from GoPro',
        'catalysts': 'CV3-AI chip adoption in ADAS, security camera wins',
        'risk': 'Cyclical semi demand, China exposure',
        'institutional': 'Wellington, Vanguard, BlackRock'
    },
    'IONQ': {
        'name': 'IonQ',
        'thesis': 'Trapped-ion quantum computing, Amazon/Azure partnerships',
        'catalysts': 'Q4 bookings update, new enterprise contracts, DARPA awards',
        'risk': 'Pre-revenue quantum, 5-10 year commercialization',
        'institutional': 'Fidelity, Blackrock, ARK Innovation (ARKQ)'
    },
    'RGTI': {
        'name': 'Rigetti Computing',
        'thesis': 'Superconducting quantum + hybrid cloud, AWS partnership',
        'catalysts': 'Ankaa-2 84-qubit launch Q4 2024, UK government contracts',
        'risk': 'Dilution risk (SPAC), negative margins',
        'institutional': 'ARK Next Gen (ARKW), Palantir early investor overlap'
    },
    'QUBT': {
        'name': 'Quantum Computing Inc',
        'thesis': 'Affordable quantum (Dirac-3), entropy computing, NASA contracts',
        'catalysts': 'Dirac-3 commercial availability Q1 2025, Los Alamos partnership',
        'risk': 'Penny stock volatility, limited financials',
        'institutional': 'Retail heavy, limited institutional'
    },
    'SOUN': {
        'name': 'SoundHound AI',
        'thesis': 'Voice AI for automotive/restaurants, strategic Nvidia partnership',
        'catalysts': 'Q4 bookings (automotive ramp), restaurant AI adoption',
        'risk': 'Competition (Google Assistant, Alexa), customer concentration',
        'institutional': 'Nvidia, ARK Next Gen (ARKW), Tencent early investor'
    },
    'BBAI': {
        'name': 'BigBear.ai',
        'thesis': 'AI decision intelligence for defense/intel agencies',
        'catalysts': 'Defense budget FY25, classified contract wins',
        'risk': 'Government customer concentration, security clearance delays',
        'institutional': 'AE Industrial Partners (sponsor), defense-focused funds'
    },
    'REKR': {
        'name': 'Rekor Systems',
        'thesis': 'AI roadway intelligence (license plate recognition), government contracts',
        'catalysts': 'Infrastructure bill spending, state DOT contracts',
        'risk': 'Privacy concerns, limited competitive moat',
        'institutional': 'ARK Next Gen (ARKW), retail heavy'
    },
    'LAZR': {
        'name': 'Luminar Technologies',
        'thesis': 'Lidar for autonomy, Volvo production win (EX90), Mercedes partnership',
        'catalysts': 'Volvo EX90 production ramp Q1 2025, new OEM partnerships',
        'risk': 'Dilution, cash burn $200M/yr, Tesla no-lidar thesis',
        'institutional': 'Volvo strategic stake, ARK Autonomous (ARKQ)'
    },
    'INVZ': {
        'name': 'Innoviz Technologies',
        'thesis': 'InnovizTwo lidar, BMW iX production, solid-state roadmap',
        'catalysts': 'BMW iX volume ramp, InnovizThree mass production 2025',
        'risk': 'OEM delays (VW postponed), gross margin pressure',
        'institutional': 'ARK Autonomous (ARKQ), Fidelity'
    },
    'OUST': {
        'name': 'Ouster',
        'thesis': 'Digital lidar for industrial/automotive, acquisition of Velodyne',
        'catalysts': 'Cost reductions post-merger, automotive design wins Q4',
        'risk': 'Integration execution, negative gross margins',
        'institutional': 'ARK Autonomous (ARKQ), Vanguard'
    },
    'CRNC': {
        'name': 'Cerence',
        'thesis': 'AI voice for 400M+ cars, Microsoft Nuance partnership',
        'catalysts': 'Edge AI pivot, new automotive platform wins',
        'risk': 'OEM headwinds (VW software delays), subscription transition',
        'institutional': 'Wellington, Vanguard, Icahn 10% stake (activist)'
    },
    'AEVA': {
        'name': 'Aeva Technologies',
        'thesis': '4D lidar (velocity + 3D), Daimler Truck partnership',
        'catalysts': 'Daimler Truck production 2026, Atlas sensor launch',
        'risk': 'Cash burn, production delays, limited traction',
        'institutional': 'ARK Autonomous (ARKQ), Sylebra Capital'
    }
}

# 2. SPACE ECONOMY (12 tickers)
# Defense contracts + constellation launches
SPACE_ECONOMY = {
    'RKLB': {
        'name': 'Rocket Lab',
        'thesis': '#2 launch provider (behind SpaceX), Neutron rocket 2025, space systems',
        'catalysts': 'Neutron test flights Q2 2025, NASA missions, Electron cadence',
        'risk': 'SpaceX competition, Neutron development delays',
        'institutional': 'ARK Space (ARKX), Vanguard, Baillie Gifford (early Tesla investor)'
    },
    'ASTS': {
        'name': 'AST SpaceMobile',
        'thesis': 'Direct-to-cell satellite (AT&T, Verizon, Vodafone partnerships)',
        'catalysts': 'BlueBird 1-5 launch Q4 2024, commercial service beta Q1 2025',
        'risk': 'Technology unproven at scale, capital intensive ($1B+ needed)',
        'institutional': 'AT&T 9% stake, Vodafone strategic, ARK Space (ARKX)'
    },
    'LUNR': {
        'name': 'Intuitive Machines',
        'thesis': 'Lunar landers/services, NASA Artemis missions, first private moon landing',
        'catalysts': 'IM-2 mission Q1 2025, Artemis contracts, commercial payload',
        'risk': 'Mission failures (IM-1 tipped over), government customer concentration',
        'institutional': 'ARK Space (ARKX), Cathie Wood "conviction buy"'
    },
    'SPIR': {
        'name': 'Spire Global',
        'thesis': 'Space-based data (weather, AIS tracking), 100+ satellite constellation',
        'catalysts': 'Maritime AI contracts, weather forecasting for insurance',
        'risk': 'Low margins, commodity data business',
        'institutional': 'ARK Space (ARKX), limited institutional'
    },
    'PL': {
        'name': 'Planet Labs',
        'thesis': 'Daily earth imagery (200+ satellites), agriculture/defense use cases',
        'catalysts': 'Defense contracts (Ukraine war demand), Pelican constellation',
        'risk': 'Cash burn, commercial customer churn',
        'institutional': 'ARK Space (ARKX), Google early investor'
    },
    'RDW': {
        'name': 'Redwire',
        'thesis': 'Space infrastructure (solar arrays, 3D printing), ISS heritage',
        'catalysts': 'Lunar infrastructure for Artemis, Mars projects',
        'risk': 'Project-based revenue, government delays',
        'institutional': 'ARK Space (ARKX), defense-focused funds'
    },
    'BKSY': {
        'name': 'BlackSky Technology',
        'thesis': 'Real-time geospatial intel, 14-satellite constellation, AI analytics',
        'catalysts': 'Defense/intel contract renewals, Gen-3 satellites Q1 2025',
        'risk': 'Government concentration (70%+), negative margins',
        'institutional': 'ARK Space (ARKX), Mithril Capital (Peter Thiel)'
    },
    'MNTS': {
        'name': 'Momentus',
        'thesis': 'Space transportation/tugs (Vigoride), last-mile orbital delivery',
        'catalysts': 'Successful Vigoride missions 2025, commercial contracts',
        'risk': 'Execution issues (past delays), small cash balance',
        'institutional': 'Retail heavy, limited institutional'
    },
    'LLAP': {
        'name': 'Terran Orbital',
        'thesis': 'Satellite manufacturing for DoD/commercial, Rivada Space contract',
        'catalysts': 'Rivada constellation production, defense awards',
        'risk': 'Lockheed Martin acquisition fell through, dilution risk',
        'institutional': 'Lockheed Martin strategic stake, ARK Space (ARKX)'
    },
    'ACHR': {
        'name': 'Archer Aviation',
        'thesis': 'eVTOL (flying taxis), Midnight aircraft FAA certification path',
        'catalysts': 'FAA Part 135 certification Q2 2025, United Airlines order',
        'risk': 'Certification delays, infrastructure buildout, cash burn',
        'institutional': 'United Airlines, Stellantis strategic stakes, ARK Space (ARKX)'
    },
    'JOBY': {
        'name': 'Joby Aviation',
        'thesis': 'eVTOL leader, Toyota partnership ($900M invested), Delta Air Lines MOU',
        'catalysts': 'Stage 4 FAA testing Q1 2025, NYC service announcement',
        'risk': 'Certification timeline (2025 optimistic), infrastructure',
        'institutional': 'Toyota, Uber, ARK Space (ARKX), Fidelity'
    },
    'LILM': {
        'name': 'Lilium',
        'thesis': 'Electric jet eVTOL (7-seater), European focus, Saudi Arabia order',
        'catalysts': 'Flight testing completion Q4 2024, Saudi production facility',
        'risk': 'Cash crisis (needs funding), tech unproven, bankruptcy risk HIGH',
        'institutional': 'Tencent, Baillie Gifford, ARK Space (ARKX)'
    }
}

# 3. BIOTECH: GENE EDITING & RARE DISEASE (16 tickers)
# Clinical trial catalysts in H1 2025
BIOTECH_HIGH_BETA = {
    'KOD': {
        'name': 'Kodiak Sciences',
        'thesis': 'Retina therapeutics (wet AMD), tarcocimab trials, Bayer partnership',
        'catalysts': 'Tarcocimab Phase 3 data Q1 2025, FDA filing decision',
        'risk': 'Trial failure history (2021 setback), competitive (REGN, Roche)',
        'institutional': 'Baker Bros (12%), ARK Genomic (ARKG), Fidelity'
    },
    'AKRO': {
        'name': 'Akero Therapeutics',
        'thesis': 'Efruxifermin for NASH (fatty liver disease), Phase 3 ongoing',
        'catalysts': 'HARMONY Phase 3 interim data Q2 2025, biopsy results',
        'risk': 'Competitive (MDGL, VKTX), liver biopsy endpoints challenging',
        'institutional': 'RA Capital, Cormorant, ARK Genomic (ARKG)'
    },
    'AKYA': {
        'name': 'Akoya Biosciences',
        'thesis': 'Spatial biology tools (PhenoCycler), cancer research applications',
        'catalysts': 'New instrument launches Q1 2025, pharma partnerships',
        'risk': 'Capital equipment cycles, research budget cuts',
        'institutional': 'ARK Genomic (ARKG), Wellington'
    },
    'HALO': {
        'name': 'Halozyme Therapeutics',
        'thesis': 'ENHANZE drug delivery platform, royalties from Roche/BMS/Pfizer',
        'catalysts': 'New partnerships Q4 2024, royalty ramp from approved drugs',
        'risk': 'Royalty model limits upside, patent expiration 2027',
        'institutional': 'Wellington, Vanguard, BlackRock (large cap biotech)'
    },
    'VRDN': {
        'name': 'Viridian Therapeutics',
        'thesis': 'VRDN-001 for thyroid eye disease (TED), IGF-1R antagonist',
        'catalysts': 'Phase 3 THRIVE-2 data Q1 2025, BLA filing timeline',
        'risk': 'Competitive (IMCR Tepezza), TED small market ($2B)',
        'institutional': 'RA Capital, Farallon, ARK Genomic (ARKG)'
    },
    'URGN': {
        'name': 'UroGen Pharma',
        'thesis': 'UGN-102 for low-grade bladder cancer, RTGel platform',
        'catalysts': 'ATLAS Phase 3 data Q2 2025, FDA approval decision',
        'risk': 'Narrow indication, reimbursement challenges',
        'institutional': 'ARK Genomic (ARKG), Acuta Capital'
    },
    'LQDA': {
        'name': 'Liquidia Technologies',
        'thesis': 'YUTREPIA for pulmonary arterial hypertension, vs United Therapeutics',
        'catalysts': 'Patent litigation resolution 2025, market share gains',
        'risk': 'Patent lawsuit (UTHR), small patient population',
        'institutional': 'ARK Genomic (ARKG), Broadfin Capital'
    },
    'TLSA': {
        'name': 'Tiziana Life Sciences',
        'thesis': 'Foralumab for MS/Alzheimer\'s (intranasal anti-CD3)',
        'catalysts': 'INIMS trial data Q1 2025, Alzheimer\'s trial initiation',
        'risk': 'Early-stage, small company, dilution risk',
        'institutional': 'Limited institutional, retail heavy'
    },
    'PVLA': {
        'name': 'Palvella Therapeutics',
        'thesis': 'QTORIN rapamycin for microcystic lymphatic malformations',
        'catalysts': 'Phase 2/3 data Q2 2025, orphan drug designation',
        'risk': 'Ultra-rare disease (small TAM), trial enrollment slow',
        'institutional': 'RA Capital, Venrock'
    },
    'OKYO': {
        'name': 'OKYO Pharma',
        'thesis': 'OK-101 for dry eye disease, chemerin modulation',
        'catalysts': 'Phase 2 ELEMY data Q1 2025, partnering discussions',
        'risk': 'Dry eye crowded (OCUL, EYEG), partnership dependent',
        'institutional': 'Limited institutional'
    },
    'IOBT': {
        'name': 'IO Biotech',
        'thesis': 'Cancer vaccines (IO102/IO112), PD-L1 checkpoint modulation',
        'catalysts': 'Phase 2 melanoma data Q2 2025, combination trial starts',
        'risk': 'Early-stage, cancer vaccine skepticism',
        'institutional': 'Novo Holdings (Novo Nordisk), Sunstone'
    },
    'SPRO': {
        'name': 'Spero Therapeutics',
        'thesis': 'Tebipenem for UTIs, oral carbapenem',
        'catalysts': 'FDA PDUFA date March 2025, commercial launch prep',
        'risk': 'Antibiotic reimbursement, limited peak sales ($200M)',
        'institutional': 'ARK Genomic (ARKG), RA Capital'
    },
    'VKTX': {
        'name': 'Viking Therapeutics',
        'thesis': 'VK2809 (NASH), VK2735 (obesity GLP-1 oral)',
        'catalysts': 'VK2735 Phase 2 obesity data Q1 2025, licensing deals',
        'risk': 'GLP-1 competition (LLY, NVO), oral bioavailability questions',
        'institutional': 'ARK Genomic (ARKG), Farallon, Foresite Capital'
    },
    'CYTK': {
        'name': 'Cytokinetics',
        'thesis': 'Aficamten for HCM (vs MYBPC3 mutation), cardiac myosin inhibitor',
        'catalysts': 'SEQUOIA-HCM Phase 3 data Q4 2024, NDA filing Q1 2025',
        'risk': 'BMS competing drug (camzyos), dosing complexity',
        'institutional': 'Wellington, Vanguard, ARK Genomic (ARKG)'
    },
    'NTLA': {
        'name': 'Intellia Therapeutics',
        'thesis': 'In vivo CRISPR (NTLA-2001 for ATTR), Regeneron partnership',
        'catalysts': 'NTLA-2001 Phase 3 initiation Q1 2025, new programs',
        'risk': 'Gene editing safety, competition (CRSP, BEAM)',
        'institutional': 'ARK Genomic (ARKG), Regeneron strategic stake'
    },
    'BEAM': {
        'name': 'Beam Therapeutics',
        'thesis': 'Base editing (precision CRISPR), sickle cell program',
        'catalysts': 'BEAM-101 sickle cell data Q2 2025, IND submissions',
        'risk': 'Editas patent dispute, early-stage tech',
        'institutional': 'ARK Genomic (ARKG), Fidelity, ARCH Venture'
    }
}

# 4. GREEN ENERGY & GRID INFRASTRUCTURE (12 tickers)
# AI data center power demand
GREEN_ENERGY_GRID = {
    'FLNC': {
        'name': 'Fluence Energy',
        'thesis': 'Energy storage (Siemens/AES JV), AI data center backup power',
        'catalysts': 'Q4 bookings (data center deals), IRA credits, grid projects',
        'risk': 'Supply chain (battery cells), project delays',
        'institutional': 'Siemens 25%, AES 25%, Vanguard'
    },
    'STEM': {
        'name': 'Stem Inc',
        'thesis': 'AI-driven energy storage software (Athena), behind-meter focus',
        'catalysts': 'Data center partnerships Q1 2025, Athena AI upgrades',
        'risk': 'Cash burn, hardware commoditization',
        'institutional': 'ARK Next Gen (ARKW), Fidelity'
    },
    'AMSC': {
        'name': 'American Superconductor',
        'thesis': 'Grid resiliency (ship defense, wind), superconductor wires',
        'catalysts': 'Navy contracts Q4 2024, wind energy rebound',
        'risk': 'Lumpy project revenue, China competition',
        'institutional': 'ARK Next Gen (ARKW), Wellington'
    },
    'NXT': {
        'name': 'Nextracker',
        'thesis': 'Solar tracking systems (#1 market share), IRA beneficiary',
        'catalysts': 'Q1 guidance (solar project pipeline), margin expansion',
        'risk': 'Solar panel tariffs, Flex acquisition integration',
        'institutional': 'TPG (sponsor), Vanguard, BlackRock'
    },
    'ARRY': {
        'name': 'Array Technologies',
        'thesis': 'Solar tracking (#2 to NXT), bifacial module optimization',
        'catalysts': 'U.S. solar project acceleration, international expansion',
        'risk': 'NXT competition, commodity steel prices',
        'institutional': 'Oaktree Capital (sponsor), Vanguard'
    },
    'SHLS': {
        'name': 'Shoals Technologies',
        'thesis': 'EBOS (electrical balance of systems) for solar, plug-and-play',
        'catalysts': 'Utility-scale solar ramp, data center solar co-location',
        'risk': 'Customer concentration, commodity pricing',
        'institutional': 'Oaktree, BlackRock'
    },
    'BE': {
        'name': 'Bloom Energy',
        'thesis': 'Solid oxide fuel cells for data centers (24/7 power), hydrogen pivot',
        'catalysts': 'SK ecoplant Korea deal, data center wins (MSFT, GOOGL)',
        'risk': 'Negative FCF, hydrogen unproven, dilution',
        'institutional': 'Vanguard, BlackRock, SK Group strategic'
    },
    'PLUG': {
        'name': 'Plug Power',
        'thesis': 'Green hydrogen ecosystem (electrolyzers, fuel cells, fueling)',
        'catalysts': 'IRA hydrogen credits, Louisiana plant startup Q1 2025',
        'risk': 'Massive cash burn ($1B+/yr), dilution, customer losses (AMZN, WMT)',
        'institutional': 'SK Group strategic, Vanguard (passive only)'
    },
    'FCEL': {
        'name': 'FuelCell Energy',
        'thesis': 'Stationary fuel cells + carbon capture, long-duration storage',
        'catalysts': 'Toyota partnership, power purchase agreements',
        'risk': 'Dilution history, negative margins, bankruptcy watch',
        'institutional': 'Limited institutional, retail speculation'
    },
    'BLDP': {
        'name': 'Ballard Power Systems',
        'thesis': 'Heavy-duty fuel cells (trucks, buses, marine)',
        'catalysts': 'Weichai partnership China, European bus orders',
        'risk': 'Hydrogen infrastructure delay, BEV competition',
        'institutional': 'Limited institutional, Canadian funds'
    },
    'ENOV': {
        'name': 'Enovix',
        'thesis': 'Silicon anode batteries (3D architecture), IoT/wearables focus',
        'catalysts': 'Fab2 ramp Q1 2025, automotive customer announcements',
        'risk': 'Manufacturing scale-up, limited revenue ($10M TTM)',
        'institutional': 'Intel Capital, Qualcomm Ventures, ARK Next Gen (ARKW)'
    },
    'QS': {
        'name': 'QuantumScape',
        'thesis': 'Solid-state batteries (VW partnership), ceramic separator',
        'catalysts': 'QSE-5 24-layer cell validation Q4 2024, pilot line scale',
        'risk': 'Commercialization 2025 (was 2024), cash burn',
        'institutional': 'VW 18% stake, ARK Next Gen (ARKW), Bill Gates'
    }
}

# 5. FINTECH & DIGITAL ASSETS (10 tickers)
# Rate cuts + crypto cycle beneficiaries
FINTECH_CRYPTO = {
    'SOFI': {
        'name': 'SoFi Technologies',
        'thesis': 'Digital bank (GAAP profitable Q3 2024), student loan refinancing',
        'catalysts': 'Member growth Q4, credit card launch, lending rebound',
        'risk': 'Credit losses, student loan pause extension',
        'institutional': 'SoftBank (diluted to <5%), Vanguard'
    },
    'UPST': {
        'name': 'Upstart Holdings',
        'thesis': 'AI lending platform, rate cut beneficiary (origination volume)',
        'catalysts': 'Q4 originations rebound, auto lending scale',
        'risk': 'Credit losses if recession, partner bank concentration',
        'institutional': 'ARK Fintech (ARKF), Baillie Gifford'
    },
    'AFRM': {
        'name': 'Affirm Holdings',
        'thesis': 'BNPL leader (Apple partnership), card launch (debit + credit)',
        'catalysts': 'Apple Pay Later integration, Affirm Card adoption',
        'risk': 'Credit losses, BNPL regulation, Apple competitive threat',
        'institutional': 'Founders Fund, Lightspeed, ARK Fintech (ARKF)'
    },
    'MQ': {
        'name': 'Marqeta',
        'thesis': 'Modern card issuing platform (Square, Uber, DoorDash)',
        'catalysts': 'New fintech launches, volume growth Q4',
        'risk': 'Square concentration (30%+), commoditization',
        'institutional': 'Coatue, Lone Pine, ARK Fintech (ARKF)'
    },
    'HOOD': {
        'name': 'Robinhood Markets',
        'thesis': 'Retail trading flow, crypto trading (Bitcoin, Ethereum)',
        'catalysts': 'Crypto wallet expansion, election volatility 2024/2025',
        'risk': 'PFOF regulation, crypto regulatory uncertainty',
        'institutional': 'Andreessen Horowitz, Ribbit Capital, Sequoia'
    },
    'CLSK': {
        'name': 'CleanSpark',
        'thesis': 'Bitcoin mining (efficient operations), renewables-focused',
        'catalysts': 'Bitcoin halving Apr 2024, facility expansion Georgia',
        'risk': 'Bitcoin price, energy costs, dilution',
        'institutional': 'Limited institutional, retail heavy'
    },
    'MARA': {
        'name': 'Marathon Digital',
        'thesis': 'Bitcoin mining scale (40 EH/s target), infrastructure plays',
        'catalysts': 'Facility buildout Q1 2025, Bitcoin halving effects',
        'risk': 'Bitcoin price, hosting costs, shareholder dilution',
        'institutional': 'Vanguard (passive), BlackRock (iShares)'
    },
    'RIOT': {
        'name': 'Riot Platforms',
        'thesis': 'Bitcoin mining + engineering services, Texas facilities',
        'catalysts': 'Rockdale expansion, hash rate growth Q4',
        'risk': 'Texas energy prices (ERCOT), Bitcoin correlation',
        'institutional': 'Vanguard (passive), State Street'
    },
    'COIN': {
        'name': 'Coinbase Global',
        'thesis': 'Crypto infrastructure (exchange + custody + Base L2)',
        'catalysts': 'Bitcoin ETF trading fees, Base chain growth, staking revenue',
        'risk': 'SEC lawsuit, crypto winter, competition',
        'institutional': 'ARK Innovation (ARKK), Vanguard, BlackRock'
    },
    'ALKT': {
        'name': 'Alkami Technology',
        'thesis': 'Digital banking cloud for credit unions/regional banks',
        'catalysts': 'Customer additions Q4, ARPU expansion',
        'risk': 'Bank failures (regional banks), IT budget cuts',
        'institutional': 'General Atlantic (sponsor), Vanguard'
    }
}

# 6. NEXT-GEN CONSUMER & SOFTWARE (11 tickers)
# High-growth consumer brands + enterprise SaaS
CONSUMER_SOFTWARE = {
    'DUOL': {
        'name': 'Duolingo',
        'thesis': 'Gamified language learning, AI tutor (GPT-4 integration)',
        'catalysts': 'Max tier adoption, international expansion, AI features',
        'risk': 'High valuation (15x sales), competition (Babbel, Busuu)',
        'institutional': 'General Atlantic, Union Square Ventures'
    },
    'ONON': {
        'name': 'On Holding',
        'thesis': 'Premium running shoes (Roger Federer), DTC + wholesale',
        'catalysts': 'U.S. growth (Under Armour share gain), innovation pipeline',
        'risk': 'Nike/Adidas competition, tariffs, wholesale inventory',
        'institutional': 'Roger Federer (investor), BlackRock'
    },
    'CELH': {
        'name': 'Celsius Holdings',
        'thesis': 'Energy drinks (Pepsi distribution), fitness positioning vs Red Bull',
        'catalysts': 'International expansion, Pepsi incentives, new flavors',
        'risk': 'Valuation correction (down 50% from highs), inventory destocking',
        'institutional': 'Fidelity, Wellington, ARK Innovation (ARKK)'
    },
    'ELF': {
        'name': 'e.l.f. Beauty',
        'thesis': 'Gen Z cosmetics (viral TikTok), prestige quality at mass prices',
        'catalysts': 'International expansion, skincare line growth',
        'risk': 'Teen spending slowdown, China manufacturing risk',
        'institutional': 'Wellington, Fidelity, ARK Innovation (ARKK)'
    },
    'AMPL': {
        'name': 'Amplitude',
        'thesis': 'Digital analytics (product analytics), freemium model',
        'catalysts': 'Enterprise customer wins, AI-powered insights',
        'risk': 'Competition (DDOG, SNOW), slowing growth',
        'institutional': 'Sequoia, Benchmark'
    },
    'PATH': {
        'name': 'UiPath',
        'thesis': 'Enterprise RPA (robotic process automation), AI automation pivot',
        'catalysts': 'AI agent integration Q1 2025, Oracle partnership',
        'risk': 'Microsoft Power Automate competition, macro softness',
        'institutional': 'Accel, Sequoia, ARK Innovation (ARKK)'
    },
    'S': {
        'name': 'SentinelOne',
        'thesis': 'AI-powered cybersecurity (EDR/XDR), vs CrowdStrike',
        'catalysts': 'Scalyr acquisition integration, ARR growth >30%',
        'risk': 'CRWD competition, macro IT spend cuts',
        'institutional': 'Tiger Global, Insight Partners, ARK Fintech (ARKF)'
    },
    'ESTC': {
        'name': 'Elastic',
        'thesis': 'Search + observability + security (Elastic Cloud)',
        'catalysts': 'GenAI search use cases, ESRE (Elastic Security)',
        'risk': 'Amazon OpenSearch competition, customer churn',
        'institutional': 'Benchmark, Index Ventures'
    },
    'DOCN': {
        'name': 'DigitalOcean',
        'thesis': 'Cloud for SMBs/developers, simplicity vs AWS complexity',
        'catalysts': 'AI/ML platform launch Q4, customer ARPU expansion',
        'risk': 'AWS/GCP aggressive SMB pricing, limited moat',
        'institutional': 'Access Industries (sponsor), Vanguard'
    },
    'APP': {
        'name': 'AppLovin',
        'thesis': 'Mobile adtech + gaming (AXON AI engine), ROAS optimization',
        'catalysts': 'AXON 2.0 rollout Q1 2025, e-commerce ad expansion',
        'risk': 'Privacy regulation (ATT), gaming cyclicality',
        'institutional': 'KKR (sponsor), Tiger Global'
    },
    'VRNS': {
        'name': 'Varonis Systems',
        'thesis': 'Data security (DLP), SaaS transition, M365 integration',
        'catalysts': 'SaaS ARR growth 50%+, DDSPM platform adoption',
        'risk': 'Competition (PANW, CRWD), billings lumpiness',
        'institutional': 'Wellington, Vanguard'
    }
}

# ============================================================================
# CONSOLIDATED ALPHA 76 LIST
# ============================================================================

def get_alpha_76_tickers() -> List[str]:
    """Get flat list of all 76 tickers."""
    all_sectors = [
        AUTONOMOUS_AI_HARDWARE,
        SPACE_ECONOMY,
        BIOTECH_HIGH_BETA,
        GREEN_ENERGY_GRID,
        FINTECH_CRYPTO,
        CONSUMER_SOFTWARE
    ]
    
    tickers = []
    for sector in all_sectors:
        tickers.extend(sector.keys())
    
    return tickers

def get_alpha_76_by_sector() -> Dict[str, Dict]:
    """Get tickers organized by sector with full metadata."""
    return {
        'autonomous_ai_hardware': AUTONOMOUS_AI_HARDWARE,
        'space_economy': SPACE_ECONOMY,
        'biotech_high_beta': BIOTECH_HIGH_BETA,
        'green_energy_grid': GREEN_ENERGY_GRID,
        'fintech_crypto': FINTECH_CRYPTO,
        'consumer_software': CONSUMER_SOFTWARE
    }

def get_ark_overlap() -> List[str]:
    """
    Get tickers that overlap with ARK Invest ETFs.
    Monitor ARKK/ARKQ/ARKW/ARKG flows as sector sentiment.
    """
    ark_tickers = []
    all_data = get_alpha_76_by_sector()
    
    for sector, stocks in all_data.items():
        for ticker, data in stocks.items():
            if 'ARK' in data['institutional']:
                ark_tickers.append(ticker)
    
    return ark_tickers

def get_high_risk_tickers() -> List[str]:
    """Get tickers with bankruptcy/dilution risk (filter carefully)."""
    high_risk = []
    all_data = get_alpha_76_by_sector()
    
    risk_keywords = ['bankruptcy', 'dilution', 'cash burn', 'negative FCF', 'cash crisis']
    
    for sector, stocks in all_data.items():
        for ticker, data in stocks.items():
            risk_text = data['risk'].lower()
            if any(keyword in risk_text for keyword in risk_keywords):
                high_risk.append(ticker)
    
    return high_risk

def get_catalyst_calendar_q1_2025() -> Dict[str, List[str]]:
    """Get tickers organized by catalyst timing."""
    calendar = {
        'q4_2024': [],  # Dec 2024
        'q1_2025': [],  # Jan-Mar 2025
        'q2_2025': [],  # Apr-Jun 2025
        'ongoing': []   # No specific date
    }
    
    all_data = get_alpha_76_by_sector()
    
    for sector, stocks in all_data.items():
        for ticker, data in stocks.items():
            catalysts = data['catalysts'].lower()
            
            if 'q4 2024' in catalysts or 'dec' in catalysts:
                calendar['q4_2024'].append(ticker)
            elif 'q1 2025' in catalysts or 'jan' in catalysts or 'feb' in catalysts or 'mar' in catalysts or 'march' in catalysts:
                calendar['q1_2025'].append(ticker)
            elif 'q2 2025' in catalysts or 'apr' in catalysts or 'may' in catalysts or 'jun' in catalysts:
                calendar['q2_2025'].append(ticker)
            else:
                calendar['ongoing'].append(ticker)
    
    return calendar

# ============================================================================
# EXPORT & VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ALPHA 76 HIGH-VELOCITY WATCHLIST")
    print("=" * 80)
    
    tickers = get_alpha_76_tickers()
    print(f"\n📊 Total Tickers: {len(tickers)}")
    
    sectors = get_alpha_76_by_sector()
    print(f"\n🎯 Sector Breakdown:")
    for sector_name, stocks in sectors.items():
        print(f"   {sector_name.replace('_', ' ').title()}: {len(stocks)} tickers")
    
    # ARK Invest overlap
    ark_overlap = get_ark_overlap()
    print(f"\n🏹 ARK Invest Overlap: {len(ark_overlap)} tickers")
    print(f"   Monitor ARKK/ARKQ/ARKW/ARKG flows for sector sentiment")
    print(f"   ARK holdings: {', '.join(ark_overlap[:10])}...")
    
    # High-risk tickers
    high_risk = get_high_risk_tickers()
    print(f"\n⚠️  High-Risk Tickers (bankruptcy/dilution): {len(high_risk)} tickers")
    print(f"   Use smaller position sizes: {', '.join(high_risk[:10])}...")
    
    # Catalyst calendar
    calendar = get_catalyst_calendar_q1_2025()
    print(f"\n📅 Catalyst Calendar:")
    print(f"   Q4 2024 (Dec):  {len(calendar['q4_2024'])} tickers")
    print(f"   Q1 2025 (Jan-Mar): {len(calendar['q1_2025'])} tickers")
    print(f"   Q2 2025 (Apr-Jun): {len(calendar['q2_2025'])} tickers")
    print(f"   Ongoing: {len(calendar['ongoing'])} tickers")
    
    # Save to file
    print(f"\n💾 Saving to alpha_76_watchlist.txt...")
    with open('alpha_76_watchlist.txt', 'w') as f:
        for ticker in tickers:
            f.write(f"{ticker}\n")
    
    print(f"✅ Saved {len(tickers)} tickers to alpha_76_watchlist.txt")
    
    # Export detailed CSV
    print(f"\n💾 Creating detailed CSV with metadata...")
    rows = []
    for sector_name, stocks in sectors.items():
        for ticker, data in stocks.items():
            rows.append({
                'ticker': ticker,
                'name': data['name'],
                'sector': sector_name,
                'thesis': data['thesis'],
                'catalysts': data['catalysts'],
                'risk': data['risk'],
                'institutional': data['institutional']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv('alpha_76_detailed.csv', index=False)
    print(f"✅ Saved detailed metadata to alpha_76_detailed.csv")
    
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run microstructure proxy filtering (Q8) to find institutional activity")
    print(f"   2. Run drift detection (Q12) to identify regime changes")
    print(f"   3. Monitor ARK flows as sector sentiment indicator")
    print(f"   4. Size positions based on volatility regime")
