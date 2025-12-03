"""
Short Squeeze Scanner - Find stocks about to squeeze (GME-style)
Uses: Finviz scraper (no API key)
"""

from universal_scraper_engine import UniversalScraperEngine
from typing import Dict, List
import logging
from backend.modules.optimized_config_loader import get_scanner_config, get_forecast_config, get_exit_config


logger = logging.getLogger(__name__)

class ShortSqueezeScanner:
    def __init__(self):
        self.scraper = UniversalScraperEngine()
    
    def analyze_ticker(self, ticker: str) -> Dict:
        data = self.scraper.scrape_short_interest(ticker)
        
        if 'error' in data:
            return {'ticker': ticker, 'signal': 'NO_DATA', 'confidence': 0}
        
        short_float = data.get('short_float_pct', 0)
        squeeze_risk = data.get('squeeze_risk', 'LOW')
        
        if short_float > 30:
            signal = 'EXTREME_SQUEEZE'
            confidence = 0.90
            alert = '🚀 MASSIVE SHORT SQUEEZE POTENTIAL!'
        elif short_float > 20:
            signal = 'HIGH_SQUEEZE'
            confidence = 0.80
            alert = '⚡ High squeeze potential'
        elif short_float > 10:
            signal = 'MODERATE_SQUEEZE'
            confidence = 0.65
            alert = 'Watch for squeeze'
        else:
            signal = 'LOW_SQUEEZE'
            confidence = 0.40
            alert = 'Low squeeze risk'
        
        return {**data, 'signal': signal, 'confidence': confidence, 'alert': alert}
    
    def find_squeezes(self, universe: List[str] = None) -> List[Dict]:
        if not universe:
            universe = ['GME', 'AMC', 'BBBY', 'CLOV', 'SPCE', 'PLTR', 'WISH']
        
        squeezes = []
        for ticker in universe:
            try:
                result = self.analyze_ticker(ticker)
                if result.get('short_float_pct', 0) > 15:
                    squeezes.append(result)
            except: continue
        
        return sorted(squeezes, key=lambda x: x.get('short_float_pct', 0), reverse=True)

