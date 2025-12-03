"""
Quantum API Configuration Manager
Loads, validates, and manages API keys and source metadata for the trading system.
"""

import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class APISource:
    """Metadata for a data API source."""
    name: str
    key: str
    priority: int
    rate_limit_per_minute: int
    supports_intraday: bool
    supports_fundamentals: bool
    supports_options: bool
    coverage: str  # "US", "Global", etc.
    base_url: str
    
    def is_valid(self) -> bool:
        """Check if API key is present."""
        return bool(self.key and len(self.key) > 10)


class QuantumAPIConfig:
    """
    Central API configuration manager for Quantum AI Trader.
    Loads API keys from environment, validates them, and provides
    priority-ordered source lists for data orchestration.
    """
    
    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize API configuration.
        
        Args:
            env_path: Optional path to .env file. If None, searches in standard locations.
        """
        self.sources: Dict[str, APISource] = {}
        self.valid_sources: List[APISource] = []
        self._load_environment(env_path)
        self._initialize_sources()
        self._validate_sources()
        
    def _load_environment(self, env_path: Optional[str] = None):
        """Load environment variables from .env file."""
        if env_path:
            load_dotenv(env_path)
        else:
            # Try multiple locations
            possible_paths = [
                ".env",
                "../.env",
                "../../.env",
                "E:/quantum-ai-trader-v1.1/.env"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    load_dotenv(path)
                    logger.info(f"✓ Loaded environment from: {path}")
                    break
        
    def _initialize_sources(self):
        """Initialize all API sources with their metadata."""
        
        # Priority 1: Polygon.io (best for US equities, intraday data)
        self.sources['polygon'] = APISource(
            name='Polygon',
            key=os.getenv('POLYGON_API_KEY', ''),
            priority=1,
            rate_limit_per_minute=5,  # Free tier
            supports_intraday=True,
            supports_fundamentals=True,
            supports_options=True,
            coverage='US',
            base_url='https://api.polygon.io'
        )
        
        # Priority 2: Financial Modeling Prep (excellent fundamentals + technicals)
        self.sources['fmp'] = APISource(
            name='FMP',
            key=os.getenv('FMP_API_KEY', ''),
            priority=2,
            rate_limit_per_minute=300,  # Standard tier
            supports_intraday=True,
            supports_fundamentals=True,
            supports_options=False,
            coverage='Global',
            base_url='https://financialmodelingprep.com/api/v3'
        )
        
        # Priority 3: Alpha Vantage (reliable daily data)
        self.sources['alphavantage'] = APISource(
            name='AlphaVantage',
            key=os.getenv('ALPHAVANTAGE_API_KEY', ''),
            priority=3,
            rate_limit_per_minute=5,  # Free tier
            supports_intraday=True,
            supports_fundamentals=True,
            supports_options=False,
            coverage='Global',
            base_url='https://www.alphavantage.co'
        )
        
        # Priority 4: EODHD (good historical data)
        self.sources['eodhd'] = APISource(
            name='EODHD',
            key=os.getenv('EODHD_API_TOKEN', ''),
            priority=4,
            rate_limit_per_minute=20,  # Free tier
            supports_intraday=False,
            supports_fundamentals=True,
            supports_options=False,
            coverage='Global',
            base_url='https://eodhistoricaldata.com/api'
        )
        
    def _validate_sources(self):
        """Validate API keys and build priority-ordered valid source list."""
        logger.info("\n" + "="*60)
        logger.info("QUANTUM API CONFIGURATION - VALIDATION")
        logger.info("="*60)
        
        valid_count = 0
        for source_key, source in self.sources.items():
            if source.is_valid():
                self.valid_sources.append(source)
                valid_count += 1
                logger.info(f"✓ {source.name:15} | Priority {source.priority} | Rate: {source.rate_limit_per_minute}/min | Coverage: {source.coverage}")
            else:
                logger.warning(f"✗ {source.name:15} | No valid API key found")
        
        # Sort by priority
        self.valid_sources.sort(key=lambda x: x.priority)
        
        logger.info("="*60)
        logger.info(f"VALIDATED: {valid_count}/{len(self.sources)} sources ready")
        logger.info("="*60 + "\n")
        
        if valid_count == 0:
            logger.error("⚠️  NO VALID API SOURCES - System cannot fetch data!")
            
    def get_source(self, name: str) -> Optional[APISource]:
        """
        Get a specific API source by name.
        
        Args:
            name: Source name (polygon, fmp, alphavantage, eodhd)
            
        Returns:
            APISource if valid, None otherwise
        """
        source = self.sources.get(name.lower())
        return source if source and source.is_valid() else None
    
    def get_sources_by_priority(self, 
                                intraday_only: bool = False,
                                fundamentals_only: bool = False) -> List[APISource]:
        """
        Get valid sources ordered by priority, optionally filtered by capabilities.
        
        Args:
            intraday_only: Only return sources supporting intraday data
            fundamentals_only: Only return sources supporting fundamentals
            
        Returns:
            Priority-ordered list of valid APISource objects
        """
        sources = self.valid_sources.copy()
        
        if intraday_only:
            sources = [s for s in sources if s.supports_intraday]
            
        if fundamentals_only:
            sources = [s for s in sources if s.supports_fundamentals]
            
        return sources
    
    def get_primary_source(self) -> Optional[APISource]:
        """Get the highest priority valid source."""
        return self.valid_sources[0] if self.valid_sources else None
    
    def has_valid_sources(self) -> bool:
        """Check if at least one valid API source is available."""
        return len(self.valid_sources) > 0
    
    def get_config_summary(self) -> Dict:
        """
        Get configuration summary for logging/debugging.
        
        Returns:
            Dictionary with config stats
        """
        return {
            'total_sources': len(self.sources),
            'valid_sources': len(self.valid_sources),
            'primary_source': self.valid_sources[0].name if self.valid_sources else None,
            'intraday_sources': sum(1 for s in self.valid_sources if s.supports_intraday),
            'fundamentals_sources': sum(1 for s in self.valid_sources if s.supports_fundamentals),
            'global_coverage': sum(1 for s in self.valid_sources if s.coverage == 'Global')
        }


# Global singleton instance
_config_instance: Optional[QuantumAPIConfig] = None


def get_config(env_path: Optional[str] = None) -> QuantumAPIConfig:
    """
    Get or create the global API configuration instance.
    
    Args:
        env_path: Optional path to .env file (only used on first call)
        
    Returns:
        QuantumAPIConfig singleton instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = QuantumAPIConfig(env_path)
    return _config_instance


def reset_config():
    """Reset the global config instance (useful for testing)."""
    global _config_instance
    _config_instance = None


# ============================================================================
# TEST FUNCTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUANTUM API CONFIG - VALIDATION TEST")
    print("="*70 + "\n")
    
    # Initialize config
    config = get_config()
    
    # Print summary
    summary = config.get_config_summary()
    print("\n📊 CONFIGURATION SUMMARY:")
    print(f"   Total Sources: {summary['total_sources']}")
    print(f"   Valid Sources: {summary['valid_sources']}")
    print(f"   Primary Source: {summary['primary_source']}")
    print(f"   Intraday Capable: {summary['intraday_sources']}")
    print(f"   Fundamentals Capable: {summary['fundamentals_sources']}")
    print(f"   Global Coverage: {summary['global_coverage']}")
    
    # Test priority ordering
    print("\n🎯 PRIORITY-ORDERED SOURCES:")
    for i, source in enumerate(config.get_sources_by_priority(), 1):
        print(f"   {i}. {source.name} (Rate: {source.rate_limit_per_minute}/min, Coverage: {source.coverage})")
    
    # Test intraday filter
    print("\n⚡ INTRADAY-CAPABLE SOURCES:")
    for source in config.get_sources_by_priority(intraday_only=True):
        print(f"   • {source.name}")
    
    # Test individual source retrieval
    print("\n🔍 INDIVIDUAL SOURCE TEST:")
    polygon = config.get_source('polygon')
    if polygon:
        print(f"   ✓ Polygon: {polygon.name} | Priority {polygon.priority} | Key: {polygon.key[:10]}...")
    else:
        print("   ✗ Polygon: Not available")
    
    print("\n" + "="*70)
    print("✅ API CONFIGURATION TEST COMPLETE")
    print("="*70 + "\n")
