"""
Quantum API Configuration Manager v2
Enhanced with metrics tracking, circuit breaker support, and extensible source registry.

DESIGN DECISIONS:
1. Plugin pattern for sources → Easy to add Reddit, insider feeds, etc.
2. Dataclass-based config → Type safety and clarity
3. Singleton with DI support → Simple usage but testable
4. Source registry → Dynamic source discovery and registration
"""

import os
import logging
from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod
from enum import Enum
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Data source types for categorization."""
    MARKET_DATA = "market_data"
    SENTIMENT = "sentiment"
    FUNDAMENTAL = "fundamental"
    INSIDER = "insider"
    ALTERNATIVE = "alternative"


@dataclass
class SourceMetadata:
    """
    Metadata for a data source.
    
    WHY: Centralized source configuration makes it easy to adjust priorities,
    rate limits, and capabilities without touching orchestrator code.
    """
    name: str
    key_env_var: str
    priority: int
    rate_limit_per_minute: int
    rate_limit_per_second: float  # For finer control
    supports_intraday: bool
    supports_fundamentals: bool
    supports_options: bool
    coverage: str  # "US", "Global", etc.
    base_url: str
    source_type: SourceType
    typical_latency_ms: int  # Observed average latency
    reliability_score: float = 0.95  # Track this dynamically
    
    def __post_init__(self):
        """Load and validate API key."""
        self.key = os.getenv(self.key_env_var, '')
        self.is_valid = bool(self.key and len(self.key) > 10)
        self.last_validated = datetime.now()
    
    def refresh_key(self):
        """Reload API key from environment (useful after .env changes)."""
        self.key = os.getenv(self.key_env_var, '')
        self.is_valid = bool(self.key and len(self.key) > 10)
        self.last_validated = datetime.now()


class SourceRegistry:
    """
    Registry of all available data sources.
    
    WHY: Plugin pattern makes it trivial to add new sources (Reddit sentiment,
    insider feeds, etc.) without modifying core orchestrator logic.
    """
    
    def __init__(self):
        self._sources: Dict[str, SourceMetadata] = {}
        self._register_default_sources()
    
    def _register_default_sources(self):
        """Register built-in market data sources."""
        
        # Priority 1: Polygon - Best for US equities, high quality
        self.register(SourceMetadata(
            name='polygon',
            key_env_var='POLYGON_API_KEY',
            priority=1,
            rate_limit_per_minute=5,  # Free tier
            rate_limit_per_second=0.083,  # ~5/min
            supports_intraday=True,
            supports_fundamentals=True,
            supports_options=True,
            coverage='US',
            base_url='https://api.polygon.io',
            source_type=SourceType.MARKET_DATA,
            typical_latency_ms=300
        ))
        
        # Priority 2: FMP - Excellent for fundamentals + global coverage
        self.register(SourceMetadata(
            name='fmp',
            key_env_var='FMP_API_KEY',
            priority=2,
            rate_limit_per_minute=300,  # Standard tier
            rate_limit_per_second=5.0,
            supports_intraday=True,
            supports_fundamentals=True,
            supports_options=False,
            coverage='Global',
            base_url='https://financialmodelingprep.com/api/v3',
            source_type=SourceType.MARKET_DATA,
            typical_latency_ms=400
        ))
        
        # Priority 3: AlphaVantage - Reliable fallback
        self.register(SourceMetadata(
            name='alphavantage',
            key_env_var='ALPHAVANTAGE_API_KEY',
            priority=3,
            rate_limit_per_minute=5,  # Free tier
            rate_limit_per_second=0.083,
            supports_intraday=True,
            supports_fundamentals=True,
            supports_options=False,
            coverage='Global',
            base_url='https://www.alphavantage.co',
            source_type=SourceType.MARKET_DATA,
            typical_latency_ms=600
        ))
        
        # Priority 4: EODHD - Good historical data
        self.register(SourceMetadata(
            name='eodhd',
            key_env_var='EODHD_API_TOKEN',
            priority=4,
            rate_limit_per_minute=20,  # Free tier
            rate_limit_per_second=0.33,
            supports_intraday=False,
            supports_fundamentals=True,
            supports_options=False,
            coverage='Global',
            base_url='https://eodhistoricaldata.com/api',
            source_type=SourceType.MARKET_DATA,
            typical_latency_ms=500
        ))
    
    def register(self, source: SourceMetadata):
        """
        Register a new data source.
        
        WHY: Allows dynamic source addition without code changes.
        Example: registry.register(RedditSentimentSource(...))
        """
        self._sources[source.name.lower()] = source
        logger.debug(f"Registered source: {source.name}")
    
    def get(self, name: str) -> Optional[SourceMetadata]:
        """Get source by name."""
        return self._sources.get(name.lower())
    
    def get_all(self) -> List[SourceMetadata]:
        """Get all registered sources."""
        return list(self._sources.values())
    
    def get_valid(self) -> List[SourceMetadata]:
        """Get only sources with valid API keys."""
        return [s for s in self._sources.values() if s.is_valid]
    
    def get_by_type(self, source_type: SourceType) -> List[SourceMetadata]:
        """Get sources by type (market_data, sentiment, etc.)."""
        return [s for s in self._sources.values() if s.source_type == source_type]
    
    def get_by_priority(self, 
                       intraday_only: bool = False,
                       fundamentals_only: bool = False) -> List[SourceMetadata]:
        """
        Get valid sources ordered by priority with optional filtering.
        
        WHY: Orchestrator needs priority-ordered list for fallback chain.
        """
        sources = self.get_valid()
        
        if intraday_only:
            sources = [s for s in sources if s.supports_intraday]
        
        if fundamentals_only:
            sources = [s for s in sources if s.supports_fundamentals]
        
        return sorted(sources, key=lambda x: x.priority)


class QuantumAPIConfig:
    """
    Central API configuration manager with advanced features.
    
    DESIGN DECISIONS:
    1. Uses SourceRegistry for extensibility (easy to add new sources)
    2. Tracks source health metrics (for circuit breaker pattern)
    3. Supports dependency injection (testable) but defaults to singleton (simple)
    """
    
    def __init__(self, env_path: Optional[str] = None, registry: Optional[SourceRegistry] = None):
        """
        Initialize configuration.
        
        Args:
            env_path: Optional path to .env file
            registry: Optional SourceRegistry (for testing/custom sources)
        
        WHY: Dependency injection allows testing with mock registry,
        but defaults make it simple to use in production.
        """
        self.registry = registry or SourceRegistry()
        self._load_environment(env_path)
        self._validate_and_report()
        
        # Track source health for circuit breaker pattern
        self.source_health: Dict[str, Dict] = {}
        for source in self.registry.get_all():
            self.source_health[source.name] = {
                'consecutive_failures': 0,
                'last_failure': None,
                'circuit_open': False
            }
    
    def _load_environment(self, env_path: Optional[str] = None):
        """
        Load environment variables with fallback paths.
        
        WHY: Supports multiple deployment scenarios (local dev, Colab, production).
        """
        if env_path:
            load_dotenv(env_path)
            logger.info(f"✓ Loaded environment from: {env_path}")
        else:
            # Try multiple common locations
            possible_paths = [
                ".env",
                "../.env",
                "../../.env",
                "E:/quantum-ai-trader-v1.1/.env",
                "/content/.env"  # Colab
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    load_dotenv(path)
                    logger.info(f"✓ Loaded environment from: {path}")
                    break
        
        # Refresh all source keys after loading env
        for source in self.registry.get_all():
            source.refresh_key()
    
    def _validate_and_report(self):
        """Validate all sources and log detailed report."""
        logger.info("\n" + "="*70)
        logger.info("QUANTUM API CONFIGURATION v2 - VALIDATION")
        logger.info("="*70)
        
        all_sources = self.registry.get_all()
        valid_sources = self.registry.get_valid()
        
        for source in sorted(all_sources, key=lambda x: x.priority):
            if source.is_valid:
                logger.info(
                    f"✓ {source.name:15} | Priority {source.priority} | "
                    f"Rate: {source.rate_limit_per_minute:3}/min | "
                    f"Coverage: {source.coverage:7} | "
                    f"Latency: ~{source.typical_latency_ms}ms"
                )
            else:
                logger.warning(f"✗ {source.name:15} | No valid API key")
        
        logger.info("="*70)
        logger.info(f"VALIDATED: {len(valid_sources)}/{len(all_sources)} sources ready")
        logger.info("="*70 + "\n")
    
    def get_source(self, name: str) -> Optional[SourceMetadata]:
        """Get a specific source by name."""
        return self.registry.get(name)
    
    def get_valid_sources(self, **filters) -> List[SourceMetadata]:
        """Get valid sources with optional filters."""
        return self.registry.get_by_priority(**filters)
    
    def get_primary_source(self) -> Optional[SourceMetadata]:
        """Get highest priority valid source."""
        valid = self.registry.get_valid()
        return min(valid, key=lambda x: x.priority) if valid else None
    
    def mark_source_failure(self, source_name: str):
        """
        Mark a source failure for circuit breaker pattern.
        
        WHY: If a source fails repeatedly, temporarily disable it to avoid
        wasting time on broken sources.
        """
        if source_name in self.source_health:
            health = self.source_health[source_name]
            health['consecutive_failures'] += 1
            health['last_failure'] = datetime.now()
            
            # Open circuit after 3 consecutive failures
            if health['consecutive_failures'] >= 3:
                health['circuit_open'] = True
                logger.warning(f"⚠️  Circuit breaker opened for {source_name}")
    
    def mark_source_success(self, source_name: str):
        """Mark a source success (resets failure count)."""
        if source_name in self.source_health:
            health = self.source_health[source_name]
            health['consecutive_failures'] = 0
            health['circuit_open'] = False
    
    def is_source_available(self, source_name: str) -> bool:
        """
        Check if source is available (not circuit broken).
        
        WHY: Orchestrator can skip sources with open circuits.
        """
        if source_name not in self.source_health:
            return True
        
        health = self.source_health[source_name]
        
        # Auto-reset circuit after 5 minutes
        if health['circuit_open'] and health['last_failure']:
            minutes_since_failure = (datetime.now() - health['last_failure']).seconds / 60
            if minutes_since_failure > 5:
                health['circuit_open'] = False
                health['consecutive_failures'] = 0
                logger.info(f"✓ Circuit breaker reset for {source_name}")
        
        return not health['circuit_open']
    
    def get_stats(self) -> Dict:
        """Get configuration statistics."""
        all_sources = self.registry.get_all()
        valid_sources = self.registry.get_valid()
        
        return {
            'total_sources': len(all_sources),
            'valid_sources': len(valid_sources),
            'primary_source': self.get_primary_source().name if valid_sources else None,
            'sources_by_type': {
                st.value: len(self.registry.get_by_type(st))
                for st in SourceType
            },
            'circuit_breakers_open': sum(
                1 for h in self.source_health.values() if h['circuit_open']
            )
        }


# ============================================================================
# GLOBAL SINGLETON (but supports DI for testing)
# ============================================================================

_config_instance: Optional[QuantumAPIConfig] = None


def get_config(env_path: Optional[str] = None, 
               registry: Optional[SourceRegistry] = None) -> QuantumAPIConfig:
    """
    Get or create global config instance.
    
    WHY: Singleton pattern for simplicity, but allows dependency injection
    for testing by passing custom registry.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = QuantumAPIConfig(env_path, registry)
    return _config_instance


def reset_config():
    """Reset global config (for testing)."""
    global _config_instance
    _config_instance = None


# ============================================================================
# EXTENSIBILITY EXAMPLE: How to add a new source
# ============================================================================

def register_custom_source(name: str, key_env_var: str, base_url: str, 
                          source_type: SourceType = SourceType.ALTERNATIVE):
    """
    Helper to register custom sources (Reddit, insider feeds, etc.).
    
    Example:
        register_custom_source(
            name='reddit_sentiment',
            key_env_var='REDDIT_API_KEY',
            base_url='https://api.reddit.com',
            source_type=SourceType.SENTIMENT
        )
    """
    config = get_config()
    custom_source = SourceMetadata(
        name=name,
        key_env_var=key_env_var,
        priority=99,  # Custom sources get lower priority
        rate_limit_per_minute=60,
        rate_limit_per_second=1.0,
        supports_intraday=False,
        supports_fundamentals=False,
        supports_options=False,
        coverage='N/A',
        base_url=base_url,
        source_type=source_type,
        typical_latency_ms=1000
    )
    config.registry.register(custom_source)
    logger.info(f"✓ Registered custom source: {name}")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("QUANTUM API CONFIG v2 - TEST")
    print("="*70 + "\n")
    
    # Test configuration
    config = get_config()
    
    # Test stats
    stats = config.get_stats()
    print("📊 Configuration Stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Test source retrieval
    print("\n🎯 Valid Sources (Priority Order):")
    for source in config.get_valid_sources():
        print(f"   {source.priority}. {source.name} ({source.rate_limit_per_minute}/min)")
    
    # Test circuit breaker
    print("\n🔌 Testing Circuit Breaker:")
    config.mark_source_failure('polygon')
    config.mark_source_failure('polygon')
    config.mark_source_failure('polygon')
    print(f"   Polygon available: {config.is_source_available('polygon')}")
    
    # Test extensibility
    print("\n🔧 Testing Custom Source Registration:")
    register_custom_source(
        name='reddit_sentiment',
        key_env_var='REDDIT_API_KEY',
        base_url='https://api.reddit.com',
        source_type=SourceType.SENTIMENT
    )
    print(f"   Total sources: {len(config.registry.get_all())}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70 + "\n")
