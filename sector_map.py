#!/usr/bin/env python3
"""
SECTOR MAP - Sector Relative Strength Analysis
===============================================
Market Physics Companion: Understanding Sector Rotation

Key Insight: Money flows between sectors predictably.
When one sector leads, others follow in sequence.

This engine:
1. Calculates relative strength for all major sectors
2. Identifies sector leadership and laggards
3. Maps sector rotation patterns
4. Provides sector-conditioned signals

Sector Rotation Model (Classic Business Cycle):
- Early Recovery: Financials, Consumer Discretionary
- Mid Cycle: Industrials, Materials, Technology
- Late Cycle: Energy, Materials
- Recession: Consumer Staples, Healthcare, Utilities

Author: Quantum Trading Research Team
Date: December 20, 2025
"""

import sqlite3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import pdist, squareform
from datetime import datetime, timedelta
import os
import json
import warnings
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DB_PATH = 'data/market_data.db'
OUTPUT_DIR = 'data/sector_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sector ETFs (main proxies)
SECTOR_ETFS = {
    'XLF': 'Financials',
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Discretionary',
    'XLB': 'Materials',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services'
}

# Backup sector proxies (if ETFs not available)
SECTOR_STOCKS = {
    'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
    'Healthcare': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV'],
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
    'Industrials': ['CAT', 'HON', 'UNP', 'BA', 'GE'],
    'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST'],
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
    'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM'],
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP'],
    'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA'],
    'Communication Services': ['GOOGL', 'META', 'DIS', 'NFLX', 'VZ']
}

# Business cycle sector rotation
CYCLE_ROTATION = {
    'Early Recovery': ['Financials', 'Consumer Discretionary', 'Real Estate'],
    'Mid Cycle': ['Technology', 'Industrials', 'Materials'],
    'Late Cycle': ['Energy', 'Materials', 'Healthcare'],
    'Recession': ['Consumer Staples', 'Healthcare', 'Utilities']
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class SectorStrength:
    """Relative strength metrics for a sector"""
    sector: str
    ticker: str
    rs_rank: int  # 1-11 rank (1 = strongest)
    rs_score: float  # Normalized score
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    relative_to_spy: float
    trend: str  # UP, DOWN, SIDEWAYS
    volatility: float


@dataclass
class SectorRotationSignal:
    """Sector rotation signal"""
    date: str
    leaders: List[str]
    laggards: List[str]
    rotation_direction: str  # RISK_ON, RISK_OFF, NEUTRAL
    cycle_phase: str
    recommended_sectors: List[str]
    avoid_sectors: List[str]


# ============================================================
# RELATIVE STRENGTH CALCULATOR
# ============================================================

class RelativeStrengthCalculator:
    """
    Calculates relative strength for sectors vs SPY.
    
    RS = (Sector Return / SPY Return) normalized
    
    Higher RS = Outperforming market
    Lower RS = Underperforming market
    """
    
    def __init__(self):
        self.sector_data = {}
        self.spy_data = None
        
    def load_data(self, df: pd.DataFrame):
        """Load and organize sector data"""
        available_tickers = df['ticker'].unique()
        
        # Load SPY as benchmark
        if 'SPY' in available_tickers:
            self.spy_data = df[df['ticker'] == 'SPY'].copy()
            self.spy_data = self.spy_data.sort_values('date').reset_index(drop=True)
        
        # Load sector ETFs or proxies
        for etf, sector in SECTOR_ETFS.items():
            if etf in available_tickers:
                sector_df = df[df['ticker'] == etf].copy()
                sector_df = sector_df.sort_values('date').reset_index(drop=True)
                self.sector_data[sector] = {
                    'ticker': etf,
                    'data': sector_df
                }
            else:
                # Try backup stocks
                backup_stocks = SECTOR_STOCKS.get(sector, [])
                for stock in backup_stocks:
                    if stock in available_tickers:
                        sector_df = df[df['ticker'] == stock].copy()
                        sector_df = sector_df.sort_values('date').reset_index(drop=True)
                        self.sector_data[sector] = {
                            'ticker': stock,
                            'data': sector_df
                        }
                        break
        
        logger.info(f"Loaded {len(self.sector_data)} sectors")
        
    def calculate_rs_scores(self, as_of_date: str = None) -> Dict[str, SectorStrength]:
        """Calculate relative strength scores for all sectors"""
        results = {}
        
        if self.spy_data is None or len(self.sector_data) == 0:
            logger.warning("Insufficient data for RS calculation")
            return results
        
        # Filter to date
        spy = self.spy_data.copy()
        if as_of_date:
            spy = spy[spy['date'] <= as_of_date]
        
        if len(spy) < 60:
            logger.warning("Insufficient SPY history")
            return results
        
        # SPY returns
        spy_returns = {
            '1m': spy['close'].iloc[-1] / spy['close'].iloc[-21] - 1 if len(spy) >= 21 else 0,
            '3m': spy['close'].iloc[-1] / spy['close'].iloc[-63] - 1 if len(spy) >= 63 else 0,
            '6m': spy['close'].iloc[-1] / spy['close'].iloc[-126] - 1 if len(spy) >= 126 else 0
        }
        
        rs_scores = []
        
        for sector, info in self.sector_data.items():
            df = info['data'].copy()
            if as_of_date:
                df = df[df['date'] <= as_of_date]
            
            if len(df) < 60:
                continue
            
            # Sector returns
            ret_1m = df['close'].iloc[-1] / df['close'].iloc[-21] - 1 if len(df) >= 21 else 0
            ret_3m = df['close'].iloc[-1] / df['close'].iloc[-63] - 1 if len(df) >= 63 else 0
            ret_6m = df['close'].iloc[-1] / df['close'].iloc[-126] - 1 if len(df) >= 126 else 0
            
            # Relative strength vs SPY
            rs_1m = ret_1m - spy_returns['1m']
            rs_3m = ret_3m - spy_returns['3m']
            rs_6m = ret_6m - spy_returns['6m']
            
            # Composite RS score (weighted)
            rs_composite = 0.5 * rs_1m + 0.3 * rs_3m + 0.2 * rs_6m
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # Trend (20-day momentum sign)
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            current = df['close'].iloc[-1]
            
            if current > sma_20 > sma_50:
                trend = 'UP'
            elif current < sma_20 < sma_50:
                trend = 'DOWN'
            else:
                trend = 'SIDEWAYS'
            
            rs_scores.append({
                'sector': sector,
                'ticker': info['ticker'],
                'rs_composite': rs_composite,
                'momentum_1m': ret_1m,
                'momentum_3m': ret_3m,
                'momentum_6m': ret_6m,
                'relative_to_spy': rs_1m,
                'trend': trend,
                'volatility': volatility
            })
        
        # Rank by RS composite
        rs_scores = sorted(rs_scores, key=lambda x: x['rs_composite'], reverse=True)
        
        for i, score in enumerate(rs_scores, 1):
            results[score['sector']] = SectorStrength(
                sector=score['sector'],
                ticker=score['ticker'],
                rs_rank=i,
                rs_score=score['rs_composite'] * 100,  # Convert to %
                momentum_1m=score['momentum_1m'] * 100,
                momentum_3m=score['momentum_3m'] * 100,
                momentum_6m=score['momentum_6m'] * 100,
                relative_to_spy=score['relative_to_spy'] * 100,
                trend=score['trend'],
                volatility=score['volatility'] * 100
            )
        
        return results


# ============================================================
# SECTOR CORRELATION ANALYZER
# ============================================================

class SectorCorrelationAnalyzer:
    """
    Analyzes correlation structure between sectors.
    Uses hierarchical clustering to identify sector groups.
    """
    
    def __init__(self):
        self.correlation_matrix = None
        self.clusters = None
        
    def compute_correlations(self, sector_data: Dict, lookback: int = 60) -> pd.DataFrame:
        """Compute correlation matrix between sectors"""
        returns_dict = {}
        
        for sector, info in sector_data.items():
            df = info['data'].copy()
            if len(df) >= lookback:
                returns = df['close'].pct_change().dropna().tail(lookback)
                returns_dict[sector] = returns.values
        
        if len(returns_dict) < 2:
            return pd.DataFrame()
        
        # Align lengths
        min_len = min(len(r) for r in returns_dict.values())
        returns_df = pd.DataFrame({k: v[-min_len:] for k, v in returns_dict.items()})
        
        self.correlation_matrix = returns_df.corr()
        return self.correlation_matrix
    
    def cluster_sectors(self, n_clusters: int = 4) -> Dict[int, List[str]]:
        """Cluster sectors using hierarchical clustering"""
        if self.correlation_matrix is None or len(self.correlation_matrix) < 2:
            return {}
        
        # Convert correlation to distance
        distance_matrix = 1 - self.correlation_matrix.abs()
        
        # Hierarchical clustering
        condensed = squareform(distance_matrix.values)
        Z = linkage(condensed, method='ward')
        
        # Cut into clusters
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Map sectors to clusters
        self.clusters = {}
        for i, sector in enumerate(self.correlation_matrix.columns):
            cluster_id = clusters[i]
            if cluster_id not in self.clusters:
                self.clusters[cluster_id] = []
            self.clusters[cluster_id].append(sector)
        
        return self.clusters
    
    def find_diversifying_sectors(self, target_sector: str, n: int = 3) -> List[str]:
        """Find sectors least correlated with target for diversification"""
        if self.correlation_matrix is None or target_sector not in self.correlation_matrix.columns:
            return []
        
        correlations = self.correlation_matrix[target_sector].drop(target_sector)
        return correlations.nsmallest(n).index.tolist()


# ============================================================
# SECTOR ROTATION DETECTOR
# ============================================================

class SectorRotationDetector:
    """
    Detects sector rotation patterns and cycle phase.
    
    Signals:
    - RISK_ON: Money flowing to cyclical/growth sectors
    - RISK_OFF: Money flowing to defensive sectors
    - NEUTRAL: No clear rotation
    """
    
    CYCLICAL_SECTORS = ['Technology', 'Consumer Discretionary', 'Industrials', 
                        'Financials', 'Materials', 'Energy']
    DEFENSIVE_SECTORS = ['Consumer Staples', 'Healthcare', 'Utilities', 'Real Estate']
    
    def __init__(self):
        self.rotation_history = []
        
    def detect_rotation(self, rs_scores: Dict[str, SectorStrength]) -> SectorRotationSignal:
        """Detect current sector rotation signal"""
        if len(rs_scores) < 4:
            return SectorRotationSignal(
                date=datetime.now().strftime('%Y-%m-%d'),
                leaders=[],
                laggards=[],
                rotation_direction='INSUFFICIENT_DATA',
                cycle_phase='UNKNOWN',
                recommended_sectors=[],
                avoid_sectors=[]
            )
        
        # Sort by RS rank
        sorted_sectors = sorted(rs_scores.values(), key=lambda x: x.rs_rank)
        
        # Top 3 leaders, bottom 3 laggards
        leaders = [s.sector for s in sorted_sectors[:3]]
        laggards = [s.sector for s in sorted_sectors[-3:]]
        
        # Count cyclical vs defensive in leaders
        cyclical_leaders = sum(1 for s in leaders if s in self.CYCLICAL_SECTORS)
        defensive_leaders = sum(1 for s in leaders if s in self.DEFENSIVE_SECTORS)
        
        cyclical_laggards = sum(1 for s in laggards if s in self.CYCLICAL_SECTORS)
        defensive_laggards = sum(1 for s in laggards if s in self.DEFENSIVE_SECTORS)
        
        # Determine rotation direction
        if cyclical_leaders >= 2 and defensive_laggards >= 2:
            rotation = 'RISK_ON'
        elif defensive_leaders >= 2 and cyclical_laggards >= 2:
            rotation = 'RISK_OFF'
        else:
            rotation = 'NEUTRAL'
        
        # Infer cycle phase
        cycle_phase = self._infer_cycle_phase(leaders, rotation)
        
        # Recommendations
        if rotation == 'RISK_ON':
            recommended = [s for s in leaders if s in self.CYCLICAL_SECTORS]
            avoid = [s for s in laggards if s in self.DEFENSIVE_SECTORS]
        elif rotation == 'RISK_OFF':
            recommended = [s for s in leaders if s in self.DEFENSIVE_SECTORS]
            avoid = [s for s in laggards if s in self.CYCLICAL_SECTORS]
        else:
            recommended = leaders[:2]
            avoid = laggards[:2]
        
        signal = SectorRotationSignal(
            date=datetime.now().strftime('%Y-%m-%d'),
            leaders=leaders,
            laggards=laggards,
            rotation_direction=rotation,
            cycle_phase=cycle_phase,
            recommended_sectors=recommended,
            avoid_sectors=avoid
        )
        
        self.rotation_history.append(signal)
        
        return signal
    
    def _infer_cycle_phase(self, leaders: List[str], rotation: str) -> str:
        """Infer business cycle phase from leading sectors"""
        for phase, sectors in CYCLE_ROTATION.items():
            matches = sum(1 for s in leaders if s in sectors)
            if matches >= 2:
                return phase
        
        if rotation == 'RISK_ON':
            return 'Early/Mid Cycle'
        elif rotation == 'RISK_OFF':
            return 'Late Cycle/Recession'
        else:
            return 'Transition'


# ============================================================
# MAIN SECTOR MAP ENGINE
# ============================================================

class SectorMap:
    """
    Main orchestrator for sector analysis.
    
    Usage:
        smap = SectorMap()
        analysis = smap.analyze()
    """
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.df = None
        
        # Components
        self.rs_calculator = RelativeStrengthCalculator()
        self.correlation_analyzer = SectorCorrelationAnalyzer()
        self.rotation_detector = SectorRotationDetector()
        
        # Results
        self.rs_scores = None
        self.rotation_signal = None
        self.sector_clusters = None
        
    def load_data(self):
        """Load market data"""
        logger.info(f"Loading market data from {self.db_path}")
        conn = sqlite3.connect(self.db_path)
        self.df = pd.read_sql("SELECT * FROM ohlcv", conn)
        conn.close()
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        logger.info(f"Loaded {len(self.df):,} records")
        
    def analyze(self, as_of_date: str = None) -> Dict:
        """Run full sector analysis"""
        logger.info("="*60)
        logger.info("SECTOR MAP - Sector Analysis")
        logger.info("="*60)
        
        # Load data
        if self.df is None:
            self.load_data()
        
        # Initialize RS calculator
        logger.info("\n📊 Loading sector data...")
        self.rs_calculator.load_data(self.df)
        
        # Calculate RS scores
        logger.info("\n📈 Calculating relative strength...")
        self.rs_scores = self.rs_calculator.calculate_rs_scores(as_of_date)
        
        if len(self.rs_scores) == 0:
            logger.warning("No sector data available")
            return {'error': 'No sector data available'}
        
        # Display rankings
        logger.info("\n🏆 Sector Rankings (by Relative Strength):")
        for sector, strength in sorted(self.rs_scores.items(), key=lambda x: x[1].rs_rank):
            emoji = "🟢" if strength.rs_rank <= 3 else ("🟡" if strength.rs_rank <= 7 else "🔴")
            logger.info(f"   {strength.rs_rank:2d}. {emoji} {sector:25s} "
                       f"RS={strength.rs_score:+5.1f}% "
                       f"1M={strength.momentum_1m:+5.1f}% "
                       f"Trend={strength.trend}")
        
        # Correlation analysis
        logger.info("\n🔗 Computing sector correlations...")
        corr_matrix = self.correlation_analyzer.compute_correlations(
            self.rs_calculator.sector_data
        )
        
        if len(corr_matrix) > 0:
            self.sector_clusters = self.correlation_analyzer.cluster_sectors(4)
            logger.info("\n📊 Sector Clusters:")
            for cluster_id, sectors in self.sector_clusters.items():
                logger.info(f"   Cluster {cluster_id}: {', '.join(sectors)}")
        
        # Rotation detection
        logger.info("\n🔄 Detecting sector rotation...")
        self.rotation_signal = self.rotation_detector.detect_rotation(self.rs_scores)
        
        logger.info(f"\n🎯 ROTATION SIGNAL: {self.rotation_signal.rotation_direction}")
        logger.info(f"   Cycle Phase: {self.rotation_signal.cycle_phase}")
        logger.info(f"   Leaders: {', '.join(self.rotation_signal.leaders)}")
        logger.info(f"   Laggards: {', '.join(self.rotation_signal.laggards)}")
        logger.info(f"   Recommended: {', '.join(self.rotation_signal.recommended_sectors)}")
        logger.info(f"   Avoid: {', '.join(self.rotation_signal.avoid_sectors)}")
        
        # Save outputs
        self._save_outputs()
        
        logger.info(f"\n📁 Saved to {OUTPUT_DIR}/")
        
        return {
            'rs_scores': {k: asdict(v) for k, v in self.rs_scores.items()},
            'rotation_signal': asdict(self.rotation_signal),
            'clusters': self.sector_clusters
        }
    
    def _save_outputs(self):
        """Save analysis outputs"""
        # RS scores
        rs_df = pd.DataFrame([asdict(v) for v in self.rs_scores.values()])
        rs_df.to_csv(f'{OUTPUT_DIR}/sector_rs_scores.csv', index=False)
        
        # Rotation signal
        with open(f'{OUTPUT_DIR}/rotation_signal.json', 'w') as f:
            json.dump(asdict(self.rotation_signal), f, indent=2)
        
        # Correlation matrix
        if self.correlation_analyzer.correlation_matrix is not None:
            self.correlation_analyzer.correlation_matrix.to_csv(
                f'{OUTPUT_DIR}/sector_correlations.csv'
            )
    
    def get_sector_for_stock(self, ticker: str) -> Optional[str]:
        """Get sector for a given stock"""
        for sector, stocks in SECTOR_STOCKS.items():
            if ticker in stocks:
                return sector
        return None
    
    def recommend_for_regime(self, regime: str) -> List[str]:
        """Get sector recommendations for given market regime"""
        if regime in ['BULL', 'BULL_LOWVOL', 'BULL_HIGHVOL']:
            return ['Technology', 'Consumer Discretionary', 'Financials']
        elif regime in ['BEAR', 'BEAR_LOWVOL', 'BEAR_HIGHVOL']:
            return ['Consumer Staples', 'Healthcare', 'Utilities']
        else:  # RANGE
            return ['Healthcare', 'Utilities', 'Consumer Staples']
    
    def generate_sector_report(self) -> str:
        """Generate markdown sector report"""
        if self.rs_scores is None:
            return "No analysis run yet. Call analyze() first."
        
        lines = [
            "# Sector Analysis Report",
            f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## 🔄 Rotation Signal",
            f"**Direction:** {self.rotation_signal.rotation_direction}",
            f"**Cycle Phase:** {self.rotation_signal.cycle_phase}",
            "",
            "### Leaders",
        ]
        
        for sector in self.rotation_signal.leaders:
            if sector in self.rs_scores:
                s = self.rs_scores[sector]
                lines.append(f"- **{sector}**: RS={s.rs_score:+.1f}%, 1M={s.momentum_1m:+.1f}%")
        
        lines.extend([
            "",
            "### Laggards",
        ])
        
        for sector in self.rotation_signal.laggards:
            if sector in self.rs_scores:
                s = self.rs_scores[sector]
                lines.append(f"- **{sector}**: RS={s.rs_score:+.1f}%, 1M={s.momentum_1m:+.1f}%")
        
        lines.extend([
            "",
            "## 🏆 Full Sector Rankings",
            "",
            "| Rank | Sector | RS Score | 1M | 3M | Trend |",
            "|------|--------|----------|----|----|-------|"
        ])
        
        for sector, strength in sorted(self.rs_scores.items(), key=lambda x: x[1].rs_rank):
            lines.append(
                f"| {strength.rs_rank} | {sector} | {strength.rs_score:+.1f}% | "
                f"{strength.momentum_1m:+.1f}% | {strength.momentum_3m:+.1f}% | {strength.trend} |"
            )
        
        lines.extend([
            "",
            "## 📊 Recommendations",
            "",
            "### Sectors to Consider",
        ])
        
        for sector in self.rotation_signal.recommended_sectors:
            lines.append(f"- ✅ **{sector}**")
        
        lines.extend([
            "",
            "### Sectors to Avoid",
        ])
        
        for sector in self.rotation_signal.avoid_sectors:
            lines.append(f"- ❌ **{sector}**")
        
        lines.append("\n---\n*This is research output, not financial advice.*")
        
        report = '\n'.join(lines)
        
        # Save report
        with open(f'{OUTPUT_DIR}/sector_report.md', 'w') as f:
            f.write(report)
        
        return report


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Sector Map - Sector Analysis')
    parser.add_argument('--date', type=str, default=None,
                       help='Analysis date (YYYY-MM-DD)')
    parser.add_argument('--report', action='store_true',
                       help='Generate markdown report')
    args = parser.parse_args()
    
    smap = SectorMap()
    analysis = smap.analyze(as_of_date=args.date)
    
    # Print summary
    print("\n" + "="*60)
    print("📊 SECTOR MAP SUMMARY")
    print("="*60)
    
    signal = smap.rotation_signal
    print(f"\n🔄 Rotation: {signal.rotation_direction}")
    print(f"   Cycle Phase: {signal.cycle_phase}")
    
    print(f"\n🏆 Top Sectors:")
    for sector in signal.leaders:
        if sector in smap.rs_scores:
            s = smap.rs_scores[sector]
            print(f"   {s.rs_rank}. {sector}: RS={s.rs_score:+.1f}%, Trend={s.trend}")
    
    print(f"\n📉 Bottom Sectors:")
    for sector in signal.laggards:
        if sector in smap.rs_scores:
            s = smap.rs_scores[sector]
            print(f"   {s.rs_rank}. {sector}: RS={s.rs_score:+.1f}%, Trend={s.trend}")
    
    print(f"\n✅ Recommended: {', '.join(signal.recommended_sectors)}")
    print(f"❌ Avoid: {', '.join(signal.avoid_sectors)}")
    
    if args.report:
        print("\n📝 Generating report...")
        report = smap.generate_sector_report()
        print(f"   Saved to {OUTPUT_DIR}/sector_report.md")
    
    print(f"\n📁 Output saved to: {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
