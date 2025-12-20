#!/usr/bin/env python3
"""
=============================================================================
IDEAL TICKER SCANNER
=============================================================================

Finds stocks that are IDEAL for your validated signal types:
- Mean reversion signals need: High volatility, range-bound behavior
- Momentum signals need: Strong trends, volume
- VIX-correlation signals need: High beta to market

This scans universes of stocks to find ones with characteristics
that match your signal types.

NOT for trading - just for finding candidates for your watchlist.

Usage:
    python IDEAL_TICKER_SCANNER.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

import pandas as pd
import numpy as np
import yfinance as yf

warnings.filterwarnings('ignore')


# =============================================================================
# STOCK UNIVERSES TO SCAN
# =============================================================================

# Popular ETFs and their characteristics
ETFS = {
    'high_beta': ['TQQQ', 'SOXL', 'TECL', 'FNGU', 'LABU'],  # 3x leveraged
    'sector': ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE'],
    'thematic': ['ARKK', 'ARKG', 'ARKF', 'ARKW', 'ARKQ', 'TAN', 'LIT', 'ICLN', 'BOTZ', 'ROBO'],
    'commodity': ['GLD', 'SLV', 'USO', 'UNG', 'WEAT', 'CORN', 'DBA'],
    'broad': ['SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO'],
}

# High-volatility individual stocks by sector (candidates for mean reversion)
HIGH_VOL_STOCKS = {
    'tech_volatile': ['NVDA', 'AMD', 'TSLA', 'PLTR', 'SNOW', 'NET', 'DDOG', 'CRWD', 'ZS'],
    'biotech': ['MRNA', 'BNTX', 'REGN', 'VRTX', 'BIIB', 'GILD', 'ILMN'],
    'crypto_related': ['COIN', 'MSTR', 'MARA', 'RIOT', 'HUT', 'CLSK'],
    'ev_clean': ['RIVN', 'LCID', 'NIO', 'XPEV', 'LI', 'PLUG', 'FCEL', 'BE'],
    'space_defense': ['RKLB', 'ASTS', 'LUNR', 'SPCE', 'BKSY', 'RDW'],
    'semiconductors': ['MU', 'AVGO', 'MRVL', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC'],
    'software': ['CRM', 'NOW', 'ADBE', 'PANW', 'FTNT', 'WDAY', 'ZM', 'DOCN'],
}

# Quality dividend stocks (for stability comparison)
QUALITY_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JNJ', 'PG', 'KO', 'PEP', 'WMT']


# =============================================================================
# METRICS CALCULATOR
# =============================================================================

class StockMetrics:
    """Calculate metrics that indicate suitability for different signal types."""
    
    @staticmethod
    def calculate_all(data: pd.DataFrame, spy_data: pd.DataFrame = None) -> Dict:
        """Calculate all metrics for a stock."""
        if data.empty or len(data) < 252:
            return {'valid': False}
        
        close = data['close']
        returns = close.pct_change().dropna()
        
        metrics = {'valid': True}
        
        # Volatility metrics
        metrics['volatility_annual'] = returns.std() * np.sqrt(252)
        metrics['volatility_20d'] = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Mean reversion metrics
        # Higher = more mean reverting
        ma20 = close.rolling(20).mean()
        deviation_from_ma = (close - ma20) / ma20
        metrics['mean_reversion_score'] = abs(deviation_from_ma).mean() * 100  # Avg % deviation
        
        # Count times it crossed MA20
        crosses = ((close > ma20) != (close > ma20).shift(1)).sum()
        metrics['ma_crosses_per_year'] = crosses / (len(data) / 252)
        
        # Range-bound vs trending
        # ADX-like measure: if stock moves a lot but ends up nowhere, it's range-bound
        total_move = abs(close.iloc[-1] / close.iloc[0] - 1)
        sum_daily_moves = returns.abs().sum()
        metrics['efficiency_ratio'] = total_move / sum_daily_moves if sum_daily_moves > 0 else 0
        # Low efficiency = range-bound (good for mean reversion)
        # High efficiency = trending (good for momentum)
        
        # Momentum metrics
        metrics['return_1y'] = (close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else None
        metrics['return_6m'] = (close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else None
        metrics['return_3m'] = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else None
        
        # Beta (if SPY provided)
        if spy_data is not None and len(spy_data) > 0:
            spy_ret = spy_data['close'].pct_change().dropna()
            common_idx = returns.index.intersection(spy_ret.index)
            if len(common_idx) > 100:
                stock_aligned = returns.loc[common_idx]
                spy_aligned = spy_ret.loc[common_idx]
                cov = np.cov(stock_aligned, spy_aligned)[0, 1]
                var = np.var(spy_aligned)
                metrics['beta'] = cov / var if var > 0 else 1.0
            else:
                metrics['beta'] = None
        
        # Volume metrics
        if 'volume' in data.columns:
            avg_volume = data['volume'].mean()
            metrics['avg_daily_volume'] = avg_volume
            metrics['dollar_volume'] = avg_volume * close.iloc[-1]
            
            # Volume stability
            vol_std = data['volume'].std()
            metrics['volume_cv'] = vol_std / avg_volume if avg_volume > 0 else 0
        
        # RSI characteristics
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        # How often is it oversold/overbought?
        metrics['pct_time_oversold'] = (rsi < 30).mean() * 100
        metrics['pct_time_overbought'] = (rsi > 70).mean() * 100
        metrics['pct_time_extreme'] = ((rsi < 30) | (rsi > 70)).mean() * 100
        
        # Gap frequency (for gap reversal signal)
        gaps = (data['open'] - close.shift(1)) / close.shift(1)
        metrics['gap_down_freq'] = (gaps < -0.02).mean() * 252  # Gap downs per year
        metrics['gap_up_freq'] = (gaps > 0.02).mean() * 252
        
        return metrics


# =============================================================================
# IDEAL STOCK FINDER
# =============================================================================

class IdealTickerScanner:
    """
    Scans stock universes to find ideal candidates for your signal types.
    """
    
    def __init__(self):
        self.results = {}
        self.spy_data = None
    
    def fetch_spy(self):
        """Fetch SPY for beta calculations."""
        if self.spy_data is None:
            self.spy_data = yf.download('SPY', period='2y', progress=False)
            if isinstance(self.spy_data.columns, pd.MultiIndex):
                self.spy_data.columns = self.spy_data.columns.get_level_values(0)
            self.spy_data.columns = [c.lower() for c in self.spy_data.columns]
    
    def scan_symbol(self, symbol: str) -> Dict:
        """Scan a single symbol."""
        try:
            data = yf.download(symbol, period='2y', progress=False)
            if data.empty:
                return {'symbol': symbol, 'valid': False, 'reason': 'No data'}
            
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            data.columns = [c.lower() for c in data.columns]
            
            if len(data) < 252:
                return {'symbol': symbol, 'valid': False, 'reason': 'Insufficient history'}
            
            metrics = StockMetrics.calculate_all(data, self.spy_data)
            metrics['symbol'] = symbol
            metrics['current_price'] = data['close'].iloc[-1]
            
            return metrics
        
        except Exception as e:
            return {'symbol': symbol, 'valid': False, 'reason': str(e)}
    
    def scan_universe(self, symbols: List[str], category: str = None) -> List[Dict]:
        """Scan a list of symbols."""
        self.fetch_spy()
        
        results = []
        for symbol in symbols:
            result = self.scan_symbol(symbol)
            if category:
                result['category'] = category
            results.append(result)
        
        return results
    
    def find_ideal_for_mean_reversion(self, all_results: List[Dict]) -> List[Dict]:
        """
        Find stocks ideal for mean reversion signals.
        
        Criteria:
        - High volatility (more opportunities)
        - Frequent MA crosses (mean reverting behavior)
        - Low efficiency ratio (range-bound)
        - Frequent RSI extremes (oversold/overbought)
        """
        valid = [r for r in all_results if r.get('valid')]
        
        scored = []
        for r in valid:
            score = 0
            
            # High volatility is good (0-30 points)
            vol = r.get('volatility_annual', 0)
            if vol > 0.8:
                score += 30
            elif vol > 0.5:
                score += 20
            elif vol > 0.3:
                score += 10
            
            # Frequent MA crosses (0-25 points)
            crosses = r.get('ma_crosses_per_year', 0)
            if crosses > 20:
                score += 25
            elif crosses > 15:
                score += 20
            elif crosses > 10:
                score += 15
            elif crosses > 5:
                score += 10
            
            # Low efficiency (range-bound) (0-25 points)
            eff = r.get('efficiency_ratio', 1)
            if eff < 0.1:
                score += 25
            elif eff < 0.2:
                score += 20
            elif eff < 0.3:
                score += 15
            elif eff < 0.4:
                score += 10
            
            # RSI extremes (0-20 points)
            extreme = r.get('pct_time_extreme', 0)
            if extreme > 20:
                score += 20
            elif extreme > 15:
                score += 15
            elif extreme > 10:
                score += 10
            
            r['mean_reversion_ideal_score'] = score
            scored.append(r)
        
        return sorted(scored, key=lambda x: -x['mean_reversion_ideal_score'])
    
    def find_ideal_for_momentum(self, all_results: List[Dict]) -> List[Dict]:
        """
        Find stocks ideal for momentum signals.
        
        Criteria:
        - Strong recent returns
        - High efficiency ratio (trending)
        - Good volume
        """
        valid = [r for r in all_results if r.get('valid')]
        
        scored = []
        for r in valid:
            score = 0
            
            # Strong momentum (0-40 points)
            ret_3m = r.get('return_3m', 0) or 0
            if ret_3m > 0.3:
                score += 40
            elif ret_3m > 0.2:
                score += 30
            elif ret_3m > 0.1:
                score += 20
            elif ret_3m > 0:
                score += 10
            
            # High efficiency (trending) (0-30 points)
            eff = r.get('efficiency_ratio', 0)
            if eff > 0.5:
                score += 30
            elif eff > 0.4:
                score += 25
            elif eff > 0.3:
                score += 20
            elif eff > 0.2:
                score += 15
            
            # Good volume (0-30 points)
            dv = r.get('dollar_volume', 0)
            if dv > 1e9:
                score += 30
            elif dv > 500e6:
                score += 25
            elif dv > 100e6:
                score += 20
            elif dv > 50e6:
                score += 15
            elif dv > 10e6:
                score += 10
            
            r['momentum_ideal_score'] = score
            scored.append(r)
        
        return sorted(scored, key=lambda x: -x['momentum_ideal_score'])
    
    def find_ideal_for_gap_reversal(self, all_results: List[Dict]) -> List[Dict]:
        """
        Find stocks ideal for gap reversal signals.
        
        Criteria:
        - Frequent gaps
        - High volatility
        - Sufficient volume
        """
        valid = [r for r in all_results if r.get('valid')]
        
        scored = []
        for r in valid:
            score = 0
            
            # Gap frequency (0-50 points)
            gaps = r.get('gap_down_freq', 0) + r.get('gap_up_freq', 0)
            if gaps > 50:
                score += 50
            elif gaps > 30:
                score += 40
            elif gaps > 20:
                score += 30
            elif gaps > 10:
                score += 20
            
            # High volatility (0-30 points)
            vol = r.get('volatility_annual', 0)
            if vol > 0.8:
                score += 30
            elif vol > 0.5:
                score += 20
            elif vol > 0.3:
                score += 10
            
            # Volume (0-20 points)
            dv = r.get('dollar_volume', 0)
            if dv > 100e6:
                score += 20
            elif dv > 50e6:
                score += 15
            elif dv > 10e6:
                score += 10
            
            r['gap_reversal_ideal_score'] = score
            scored.append(r)
        
        return sorted(scored, key=lambda x: -x['gap_reversal_ideal_score'])


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*70)
    print("IDEAL TICKER SCANNER")
    print("="*70)
    print("Finding stocks ideal for your validated signal types...")
    print("="*70)
    
    scanner = IdealTickerScanner()
    all_results = []
    
    # Scan all universes
    print("\n📊 Scanning stock universes...")
    
    # Scan ETFs
    for category, symbols in ETFS.items():
        print(f"   Scanning {category} ETFs...")
        results = scanner.scan_universe(symbols, f"ETF_{category}")
        all_results.extend(results)
    
    # Scan high-vol stocks
    for category, symbols in HIGH_VOL_STOCKS.items():
        print(f"   Scanning {category}...")
        results = scanner.scan_universe(symbols, category)
        all_results.extend(results)
    
    # Scan quality stocks for comparison
    print("   Scanning quality stocks...")
    results = scanner.scan_universe(QUALITY_STOCKS, "quality")
    all_results.extend(results)
    
    # Load user's watchlist
    watchlist_file = Path('./watchlist_config.json')
    if watchlist_file.exists():
        import json
        with open(watchlist_file) as f:
            config = json.load(f)
        user_symbols = config.get('watchlist', {}).get('symbols', [])
        if user_symbols:
            print(f"   Scanning YOUR watchlist ({len(user_symbols)} symbols)...")
            results = scanner.scan_universe(user_symbols, "your_portfolio")
            all_results.extend(results)
    
    valid_results = [r for r in all_results if r.get('valid')]
    print(f"\n✅ Scanned {len(all_results)} symbols, {len(valid_results)} valid")
    
    # Find ideal stocks for each signal type
    print("\n" + "="*70)
    print("🎯 IDEAL FOR MEAN REVERSION (H16, H19, RSI)")
    print("="*70)
    mr_ideal = scanner.find_ideal_for_mean_reversion(valid_results)
    print(f"{'Symbol':<10} {'Category':<20} {'Vol':<8} {'MA Crosses':<12} {'RSI Ext%':<10} {'Score':<8}")
    print("-"*70)
    for r in mr_ideal[:15]:
        print(f"{r['symbol']:<10} {r.get('category', 'N/A'):<20} "
              f"{r.get('volatility_annual', 0)*100:>5.0f}% "
              f"{r.get('ma_crosses_per_year', 0):>10.0f} "
              f"{r.get('pct_time_extreme', 0):>8.1f}% "
              f"{r.get('mean_reversion_ideal_score', 0):>6}")
    
    print("\n" + "="*70)
    print("🚀 IDEAL FOR MOMENTUM")
    print("="*70)
    mom_ideal = scanner.find_ideal_for_momentum(valid_results)
    print(f"{'Symbol':<10} {'Category':<20} {'3M Ret':<10} {'Efficiency':<12} {'$Vol/day':<12} {'Score':<8}")
    print("-"*70)
    for r in mom_ideal[:15]:
        ret = r.get('return_3m', 0) or 0
        dv = r.get('dollar_volume', 0)
        print(f"{r['symbol']:<10} {r.get('category', 'N/A'):<20} "
              f"{ret*100:>+7.1f}% "
              f"{r.get('efficiency_ratio', 0):>10.2f} "
              f"${dv/1e6:>9.0f}M "
              f"{r.get('momentum_ideal_score', 0):>6}")
    
    print("\n" + "="*70)
    print("📉 IDEAL FOR GAP REVERSAL")
    print("="*70)
    gap_ideal = scanner.find_ideal_for_gap_reversal(valid_results)
    print(f"{'Symbol':<10} {'Category':<20} {'Gaps/Yr':<10} {'Vol':<10} {'Score':<8}")
    print("-"*70)
    for r in gap_ideal[:15]:
        gaps = r.get('gap_down_freq', 0) + r.get('gap_up_freq', 0)
        print(f"{r['symbol']:<10} {r.get('category', 'N/A'):<20} "
              f"{gaps:>8.0f} "
              f"{r.get('volatility_annual', 0)*100:>7.0f}% "
              f"{r.get('gap_reversal_ideal_score', 0):>6}")
    
    # Check YOUR portfolio stocks specifically
    your_stocks = [r for r in valid_results if r.get('category') == 'your_portfolio']
    if your_stocks:
        print("\n" + "="*70)
        print("📋 YOUR PORTFOLIO ANALYSIS")
        print("="*70)
        
        # Add scores to your stocks
        for r in your_stocks:
            # Find in scored lists
            for mr in mr_ideal:
                if mr['symbol'] == r['symbol']:
                    r['mr_score'] = mr.get('mean_reversion_ideal_score', 0)
            for mom in mom_ideal:
                if mom['symbol'] == r['symbol']:
                    r['mom_score'] = mom.get('momentum_ideal_score', 0)
            for gap in gap_ideal:
                if gap['symbol'] == r['symbol']:
                    r['gap_score'] = gap.get('gap_reversal_ideal_score', 0)
        
        print(f"{'Symbol':<10} {'Vol':<8} {'MR Score':<10} {'Mom Score':<10} {'Gap Score':<10} {'Best For':<15}")
        print("-"*70)
        for r in sorted(your_stocks, key=lambda x: -(x.get('mr_score', 0) + x.get('mom_score', 0))):
            mr_s = r.get('mr_score', 0)
            mom_s = r.get('mom_score', 0)
            gap_s = r.get('gap_score', 0)
            
            best = 'Unknown'
            if mr_s >= mom_s and mr_s >= gap_s:
                best = 'Mean Reversion'
            elif mom_s >= mr_s and mom_s >= gap_s:
                best = 'Momentum'
            else:
                best = 'Gap Reversal'
            
            print(f"{r['symbol']:<10} {r.get('volatility_annual', 0)*100:>5.0f}% "
                  f"{mr_s:>8} {mom_s:>10} {gap_s:>10} {best:<15}")
    
    # Save results
    print("\n" + "="*70)
    print("RECOMMENDED ADDITIONS TO WATCHLIST")
    print("="*70)
    
    # Find stocks not in user's portfolio but highly rated
    if your_stocks:
        your_symbols = set(r['symbol'] for r in your_stocks)
    else:
        your_symbols = set()
    
    recommendations = []
    
    # Top mean reversion candidates not in portfolio
    for r in mr_ideal[:30]:
        if r['symbol'] not in your_symbols and r.get('mean_reversion_ideal_score', 0) >= 60:
            recommendations.append({
                'symbol': r['symbol'],
                'category': r.get('category', 'N/A'),
                'best_for': 'Mean Reversion',
                'score': r['mean_reversion_ideal_score']
            })
    
    # Top gap candidates not in portfolio
    for r in gap_ideal[:20]:
        if r['symbol'] not in your_symbols and r.get('gap_reversal_ideal_score', 0) >= 60:
            if not any(rec['symbol'] == r['symbol'] for rec in recommendations):
                recommendations.append({
                    'symbol': r['symbol'],
                    'category': r.get('category', 'N/A'),
                    'best_for': 'Gap Reversal',
                    'score': r['gap_reversal_ideal_score']
                })
    
    print("\nStocks to consider adding:")
    for rec in sorted(recommendations, key=lambda x: -x['score'])[:15]:
        print(f"   {rec['symbol']:<8} ({rec['category']}) - Best for: {rec['best_for']} (Score: {rec['score']})")
    
    # Save full results to CSV
    results_file = Path('./watchlist_training_results/ideal_tickers_scan.csv')
    results_file.parent.mkdir(exist_ok=True)
    
    df_rows = []
    for r in valid_results:
        df_rows.append({
            'symbol': r['symbol'],
            'category': r.get('category', 'N/A'),
            'volatility': r.get('volatility_annual', 0),
            'ma_crosses_per_year': r.get('ma_crosses_per_year', 0),
            'efficiency_ratio': r.get('efficiency_ratio', 0),
            'pct_time_extreme': r.get('pct_time_extreme', 0),
            'gaps_per_year': r.get('gap_down_freq', 0) + r.get('gap_up_freq', 0),
            'return_3m': r.get('return_3m', 0),
            'dollar_volume': r.get('dollar_volume', 0),
            'beta': r.get('beta', None),
        })
    
    df = pd.DataFrame(df_rows)
    df.to_csv(results_file, index=False)
    print(f"\n✅ Full results saved to {results_file}")


if __name__ == "__main__":
    main()
