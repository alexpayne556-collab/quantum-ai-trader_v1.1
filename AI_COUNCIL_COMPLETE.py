"""
AI COUNCIL + OUR INNOVATIONS - COMPLETE PYTHON CODE
====================================================

Copy these cells into Jupyter Lab and test EVERYTHING.
We're the architects. We choose what works.

Sections:
1. Setup & Imports
2. Perplexity's Solutions
3. DeepSeek's Solutions  
4. Claude's Solutions
5. Our Original 6 Modules
6. Test Cases
7. Hybrid System

"""

# ============================================
# CELL 1: SETUP & IMPORTS
# ============================================

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Check GPU (if available)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU DETECTED: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️ No GPU - Running on CPU (fine for testing)")
except:
    print("⚠️ PyTorch not installed - No GPU acceleration")

print(f"\n✅ All imports successful")
print(f"📅 Today: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# ============================================
# CELL 2: PERPLEXITY'S INSTITUTIONAL DETECTOR
# ============================================

class InstitutionalVsRetailDetector:
    """
    PERPLEXITY'S COMPLETE IMPLEMENTATION
    
    Distinguishes institutional accumulation from retail chasing based on:
    - Block trade analysis (50K+ share trades)
    - VWAP relationships (institutions buy BELOW, retail buys ABOVE)
    - Volume consistency (institutions = steady, retail = spiky)
    - Order type inference (limit vs market)
    """
    
    def __init__(self):
        self.vwap_lookback = 60  # minutes
        self.block_threshold = 50000  # shares
    
    def analyze_volume_signature(
        self, 
        time_and_sales: List[dict], 
        vwap: float, 
        time_window: Tuple[datetime, datetime]
    ) -> Tuple[str, float, dict]:
        """
        Main analysis function.
        
        Returns: (trader_type, confidence, evidence)
        """
        
        window_trades = [
            t for t in time_and_sales
            if time_window[0] <= t['time'] <= time_window[1]
        ]
        
        if not window_trades:
            return 'UNKNOWN', 0, {}
        
        # Score 4 components
        blocks = [t for t in window_trades if t['shares'] >= self.block_threshold]
        
        block_score = self._score_blocks(blocks, vwap)
        vwap_score = self._score_vwap_relationship(window_trades, vwap)
        volume_score = self._score_volume_pattern(window_trades)
        order_score = self._score_order_types(window_trades)
        
        # Weighted composite
        institutional_confidence = (
            0.40 * block_score +
            0.30 * vwap_score +
            0.20 * volume_score +
            0.10 * order_score
        )
        
        evidence = {
            'total_blocks': len(blocks),
            'blocks_below_vwap': sum(1 for b in blocks if b['price'] < vwap),
            'total_volume': sum(t['shares'] for t in window_trades),
            'scores': {
                'block': round(block_score, 2),
                'vwap': round(vwap_score, 2),
                'volume_pattern': round(volume_score, 2),
                'order_type': round(order_score, 2)
            }
        }
        
        if institutional_confidence > 0.70:
            return 'INSTITUTIONAL', institutional_confidence, evidence
        elif institutional_confidence < 0.30:
            return 'RETAIL', 1 - institutional_confidence, evidence
        else:
            return 'MIXED', institutional_confidence, evidence
    
    def _score_blocks(self, blocks: List[dict], vwap: float) -> float:
        """Institutions use block trades at/below VWAP"""
        if len(blocks) < 2:
            return 0.2
        
        blocks_at_vwap = sum(
            1 for b in blocks 
            if abs(b['price'] - vwap) / vwap < 0.005
        )
        blocks_below_vwap = sum(1 for b in blocks if b['price'] < vwap)
        
        vwap_adherence = (blocks_at_vwap + blocks_below_vwap * 0.5) / len(blocks)
        block_consistency = min(len(blocks) / 5, 1.0)
        
        return vwap_adherence * 0.7 + block_consistency * 0.3
    
    def _score_vwap_relationship(self, trades: List[dict], vwap: float) -> float:
        """% of volume below VWAP (institutions = 45-60%, retail = <30%)"""
        total_volume = sum(t['shares'] for t in trades)
        if total_volume == 0:
            return 0.5
        
        volume_below_vwap = sum(
            t['shares'] for t in trades if t['price'] < vwap
        )
        pct_below = volume_below_vwap / total_volume
        
        if pct_below > 0.60:
            return 1.0
        elif pct_below > 0.45:
            return 0.75
        elif pct_below > 0.30:
            return 0.40
        else:
            return 0.1
    
    def _score_volume_pattern(self, trades: List[dict]) -> float:
        """Coefficient of variation (low = institutional, high = retail)"""
        minute_volumes = {}
        for t in trades:
            minute = t['time'].replace(second=0, microsecond=0)
            minute_volumes[minute] = minute_volumes.get(minute, 0) + t['shares']
        
        volumes = list(minute_volumes.values())
        if len(volumes) < 2:
            return 0.5
        
        mean_vol = statistics.mean(volumes)
        std_vol = statistics.stdev(volumes)
        cv = std_vol / mean_vol if mean_vol > 0 else 1.0
        
        if cv < 0.5:
            return 1.0
        elif cv < 1.0:
            return 0.6
        else:
            return 0.2
    
    def _score_order_types(self, trades: List[dict]) -> float:
        """Price clustering = limit orders = institutional"""
        price_clusters = {}
        for t in trades:
            price = round(t['price'], 2)
            price_clusters[price] = price_clusters.get(price, 0) + t['shares']
        
        if len(price_clusters) == 0:
            return 0.5
        
        sorted_prices = sorted(price_clusters.items(), key=lambda x: x[1], reverse=True)
        top_3_volume = sum(p[1] for p in sorted_prices[:3])
        total_volume = sum(t['shares'] for t in trades)
        
        pct_top_3 = top_3_volume / total_volume if total_volume > 0 else 0
        
        if pct_top_3 > 0.70:
            return 0.85
        elif pct_top_3 > 0.50:
            return 0.60
        else:
            return 0.20

print("✅ Perplexity's InstitutionalVsRetailDetector loaded")


# ============================================  
# CELL 3: DEEPSEEK'S DIP SYSTEM
# ============================================

def is_buyable_dip_v2_deepseek(
    stock_data: dict,
    spy_data: dict,
    sector_data: dict,
    news_data: dict,
    vix_data: dict,
    current_time: datetime
) -> Tuple[bool, str]:
    """
    DEEPSEEK'S COMPLETE 5-FILTER SYSTEM
    
    Filters:
    1. Sector relative strength
    2. Volume pattern (LOW volume pullback after HIGH volume up-day)
    3. News sentiment (blacklist check)
    4. VIX containment
    5. Time of day (10:30 AM - 2:30 PM ET)
    
    Returns: (is_buyable, reason)
    """
    
    # BASE CONDITIONS
    base_ok = (
        stock_data['close'] > stock_data['ma_200'] and
        spy_data['close'] > spy_data['ma_50'] and
        -8.0 <= stock_data['pct_change'] <= -3.5
    )
    
    if not base_ok:
        return False, \"Base conditions not met (need -3.5% to -8% pullback, above 200MA)\"
    
    failures = []
    
    # FILTER 1: Sector holding up better than stock
    sector_ok = sector_data['pct_change'] > stock_data['pct_change']
    if not sector_ok:
        failures.append(\"sector_weakness\")
    
    # FILTER 2: Volume pattern (THE MOST CRITICAL)
    volume_ok = (
        stock_data['volume'] < (stock_data['avg_volume_20'] * 0.85) and
        stock_data['volume_prev'] > (stock_data['avg_volume_20'] * 1.2) and
        stock_data['close_prev'] > stock_data['open_prev']
    )
    if not volume_ok:
        failures.append(\"bad_volume_pattern\")
    
    # FILTER 3: News sentiment
    blacklist = ['fraud', 'investigation', 'sec', 'downgrade', 'cut', 'miss', 'warning', 'sues']
    headline = news_data.get('headline', '').lower()
    body = news_data.get('body', '').lower()
    news_ok = not any(kw in headline or kw in body for kw in blacklist)
    if not news_ok:
        failures.append(\"bad_news\")
    
    # FILTER 4: VIX containment
    vix_ok = (
        vix_data['current'] < 22 and
        vix_data['current'] < (vix_data['ma_20'] * 1.15)
    )
    if not vix_ok:
        failures.append(\"vix_spike\")
    
    # FILTER 5: Time of day
    hour_min = current_time.hour + current_time.minute / 60
    time_ok = 10.5 <= hour_min <= 14.5
    if not time_ok:
        failures.append(\"bad_entry_time\")
    
    # DECISION
    all_ok = sector_ok and volume_ok and news_ok and vix_ok and time_ok
    
    if all_ok:
        return True, \"✅ All 5 filters passed - BUYABLE DIP\"
    else:
        return False, f\"❌ Failed: {', '.join(failures)}\"

print(\"✅ DeepSeek's is_buyable_dip_v2_deepseek loaded\")


# ============================================
# CELL 4: DEEPSEEK'S EXPECTED VALUE CALCULATOR
# ============================================

def calculate_expected_value_and_position(
    historical_trades: pd.DataFrame,
    capital: float,
    max_kelly_fraction: float = 0.25
) -> Tuple[float, float, str]:
    \"\"\"
    DEEPSEEK'S KELLY CRITERION POSITION SIZER
    
    Uses Half-Kelly for conservative sizing.
    
    historical_trades must have 'gain' and 'loss' columns.
    \"\"\"
    
    if len(historical_trades) < 10:
        return 0, 0, \"⚠️ Insufficient data: need ≥10 trades\"
    
    wins = historical_trades[historical_trades['gain'] > 0]
    losses = historical_trades[historical_trades['loss'] < 0]
    
    # Win rate
    W = len(wins) / len(historical_trades)
    
    # Median win/loss (robust to outliers)
    avg_win = wins['gain'].median() if len(wins) > 0 else 0
    avg_loss = abs(losses['loss'].median()) if len(losses) > 0 else 1
    
    if avg_loss == 0:
        return 0, 0, \"⚠️ Average loss is zero\"
    
    # Win/Loss ratio
    R = avg_win / avg_loss
    
    # Kelly Criterion: K = W - (1-W)/R
    K = W - ((1 - W) / R)
    
    # Half-Kelly for safety
    conservative_K = K / 2
    position_fraction = min(conservative_K, max_kelly_fraction)
    position_fraction = max(position_fraction, 0)
    
    position_size = capital * position_fraction
    expected_value = (W * avg_win) - ((1 - W) * avg_loss)
    
    explanation = (
        f\"Win Rate: {W:.1%} | Avg W/L: {avg_win:.2f}/{avg_loss:.2f} | \"
        f\"Kelly: {K:.1%} | Using: {position_fraction:.1%}\"
    )
    
    return expected_value, position_size, explanation

print(\"✅ DeepSeek's calculate_expected_value_and_position loaded\")


# ============================================
# CELL 5: DEEPSEEK'S PDT SIMULATION
# ============================================

def simulate_day_trade_usage(
    num_simulations: int = 10000,
    positions_per_week: int = 3,
    weeks: int = 52,
    stop_hit_rate: float = 0.25,
    emergency_rate: float = 0.05
) -> dict:
    \"\"\"
    DEEPSEEK'S PDT CONSTRAINT SIMULATION
    
    Monte Carlo simulation: How often do we run out of day trades?
    \"\"\"
    
    results = []
    
    for _ in range(num_simulations):
        day_trades_available = 3
        stuck_positions = 0
        
        for week in range(weeks):
            # Replenish every 5 trading days
            day_trades_available = 3
            
            # Simulate trades
            for pos in range(positions_per_week):
                needs_day_trade = (
                    np.random.rand() < emergency_rate or 
                    np.random.rand() < stop_hit_rate
                )
                
                if needs_day_trade:
                    if day_trades_available > 0:
                        day_trades_available -= 1
                    else:
                        stuck_positions += 1
        
        results.append(stuck_positions)
    
    avg_stuck = np.mean(results)
    pct_with_stuck = np.mean(np.array(results) > 0) * 100
    
    print(f\"\\n=== PDT SIMULATION ({num_simulations:,} runs) ===\")
    print(f\"Avg stuck positions per year: {avg_stuck:.2f}\")
    print(f\"% of years with ≥1 stuck position: {pct_with_stuck:.1f}%\")
    print(f\"\\nWith {stop_hit_rate:.0%} stops + {emergency_rate:.0%} emergencies:\")
    
    if pct_with_stuck > 30:
        print(f\"⚠️ PDT constraint is a REAL PROBLEM\")
    elif pct_with_stuck > 10:
        print(f\"⚠️ PDT constraint is manageable but risky\")
    else:
        print(f\"✅ PDT constraint is minor\")
    
    return {
        'avg_stuck': avg_stuck,
        'pct_with_stuck': pct_with_stuck,
        'results': results
    }

print(\"✅ DeepSeek's simulate_day_trade_usage loaded\")

