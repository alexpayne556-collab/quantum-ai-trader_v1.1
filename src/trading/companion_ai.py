"""
🤖 COMPANION AI - Your Trading Guardian

Monitors positions and warns you:
"Hey, dump it while you're up - it will drop because of XYZ"

Features:
- Signal decay detection (30-min half-life)
- Regime shift monitoring
- Volume analysis
- Round number psychology
- Exit recommendations
- Real-time alerts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Warning:
    """Warning signal from Companion AI"""
    severity: str  # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    message: str
    reason: str
    timestamp: datetime
    recommended_action: str


@dataclass
class Position:
    """Active trading position"""
    ticker: str
    entry_price: float
    entry_time: datetime
    current_price: float
    shares: int
    signal_confidence: float
    target_profit: float
    stop_loss: float
    cluster_id: int


class CompanionAI:
    """
    AI companion that monitors your positions and warns about risks
    
    Philosophy: "Hey, dump it while you're up"
    """
    
    def __init__(
        self,
        signal_decay_halflife_minutes: int = 30,
        regime_shift_threshold: float = 0.7,
        volume_decline_threshold: float = 0.5,
        profit_take_threshold: float = 0.08,  # 8% (near 10% target)
        warning_cooldown_minutes: int = 5
    ):
        self.signal_decay_halflife = signal_decay_halflife_minutes
        self.regime_threshold = regime_shift_threshold
        self.volume_threshold = volume_decline_threshold
        self.profit_threshold = profit_take_threshold
        self.warning_cooldown = warning_cooldown_minutes
        
        # Track warnings to avoid spam
        self.last_warning_time = {}
        
        # Market regime tracking
        self.current_regime = 'NORMAL'
        self.regime_history = []
        
        logger.info("🤖 Companion AI initialized")
        logger.info(f"   Signal decay: {self.signal_decay_halflife}min half-life")
        logger.info(f"   Regime threshold: {self.regime_threshold}")
        logger.info(f"   Profit take: {self.profit_threshold:.1%}")
    
    def monitor_position(
        self,
        position: Position,
        current_data: pd.DataFrame
    ) -> List[Warning]:
        """
        Monitor a position for warning signals
        
        Returns list of warnings (if any)
        """
        warnings = []
        
        # Calculate position metrics
        profit_pct = (position.current_price - position.entry_price) / position.entry_price
        signal_age_minutes = (datetime.now() - position.entry_time).total_seconds() / 60
        
        # 1. SIGNAL DECAY CHECK
        decay_warning = self._check_signal_decay(position, signal_age_minutes)
        if decay_warning:
            warnings.append(decay_warning)
        
        # 2. PROFIT TARGET CHECK
        profit_warning = self._check_profit_target(position, profit_pct)
        if profit_warning:
            warnings.append(profit_warning)
        
        # 3. REGIME SHIFT CHECK
        regime_warning = self._check_regime_shift(position.ticker, current_data)
        if regime_warning:
            warnings.append(regime_warning)
        
        # 4. VOLUME DECLINE CHECK
        volume_warning = self._check_volume_decline(current_data)
        if volume_warning:
            warnings.append(volume_warning)
        
        # 5. ROUND NUMBER CHECK
        round_warning = self._check_round_number_resistance(position.current_price)
        if round_warning:
            warnings.append(round_warning)
        
        # 6. STOP LOSS PROXIMITY CHECK
        stop_warning = self._check_stop_loss_proximity(position, profit_pct)
        if stop_warning:
            warnings.append(stop_warning)
        
        return warnings
    
    def _check_signal_decay(
        self,
        position: Position,
        signal_age_minutes: float
    ) -> Optional[Warning]:
        """Check if signal has decayed (30-min half-life)"""
        
        if signal_age_minutes <= self.signal_decay_halflife:
            return None
        
        # Calculate decay factor
        half_lives = signal_age_minutes / self.signal_decay_halflife
        decay_factor = 0.5 ** half_lives
        effective_confidence = position.signal_confidence * decay_factor
        
        if effective_confidence < 0.5:  # Below 50%
            return Warning(
                severity='HIGH',
                message=f"⚠️ {position.ticker} signal has decayed significantly",
                reason=f"Signal age: {signal_age_minutes:.0f}min (effective confidence: {effective_confidence:.1%})",
                timestamp=datetime.now(),
                recommended_action='CONSIDER_EXIT'
            )
        elif effective_confidence < 0.6:  # Below 60%
            return Warning(
                severity='MEDIUM',
                message=f"⚠️ {position.ticker} signal aging",
                reason=f"Signal age: {signal_age_minutes:.0f}min (effective confidence: {effective_confidence:.1%})",
                timestamp=datetime.now(),
                recommended_action='WATCH_CLOSELY'
            )
        
        return None
    
    def _check_profit_target(
        self,
        position: Position,
        profit_pct: float
    ) -> Optional[Warning]:
        """Check if near profit target - recommend taking profit"""
        
        target_pct = (position.target_profit - position.entry_price) / position.entry_price
        
        # Near target (80% of way there)
        if profit_pct >= target_pct * 0.8 and profit_pct > 0:
            return Warning(
                severity='HIGH',
                message=f"✅ {position.ticker} near profit target (+{profit_pct:.1%})",
                reason=f"Target: +{target_pct:.1%}, Current: +{profit_pct:.1%}",
                timestamp=datetime.now(),
                recommended_action='TAKE_PROFIT'
            )
        
        # Good profit (>5%)
        elif profit_pct >= 0.05:
            return Warning(
                severity='MEDIUM',
                message=f"✅ {position.ticker} in profit (+{profit_pct:.1%})",
                reason="Consider scaling out or tightening stop",
                timestamp=datetime.now(),
                recommended_action='CONSIDER_PARTIAL_EXIT'
            )
        
        return None
    
    def _check_regime_shift(
        self,
        ticker: str,
        current_data: pd.DataFrame
    ) -> Optional[Warning]:
        """
        Detect regime shifts using adversarial validation
        
        If current market looks different from training data,
        model predictions become unreliable
        """
        
        # Simple regime detection: volatility regime change
        if len(current_data) < 20:
            return None
        
        recent_vol = current_data['close'].pct_change().tail(5).std()
        historical_vol = current_data['close'].pct_change().std()
        
        vol_ratio = recent_vol / historical_vol
        
        # Volatility explosion (regime shift)
        if vol_ratio > 2.0:
            return Warning(
                severity='CRITICAL',
                message=f"🚨 {ticker} REGIME SHIFT DETECTED",
                reason=f"Volatility spiked {vol_ratio:.1f}x (historical: {historical_vol:.2%}, recent: {recent_vol:.2%})",
                timestamp=datetime.now(),
                recommended_action='EXIT_IMMEDIATELY'
            )
        
        # Volatility increase
        elif vol_ratio > 1.5:
            return Warning(
                severity='HIGH',
                message=f"⚠️ {ticker} volatility increasing",
                reason=f"Vol ratio: {vol_ratio:.1f}x",
                timestamp=datetime.now(),
                recommended_action='TIGHTEN_STOP'
            )
        
        return None
    
    def _check_volume_decline(
        self,
        current_data: pd.DataFrame
    ) -> Optional[Warning]:
        """Check if volume is declining (momentum fading)"""
        
        if len(current_data) < 10:
            return None
        
        recent_volume = current_data['volume'].tail(3).mean()
        avg_volume = current_data['volume'].tail(20).mean()
        
        volume_ratio = recent_volume / avg_volume
        
        # Volume drying up
        if volume_ratio < self.volume_threshold:
            return Warning(
                severity='MEDIUM',
                message="⚠️ Volume declining",
                reason=f"Recent volume {volume_ratio:.1%} of average (momentum fading)",
                timestamp=datetime.now(),
                recommended_action='WATCH_CLOSELY'
            )
        
        return None
    
    def _check_round_number_resistance(
        self,
        current_price: float
    ) -> Optional[Warning]:
        """Check if near round number (psychological resistance)"""
        
        # Find nearest round numbers
        round_10 = round(current_price / 10) * 10
        round_50 = round(current_price / 50) * 50
        round_100 = round(current_price / 100) * 100
        
        # Distance to nearest significant round number
        dist_10 = abs(current_price - round_10) / current_price
        dist_50 = abs(current_price - round_50) / current_price
        dist_100 = abs(current_price - round_100) / current_price
        
        min_dist = min(dist_10, dist_50, dist_100)
        
        # Within 2% of round number
        if min_dist < 0.02:
            if dist_100 == min_dist:
                level = round_100
                significance = "MAJOR"
            elif dist_50 == min_dist:
                level = round_50
                significance = "SIGNIFICANT"
            else:
                level = round_10
                significance = "MINOR"
            
            return Warning(
                severity='MEDIUM',
                message=f"⚠️ Near {significance} round number (${level:.0f})",
                reason=f"Price: ${current_price:.2f}, Distance: {min_dist:.1%}",
                timestamp=datetime.now(),
                recommended_action='WATCH_FOR_RESISTANCE'
            )
        
        return None
    
    def _check_stop_loss_proximity(
        self,
        position: Position,
        profit_pct: float
    ) -> Optional[Warning]:
        """Check if approaching stop loss"""
        
        stop_pct = (position.stop_loss - position.entry_price) / position.entry_price
        
        # Within 2% of stop loss
        if profit_pct < stop_pct * 1.2 and profit_pct < 0:
            return Warning(
                severity='CRITICAL',
                message=f"🚨 {position.ticker} approaching stop loss",
                reason=f"Current: {profit_pct:.1%}, Stop: {stop_pct:.1%}",
                timestamp=datetime.now(),
                recommended_action='PREPARE_TO_EXIT'
            )
        
        return None
    
    def recommend_exit(
        self,
        position: Position,
        warnings: List[Warning]
    ) -> Dict:
        """
        Analyze warnings and recommend exit strategy
        
        Returns:
            {
                'action': 'HOLD' | 'WATCH' | 'PARTIAL_EXIT' | 'FULL_EXIT' | 'EMERGENCY_EXIT',
                'urgency': 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL',
                'score': int,
                'reasons': List[str],
                'message': str
            }
        """
        
        # Calculate exit score
        score = 0
        reasons = []
        
        for warning in warnings:
            if warning.severity == 'CRITICAL':
                score += 4
            elif warning.severity == 'HIGH':
                score += 3
            elif warning.severity == 'MEDIUM':
                score += 2
            else:
                score += 1
            
            reasons.append(warning.reason)
        
        # Determine action
        if score >= 8:
            action = 'EMERGENCY_EXIT'
            urgency = 'CRITICAL'
            message = f"🚨 DUMP {position.ticker} NOW: {', '.join(reasons)}"
        
        elif score >= 5:
            action = 'FULL_EXIT'
            urgency = 'HIGH'
            message = f"⚠️ EXIT {position.ticker}: {', '.join(reasons)}"
        
        elif score >= 3:
            action = 'PARTIAL_EXIT'
            urgency = 'MEDIUM'
            message = f"📉 Scale out of {position.ticker}: {', '.join(reasons)}"
        
        elif score >= 1:
            action = 'WATCH'
            urgency = 'MEDIUM'
            message = f"👀 Watch {position.ticker}: {', '.join(reasons)}"
        
        else:
            action = 'HOLD'
            urgency = 'LOW'
            message = f"✅ {position.ticker} looking good"
            reasons = ['Position healthy']
        
        return {
            'action': action,
            'urgency': urgency,
            'score': score,
            'reasons': reasons,
            'message': message,
            'warnings': warnings
        }
    
    def send_alert(
        self,
        message: str,
        urgency: str = 'MEDIUM',
        ticker: str = None
    ):
        """
        Send alert to trader
        
        Urgency levels:
        - CRITICAL: SMS + Email + Desktop
        - HIGH: Email + Desktop
        - MEDIUM: Desktop
        - LOW: Log only
        """
        
        # Check cooldown
        if ticker and ticker in self.last_warning_time:
            time_since_last = (datetime.now() - self.last_warning_time[ticker]).total_seconds() / 60
            if time_since_last < self.warning_cooldown:
                logger.debug(f"Skipping alert for {ticker} (cooldown)")
                return
        
        # Send based on urgency
        if urgency == 'CRITICAL':
            logger.critical(message)
            # TODO: Integrate SMS (Twilio)
            # TODO: Integrate Email (SendGrid)
            # TODO: Desktop notification
            print(f"\n{'='*60}")
            print(f"🚨 CRITICAL ALERT: {message}")
            print(f"{'='*60}\n")
        
        elif urgency == 'HIGH':
            logger.error(message)
            # TODO: Email + Desktop
            print(f"\n⚠️  HIGH ALERT: {message}\n")
        
        elif urgency == 'MEDIUM':
            logger.warning(message)
            print(f"⚠️  {message}")
        
        else:
            logger.info(message)
        
        # Update cooldown
        if ticker:
            self.last_warning_time[ticker] = datetime.now()
    
    def monitor_all_positions(
        self,
        positions: List[Position],
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Monitor all active positions
        
        Returns:
            {
                'NVDA': {
                    'warnings': [...],
                    'recommendation': {...}
                },
                ...
            }
        """
        
        results = {}
        
        for position in positions:
            # Get current data
            if position.ticker not in market_data:
                logger.warning(f"No data for {position.ticker}")
                continue
            
            current_data = market_data[position.ticker]
            
            # Monitor position
            warnings = self.monitor_position(position, current_data)
            
            # Get recommendation
            recommendation = self.recommend_exit(position, warnings)
            
            # Send alert if urgent
            if recommendation['urgency'] in ['CRITICAL', 'HIGH']:
                self.send_alert(
                    recommendation['message'],
                    recommendation['urgency'],
                    position.ticker
                )
            
            results[position.ticker] = {
                'warnings': warnings,
                'recommendation': recommendation,
                'position': position
            }
        
        return results


# Example usage
if __name__ == '__main__':
    # Setup
    companion = CompanionAI(
        signal_decay_halflife_minutes=30,
        profit_take_threshold=0.08
    )
    
    # Mock position
    position = Position(
        ticker='NVDA',
        entry_price=475.00,
        entry_time=datetime.now() - timedelta(minutes=45),
        current_price=490.50,
        shares=100,
        signal_confidence=0.85,
        target_profit=522.50,  # +10%
        stop_loss=451.25,      # -5%
        cluster_id=0
    )
    
    # Mock data
    mock_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 480,
        'volume': np.random.randint(1000000, 5000000, 100)
    })
    
    # Monitor
    warnings = companion.monitor_position(position, mock_data)
    recommendation = companion.recommend_exit(position, warnings)
    
    print(f"\n{'='*60}")
    print(f"Monitoring {position.ticker}")
    print(f"{'='*60}")
    print(f"Entry: ${position.entry_price:.2f}")
    print(f"Current: ${position.current_price:.2f}")
    print(f"Profit: {((position.current_price - position.entry_price) / position.entry_price):.1%}")
    print(f"\nWarnings: {len(warnings)}")
    for w in warnings:
        print(f"  [{w.severity}] {w.message}")
    print(f"\nRecommendation: {recommendation['action']}")
    print(f"Urgency: {recommendation['urgency']}")
    print(f"Message: {recommendation['message']}")
