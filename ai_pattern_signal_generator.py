"""
AI Pattern Signal Generator
Uses the battle-tested AI-discovered patterns from pattern_battle_results.json

Patterns:
- AI:quantum_mom: Strong 5d momentum + MACD + bounce (quantum stocks)
- AI:nuclear_dip: 21d return < -5% + MACD rising (nuclear plays)
- AI:trend_cont: All timeframes aligned + bullish ribbon
- H:bounce: Bounce > 5% from 5d low + EMA8 rising
- H:ribbon_mom: Bullish ribbon + MACD rising + RSI 50-70
- H:dip_buy: RSI < 35 AND 5d momentum < -5%

Optimized for real-time signal generation
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')


class AIPatternSignalGenerator:
    """
    Generate signals using AI-discovered patterns from battle testing.
    """

    def __init__(self, tickers: List[str] = None):
        """
        Initialize with tickers to monitor.
        """
        if tickers is None:
            # Default watchlist from battle results
            tickers = [
                'APLD', 'SERV', 'MRVL', 'NVDA', 'AMD', 'MU', 'QCOM', 'CRDO',
                'SMR', 'OKLO', 'LEU', 'UUUU', 'CCJ',
                'HOOD', 'LUNR', 'SNOW', 'NOW',
                'IONQ', 'RGTI', 'QUBT',
                'TSLA', 'META', 'GOOGL',
                'SPY', 'QQQ',
                'BA', 'RIVN', 'LYFT'
            ]

        self.tickers = tickers
        self.signals = {}

        # Load battle results for pattern performance
        try:
            with open('pattern_battle_results.json', 'r') as f:
                self.battle_results = json.load(f)
        except:
            print("⚠️ pattern_battle_results.json not found - using default weights")
            self.battle_results = None

    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all features needed for AI patterns"""

        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']

        # Basic returns
        df['ret_1d'] = close.pct_change(1) * 100
        df['ret_5d'] = close.pct_change(5) * 100
        df['ret_10d'] = close.pct_change(10) * 100
        df['ret_21d'] = close.pct_change(21) * 100

        # EMAs for ribbon
        df['ema_8'] = close.ewm(span=8).mean()
        df['ema_13'] = close.ewm(span=13).mean()
        df['ema_21'] = close.ewm(span=21).mean()
        df['ema_34'] = close.ewm(span=34).mean()
        df['ema_55'] = close.ewm(span=55).mean()

        df['ema_8_rising'] = (df['ema_8'] > df['ema_8'].shift(3)).astype(float)

        # Ribbon
        df['ribbon_bullish'] = ((df['ema_8'] > df['ema_13']) & (df['ema_13'] > df['ema_21'])).astype(float)
        df['ribbon_tight'] = ((df['ema_21'] - df['ema_55']) / df['ema_55'] * 100 < 5).astype(float)

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['macd_rising'] = (df['macd_hist'] > df['macd_hist'].shift(1)).astype(float)

        # Momentum
        df['mom_5d'] = close.pct_change(5) * 100

        # Bounce
        df['low_5d'] = low.rolling(5).min()
        df['bounce'] = (close / (df['low_5d'] + 1e-10) - 1) * 100
        df['bounce_signal'] = ((df['bounce'] > 3) & (df['ema_8_rising'] > 0)).astype(float)

        # Trend alignment
        df['trend_5d'] = np.sign(df['ret_5d'])
        df['trend_10d'] = np.sign(df['ret_10d'])
        df['trend_21d'] = np.sign(df['ret_21d'])
        df['trend_align'] = (df['trend_5d'] + df['trend_10d'] + df['trend_21d']) / 3

        # Clean
        df = df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

        return df

    def check_ai_patterns(self, df: pd.DataFrame, idx: int) -> List[str]:
        """Check AI-discovered patterns"""

        rsi = float(df['rsi_14'].iloc[idx])
        mom_5d = float(df['mom_5d'].iloc[idx])
        ret_21d = float(df['ret_21d'].iloc[idx])
        macd_rising = float(df['macd_rising'].iloc[idx])
        bounce_signal = float(df['bounce_signal'].iloc[idx])
        trend_align = float(df['trend_align'].iloc[idx])
        ribbon_bullish = float(df['ribbon_bullish'].iloc[idx])

        signals = []

        # AI:quantum_mom - Strong 5d momentum + MACD + bounce
        if mom_5d > 10 and macd_rising > 0 and bounce_signal > 0:
            signals.append('AI:quantum_mom')

        # AI:nuclear_dip - 21d return < -5% + MACD rising
        if ret_21d < -5 and macd_rising > 0:
            signals.append('AI:nuclear_dip')

        # AI:trend_cont - All timeframes aligned + bullish ribbon
        if trend_align >= 0.67 and ribbon_bullish > 0 and 45 < rsi < 70:
            signals.append('AI:trend_cont')

        return signals

    def check_human_patterns(self, df: pd.DataFrame, idx: int) -> List[str]:
        """Check human patterns that performed well"""

        rsi = float(df['rsi_14'].iloc[idx])
        mom_5d = float(df['mom_5d'].iloc[idx])
        bounce = float(df['bounce'].iloc[idx])
        ema8_rising = float(df['ema_8_rising'].iloc[idx])
        ribbon_bullish = float(df['ribbon_bullish'].iloc[idx])
        macd_rising = float(df['macd_rising'].iloc[idx])

        signals = []

        # H:bounce - Bounce > 5% from 5d low + EMA8 rising
        if bounce > 5 and ema8_rising > 0:
            signals.append('H:bounce')

        # H:ribbon_mom - Bullish ribbon + MACD rising + RSI 50-70
        if ribbon_bullish > 0 and macd_rising > 0 and 50 < rsi < 70:
            signals.append('H:ribbon_mom')

        # H:dip_buy - RSI < 35 AND 5d momentum < -5%
        if rsi < 35 and mom_5d < -5:
            signals.append('H:dip_buy')

        return signals

    def generate_signals(self, lookback_days: int = 250) -> Dict:
        """
        Generate today's buy signals for all tickers.
        Returns dict with signal details.
        """

        print(f"\n🎯 GENERATING AI PATTERN SIGNALS ({len(self.tickers)} TICKERS)")
        print("=" * 70)

        all_signals = {}

        for ticker in self.tickers:
            try:
                # Get data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=lookback_days)

                df = yf.download(
                    ticker,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=end_date.strftime("%Y-%m-%d"),
                    progress=False,
                    auto_adjust=True
                )

                if df.empty or len(df) < 100:
                    continue

                # Handle columns
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Compute features
                df = self.compute_features(df)

                # Check patterns on latest data
                latest_idx = -1
                ai_signals = self.check_ai_patterns(df, latest_idx)
                human_signals = self.check_human_patterns(df, latest_idx)
                all_patterns = ai_signals + human_signals

                if all_patterns:
                    # Calculate confidence based on pattern performance
                    confidence = self.calculate_confidence(all_patterns)

                    # Get current price and metrics
                    current_price = float(df['Close'].iloc[-1])
                    rsi = float(df['rsi_14'].iloc[-1])
                    mom_5d = float(df['mom_5d'].iloc[-1])
                    ret_21d = float(df['ret_21d'].iloc[-1])

                    signal = {
                        'ticker': ticker,
                        'price': current_price,
                        'patterns': all_patterns,
                        'confidence': confidence,
                        'rsi': rsi,
                        'mom_5d': mom_5d,
                        'ret_21d': ret_21d,
                        'timestamp': datetime.now().isoformat()
                    }

                    all_signals[ticker] = signal
                    print(f"✅ {ticker:6s} ${current_price:>8.2f} | Conf: {confidence:.1f} | Patterns: {', '.join(all_patterns)}")

            except Exception as e:
                print(f"❌ {ticker}: {e}")

        # Sort by confidence
        sorted_signals = dict(sorted(all_signals.items(),
                                   key=lambda x: x[1]['confidence'], reverse=True))

        self.signals = sorted_signals

        print(f"\n🏆 TOP 10 SIGNALS:")
        for i, (ticker, sig) in enumerate(list(sorted_signals.items())[:10], 1):
            stars = '⭐' * min(int(sig['confidence'] / 20), 5)
            print(f"   {i}. {stars} {ticker} (${sig['price']:.2f}) - {', '.join(sig['patterns'])}")

        return sorted_signals

    def calculate_confidence(self, patterns: List[str]) -> float:
        """Calculate confidence score based on pattern performance"""

        if not self.battle_results:
            # Default weights if no battle results
            pattern_weights = {
                'AI:quantum_mom': 80,
                'AI:nuclear_dip': 85,
                'AI:trend_cont': 75,
                'H:bounce': 70,
                'H:ribbon_mom': 65,
                'H:dip_buy': 60
            }
        else:
            # Use actual win rates from battle results
            perf = self.battle_results.get('pattern_performance', {})
            pattern_weights = {}
            for pattern in patterns:
                if pattern in perf:
                    wr = perf[pattern].get('win_rate', 50)
                    pattern_weights[pattern] = wr
                else:
                    pattern_weights[pattern] = 50

        # Average win rate across all triggered patterns
        if patterns:
            avg_win_rate = np.mean([pattern_weights.get(p, 50) for p in patterns])
            # Boost confidence for multiple patterns
            confidence = avg_win_rate * (1 + 0.1 * len(patterns))
            return min(confidence, 100)
        else:
            return 0

    def save_signals(self, filename: str = 'ai_pattern_signals.json'):
        """Save signals to JSON file"""
        if self.signals:
            with open(filename, 'w') as f:
                json.dump(self.signals, f, indent=2, default=str)
            print(f"✅ Signals saved to {filename}")
        else:
            print("❌ No signals to save")


# Quick test
if __name__ == '__main__':
    generator = AIPatternSignalGenerator()
    signals = generator.generate_signals()
    generator.save_signals()