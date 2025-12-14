#!/usr/bin/env python3
"""
🔮 QUANTUM VISUAL RECOMMENDER
=============================
- 21-day forward projections
- Interactive charts with indicators
- Self-correcting warnings when predictions go wrong
- Full logging for learning and improvement
- Daily recommendations with confidence scoring
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create logs directory
LOG_DIR = Path('prediction_logs')
LOG_DIR.mkdir(exist_ok=True)

TICKERS = [
    'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'BAC', 'QCOM', 'UUUU',
    'TSLA', 'AMD', 'NOW', 'NVDA', 'MU', 'PG', 'DLB', 'XME',
    'KRYS', 'LEU', 'QTUM', 'SPY', 'UNH', 'WMT', 'OKLO', 'RXRX',
    'MTZ', 'SNOW', 'GRRR', 'BSX', 'LLY', 'VOO', 'GEO', 'CXW',
    'LYFT', 'MNDY', 'BA', 'LAC', 'INTC', 'ALK', 'LMT', 'CRDO',
    'ANET', 'META', 'RIVN', 'GOOGL', 'HL', 'TEM', 'TDOC', 'KMTS'
]


class VisualRecommender:
    """
    Full visual analysis with 21-day projections and self-correction
    """
    
    def __init__(self):
        self.predictions_file = LOG_DIR / 'all_predictions.json'
        self.corrections_file = LOG_DIR / 'corrections_log.json'
        self.load_history()
    
    def load_history(self):
        """Load past predictions for self-correction analysis"""
        if self.predictions_file.exists():
            with open(self.predictions_file, 'r') as f:
                self.prediction_history = json.load(f)
        else:
            self.prediction_history = []
        
        if self.corrections_file.exists():
            with open(self.corrections_file, 'r') as f:
                self.corrections = json.load(f)
        else:
            self.corrections = []
    
    def save_prediction(self, prediction):
        """Save prediction for future validation"""
        self.prediction_history.append(prediction)
        with open(self.predictions_file, 'w') as f:
            json.dump(self.prediction_history[-500:], f, indent=2, default=str)  # Keep last 500
    
    def check_past_predictions(self, ticker, df):
        """Check if past predictions were accurate - SELF CORRECTION"""
        warnings_list = []
        
        current_price = float(df['Close'].iloc[-1])
        
        # Find predictions made for this ticker
        for pred in self.prediction_history[-50:]:
            if pred.get('ticker') != ticker:
                continue
            
            pred_date = datetime.fromisoformat(pred['date'])
            days_ago = (datetime.now() - pred_date).days
            
            if days_ago < 1 or days_ago > 21:
                continue
            
            pred_price = pred.get('price_at_prediction', 0)
            pred_signal = pred.get('signal', '')
            pred_target = pred.get('target_price', 0)
            
            if pred_price == 0:
                continue
            
            actual_change = (current_price - pred_price) / pred_price * 100
            
            # Check if prediction was WRONG
            if 'BUY' in pred_signal and actual_change < -5:
                warnings_list.append({
                    'type': 'BAD_BUY',
                    'days_ago': days_ago,
                    'predicted': f"BUY at ${pred_price:.2f}",
                    'actual': f"Down {actual_change:.1f}%",
                    'lesson': "Entry was too early or trend reversed"
                })
            elif 'SELL' in pred_signal and actual_change > 5:
                warnings_list.append({
                    'type': 'MISSED_GAIN',
                    'days_ago': days_ago,
                    'predicted': f"SELL at ${pred_price:.2f}",
                    'actual': f"Up {actual_change:.1f}%",
                    'lesson': "Sold too early, trend continued"
                })
            elif 'BUY' in pred_signal and actual_change > 5:
                warnings_list.append({
                    'type': 'GOOD_CALL',
                    'days_ago': days_ago,
                    'predicted': f"BUY at ${pred_price:.2f}",
                    'actual': f"Up {actual_change:.1f}%",
                    'lesson': "Good prediction!"
                })
        
        return warnings_list
    
    def calculate_indicators(self, df):
        """Calculate all technical indicators for display"""
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        indicators = {}
        
        # EMAs
        indicators['EMA_9'] = close.ewm(span=9).mean()
        indicators['EMA_21'] = close.ewm(span=21).mean()
        indicators['EMA_50'] = close.ewm(span=50).mean()
        indicators['EMA_200'] = close.ewm(span=200).mean()
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        indicators['BB_Upper'] = sma20 + 2 * std20
        indicators['BB_Lower'] = sma20 - 2 * std20
        indicators['BB_Mid'] = sma20
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-8)
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        indicators['MACD'] = ema12 - ema26
        indicators['MACD_Signal'] = indicators['MACD'].ewm(span=9).mean()
        indicators['MACD_Hist'] = indicators['MACD'] - indicators['MACD_Signal']
        
        # Volume SMA
        indicators['Vol_SMA'] = volume.rolling(20).mean()
        
        # ATR
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        indicators['ATR'] = tr.rolling(14).mean()
        
        # Support/Resistance (20-day)
        indicators['Support'] = low.rolling(20).min()
        indicators['Resistance'] = high.rolling(20).max()
        
        return indicators
    
    def project_21_days(self, df, indicators):
        """
        Project price 21 days forward using multiple methods
        Returns: dict with projections and confidence
        """
        close = df['Close'].values
        current_price = float(close[-1])
        atr = float(indicators['ATR'].iloc[-1])
        
        # Method 1: Trend continuation (EMA slope)
        ema_21 = indicators['EMA_21'].values
        ema_slope = (ema_21[-1] - ema_21[-5]) / 5  # Daily slope
        trend_projection = current_price + (ema_slope * 21)
        
        # Method 2: Mean reversion to 50 EMA
        ema_50 = float(indicators['EMA_50'].iloc[-1])
        mean_rev_projection = current_price + (ema_50 - current_price) * 0.5
        
        # Method 3: ATR-based range
        atr_bull = current_price + atr * 3  # 3 ATR up
        atr_bear = current_price - atr * 2  # 2 ATR down (asymmetric for bullish bias)
        
        # Method 4: Historical volatility projection
        returns = pd.Series(close).pct_change().dropna()
        daily_vol = returns.std()
        vol_range = current_price * daily_vol * np.sqrt(21)  # 21-day vol
        
        # Combine projections
        rsi = float(indicators['RSI'].iloc[-1])
        macd_hist = float(indicators['MACD_Hist'].iloc[-1])
        
        # Weight based on indicators
        if rsi < 40 and macd_hist > 0:  # Oversold + MACD turning
            weight_bull = 0.7
        elif rsi > 70 and macd_hist < 0:  # Overbought + MACD turning
            weight_bull = 0.3
        elif macd_hist > 0:
            weight_bull = 0.6
        else:
            weight_bull = 0.4
        
        # Final projections
        bull_target = max(trend_projection, atr_bull) * weight_bull + mean_rev_projection * (1 - weight_bull)
        bear_target = min(trend_projection, atr_bear) * (1 - weight_bull) + mean_rev_projection * weight_bull
        
        # Most likely target
        if weight_bull > 0.5:
            likely_target = current_price + (bull_target - current_price) * weight_bull
        else:
            likely_target = current_price + (bear_target - current_price) * (1 - weight_bull)
        
        # Confidence based on trend alignment
        ema_aligned = indicators['EMA_9'].iloc[-1] > indicators['EMA_21'].iloc[-1] > indicators['EMA_50'].iloc[-1]
        confidence = 0.5
        if ema_aligned and macd_hist > 0:
            confidence = 0.75
        elif not ema_aligned and macd_hist < 0:
            confidence = 0.35
        
        return {
            'current': current_price,
            'bull_target': float(bull_target),
            'bear_target': float(bear_target),
            'likely_target': float(likely_target),
            'confidence': confidence,
            'expected_return': (likely_target - current_price) / current_price * 100,
            'upside': (bull_target - current_price) / current_price * 100,
            'downside': (bear_target - current_price) / current_price * 100,
            'vol_range': float(vol_range),
            'days': 21
        }
    
    def generate_signal(self, df, indicators, projection):
        """Generate trading signal based on all data"""
        rsi = float(indicators['RSI'].iloc[-1])
        macd_hist = float(indicators['MACD_Hist'].iloc[-1])
        current = projection['current']
        bb_upper = float(indicators['BB_Upper'].iloc[-1])
        bb_lower = float(indicators['BB_Lower'].iloc[-1])
        
        score = 0
        reasons = []
        
        # RSI
        if rsi < 30:
            score += 3
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi < 45:
            score += 1
            reasons.append(f"RSI low ({rsi:.0f})")
        elif rsi > 70:
            score -= 2
            reasons.append(f"RSI overbought ({rsi:.0f})")
        
        # MACD
        if macd_hist > 0:
            score += 2
            reasons.append("MACD bullish")
        else:
            score -= 1
            reasons.append("MACD bearish")
        
        # Bollinger Bands
        if current < bb_lower:
            score += 2
            reasons.append("Below BB lower")
        elif current > bb_upper:
            score -= 1
            reasons.append("Above BB upper")
        
        # 21-day projection
        if projection['expected_return'] > 5:
            score += 2
            reasons.append(f"21d target +{projection['expected_return']:.1f}%")
        elif projection['expected_return'] < -3:
            score -= 2
            reasons.append(f"21d target {projection['expected_return']:.1f}%")
        
        # Trend
        if projection['confidence'] > 0.6:
            score += 1
            reasons.append("Strong trend alignment")
        
        # Determine signal
        if score >= 5:
            signal = "STRONG BUY"
        elif score >= 3:
            signal = "BUY"
        elif score <= -3:
            signal = "SELL"
        elif score <= -1:
            signal = "WEAK SELL"
        else:
            signal = "HOLD"
        
        return signal, score, reasons
    
    def plot_chart(self, ticker, df, indicators, projection, signal, warnings, save_path=None):
        """Create comprehensive chart with all indicators and 21-day projection"""
        
        # Use last 90 days for display
        display_days = 90
        df_display = df.tail(display_days).copy()
        
        fig = plt.figure(figsize=(16, 12))
        
        # ====== PRICE CHART (Main) ======
        ax1 = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
        
        # Candlestick-style (simplified with close line)
        dates = df_display.index
        ax1.fill_between(dates, df_display['Low'], df_display['High'], alpha=0.3, color='gray')
        ax1.plot(dates, df_display['Close'], 'k-', linewidth=1.5, label='Close')
        
        # EMAs
        for ema, color, label in [('EMA_9', 'blue', 'EMA 9'), 
                                   ('EMA_21', 'orange', 'EMA 21'),
                                   ('EMA_50', 'purple', 'EMA 50')]:
            if ema in indicators:
                ax1.plot(dates, indicators[ema].tail(display_days), color=color, 
                        linestyle='--', alpha=0.7, label=label)
        
        # Bollinger Bands
        ax1.fill_between(dates, indicators['BB_Lower'].tail(display_days), 
                        indicators['BB_Upper'].tail(display_days), 
                        alpha=0.1, color='blue', label='BB')
        
        # Support/Resistance
        ax1.axhline(float(indicators['Support'].iloc[-1]), color='green', 
                   linestyle=':', alpha=0.5, label='Support')
        ax1.axhline(float(indicators['Resistance'].iloc[-1]), color='red', 
                   linestyle=':', alpha=0.5, label='Resistance')
        
        # ====== 21-DAY PROJECTION ======
        last_date = dates[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=21)
        
        current = projection['current']
        bull = projection['bull_target']
        bear = projection['bear_target']
        likely = projection['likely_target']
        
        # Projection cone
        for i, d in enumerate(future_dates):
            progress = (i + 1) / 21
            bull_pt = current + (bull - current) * progress
            bear_pt = current + (bear - current) * progress
            likely_pt = current + (likely - current) * progress
            
            if i == len(future_dates) - 1:
                ax1.scatter(d, bull_pt, color='green', s=100, marker='^', zorder=5)
                ax1.scatter(d, bear_pt, color='red', s=100, marker='v', zorder=5)
                ax1.scatter(d, likely_pt, color='blue', s=150, marker='*', zorder=5)
                ax1.annotate(f'${bull_pt:.2f} (+{(bull_pt-current)/current*100:.1f}%)', 
                           (d, bull_pt), textcoords="offset points", xytext=(10,5), fontsize=9, color='green')
                ax1.annotate(f'${bear_pt:.2f} ({(bear_pt-current)/current*100:.1f}%)', 
                           (d, bear_pt), textcoords="offset points", xytext=(10,-10), fontsize=9, color='red')
                ax1.annotate(f'LIKELY: ${likely_pt:.2f}', 
                           (d, likely_pt), textcoords="offset points", xytext=(10,0), fontsize=10, color='blue', fontweight='bold')
        
        # Projection lines
        ax1.plot([last_date, future_dates[-1]], [current, bull], 'g--', alpha=0.5, linewidth=2)
        ax1.plot([last_date, future_dates[-1]], [current, bear], 'r--', alpha=0.5, linewidth=2)
        ax1.plot([last_date, future_dates[-1]], [current, likely], 'b-', alpha=0.7, linewidth=2)
        
        # Fill projection cone
        ax1.fill_between([last_date, future_dates[-1]], 
                        [current, bear], [current, bull], alpha=0.1, color='yellow')
        
        ax1.axvline(last_date, color='black', linestyle='-', alpha=0.3, label='Today')
        
        # Signal annotation
        signal_color = 'green' if 'BUY' in signal else 'red' if 'SELL' in signal else 'gray'
        ax1.set_title(f'{ticker} - {signal} | 21-Day Projection | Conf: {projection["confidence"]*100:.0f}%', 
                     fontsize=14, fontweight='bold', color=signal_color)
        ax1.legend(loc='upper left', fontsize=8)
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        
        # ====== RSI ======
        ax2 = plt.subplot2grid((4, 1), (2, 0))
        ax2.plot(dates, indicators['RSI'].tail(display_days), 'purple', linewidth=1.5)
        ax2.axhline(70, color='red', linestyle='--', alpha=0.5)
        ax2.axhline(30, color='green', linestyle='--', alpha=0.5)
        ax2.fill_between(dates, 30, 70, alpha=0.1, color='gray')
        ax2.set_ylabel('RSI')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        # ====== MACD ======
        ax3 = plt.subplot2grid((4, 1), (3, 0))
        macd = indicators['MACD'].tail(display_days)
        signal_line = indicators['MACD_Signal'].tail(display_days)
        hist = indicators['MACD_Hist'].tail(display_days)
        
        ax3.plot(dates, macd, 'b-', label='MACD', linewidth=1)
        ax3.plot(dates, signal_line, 'r-', label='Signal', linewidth=1)
        colors = ['green' if h >= 0 else 'red' for h in hist]
        ax3.bar(dates, hist, color=colors, alpha=0.5, width=0.8)
        ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
        ax3.set_ylabel('MACD')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # ====== WARNINGS BOX ======
        if warnings:
            warning_text = "⚠️ SELF-CORRECTION WARNINGS:\n"
            for w in warnings[:3]:
                warning_text += f"• {w['type']}: {w['actual']} ({w['days_ago']}d ago)\n"
                warning_text += f"  Lesson: {w['lesson']}\n"
            
            fig.text(0.02, 0.02, warning_text, fontsize=9, 
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                    verticalalignment='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Chart saved: {save_path}")
        
        plt.close()
        return fig
    
    def analyze_ticker(self, ticker, save_chart=True):
        """Full analysis for a single ticker"""
        try:
            # Download data
            df = yf.download(ticker, period='1y', progress=False)
            if len(df) < 100:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Calculate indicators
            indicators = self.calculate_indicators(df)
            
            # Project 21 days
            projection = self.project_21_days(df, indicators)
            
            # Generate signal
            signal, score, reasons = self.generate_signal(df, indicators, projection)
            
            # Check past predictions for self-correction
            warnings = self.check_past_predictions(ticker, df)
            
            # Create result
            result = {
                'ticker': ticker,
                'date': datetime.now().isoformat(),
                'price_at_prediction': projection['current'],
                'signal': signal,
                'score': score,
                'reasons': reasons,
                'projection_21d': projection,
                'rsi': float(indicators['RSI'].iloc[-1]),
                'macd_hist': float(indicators['MACD_Hist'].iloc[-1]),
                'warnings': warnings,
                'target_price': projection['likely_target']
            }
            
            # Save prediction for future validation
            self.save_prediction(result)
            
            # Generate chart
            if save_chart:
                chart_path = LOG_DIR / f"{ticker}_analysis.png"
                self.plot_chart(ticker, df, indicators, projection, signal, warnings, chart_path)
            
            return result
            
        except Exception as e:
            print(f"❌ Error analyzing {ticker}: {e}")
            return None
    
    def run_full_analysis(self, tickers=None, top_n=10):
        """Analyze all tickers and show top recommendations"""
        if tickers is None:
            tickers = TICKERS
        
        print("=" * 70)
        print("🔮 QUANTUM VISUAL RECOMMENDER - 21-DAY OUTLOOK")
        print("=" * 70)
        print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Analyzing {len(tickers)} tickers...")
        print("=" * 70)
        
        all_results = []
        
        for i, ticker in enumerate(tickers):
            print(f"\r   Processing {ticker} ({i+1}/{len(tickers)})...", end="", flush=True)
            result = self.analyze_ticker(ticker)
            if result:
                all_results.append(result)
        
        print("\n")
        
        # Sort by score
        all_results.sort(key=lambda x: -x['score'])
        
        # ====== TOP BUYS ======
        print("🟢 TOP BUY RECOMMENDATIONS:")
        print("-" * 70)
        buys = [r for r in all_results if 'BUY' in r['signal']]
        for r in buys[:top_n]:
            proj = r['projection_21d']
            warn_flag = "⚠️" if any(w['type'] == 'BAD_BUY' for w in r['warnings']) else ""
            print(f"   {r['ticker']:5} | {r['signal']:12} | Score: {r['score']:+3} | "
                  f"21d: {proj['expected_return']:+5.1f}% | RSI: {r['rsi']:5.1f} {warn_flag}")
            print(f"          Current: ${proj['current']:.2f} → Target: ${proj['likely_target']:.2f}")
            print(f"          Range: ${proj['bear_target']:.2f} - ${proj['bull_target']:.2f}")
            print(f"          Why: {', '.join(r['reasons'][:3])}")
            print()
        
        # ====== SELLS ======
        sells = [r for r in all_results if 'SELL' in r['signal']]
        if sells:
            print("\n🔴 SELL/AVOID:")
            print("-" * 70)
            for r in sells[:5]:
                proj = r['projection_21d']
                print(f"   {r['ticker']:5} | {r['signal']:12} | 21d: {proj['expected_return']:+5.1f}%")
        
        # ====== SAVE SUMMARY ======
        summary = {
            'date': datetime.now().isoformat(),
            'total_analyzed': len(all_results),
            'buys': len(buys),
            'sells': len(sells),
            'top_picks': [{'ticker': r['ticker'], 'signal': r['signal'], 
                          'target': r['projection_21d']['likely_target'],
                          'expected_return': r['projection_21d']['expected_return']} 
                         for r in buys[:10]],
            'all_results': all_results
        }
        
        summary_path = LOG_DIR / f"daily_summary_{datetime.now().strftime('%Y%m%d')}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\n✅ Summary saved: {summary_path}")
        print(f"📊 Charts saved in: {LOG_DIR}/")
        
        return summary


def main():
    """Run the visual recommender"""
    recommender = VisualRecommender()
    
    # Run full analysis
    summary = recommender.run_full_analysis(top_n=10)
    
    print("\n" + "=" * 70)
    print("📈 QUICK REFERENCE - TOP 5 FOR TOMORROW:")
    print("=" * 70)
    
    for i, pick in enumerate(summary['top_picks'][:5], 1):
        print(f"   {i}. {pick['ticker']:5} → ${pick['target']:.2f} ({pick['expected_return']:+.1f}%)")
    
    print("\n💡 Charts with full indicators saved to prediction_logs/")
    print("🔄 Run daily to track prediction accuracy and self-correct!")


if __name__ == '__main__':
    main()
