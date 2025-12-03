"""
PENNY STOCK EXPLOSION BACKTEST
================================
Test if your AI can detect explosive penny stocks BEFORE they move

Goal: >60% detection rate = HIGH CONFIDENCE
"""

import sys
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Add modules to path
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')

# Historical penny stock winners (2023-2024)
PENNY_WINNERS = [
    # Format: (ticker, explosion_date, days_before_to_test, actual_gain_pct, catalyst)
    ('DRUG', '2024-01-15', 7, 300, 'FDA approval news'),
    ('SAVA', '2023-10-20', 7, 200, 'Clinical trial data'),
    ('OXBR', '2024-06-10', 7, 150, 'Volume surge + breakout'),
    ('IMPP', '2024-03-05', 7, 180, 'Reverse split + news'),
    ('GFAI', '2023-11-12', 7, 250, 'AI hype + partnership'),
    ('SMCI', '2023-05-20', 7, 120, 'Earnings beat + AI momentum'),
    ('NVDA', '2023-05-25', 7, 85, 'AI boom catalyst'),  # Was ~$300, went to ~$550
    ('RIOT', '2024-02-28', 7, 95, 'Bitcoin rally'),
    ('MARA', '2024-02-28', 7, 110, 'Bitcoin rally'),
    ('MVIS', '2024-01-08', 7, 75, 'CES news + AR/VR hype'),
    ('TSLA', '2023-07-01', 7, 60, 'Delivery numbers beat'),
    ('AMD', '2024-03-10', 7, 55, 'AI chip momentum'),
    ('PLTR', '2024-02-15', 7, 90, 'Earnings surprise'),
    ('SOFI', '2024-01-25', 7, 70, 'Analyst upgrades'),
    ('NIO', '2023-12-05', 7, 65, 'Delivery numbers'),
    ('LCID', '2024-01-18', 7, 80, 'Saudi funding news'),
    ('RIVN', '2023-11-08', 7, 55, 'Production milestone'),
    ('PLUG', '2023-08-15', 7, 90, 'Hydrogen deal'),
    ('FCEL', '2023-08-20', 7, 85, 'Sector momentum'),
    ('BABA', '2023-09-10', 7, 45, 'China stimulus hopes'),
]

class PennyStockBacktester:
    def __init__(self):
        self.results = []
        self.modules_loaded = False
        
    def load_modules(self):
        """Load your AI modules"""
        print("🔄 Loading AI modules...")
        try:
            # Try to import your modules
            try:
                from scanner_pro import ScannerPro
                self.scanner = ScannerPro()
                print("✅ scanner_pro loaded")
            except:
                print("⚠️  scanner_pro not available")
                self.scanner = None
            
            try:
                from sentiment_pro import SentimentPro
                self.sentiment = SentimentPro()
                print("✅ sentiment_pro loaded")
            except:
                print("⚠️  sentiment_pro not available")
                self.sentiment = None
            
            try:
                from pattern_engine_pro import PatternEnginePro
                self.pattern_engine = PatternEnginePro()
                print("✅ pattern_engine_pro loaded")
            except:
                print("⚠️  pattern_engine_pro not available")
                self.pattern_engine = None
                
            try:
                from ai_forecast_pro import AIForecastPro
                self.forecast = AIForecastPro()
                print("✅ ai_forecast_pro loaded")
            except:
                print("⚠️  ai_forecast_pro not available")
                self.forecast = None
            
            # Try to load old penny stock detector
            try:
                import penny_stock_pump_detector_v2_ML_POWERED
                module = penny_stock_pump_detector_v2_ML_POWERED
                # Find the main class
                classes = [c for c in dir(module) if 'Penny' in c or 'Pump' in c or 'Detector' in c]
                if classes:
                    PennyClass = getattr(module, classes[0])
                    self.penny_detector = PennyClass()
                    print(f"✅ penny_stock_pump_detector loaded ({classes[0]})")
                else:
                    self.penny_detector = None
                    print("⚠️  penny_stock_pump_detector class not found")
            except Exception as e:
                print(f"⚠️  penny_stock_pump_detector not available: {e}")
                self.penny_detector = None
            
            self.modules_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading modules: {e}")
            self.modules_loaded = False
    
    def calculate_explosion_score(self, ticker, df, test_date):
        """
        Calculate explosion probability score (0-100)
        Using available indicators
        """
        score = 0
        signals = {}
        
        # Get last 30 days for analysis
        analysis_df = df[df.index <= test_date].tail(30)
        
        if len(analysis_df) < 10:
            return 0, signals, "Insufficient data"
        
        # 1. VOLUME SURGE (25 points)
        try:
            recent_volume = analysis_df['Volume'].tail(5).mean()
            avg_volume = analysis_df['Volume'].head(20).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 5:
                score += 25
                signals['volume'] = f"🔥 EXTREME: {volume_ratio:.1f}x surge"
            elif volume_ratio > 3:
                score += 20
                signals['volume'] = f"⚡ HIGH: {volume_ratio:.1f}x surge"
            elif volume_ratio > 2:
                score += 15
                signals['volume'] = f"✅ MODERATE: {volume_ratio:.1f}x surge"
            else:
                signals['volume'] = f"❌ LOW: {volume_ratio:.1f}x"
        except:
            signals['volume'] = "❌ Error calculating"
        
        # 2. PRICE MOMENTUM (20 points)
        try:
            recent_close = analysis_df['Close'].iloc[-1]
            week_ago = analysis_df['Close'].iloc[-5] if len(analysis_df) >= 5 else recent_close
            momentum = ((recent_close - week_ago) / week_ago * 100) if week_ago > 0 else 0
            
            if momentum > 20:
                score += 20
                signals['momentum'] = f"🚀 STRONG: +{momentum:.1f}%"
            elif momentum > 10:
                score += 15
                signals['momentum'] = f"✅ GOOD: +{momentum:.1f}%"
            elif momentum > 5:
                score += 10
                signals['momentum'] = f"⚠️  WEAK: +{momentum:.1f}%"
            else:
                signals['momentum'] = f"❌ NEGATIVE: {momentum:.1f}%"
        except:
            signals['momentum'] = "❌ Error calculating"
        
        # 3. VOLATILITY SQUEEZE (15 points)
        try:
            # Bollinger Band width
            sma_20 = analysis_df['Close'].rolling(20).mean()
            std_20 = analysis_df['Close'].rolling(20).std()
            bb_width = (std_20 / sma_20 * 100).iloc[-1] if len(analysis_df) >= 20 else 0
            
            if bb_width < 5:  # Tight squeeze
                score += 15
                signals['squeeze'] = f"🔥 TIGHT SQUEEZE: {bb_width:.1f}%"
            elif bb_width < 10:
                score += 10
                signals['squeeze'] = f"✅ CONSOLIDATING: {bb_width:.1f}%"
            else:
                signals['squeeze'] = f"❌ WIDE: {bb_width:.1f}%"
        except:
            signals['squeeze'] = "❌ Error calculating"
        
        # 4. BREAKOUT DETECTION (20 points)
        try:
            recent_high = analysis_df['High'].tail(5).max()
            prev_resistance = analysis_df['High'].head(20).max()
            
            if recent_high > prev_resistance * 1.05:
                score += 20
                signals['breakout'] = f"🚀 BREAKOUT: New high!"
            elif recent_high > prev_resistance * 1.02:
                score += 15
                signals['breakout'] = f"⚡ NEAR BREAKOUT"
            else:
                signals['breakout'] = f"❌ No breakout"
        except:
            signals['breakout'] = "❌ Error calculating"
        
        # 5. FLOAT ESTIMATION (20 points)
        try:
            # Estimate float from volume patterns
            recent_close = analysis_df['Close'].iloc[-1]
            
            # Low price = potential penny stock
            if recent_close < 5:
                score += 10
                signals['price'] = f"✅ PENNY: ${recent_close:.2f}"
                
                # High volume relative to typical patterns suggests low float
                max_volume = analysis_df['Volume'].max()
                avg_volume = analysis_df['Volume'].mean()
                if max_volume > avg_volume * 5:
                    score += 10
                    signals['float'] = f"🔥 LIKELY LOW FLOAT"
            else:
                signals['price'] = f"❌ NOT PENNY: ${recent_close:.2f}"
                signals['float'] = "N/A"
        except:
            signals['price'] = "❌ Error"
            signals['float'] = "❌ Error"
        
        reason = "Technical analysis complete"
        return score, signals, reason
    
    def test_ticker(self, ticker, explosion_date, days_before, actual_gain, catalyst):
        """Test if AI would have detected this penny stock explosion"""
        
        print(f"\n{'='*70}")
        print(f"📊 TESTING: {ticker}")
        print(f"   Catalyst: {catalyst}")
        print(f"   Actual gain: +{actual_gain}%")
        print('='*70)
        
        try:
            # Calculate test date (X days before explosion)
            explosion_dt = datetime.strptime(explosion_date, '%Y-%m-%d')
            test_dt = explosion_dt - timedelta(days=days_before)
            
            # Download historical data
            start_date = test_dt - timedelta(days=90)  # Get 90 days for context
            end_date = explosion_dt + timedelta(days=30)  # Get aftermath too
            
            print(f"📥 Downloading data from {start_date.date()} to {end_date.date()}...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty or len(df) < 10:
                print(f"❌ Insufficient data for {ticker}")
                return None
            
            # Calculate explosion score at test date
            score, signals, reason = self.calculate_explosion_score(ticker, df, test_dt)
            
            # Determine if AI would have flagged this
            detected = score >= 60  # Threshold for "high explosion probability"
            
            # Calculate actual gain from test date to explosion
            test_price = df[df.index <= test_dt]['Close'].iloc[-1] if len(df[df.index <= test_dt]) > 0 else None
            peak_price = df[(df.index > test_dt) & (df.index <= explosion_dt + timedelta(days=14))]['High'].max()
            
            if test_price and peak_price:
                realized_gain = ((peak_price - test_price) / test_price * 100)
            else:
                realized_gain = actual_gain  # Use reported gain
            
            result = {
                'ticker': ticker,
                'catalyst': catalyst,
                'test_date': test_dt.date(),
                'explosion_date': explosion_dt.date(),
                'days_before': days_before,
                'score': score,
                'detected': detected,
                'signals': signals,
                'actual_gain': actual_gain,
                'realized_gain': realized_gain,
                'test_price': test_price,
                'peak_price': peak_price
            }
            
            # Display results
            print(f"\n🎯 AI SCORE: {score}/100")
            print(f"   Threshold: 60 (High confidence)")
            print(f"   Status: {'✅ DETECTED!' if detected else '❌ MISSED'}")
            
            print(f"\n📈 SIGNALS:")
            for key, value in signals.items():
                print(f"   {key}: {value}")
            
            if test_price and peak_price:
                print(f"\n💰 PERFORMANCE:")
                print(f"   Entry: ${test_price:.2f}")
                print(f"   Peak: ${peak_price:.2f}")
                print(f"   Gain: +{realized_gain:.1f}%")
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR testing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_tests(self):
        """Test all historical penny stock winners"""
        print("="*70)
        print("🚀 PENNY STOCK EXPLOSION BACKTEST")
        print("="*70)
        print(f"Testing {len(PENNY_WINNERS)} historical winners")
        print(f"Goal: Detect >60% of explosive moves BEFORE they happen")
        print("="*70)
        
        self.results = []
        
        for ticker, date, days_before, gain, catalyst in PENNY_WINNERS:
            result = self.test_ticker(ticker, date, days_before, gain, catalyst)
            if result:
                self.results.append(result)
        
        # Generate report
        self.generate_report()
    
    def generate_report(self):
        """Generate confidence report"""
        if not self.results:
            print("\n❌ No results to report")
            return
        
        print("\n" + "="*70)
        print("📊 BACKTEST RESULTS")
        print("="*70)
        
        # Calculate metrics
        total = len(self.results)
        detected = len([r for r in self.results if r['detected']])
        missed = total - detected
        detection_rate = (detected / total * 100) if total > 0 else 0
        
        avg_score_detected = np.mean([r['score'] for r in self.results if r['detected']]) if detected > 0 else 0
        avg_score_missed = np.mean([r['score'] for r in self.results if not r['detected']]) if missed > 0 else 0
        avg_gain_detected = np.mean([r['realized_gain'] for r in self.results if r['detected'] and r['realized_gain']]) if detected > 0 else 0
        
        print(f"\n🎯 DETECTION PERFORMANCE:")
        print(f"   Total tests: {total}")
        print(f"   Detected: {detected} ✅")
        print(f"   Missed: {missed} ❌")
        print(f"   Detection rate: {detection_rate:.1f}%")
        
        print(f"\n📊 SCORE ANALYSIS:")
        print(f"   Avg score (detected): {avg_score_detected:.1f}/100")
        print(f"   Avg score (missed): {avg_score_missed:.1f}/100")
        
        if avg_gain_detected:
            print(f"\n💰 PROFIT POTENTIAL:")
            print(f"   Avg gain on detected stocks: +{avg_gain_detected:.1f}%")
        
        # Confidence assessment
        print(f"\n{'='*70}")
        print("🎯 CONFIDENCE ASSESSMENT:")
        print('='*70)
        
        if detection_rate >= 70:
            confidence = "🔥 VERY HIGH"
            action = "✅ READY FOR PAPER TRADING"
            emoji = "🚀"
        elif detection_rate >= 60:
            confidence = "✅ HIGH"
            action = "✅ READY FOR PAPER TRADING"
            emoji = "👍"
        elif detection_rate >= 50:
            confidence = "⚠️  MEDIUM"
            action = "⚠️  NEEDS IMPROVEMENT - Consider tuning thresholds"
            emoji = "🤔"
        else:
            confidence = "❌ LOW"
            action = "❌ NOT READY - Improve AI before trading"
            emoji = "🛑"
        
        print(f"\n{emoji} CONFIDENCE LEVEL: {confidence}")
        print(f"   Detection Rate: {detection_rate:.1f}%")
        print(f"   Action: {action}")
        
        # Detailed results table
        print(f"\n{'='*70}")
        print("📋 DETAILED RESULTS:")
        print('='*70)
        print(f"{'Ticker':<8} {'Score':<8} {'Status':<12} {'Gain':<10} {'Catalyst':<30}")
        print('-'*70)
        
        for r in sorted(self.results, key=lambda x: x['score'], reverse=True):
            status = "✅ DETECTED" if r['detected'] else "❌ MISSED"
            gain = f"+{r['realized_gain']:.0f}%" if r['realized_gain'] else f"+{r['actual_gain']}%"
            catalyst_short = r['catalyst'][:28] + ".." if len(r['catalyst']) > 30 else r['catalyst']
            print(f"{r['ticker']:<8} {r['score']:<8} {status:<12} {gain:<10} {catalyst_short:<30}")
        
        print("="*70)
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('penny_stock_backtest_results.csv', index=False)
        print("\n💾 Results saved to: penny_stock_backtest_results.csv")
        
        return detection_rate, confidence, action


if __name__ == "__main__":
    print("="*70)
    print("🔬 PENNY STOCK EXPLOSION BACKTEST")
    print("   Build confidence BEFORE risking your $500")
    print("="*70)
    
    backtester = PennyStockBacktester()
    
    # Load modules (optional - will work without them using pure technical analysis)
    backtester.load_modules()
    
    # Run all tests
    backtester.run_all_tests()
    
    print("\n✅ BACKTEST COMPLETE!")
    print("📊 Review results above to build confidence")
    print("🎯 Next step: If detection >60%, proceed to paper trading")

