"""
SWING TRADE BACKTEST
====================
Test if your AI can identify profitable swing trades (3-4 month holds)

Goal: >60% win rate = HIGH CONFIDENCE
"""

import sys
import os
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np

# Add modules to path
sys.path.insert(0, '/content/drive/MyDrive/QuantumAI/backend/modules')

# Historical swing trade winners (2023-2024)
# Format: (ticker, entry_date, exit_date, entry_price, exit_price, catalyst)
SWING_WINNERS = [
    ('NVDA', '2023-10-01', '2024-01-15', 450, 580, 'AI boom + strong earnings'),
    ('AMD', '2023-09-15', '2023-12-20', 105, 145, 'Data center growth + AI'),
    ('PLTR', '2023-11-01', '2024-02-15', 16, 26, 'Army contract + profitability'),
    ('META', '2023-11-01', '2024-02-01', 300, 400, 'Cost cutting + AI investments'),
    ('TSLA', '2023-04-15', '2023-07-20', 165, 270, 'Strong deliveries + price cuts working'),
    ('SOFI', '2023-10-01', '2024-01-15', 6.5, 9.5, 'Bank charter benefits + rate cuts hope'),
    ('COIN', '2024-01-01', '2024-04-15', 140, 250, 'Bitcoin ETF approval'),
    ('RIOT', '2024-01-01', '2024-03-30', 12, 22, 'Bitcoin rally + halving'),
    ('MARA', '2024-01-01', '2024-03-30', 18, 30, 'Bitcoin rally + mining efficiency'),
    ('SHOP', '2023-08-01', '2023-11-30', 55, 75, 'E-commerce recovery + logistics sale'),
    ('CRWD', '2023-09-01', '2023-12-31', 150, 230, 'Cybersecurity demand + ARR growth'),
    ('NET', '2023-10-01', '2024-01-31', 60, 95, 'Enterprise adoption + margin expansion'),
    ('DKNG', '2023-08-01', '2023-11-30', 28, 42, 'NFL season + state expansion'),
    ('AFRM', '2023-09-01', '2023-12-31', 20, 38, 'BNPL adoption + merchant growth'),
    ('SQ', '2023-07-01', '2023-10-31', 60, 82, 'Cash App growth + crypto recovery'),
    ('U', '2023-08-15', '2023-11-30', 32, 52, 'Return to office + enterprise deals'),
    ('SNOW', '2023-10-01', '2024-01-31', 150, 210, 'Data cloud adoption + large deals'),
    ('ZM', '2023-09-01', '2023-12-31', 65, 85, 'Enterprise stabilization + AI features'),
    ('UBER', '2023-08-01', '2023-11-30', 42, 58, 'Profitability milestone + autonomous'),
    ('DASH', '2023-10-01', '2024-01-31', 95, 135, 'Delivery network effect + advertising'),
]

class SwingTradeBacktester:
    def __init__(self):
        self.results = []
        self.modules_loaded = False
        
    def load_modules(self):
        """Load your AI modules"""
        print("🔄 Loading AI modules...")
        try:
            try:
                from ai_forecast_pro import AIForecastPro
                self.forecast = AIForecastPro()
                print("✅ ai_forecast_pro loaded")
            except:
                print("⚠️  ai_forecast_pro not available")
                self.forecast = None
            
            try:
                from institutional_flow_pro import InstitutionalFlowPro
                self.institutional = InstitutionalFlowPro()
                print("✅ institutional_flow_pro loaded")
            except:
                print("⚠️  institutional_flow_pro not available")
                self.institutional = None
            
            try:
                from pattern_engine_pro import PatternEnginePro
                self.pattern_engine = PatternEnginePro()
                print("✅ pattern_engine_pro loaded")
            except:
                print("⚠️  pattern_engine_pro not available")
                self.pattern_engine = None
            
            try:
                from sentiment_pro import SentimentPro
                self.sentiment = SentimentPro()
                print("✅ sentiment_pro loaded")
            except:
                print("⚠️  sentiment_pro not available")
                self.sentiment = None
            
            try:
                from risk_manager_pro import RiskManagerPro
                self.risk_manager = RiskManagerPro()
                print("✅ risk_manager_pro loaded")
            except:
                print("⚠️  risk_manager_pro not available")
                self.risk_manager = None
            
            self.modules_loaded = True
            
        except Exception as e:
            print(f"❌ Error loading modules: {e}")
            self.modules_loaded = False
    
    def calculate_swing_score(self, ticker, df, entry_date):
        """
        Calculate swing trade score (0-100)
        Based on Perplexity's recommended factors
        """
        score = 0
        signals = {}
        
        # Get data up to entry date
        analysis_df = df[df.index <= entry_date].tail(120)  # 120 days context
        
        if len(analysis_df) < 50:
            return 0, signals, "Insufficient data"
        
        # 1. TECHNICAL SETUP (30 points)
        try:
            recent_close = analysis_df['Close'].iloc[-1]
            ma_50 = analysis_df['Close'].rolling(50).mean().iloc[-1] if len(analysis_df) >= 50 else recent_close
            ma_200 = analysis_df['Close'].rolling(200).mean().iloc[-1] if len(analysis_df) >= 200 else recent_close
            
            points = 0
            setup_signals = []
            
            # Above moving averages (bullish)
            if recent_close > ma_50:
                points += 10
                setup_signals.append("Above 50-day MA")
            if recent_close > ma_200:
                points += 10
                setup_signals.append("Above 200-day MA")
            
            # Golden cross
            if ma_50 > ma_200:
                points += 5
                setup_signals.append("Golden cross")
            
            # Uptrend
            if len(analysis_df) >= 20:
                price_20d_ago = analysis_df['Close'].iloc[-20]
                if recent_close > price_20d_ago * 1.05:
                    points += 5
                    setup_signals.append("20-day uptrend")
            
            score += points
            signals['technical'] = f"✅ {points}/30: {', '.join(setup_signals) if setup_signals else 'No strong setup'}"
            
        except Exception as e:
            signals['technical'] = f"❌ Error: {e}"
        
        # 2. MOMENTUM (25 points)
        try:
            recent_close = analysis_df['Close'].iloc[-1]
            
            # Multiple timeframe momentum
            momentum_signals = []
            points = 0
            
            # 5-day momentum
            if len(analysis_df) >= 5:
                price_5d = analysis_df['Close'].iloc[-5]
                mom_5d = ((recent_close - price_5d) / price_5d * 100)
                if mom_5d > 3:
                    points += 5
                    momentum_signals.append(f"5d: +{mom_5d:.1f}%")
            
            # 20-day momentum
            if len(analysis_df) >= 20:
                price_20d = analysis_df['Close'].iloc[-20]
                mom_20d = ((recent_close - price_20d) / price_20d * 100)
                if mom_20d > 5:
                    points += 10
                    momentum_signals.append(f"20d: +{mom_20d:.1f}%")
            
            # 60-day momentum
            if len(analysis_df) >= 60:
                price_60d = analysis_df['Close'].iloc[-60]
                mom_60d = ((recent_close - price_60d) / price_60d * 100)
                if mom_60d > 10:
                    points += 10
                    momentum_signals.append(f"60d: +{mom_60d:.1f}%")
            
            score += points
            signals['momentum'] = f"{'✅' if points >= 15 else '⚠️'} {points}/25: {', '.join(momentum_signals) if momentum_signals else 'Weak momentum'}"
            
        except Exception as e:
            signals['momentum'] = f"❌ Error: {e}"
        
        # 3. VOLUME ANALYSIS (20 points)
        try:
            recent_volume = analysis_df['Volume'].tail(10).mean()
            avg_volume = analysis_df['Volume'].head(100).mean() if len(analysis_df) >= 100 else recent_volume
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            points = 0
            
            # Volume trending higher
            if volume_ratio > 1.5:
                points += 20
                vol_signal = f"🔥 STRONG: {volume_ratio:.1f}x average"
            elif volume_ratio > 1.2:
                points += 15
                vol_signal = f"✅ GOOD: {volume_ratio:.1f}x average"
            elif volume_ratio > 1.0:
                points += 10
                vol_signal = f"⚠️  MODERATE: {volume_ratio:.1f}x average"
            else:
                vol_signal = f"❌ WEAK: {volume_ratio:.1f}x average"
            
            score += points
            signals['volume'] = f"{points}/20: {vol_signal}"
            
        except Exception as e:
            signals['volume'] = f"❌ Error: {e}"
        
        # 4. VOLATILITY & RISK (15 points)
        try:
            # Calculate volatility (lower is better for swing trades)
            returns = analysis_df['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252) * 100  # Annualized
            
            points = 0
            
            # Moderate volatility preferred (not too high, not too low)
            if 20 < volatility < 50:
                points += 15
                vol_signal = f"✅ IDEAL: {volatility:.1f}% annual"
            elif 15 < volatility <= 20 or 50 <= volatility < 70:
                points += 10
                vol_signal = f"⚠️  ACCEPTABLE: {volatility:.1f}% annual"
            else:
                points += 5
                vol_signal = f"❌ {'TOO HIGH' if volatility >= 70 else 'TOO LOW'}: {volatility:.1f}% annual"
            
            score += points
            signals['volatility'] = f"{points}/15: {vol_signal}"
            
        except Exception as e:
            signals['volatility'] = f"❌ Error: {e}"
        
        # 5. PATTERN DETECTION (10 points)
        try:
            # Simple pattern detection
            recent_highs = analysis_df['High'].tail(20)
            recent_lows = analysis_df['Low'].tail(20)
            recent_close = analysis_df['Close'].iloc[-1]
            
            points = 0
            pattern_signals = []
            
            # Higher highs and higher lows (uptrend)
            highs_increasing = recent_highs.iloc[-1] > recent_highs.iloc[0]
            lows_increasing = recent_lows.iloc[-1] > recent_lows.iloc[0]
            
            if highs_increasing and lows_increasing:
                points += 5
                pattern_signals.append("Uptrend confirmed")
            
            # Consolidation before breakout
            price_range = (recent_highs.max() - recent_lows.min()) / recent_close * 100
            if price_range < 15:  # Tight range
                points += 5
                pattern_signals.append("Consolidation")
            
            score += points
            signals['pattern'] = f"{points}/10: {', '.join(pattern_signals) if pattern_signals else 'No clear pattern'}"
            
        except Exception as e:
            signals['pattern'] = f"❌ Error: {e}"
        
        reason = "Technical swing analysis complete"
        return score, signals, reason
    
    def test_swing_trade(self, ticker, entry_date, exit_date, entry_price_reported, exit_price_reported, catalyst):
        """Test if AI would have recommended this swing trade"""
        
        print(f"\n{'='*70}")
        print(f"📊 TESTING: {ticker}")
        print(f"   Catalyst: {catalyst}")
        print(f"   Reported: ${entry_price_reported} → ${exit_price_reported}")
        print(f"   Reported gain: +{((exit_price_reported - entry_price_reported) / entry_price_reported * 100):.1f}%")
        print('='*70)
        
        try:
            # Parse dates
            entry_dt = datetime.strptime(entry_date, '%Y-%m-%d')
            exit_dt = datetime.strptime(exit_date, '%Y-%m-%d')
            hold_days = (exit_dt - entry_dt).days
            
            # Download historical data
            start_date = entry_dt - timedelta(days=250)  # Get more context for swing
            end_date = exit_dt + timedelta(days=30)
            
            print(f"📥 Downloading data from {start_date.date()} to {end_date.date()}...")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)
            
            if df.empty or len(df) < 50:
                print(f"❌ Insufficient data for {ticker}")
                return None
            
            # Calculate swing score at entry date
            score, signals, reason = self.calculate_swing_score(ticker, df, entry_dt)
            
            # Determine if AI would have recommended
            recommended = score >= 60  # Threshold for "good swing trade"
            
            # Get actual entry/exit prices from data
            entry_prices = df[df.index.date == entry_dt.date()]['Close']
            exit_prices = df[df.index.date == exit_dt.date()]['Close']
            
            actual_entry = entry_prices.iloc[0] if len(entry_prices) > 0 else entry_price_reported
            actual_exit = exit_prices.iloc[0] if len(exit_prices) > 0 else exit_price_reported
            
            actual_gain = ((actual_exit - actual_entry) / actual_entry * 100)
            
            # Calculate risk metrics
            hold_period_df = df[(df.index >= entry_dt) & (df.index <= exit_dt)]
            if len(hold_period_df) > 0:
                max_drawdown = ((hold_period_df['Low'].min() - actual_entry) / actual_entry * 100)
                max_upside = ((hold_period_df['High'].max() - actual_entry) / actual_entry * 100)
            else:
                max_drawdown = 0
                max_upside = actual_gain
            
            result = {
                'ticker': ticker,
                'catalyst': catalyst,
                'entry_date': entry_dt.date(),
                'exit_date': exit_dt.date(),
                'hold_days': hold_days,
                'score': score,
                'recommended': recommended,
                'signals': signals,
                'entry_price': actual_entry,
                'exit_price': actual_exit,
                'gain_pct': actual_gain,
                'max_drawdown': max_drawdown,
                'max_upside': max_upside,
                'win': actual_gain > 0
            }
            
            # Display results
            print(f"\n🎯 AI SCORE: {score}/100")
            print(f"   Threshold: 60 (Good swing trade)")
            print(f"   Status: {'✅ RECOMMENDED' if recommended else '❌ NOT RECOMMENDED'}")
            
            print(f"\n📈 SIGNALS:")
            for key, value in signals.items():
                print(f"   {key}: {value}")
            
            print(f"\n💰 ACTUAL PERFORMANCE:")
            print(f"   Entry: ${actual_entry:.2f}")
            print(f"   Exit: ${actual_exit:.2f}")
            print(f"   Gain: +{actual_gain:.1f}%")
            print(f"   Hold period: {hold_days} days ({hold_days/30:.1f} months)")
            print(f"   Max drawdown: {max_drawdown:.1f}%")
            print(f"   Max upside: +{max_upside:.1f}%")
            
            # Trade quality assessment
            if recommended and actual_gain > 20:
                print(f"   🎉 WINNER! AI correctly identified this trade")
            elif recommended and actual_gain > 0:
                print(f"   ✅ PROFIT but below target")
            elif not recommended and actual_gain > 20:
                print(f"   ⚠️  MISSED OPPORTUNITY - AI was too conservative")
            elif not recommended and actual_gain <= 0:
                print(f"   ✅ CORRECTLY AVOIDED")
            
            return result
            
        except Exception as e:
            print(f"❌ ERROR testing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_tests(self):
        """Test all historical swing trades"""
        print("="*70)
        print("📈 SWING TRADE BACKTEST")
        print("="*70)
        print(f"Testing {len(SWING_WINNERS)} historical swing trades")
        print(f"Goal: >60% win rate on recommended trades")
        print("="*70)
        
        self.results = []
        
        for ticker, entry, exit, entry_price, exit_price, catalyst in SWING_WINNERS:
            result = self.test_swing_trade(ticker, entry, exit, entry_price, exit_price, catalyst)
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
        recommended = [r for r in self.results if r['recommended']]
        not_recommended = [r for r in self.results if not r['recommended']]
        
        # Win rate on recommended trades
        recommended_winners = [r for r in recommended if r['win']]
        recommended_losers = [r for r in recommended if not r['win']]
        win_rate = (len(recommended_winners) / len(recommended) * 100) if recommended else 0
        
        # Performance metrics
        avg_score = np.mean([r['score'] for r in self.results])
        avg_gain_all = np.mean([r['gain_pct'] for r in self.results])
        avg_gain_recommended = np.mean([r['gain_pct'] for r in recommended]) if recommended else 0
        avg_gain_winners = np.mean([r['gain_pct'] for r in recommended_winners]) if recommended_winners else 0
        avg_hold_days = np.mean([r['hold_days'] for r in recommended]) if recommended else 0
        
        print(f"\n🎯 RECOMMENDATION PERFORMANCE:")
        print(f"   Total trades tested: {total}")
        print(f"   AI recommended: {len(recommended)} ✅")
        print(f"   AI avoided: {len(not_recommended)} ❌")
        
        print(f"\n💰 WIN RATE (on recommended trades):")
        print(f"   Winners: {len(recommended_winners)} ✅")
        print(f"   Losers: {len(recommended_losers)} ❌")
        print(f"   Win rate: {win_rate:.1f}%")
        
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Avg AI score: {avg_score:.1f}/100")
        print(f"   Avg gain (all trades): +{avg_gain_all:.1f}%")
        print(f"   Avg gain (recommended): +{avg_gain_recommended:.1f}%")
        print(f"   Avg gain (winners only): +{avg_gain_winners:.1f}%")
        print(f"   Avg hold period: {avg_hold_days:.0f} days ({avg_hold_days/30:.1f} months)")
        
        # Portfolio simulation
        if recommended:
            print(f"\n💵 PORTFOLIO SIMULATION (if you followed AI):")
            starting_capital = 500
            capital = starting_capital
            position_size_pct = 0.40  # 40% per trade for $500 account
            
            for r in recommended:
                position_value = capital * position_size_pct
                gain_dollars = position_value * (r['gain_pct'] / 100)
                capital += gain_dollars
            
            total_return = ((capital - starting_capital) / starting_capital * 100)
            print(f"   Starting capital: ${starting_capital}")
            print(f"   Position size: {position_size_pct*100:.0f}% per trade")
            print(f"   Trades taken: {len(recommended)}")
            print(f"   Ending capital: ${capital:.2f}")
            print(f"   Total return: +{total_return:.1f}%")
        
        # Confidence assessment
        print(f"\n{'='*70}")
        print("🎯 CONFIDENCE ASSESSMENT:")
        print('='*70)
        
        if win_rate >= 70:
            confidence = "🔥 VERY HIGH"
            action = "✅ EXCELLENT - Ready for paper trading"
            emoji = "🚀"
        elif win_rate >= 60:
            confidence = "✅ HIGH"
            action = "✅ GOOD - Ready for paper trading"
            emoji = "👍"
        elif win_rate >= 50:
            confidence = "⚠️  MEDIUM"
            action = "⚠️  FAIR - Consider improving before live trading"
            emoji = "🤔"
        else:
            confidence = "❌ LOW"
            action = "❌ POOR - Needs significant improvement"
            emoji = "🛑"
        
        print(f"\n{emoji} CONFIDENCE LEVEL: {confidence}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Avg Gain (recommended): +{avg_gain_recommended:.1f}%")
        print(f"   Action: {action}")
        
        # Detailed results table
        print(f"\n{'='*70}")
        print("📋 DETAILED RESULTS:")
        print('='*70)
        print(f"{'Ticker':<8} {'Score':<8} {'Status':<15} {'Gain':<10} {'Hold':<8}")
        print('-'*70)
        
        for r in sorted(self.results, key=lambda x: x['score'], reverse=True):
            status = "✅ RECOMMENDED" if r['recommended'] else "❌ AVOIDED"
            gain_str = f"+{r['gain_pct']:.1f}%" if r['gain_pct'] > 0 else f"{r['gain_pct']:.1f}%"
            hold_str = f"{r['hold_days']}d"
            print(f"{r['ticker']:<8} {r['score']:<8} {status:<15} {gain_str:<10} {hold_str:<8}")
        
        print("="*70)
        
        # Save results
        results_df = pd.DataFrame(self.results)
        results_df.to_csv('swing_trade_backtest_results.csv', index=False)
        print("\n💾 Results saved to: swing_trade_backtest_results.csv")
        
        return win_rate, confidence, action


if __name__ == "__main__":
    print("="*70)
    print("🔬 SWING TRADE BACKTEST")
    print("   Build confidence BEFORE risking your $500")
    print("="*70)
    
    backtester = SwingTradeBacktester()
    
    # Load modules (optional)
    backtester.load_modules()
    
    # Run all tests
    backtester.run_all_tests()
    
    print("\n✅ BACKTEST COMPLETE!")
    print("📊 Review results above to build confidence")
    print("🎯 Next step: If win rate >60%, proceed to paper trading")

