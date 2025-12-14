#!/usr/bin/env python3
"""
🔮 QUANTUM ORACLE - UNLEASHED MODE
===================================
- Scans ALL 50 tickers
- Thinks 21 days ahead
- Identifies DIP BUYING opportunities
- Shows full reasoning chain
- Logs everything for analysis
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Your full watchlist
TICKERS = [
    'APLD', 'SERV', 'MRVL', 'HOOD', 'LUNR', 'BAC', 'QCOM', 'UUUU',
    'TSLA', 'AMD', 'NOW', 'NVDA', 'MU', 'PG', 'DLB', 'XME',
    'KRYS', 'LEU', 'QTUM', 'SPY', 'UNH', 'WMT', 'OKLO', 'RXRX',
    'MTZ', 'SNOW', 'GRRR', 'BSX', 'LLY', 'VOO', 'GEO', 'CXW',
    'LYFT', 'MNDY', 'BA', 'LAC', 'INTC', 'ALK', 'LMT', 'CRDO',
    'ANET', 'META', 'RIVN', 'GOOGL', 'HL', 'TEM', 'TDOC', 'KMTS'
]

class UnleashedOracle:
    """
    OFF THE LEASH - Full analysis with multiple time horizons
    """
    
    def __init__(self):
        self.analysis_log = []
    
    def analyze_ticker(self, ticker):
        """Deep analysis with 3-day, 5-day, and 21-day outlooks"""
        try:
            df = yf.download(ticker, period='1y', progress=False)
            if len(df) < 100:
                return None
            
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values
            
            current_price = float(close[-1])
            
            # ============================================
            # MULTI-TIMEFRAME ANALYSIS
            # ============================================
            
            analysis = {
                'ticker': ticker,
                'price': current_price,
                'timestamp': datetime.now().isoformat(),
            }
            
            # --- SHORT TERM (3 days) ---
            ret_3d = (close[-1] / close[-4] - 1) * 100 if len(close) > 3 else 0
            ret_5d = (close[-1] / close[-6] - 1) * 100 if len(close) > 5 else 0
            
            # --- MEDIUM TERM (21 days) ---
            ret_21d = (close[-1] / close[-22] - 1) * 100 if len(close) > 21 else 0
            ret_63d = (close[-1] / close[-64] - 1) * 100 if len(close) > 63 else 0  # 3 months
            
            analysis['returns'] = {
                '3d': round(ret_3d, 2),
                '5d': round(ret_5d, 2),
                '21d': round(ret_21d, 2),
                '3mo': round(ret_63d, 2)
            }
            
            # --- RSI ---
            delta = pd.Series(close).diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-1]
            rsi = 100 - (100 / (1 + gain / (loss + 1e-8)))
            analysis['rsi'] = round(rsi, 1)
            
            # --- MACD ---
            ema12 = pd.Series(close).ewm(span=12).mean()
            ema26 = pd.Series(close).ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            macd_hist = float(macd.iloc[-1] - signal.iloc[-1])
            analysis['macd_hist'] = round(macd_hist, 3)
            analysis['macd_bullish'] = macd_hist > 0
            
            # --- VOLUME ---
            vol_avg = np.mean(volume[-20:])
            vol_ratio = volume[-1] / (vol_avg + 1)
            analysis['volume_ratio'] = round(vol_ratio, 2)
            analysis['volume_surge'] = vol_ratio > 1.5
            
            # --- TREND (EMAs) ---
            ema_20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
            ema_50 = pd.Series(close).ewm(span=50).mean().iloc[-1]
            ema_200 = pd.Series(close).ewm(span=200).mean().iloc[-1]
            
            analysis['above_ema20'] = current_price > ema_20
            analysis['above_ema50'] = current_price > ema_50
            analysis['above_ema200'] = current_price > ema_200
            analysis['ema_trend'] = 'BULLISH' if ema_20 > ema_50 > ema_200 else 'BEARISH' if ema_20 < ema_50 < ema_200 else 'MIXED'
            
            # --- VOLATILITY ---
            atr = np.mean([max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1])) 
                          for i in range(-14, 0)])
            analysis['atr'] = round(atr, 2)
            analysis['atr_pct'] = round(atr / current_price * 100, 2)
            
            # --- BOLLINGER BANDS ---
            sma20 = np.mean(close[-20:])
            std20 = np.std(close[-20:])
            bb_upper = sma20 + 2 * std20
            bb_lower = sma20 - 2 * std20
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower + 1e-8)
            analysis['bb_position'] = round(bb_position, 2)  # 0 = bottom, 1 = top
            
            # --- SUPPORT/RESISTANCE ---
            recent_low = min(low[-20:])
            recent_high = max(high[-20:])
            analysis['near_support'] = (current_price - recent_low) / recent_low < 0.03
            analysis['near_resistance'] = (recent_high - current_price) / current_price < 0.03
            analysis['support'] = round(recent_low, 2)
            analysis['resistance'] = round(recent_high, 2)
            
            # ============================================
            # DIP DETECTION - Is this a buying opportunity?
            # ============================================
            
            dip_score = 0
            dip_reasons = []
            
            # RSI oversold
            if rsi < 35:
                dip_score += 3
                dip_reasons.append(f"RSI oversold ({rsi:.0f})")
            elif rsi < 45:
                dip_score += 1
                dip_reasons.append(f"RSI low ({rsi:.0f})")
            
            # Price near support
            if analysis['near_support']:
                dip_score += 2
                dip_reasons.append(f"Near support ${recent_low:.2f}")
            
            # BB at bottom
            if bb_position < 0.2:
                dip_score += 2
                dip_reasons.append(f"BB bottom ({bb_position:.0%})")
            
            # Recent drop but long-term uptrend
            if ret_5d < -5 and ret_63d > 0:
                dip_score += 3
                dip_reasons.append(f"Short drop ({ret_5d:.1f}%) in uptrend ({ret_63d:.1f}%)")
            
            # Volume spike on down day (capitulation)
            if vol_ratio > 2 and ret_3d < -3:
                dip_score += 2
                dip_reasons.append("Volume capitulation")
            
            analysis['dip_score'] = dip_score
            analysis['dip_reasons'] = dip_reasons
            analysis['is_dip_buy'] = dip_score >= 4
            
            # ============================================
            # 21-DAY OUTLOOK
            # ============================================
            
            outlook_21d = 0
            outlook_reasons = []
            
            # Strong long-term trend
            if analysis['ema_trend'] == 'BULLISH':
                outlook_21d += 3
                outlook_reasons.append("EMA trend bullish")
            elif analysis['ema_trend'] == 'BEARISH':
                outlook_21d -= 2
                outlook_reasons.append("EMA trend bearish")
            
            # MACD momentum
            if macd_hist > 0 and float(macd.iloc[-1]) > float(macd.iloc[-5]):
                outlook_21d += 2
                outlook_reasons.append("MACD accelerating")
            elif macd_hist < 0 and float(macd.iloc[-1]) < float(macd.iloc[-5]):
                outlook_21d -= 2
                outlook_reasons.append("MACD declining")
            
            # RSI momentum
            rsi_5d_ago = 100 - (100 / (1 + delta.where(delta > 0, 0).rolling(14).mean().iloc[-6] / 
                                        ((-delta.where(delta < 0, 0)).rolling(14).mean().iloc[-6] + 1e-8)))
            if rsi > rsi_5d_ago and rsi < 70:
                outlook_21d += 1
                outlook_reasons.append("RSI improving")
            
            # Price above key MAs
            if analysis['above_ema200']:
                outlook_21d += 2
                outlook_reasons.append("Above 200 EMA (bullish)")
            
            # 3-month performance
            if ret_63d > 20:
                outlook_21d += 2
                outlook_reasons.append(f"Strong 3mo ({ret_63d:.0f}%)")
            elif ret_63d < -20:
                outlook_21d -= 1
                outlook_reasons.append(f"Weak 3mo ({ret_63d:.0f}%)")
            
            analysis['outlook_21d_score'] = outlook_21d
            analysis['outlook_21d_reasons'] = outlook_reasons
            
            # ============================================
            # FINAL SIGNALS
            # ============================================
            
            # Short-term signal (3-5 days)
            short_score = 0
            if analysis['macd_bullish']: short_score += 2
            if analysis['above_ema20']: short_score += 1
            if rsi < 50 and rsi > 30: short_score += 1  # Room to run
            if analysis['volume_surge']: short_score += 1
            
            if short_score >= 3:
                analysis['signal_short'] = 'BUY'
            elif short_score <= 0:
                analysis['signal_short'] = 'SELL'
            else:
                analysis['signal_short'] = 'HOLD'
            
            # Long-term signal (21 days)
            if outlook_21d >= 4:
                analysis['signal_long'] = 'STRONG BUY'
            elif outlook_21d >= 2:
                analysis['signal_long'] = 'BUY'
            elif outlook_21d <= -2:
                analysis['signal_long'] = 'SELL'
            else:
                analysis['signal_long'] = 'HOLD'
            
            # DIP BUY override
            if analysis['is_dip_buy'] and analysis['ema_trend'] != 'BEARISH':
                analysis['special_signal'] = '🔥 DIP BUY OPPORTUNITY'
            else:
                analysis['special_signal'] = None
            
            # Confidence (0-100)
            confidence = min(100, 50 + short_score * 10 + (outlook_21d * 5))
            analysis['confidence'] = confidence
            
            return analysis
            
        except Exception as e:
            return {'ticker': ticker, 'error': str(e)}
    
    def run_full_scan(self):
        """Scan all tickers and categorize"""
        print("=" * 70)
        print("🔮 QUANTUM ORACLE - UNLEASHED (ALL 50 TICKERS)")
        print("=" * 70)
        print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("🎯 Multi-timeframe: 3d, 5d, 21d, 3mo")
        print("🔍 Looking for: DIP BUYS, MOMENTUM, LONG-TERM PLAYS")
        print("=" * 70)
        
        all_results = []
        dip_buys = []
        strong_buys = []
        sells = []
        
        for i, ticker in enumerate(TICKERS):
            print(f"\r📊 Analyzing {ticker} ({i+1}/{len(TICKERS)})...", end="", flush=True)
            result = self.analyze_ticker(ticker)
            if result and 'error' not in result:
                all_results.append(result)
                
                if result.get('is_dip_buy'):
                    dip_buys.append(result)
                if result.get('signal_long') in ['STRONG BUY', 'BUY']:
                    strong_buys.append(result)
                if result.get('signal_short') == 'SELL' or result.get('signal_long') == 'SELL':
                    sells.append(result)
        
        print("\n")
        
        # ============================================
        # DISPLAY RESULTS
        # ============================================
        
        # DIP BUYING OPPORTUNITIES
        if dip_buys:
            print("🔥 DIP BUYING OPPORTUNITIES (Buy the dip!):")
            print("-" * 70)
            for r in sorted(dip_buys, key=lambda x: -x['dip_score']):
                print(f"   {r['ticker']:5} | ${r['price']:8.2f} | RSI: {r['rsi']:5.1f} | Dip Score: {r['dip_score']}")
                print(f"          Reasons: {', '.join(r['dip_reasons'][:3])}")
                print()
        
        # STRONG 21-DAY OUTLOOK
        long_plays = [r for r in all_results if r['outlook_21d_score'] >= 4]
        if long_plays:
            print("\n📈 STRONG 21-DAY OUTLOOK:")
            print("-" * 70)
            for r in sorted(long_plays, key=lambda x: -x['outlook_21d_score']):
                print(f"   {r['ticker']:5} | ${r['price']:8.2f} | Score: {r['outlook_21d_score']} | Trend: {r['ema_trend']}")
                print(f"          Returns: 5d={r['returns']['5d']:+.1f}% | 21d={r['returns']['21d']:+.1f}% | 3mo={r['returns']['3mo']:+.1f}%")
                print(f"          Why: {', '.join(r['outlook_21d_reasons'][:3])}")
                print()
        
        # SHORT-TERM MOMENTUM
        momentum = [r for r in all_results if r['signal_short'] == 'BUY' and r['volume_surge']]
        if momentum:
            print("\n⚡ MOMENTUM PLAYS (Short-term with volume):")
            print("-" * 70)
            for r in momentum[:5]:
                print(f"   {r['ticker']:5} | ${r['price']:8.2f} | Vol: {r['volume_ratio']:.1f}x | MACD: {'🟢' if r['macd_bullish'] else '🔴'}")
        
        # SELLS / AVOID
        if sells:
            print("\n🔴 AVOID / SELL:")
            print("-" * 70)
            for r in sells[:5]:
                print(f"   {r['ticker']:5} | ${r['price']:8.2f} | RSI: {r['rsi']:.0f} | Trend: {r['ema_trend']}")
        
        # FULL RANKING
        print("\n" + "=" * 70)
        print("📊 ALL TICKERS RANKED BY 21-DAY OUTLOOK:")
        print("=" * 70)
        
        sorted_results = sorted(all_results, key=lambda x: -x.get('outlook_21d_score', 0))
        for i, r in enumerate(sorted_results):
            signal = r.get('signal_long', 'HOLD')
            emoji = "🔥" if r.get('is_dip_buy') else "🟢" if 'BUY' in signal else "🔴" if signal == 'SELL' else "⚪"
            special = " 💎DIP" if r.get('is_dip_buy') else ""
            print(f"{i+1:2}. {emoji} {r['ticker']:5} | {signal:11} | 21d: {r['outlook_21d_score']:+3} | "
                  f"RSI: {r['rsi']:5.1f} | ${r['price']:8.2f}{special}")
        
        # SAVE RESULTS
        print("\n" + "=" * 70)
        output = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_tickers': len(all_results),
                'dip_buys': len(dip_buys),
                'strong_buys': len(strong_buys),
                'sells': len(sells)
            },
            'dip_opportunities': [{'ticker': r['ticker'], 'price': r['price'], 
                                   'dip_score': r['dip_score'], 'reasons': r['dip_reasons']} 
                                  for r in dip_buys],
            'all_analysis': all_results
        }
        
        with open('oracle_unleashed_results.json', 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print("✅ Full results saved to: oracle_unleashed_results.json")
        
        # INTELLIGENCE CHECK
        print("\n" + "=" * 70)
        print("🧠 ORACLE INTELLIGENCE CHECK:")
        print("=" * 70)
        
        # Does it understand dips?
        if dip_buys:
            print(f"✅ Found {len(dip_buys)} dip buying opportunities")
            print(f"   Best dip: {dip_buys[0]['ticker']} (score: {dip_buys[0]['dip_score']})")
        else:
            print("⚠️ No major dips detected - market may be extended")
        
        # Does it see long-term?
        bullish_21d = len([r for r in all_results if r['outlook_21d_score'] >= 3])
        bearish_21d = len([r for r in all_results if r['outlook_21d_score'] <= -2])
        print(f"✅ 21-day outlook: {bullish_21d} bullish, {bearish_21d} bearish")
        
        # Trend alignment
        aligned = len([r for r in all_results if r['ema_trend'] == 'BULLISH'])
        print(f"✅ {aligned}/{len(all_results)} tickers in full bullish trend (EMA alignment)")
        
        return output


if __name__ == '__main__':
    oracle = UnleashedOracle()
    results = oracle.run_full_scan()
