
#!/usr/bin/env python3
"""
🔥 DAILY MORNING SCANNER
Run this at 6 AM every trading day
"""

import requests
from datetime import datetime, timedelta
import time

FINNHUB_KEY = 'd3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0'

UNIVERSE = [
    'IONQ', 'RGTI', 'QBTS',  # Quantum
    'LEU', 'OKLO', 'UUUU', 'SMR',  # Nuclear
    'TLRY', 'ACB', 'SNDL', 'CGC',  # Cannabis
    'WULF', 'MARA', 'RIOT',  # BTC Miners
    'NVDA', 'SMCI', 'AMD', 'AVGO',  # AI/Semis
    'HOOD', 'COIN',  # Fintech
    'TSLA',  # EV
]

def get_news_velocity(ticker, days=5):
    end = datetime.now()
    start = end - timedelta(days=days)
    url = f'https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start.strftime("%Y-%m-%d")}&to={end.strftime("%Y-%m-%d")}&token={FINNHUB_KEY}'

    try:
        resp = requests.get(url, timeout=10)
        articles = resp.json()

        daily_counts = {}
        for a in articles:
            date = datetime.fromtimestamp(a['datetime']).strftime('%Y-%m-%d')
            daily_counts[date] = daily_counts.get(date, 0) + 1

        sorted_dates = sorted(daily_counts.keys(), reverse=True)
        recent = sum(daily_counts.get(d, 0) for d in sorted_dates[:3]) if len(sorted_dates) >= 3 else 0
        older = sum(daily_counts.get(d, 0) for d in sorted_dates[3:7]) if len(sorted_dates) >= 4 else 1

        return recent / max(older, 1)
    except:
        return 0

def get_finviz_data(ticker):
    headers = {'User-Agent': 'Mozilla/5.0'}
    url = f'https://finviz.com/quote.ashx?t={ticker}'

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text

        data = {}
        for field in ['RSI (14)', 'Change', 'Short Float']:
            idx = html.find(f'>{field}</td>')
            if idx > 0:
                start = html.find('<b>', idx) + 3
                end = html.find('</b>', start)
                value = html[start:end].replace('<span class="color-text is-positive">', '').replace('<span class="color-text is-negative">', '').replace('</span>', '')
                data[field] = value
        return data
    except:
        return {}

if __name__ == '__main__':
    print("=" * 60)
    print(f"🌅 MORNING SCAN - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    alerts = []

    for ticker in UNIVERSE:
        velocity = get_news_velocity(ticker)
        data = get_finviz_data(ticker)

        signals = []

        if velocity > 2:
            signals.append(f"News {velocity:.1f}x")

        try:
            rsi = float(data.get('RSI (14)', '50'))
            if rsi < 40:
                signals.append(f"RSI {rsi:.0f}")
        except:
            pass

        if signals:
            alerts.append((ticker, signals))

        time.sleep(0.3)

    print("\n🚨 ALERTS:")
    for ticker, signals in sorted(alerts, key=lambda x: len(x[1]), reverse=True):
        icon = '🔥🔥' if len(signals) >= 2 else '⚠️'
        print(f"{icon} {ticker}: {' + '.join(signals)}")
