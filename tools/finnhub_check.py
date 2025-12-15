import time
import os
import requests

SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "TSLA",
] * 5  # 30 symbols

API_KEY = os.environ.get("FINNHUB_KEY", "YOUR_FINNHUB_KEY")

def main():
    if not API_KEY or API_KEY == "YOUR_FINNHUB_KEY":
        print("Warning: set FINNHUB_KEY in environment or .env")
    t0 = time.time()
    ok = 0
    for sym in SYMBOLS:
        try:
            r = requests.get(
                f"https://finnhub.io/api/v1/company-news",
                params={"symbol": sym, "from": "2025-12-01", "to": "2025-12-14", "token": API_KEY},
                timeout=10,
            )
            if r.status_code == 200:
                ok += 1
        except Exception:
            pass
    t1 = time.time()
    print(f"Finnhub 30 stocks total time: {t1 - t0:.2f}s; successes: {ok}/30")

if __name__ == "__main__":
    main()
