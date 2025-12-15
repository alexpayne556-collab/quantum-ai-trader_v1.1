import time
import requests

# Example FRED series: Federal Funds Effective Rate
SERIES = ["DFF", "GDP", "CPIAUCSL", "UNRATE", "FEDFUNDS"]

API_KEY = "YOUR_FRED_KEY"

def main():
    t0 = time.time()
    ok = 0
    for series_id in SERIES:
        try:
            r = requests.get(
                "https://api.stlouisfed.org/fred/series/observations",
                params={"series_id": series_id, "api_key": API_KEY, "file_type": "json"},
                timeout=10,
            )
            if r.status_code == 200:
                ok += 1
        except Exception:
            pass
    t1 = time.time()
    print(f"FRED {len(SERIES)} series time: {t1 - t0:.2f}s; successes: {ok}/{len(SERIES)}")

if __name__ == "__main__":
    main()
