import time
import requests
from bs4 import BeautifulSoup

# Simple SEC search (company facts/pages). For sentiment, you’ll later target filings text.
CIKS = ["0000320193", "0000789019", "0001652044", "0001318605"]  # AAPL, MSFT, GOOGL, TSLA

def fetch_company_page(cik):
    url = f"https://www.sec.gov/edgar/browse/?CIK={cik}&owner=exclude"
    headers = {"User-Agent": "ShadowPC-Tester/1.0"}
    r = requests.get(url, headers=headers, timeout=10)
    return r.text if r.status_code == 200 else ""

def main():
    t0 = time.time()
    ok = 0
    for cik in CIKS:
        html = fetch_company_page(cik)
        if html:
            ok += 1
            soup = BeautifulSoup(html, "html.parser")
            title = soup.find("title")
            if title:
                print(f"CIK {cik} page: {title.text[:60]}")
    t1 = time.time()
    print(f"SEC fetch {len(CIKS)} pages time: {t1 - t0:.2f}s; successes: {ok}/{len(CIKS)}")

if __name__ == "__main__":
    main()
