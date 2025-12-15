import os
import time
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import pathlib

# Load symbols from a file if provided, else default sample set
def load_symbols():
    fp = os.getenv("DIGEST_TICKERS_FILE", "alpha_76_watchlist.txt")
    p = pathlib.Path(fp)
    if p.exists():
        lines = [x.strip() for x in p.read_text().splitlines() if x.strip() and not x.startswith("#")]
        return lines[:50] if lines else ["AAPL"]
    return ["AAPL","MSFT","GOOGL","NVDA","META","TSLA"]*5  # 30

SYMBOLS = load_symbols()
FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "YOUR_FINNHUB_KEY")
FRED_KEY = os.environ.get("FRED_KEY", "YOUR_FRED_KEY")

def collect_finnhub_news(symbols):
    t0=time.time(); items=0
    for s in symbols:
        try:
            r = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": s, "from": "2025-12-01", "to": "2025-12-14", "token": FINNHUB_KEY},
                timeout=10,
            )
            if r.status_code==200:
                data=r.json()
                items += len(data)
        except Exception:
            pass
    t1=time.time(); return {"time": t1-t0, "count": items}

def sentiment_gpu(texts, model_name="ProsusAI/finbert"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else None,
        low_cpu_mem_usage=True
    ).to(device)
    batch = tok(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    torch.backends.cudnn.benchmark = device.type=="cuda"
    with torch.no_grad():
        for _ in range(3): _ = mdl(**batch).logits
        t0=time.time(); logits = mdl(**batch).logits; t1=time.time()
    scores = logits.softmax(-1).tolist()
    return {"time": t1-t0, "scores": scores, "cuda": torch.cuda.is_available()}

def score_narrative_divergence(sent_scores):
    # Placeholder scoring: favor positive sentiment
    pos = [s[2] if len(s)>=3 else max(s) for s in sent_scores]
    return sum(pos)/len(pos) if pos else 0.0

def main():
    # Collect (minimal example: news only)
    api = collect_finnhub_news(SYMBOLS)
    print(f"Finnhub time: {api['time']:.2f}s, items: {api['count']}")

    # Sample texts for sentiment
    texts = ["Earnings beat expectations; guidance raised."] * 30
    sent = sentiment_gpu(texts)
    print(f"Sentiment time: {sent['time']:.4f}s, CUDA: {sent['cuda']}")

    # Score
    score = score_narrative_divergence(sent["scores"]) 
    total_time = api['time'] + sent['time']
    print(f"Narrative Divergence score: {score:.4f}")
    print(f"Total pipeline time (API+sentiment): {total_time:.2f}s")

if __name__ == "__main__":
    main()
