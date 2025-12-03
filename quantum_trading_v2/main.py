"""System entry point for quantum_trading_v2. Tests config, DB, data collection."""
from config.settings import Settings
from data.database_manager import DatabaseManager
from data.price_collector import PriceCollector
from data.news_collector import NewsCollector

def main():
    print("\n=== Quantum Trading System v2 ===\n")
    # Load config and validate API keys
    settings = Settings()

    # Initialize database
    db = DatabaseManager(settings.DATABASE_PATH)
    print("[Main] Database initialized and tables ensured.")
    assert db.validate(), "Database integrity check failed!"

    # Collect price data for AAPL
    price_collector = PriceCollector(settings.FINNHUB_API_KEY)
    prices = price_collector.fetch_historical("AAPL", days=5)
    print(f"[Main] Collected {len(prices)} price records for AAPL.")
    for p in prices:
        db.insert_price(p)
    print("[Main] Price data inserted into DB.")

    # Collect news for AAPL
    news_collector = NewsCollector(settings.NEWSAPI_API_KEY)
    news_items = news_collector.fetch_news("AAPL", limit=3)
    print(f"[Main] Collected {len(news_items)} news articles for AAPL.")
    for n in news_items:
        db.insert_news(n)
    print("[Main] News data inserted into DB.")

    # Show DB contents
    print("\n[Main] Recent price data:")
    for row in db.get_prices("AAPL")[:3]:
        print(row)
    print("\n[Main] Recent news sentiment:")
    for row in db.get_news("AAPL")[:3]:
        print(row)

    db.close()
    print("\n=== Quantum Trading System v2 - All tests passed ===\n")

if __name__ == "__main__":
    main()
