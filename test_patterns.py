"""Test pattern detection with new data fetcher"""
from data_fetcher import DataFetcher
from pattern_detector import PatternDetector

print("Testing pattern detection...")

fetcher = DataFetcher()
df = fetcher.fetch_ohlcv('MU', '60d')
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

pd = PatternDetector()
result = pd.detect_all_patterns('MU')
print(f"Patterns detected: {len(result['all_patterns'])}")
print("✅ Pattern detection working!")
