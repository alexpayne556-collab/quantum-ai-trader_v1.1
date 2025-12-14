"""
Quick test of Underdog API endpoints (without models trained)
"""

import requests
import json

API_BASE = "http://localhost:5000/api/underdog"

def test_endpoint(name, method, endpoint, data=None):
    """Test a single endpoint"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    url = f"{API_BASE}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        print(f"Status: {response.status_code}")
        
        result = response.json()
        print(json.dumps(result, indent=2))
        
        return result.get('success', False)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("🧪 Testing Underdog API Endpoints")
    print("Note: Models not trained yet - some endpoints will return 'not trained' errors")
    
    results = {}
    
    # Test 1: Health check
    results['status'] = test_endpoint(
        "System Health Check",
        "GET",
        "/status"
    )
    
    # Test 2: Watchlist
    results['alpha76'] = test_endpoint(
        "Alpha 76 Watchlist",
        "GET",
        "/alpha76"
    )
    
    # Test 3: Regime
    results['regime'] = test_endpoint(
        "Current Market Regime",
        "GET",
        "/regime"
    )
    
    # Test 4: Single prediction (will fail if models not trained)
    results['predict'] = test_endpoint(
        "Single Ticker Prediction (RKLB)",
        "POST",
        "/predict",
        {"ticker": "RKLB"}
    )
    
    # Test 5: Top signals (will fail if models not trained)
    results['top_signals'] = test_endpoint(
        "Top BUY Signals",
        "GET",
        "/top-signals"
    )
    
    # Test 6: Batch predict (will fail if models not trained)
    results['batch'] = test_endpoint(
        "Batch Predictions",
        "POST",
        "/batch-predict",
        {"tickers": ["RKLB", "IONQ", "SOFI"]}
    )
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, success in results.items():
        status = "✅ PASS" if success else "⚠️  FAIL (expected if models not trained)"
        print(f"{name:20s}: {status}")
    
    print(f"\n{passed}/{total} endpoints responding correctly")
    
    if results['status'] and results['alpha76'] and results['regime']:
        print("\n✅ Core endpoints working - ready to train models on Colab Pro!")
    else:
        print("\n⚠️  Some core endpoints failed - check API server")


if __name__ == "__main__":
    print("Starting API test...")
    print("Make sure API is running: python underdog_api.py")
    print()
    
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API")
        print("Make sure the API is running: python underdog_api.py")
