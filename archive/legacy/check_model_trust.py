"""
Quick model trust check - shows if AI models are ready for real capital
"""
import os
import joblib
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models'

print("\n" + "="*60)
print("🤖 AI MODEL TRUST EVALUATION")
print("="*60)

tickers = ['MU', 'IONQ', 'APLD', 'ANNX']
results = []

# Check if models exist
print("\n🔍 Checking for existing trained models...")
for ticker in tickers:
    model_path = MODELS_DIR / f'{ticker}_tuned.pkl'
    if model_path.exists():
        try:
            model_data = joblib.load(model_path)
            results.append({
                'ticker': ticker,
                'accuracy': model_data.get('accuracy_mean', model_data.get('accuracy', 0)),
                'trained_date': model_data.get('trained_datetime', 'Unknown'),
                'samples': model_data.get('training_samples', 'Unknown')
            })
            print(f"  ✓ Found {ticker} model")
        except Exception as e:
            print(f"  ✗ Error loading {ticker}: {e}")
    else:
        print(f"  ✗ {ticker} model not found - needs training")

if not results:
    print("\n❌ NO MODELS FOUND!")
    print("\n🔧 TO TRAIN MODELS:")
    print("   python ai_recommender_tuned.py")
    print("\n   Or use Colab Pro training notebook")
    exit(1)

# Summary
df = pd.DataFrame(results)
print("\n" + "="*60)
print("📊 MODEL PERFORMANCE SUMMARY")
print("="*60)
print(df[['ticker', 'accuracy', 'samples']].to_string(index=False))

avg_acc = df['accuracy'].mean()
print(f"\n📈 Average Accuracy: {avg_acc:.1%}")

# Trust analysis
print("\n" + "="*60)
print("⚠️  TRUST ANALYSIS - CAN YOU TRADE WITH REAL MONEY?")
print("="*60)

for _, row in df.iterrows():
    ticker = row['ticker']
    acc = row['accuracy']
    
    if acc >= 0.65:
        trust = "✅ TRUSTWORTHY"
        recommendation = "Ready for real capital with proper risk management"
    elif acc >= 0.60:
        trust = "⚠️  USE CAUTION"
        recommendation = "Paper trade first, small positions only"
    else:
        trust = "❌ NOT READY"
        recommendation = "Needs more training data or feature engineering"
    
    print(f"\n{ticker}: {acc:.1%} - {trust}")
    print(f"   → {recommendation}")

print("\n" + "="*60)
print("🎯 RECOMMENDATION:")
print("="*60)

if avg_acc >= 0.65:
    print("✅ SYSTEM READY FOR REAL CAPITAL")
    print("   - Models show consistent 65%+ accuracy")
    print("   - Start with 1% risk per trade")
    print("   - Monitor for first 2 weeks")
elif avg_acc >= 0.60:
    print("⚠️  PROCEED WITH EXTREME CAUTION")
    print("   - Paper trade for 1 week minimum")
    print("   - Use 0.5% risk per trade max")
    print("   - Consider adding more training data")
else:
    print("❌ NOT READY FOR REAL CAPITAL")
    print("   - Train on 3+ years of data")
    print("   - Add more features")
    print("   - Paper trade until 65%+ accuracy")

print("\n" + "="*60)
print("📁 Models saved to: models/")
print("🌐 Charts available in: frontend/charts/")
print("="*60 + "\n")
