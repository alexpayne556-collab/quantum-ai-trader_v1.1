# 🚀 ULTIMATE COLAB TRAINING SESSION
## Copy-Paste Ready for Google Colab Tonight

---

## Cell 1: Setup & GPU Check
```python
# Check GPU
!nvidia-smi
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
```

---

## Cell 2: Install Dependencies
```python
!pip install torch gymnasium pandas numpy yfinance ta-lib requests -q
```

---

## Cell 3: Upload Files
```python
# Upload these files from your repo:
# - ultimate_training_ring.py
# - combination_discovery.py

from google.colab import files
uploaded = files.upload()
```

---

## Cell 4: Run Ultimate Training (2000 episodes)
```python
%run ultimate_training_ring.py
```

This will:
- Load all 44 tickers
- Compute 100+ features each
- Train with mixed precision
- Find winning patterns
- Save: ultimate_model_final.pt, ultimate_predictions.json

---

## Cell 5: Run Combination Discovery (find THE winning patterns)
```python
%run combination_discovery.py
```

This will:
- Test 10,000+ strategy combinations
- Find patterns that work across multiple tickers
- Save: discovered_patterns.json

---

## Cell 6: Download Results
```python
from google.colab import files

# Download trained model
files.download('ultimate_model_final.pt')
files.download('best_model.pt')

# Download predictions
files.download('ultimate_predictions.json')

# Download discovered patterns
files.download('discovered_patterns.json')
```

---

## Cell 7: Quick Analysis of Results
```python
import json

# Load predictions
with open('ultimate_predictions.json', 'r') as f:
    predictions = json.load(f)

print("🎯 TOP BUY SIGNALS:")
buys = [p for p in predictions if p['action'] == 'BUY']
for p in sorted(buys, key=lambda x: x['buy_prob'], reverse=True)[:10]:
    signal = p.get('signal', '')
    print(f"  {p['ticker']:5s} ${p['price']:>7.2f} | {p['buy_prob']*100:.0f}% conf | {signal}")

print("\n🎯 DIP BUYS (ACT ON THESE):")
dips = [p for p in predictions if p.get('dip_buy')]
for p in dips:
    print(f"  {p['ticker']:5s} ${p['price']:>7.2f} | Down {p['drawdown']:.1f}% | RSI: {p['rsi']:.0f}")

# Load discovered patterns
try:
    with open('discovered_patterns.json', 'r') as f:
        patterns = json.load(f)
    
    print(f"\n🏆 TOP DISCOVERED PATTERNS: {patterns['total_patterns']}")
    for i, p in enumerate(patterns['patterns'][:5]):
        print(f"\n{i+1}. {p['name']}")
        print(f"   Win Rate: {p['win_rate']*100:.1f}%")
        print(f"   Avg Return: {p['avg_return']*100:.2f}%")
        print(f"   Sharpe: {p['sharpe_ratio']:.2f}")
except:
    print("Run combination_discovery.py first")
```

---

## Cell 8: Extended Training (if results are good)
```python
# If first results look promising, train more!
import torch
from ultimate_training_ring import TrainingConfig, load_market_data, VectorizedTradingEnv, UltimateTrainer

# Load checkpoint
config = TrainingConfig()
config.total_episodes = 3000  # More episodes

data = load_market_data(config.tickers)
env = VectorizedTradingEnv(data, config)
trainer = UltimateTrainer(env, config)

# Load previous best
trainer.load_checkpoint('best_model.pt')

# Continue training
trainer.train(1000)  # 1000 more episodes
trainer.save_checkpoint('extended_model.pt')
```

---

## Cell 9: Generate Final Paper Trading Signals
```python
import json
from datetime import datetime

# Load best model and generate signals
model = trainer.model
model.eval()

# Get predictions
obs = env.reset()
with torch.no_grad():
    obs_tensor = torch.FloatTensor(obs).to(trainer.model.dip_head[0].weight.device)
    logits, values = model(obs_tensor)
    probs = torch.softmax(logits, dim=-1)

final_signals = []
for i, ticker in enumerate(env.tickers):
    action_probs = probs[i].cpu().numpy()
    
    idx = int(env.step_idx[i])
    df = env.data[ticker]
    price = df['Close'].iloc[idx]
    
    signal = {
        'ticker': ticker,
        'price': float(price),
        'action': ['HOLD', 'BUY', 'SELL'][action_probs.argmax()],
        'buy_prob': float(action_probs[1]),
        'sell_prob': float(action_probs[2]),
        'timestamp': datetime.now().isoformat()
    }
    final_signals.append(signal)

# Save
with open('paper_trading_signals.json', 'w') as f:
    json.dump(final_signals, f, indent=2)

print("✅ Ready for paper trading!")
print("\n📊 TOP SIGNALS FOR TOMORROW:")
for s in sorted(final_signals, key=lambda x: x['buy_prob'], reverse=True)[:10]:
    if s['action'] == 'BUY':
        print(f"  🟢 {s['ticker']:5s} ${s['price']:>7.2f} | {s['buy_prob']*100:.0f}% confidence")

files.download('paper_trading_signals.json')
```

---

# 🎯 TONIGHT'S MISSION

1. **Upload files to Colab**
2. **Run 2000+ episode training** (Cell 4) - ~30-60 min
3. **Run combination discovery** (Cell 5) - ~20-30 min  
4. **Download all results**
5. **Analyze and prepare paper trades**

## Expected Outputs:
- `ultimate_model_final.pt` - Trained brain
- `ultimate_predictions.json` - Today's signals
- `discovered_patterns.json` - Winning pattern combinations
- `paper_trading_signals.json` - Tomorrow's trades

## Success Metrics:
- ✅ Win Rate > 60%
- ✅ Sharpe Ratio > 1.0
- ✅ Clear buy signals identified
- ✅ Dip buy opportunities flagged

Let's do this! 🔥
