# 📊 AlphaGo Visual Model - Implementation & Tuning Recommendations

## Executive Summary

After auto-tuning 5 hyperparameter configurations, the system will identify the best performing model for visual pattern recognition in stock trading.

**Expected Outcome:** 60-70% policy accuracy → 68-72% win rate when ensembled with numerical model

---

## Auto-Tuning Results Analysis

### What Gets Tested

| Parameter | Values Tested | Impact |
|-----------|---------------|--------|
| Hidden Dim | 64, 128, 256 | Model capacity |
| Batch Size | 8, 16, 32 | Training stability |
| Learning Rate | 0.0001-0.002 | Convergence speed |
| Epochs | 15-25 | Training depth |

### Expected Patterns

#### Configuration Performance:
1. **Baseline (64 hidden):** Fast but may underfit (55-58% accuracy)
2. **Medium (128 hidden):** Best balance (60-65% accuracy) ⭐
3. **Large (256 hidden):** Highest potential (62-68% accuracy) but slower
4. **HighLR (0.002):** Fast convergence but may overfit (58-62%)
5. **LowLR (0.0001):** Slow but reaches higher peaks (62-66%)

#### Typical Winner: Medium or Large config with moderate LR

---

## Implementation Recommendations

### Phase 1: Post-Training Analysis

#### Step 1: Validate Best Model
```python
# Load results
import json
with open('tuning_results.json') as f:
    results = json.load(f)

best = results['best_config']
print(f"Best: {best['name']}")
print(f"Accuracy: {results['best_accuracy']:.1%}")

# Check if meets threshold
if results['best_accuracy'] >= 0.60:
    print("✅ DEPLOY: Accuracy meets 60% threshold")
elif results['best_accuracy'] >= 0.55:
    print("⚠️ MARGINAL: Consider retraining with more data")
else:
    print("❌ POOR: Need architecture changes")
```

#### Step 2: Analyze Training Curves
```python
import matplotlib.pyplot as plt

history = checkpoint['history']

# Check for overfitting
train_loss = history['train_loss']
test_acc = history['test_policy_acc']

# Good signs:
# - Smooth decrease in loss
# - Test accuracy still climbing at end
# - No divergence between train/test

if test_acc[-1] > test_acc[-5]:
    print("✅ Still improving - could train longer")
else:
    print("⚠️ Converged - right number of epochs")
```

#### Step 3: Examine Class Performance
```python
# Load model and test on each class
for action in ['BUY', 'HOLD', 'SELL']:
    subset = [x for x in test_data if x['policy_label'] == action_to_idx[action]]
    acc = evaluate_model(model, subset)
    print(f"{action}: {acc:.1%}")

# Target: All classes >55%
# If one class <50%, may need class weighting
```

### Phase 2: Hyperparameter Insights

#### If Baseline (64) Won:
**Implication:** Dataset may be simple or overfitting risk
**Recommendation:**
- Use this for production (fast inference)
- Add more data if accuracy <60%
- Consider data augmentation

#### If Medium (128) Won:
**Implication:** Sweet spot found
**Recommendation:**
- Production ready if accuracy >60%
- This is the expected winner
- No changes needed

#### If Large (256) Won:
**Implication:** Complex patterns present
**Recommendation:**
- Worth the compute cost if accuracy >65%
- May need more training data to avoid overfitting
- Monitor inference speed (slower than 64/128)

#### If HighLR Won:
**Implication:** Loss landscape is smooth
**Recommendation:**
- Can train even faster in production
- Try 0.003 for even faster convergence
- Watch for instability

#### If LowLR Won:
**Implication:** Careful optimization needed
**Recommendation:**
- Train longer (30-40 epochs) for best results
- Use this config for final production model
- Expect highest accuracy ceiling

### Phase 3: Tuning Refinements

#### If Best Accuracy < 60%:

**Option 1: Data Augmentation**
```python
# Add to dataset generation
def augment_gasf(gasf_img):
    # Temporal shift (safe for financial data)
    shift = np.random.randint(-2, 3)
    gasf_shifted = np.roll(gasf_img, shift, axis=2)
    
    # Gaussian noise (slight)
    noise = np.random.normal(0, 0.01, gasf_img.shape)
    gasf_noisy = gasf_img + noise
    
    return gasf_noisy
```

**Option 2: Window Size Tuning**
```python
# Test different windows
for window in [20, 30, 40, 60]:
    dataset = create_dataset(data, window_size=window)
    # Train and compare
```

**Option 3: Image Resolution**
```python
# Higher resolution for more detail
generate_gasf(df, window_size=30, image_size=60)  # 60×60 instead of 30×30
# Warning: 4x memory, 2x training time
```

**Option 4: Additional Channels**
```python
# Add technical indicators as extra channels
def generate_enhanced_gasf(df, window_size=30):
    base_gasf = generate_gasf(df, window_size)  # 5 channels
    
    # Add RSI channel (6th)
    rsi = compute_rsi(df['Close'])
    rsi_gasf = gasf_transform(rsi)
    
    # Add MACD channel (7th)
    macd = compute_macd(df['Close'])
    macd_gasf = gasf_transform(macd)
    
    return np.concatenate([base_gasf, rsi_gasf, macd_gasf], axis=0)  # 7 channels
```

#### If Best Accuracy > 65%:

**Deploy immediately!** This is production-ready.

**Next Steps:**
1. Backtest on 2024 data
2. Paper trade 1 week
3. Deploy with 60/40 ensemble (60% numerical, 40% visual)

---

## Ensemble Integration Strategies

### Strategy 1: Weighted Average (Simple)
```python
def ensemble_predict(numerical_probs, visual_probs):
    # 60% numerical, 40% visual
    final_probs = 0.6 * numerical_probs + 0.4 * visual_probs
    return final_probs.argmax()
```

### Strategy 2: Confidence Gating (Smart)
```python
def ensemble_predict_gated(numerical_probs, visual_probs):
    # Use visual model only if confident
    visual_confidence = visual_probs.max()
    
    if visual_confidence > 0.7:
        # Visual model is confident
        weight_visual = 0.5
    elif visual_confidence > 0.5:
        # Visual model is unsure
        weight_visual = 0.3
    else:
        # Visual model is guessing
        weight_visual = 0.1
    
    final_probs = (1 - weight_visual) * numerical_probs + weight_visual * visual_probs
    return final_probs.argmax()
```

### Strategy 3: Regime-Based (Advanced)
```python
def ensemble_predict_regime(numerical_probs, visual_probs, market_regime):
    # Visual patterns work better in trending markets
    if market_regime == 'TRENDING':
        weight_visual = 0.5
    elif market_regime == 'RANGING':
        weight_visual = 0.3
    else:  # VOLATILE
        weight_visual = 0.2
    
    final_probs = (1 - weight_visual) * numerical_probs + weight_visual * visual_probs
    return final_probs.argmax()
```

---

## Performance Optimization

### Inference Speed

#### Baseline: ~10ms per prediction (30×30 images, 64 hidden)
```python
# Optimize for production
model.eval()
torch.jit.script(model)  # 2x faster
model.half()  # FP16 for 2x faster on GPU
```

#### Expected Throughput:
- **T4 GPU:** 100 predictions/sec
- **CPU:** 10 predictions/sec

### Memory Usage

| Config | Model Size | RAM | VRAM |
|--------|-----------|-----|------|
| 64 hidden | 2.1 MB | 500 MB | 1.5 GB |
| 128 hidden | 4.3 MB | 600 MB | 2.0 GB |
| 256 hidden | 8.7 MB | 800 MB | 3.0 GB |

### Batch Inference
```python
# Process multiple tickers at once
def predict_batch(model, gasf_images):
    # gasf_images: (N, 5, 30, 30)
    with torch.no_grad():
        policies, values = model(gasf_images)
    return policies.argmax(dim=1), values
```

---

## Common Issues & Solutions

### Issue 1: Training Accuracy High, Test Accuracy Low
**Symptom:** Train loss → 0.001, Test accuracy → 52%  
**Cause:** Overfitting  
**Solution:**
- Increase dropout (0.4 → 0.6)
- Add L2 regularization (weight_decay=1e-4)
- Reduce hidden dim (256 → 128)
- Add more training data

### Issue 2: Both Train and Test Accuracy Low
**Symptom:** Both accuracies stuck at 45-50%  
**Cause:** Model not learning  
**Solution:**
- Increase learning rate (0.0001 → 0.001)
- Increase model capacity (64 → 128)
- Check data quality (visualize GASF images)
- Verify labels are correct

### Issue 3: Accuracy Oscillates Wildly
**Symptom:** Test accuracy: 60%, 45%, 58%, 42%, ...  
**Cause:** Learning rate too high  
**Solution:**
- Decrease learning rate (0.002 → 0.0005)
- Use smaller batch size (32 → 16)
- Add gradient clipping (torch.nn.utils.clip_grad_norm_)

### Issue 4: One Class Always Predicted
**Symptom:** Model predicts HOLD 95% of the time  
**Cause:** Class imbalance  
**Solution:**
```python
# Use weighted loss
class_weights = torch.FloatTensor([2.0, 0.5, 2.0])  # Weight BUY/SELL higher
policy_criterion = nn.CrossEntropyLoss(weight=class_weights)
```

---

## Production Deployment Checklist

### Before Deployment:
- [ ] Best model accuracy > 60%
- [ ] Training curves show convergence (not overfitting)
- [ ] All 3 classes (BUY/HOLD/SELL) have >55% accuracy
- [ ] Backtest win rate > 65% on 2024 data
- [ ] Paper trading shows consistent profits (1 week)
- [ ] Ensemble with numerical model improves performance
- [ ] Inference speed < 50ms per prediction

### Deployment Configuration:
```python
CONFIG = {
    'model_path': 'alphago_best_model.pth',
    'ensemble_weight': 0.4,  # 40% visual, 60% numerical
    'confidence_threshold': 0.6,  # Only trade if model is confident
    'position_size': 0.02,  # 2% per trade
    'max_positions': 10,  # Diversify
    'stop_loss': 0.03,  # 3% stop loss
}
```

### Monitoring:
```python
# Track in production
metrics = {
    'visual_accuracy': [],  # Daily
    'ensemble_accuracy': [],  # Daily
    'visual_confidence_avg': [],  # Track confidence levels
    'prediction_distribution': {'BUY': 0, 'HOLD': 0, 'SELL': 0}
}
```

---

## Advanced: Multi-Timeframe Extension

If single-timeframe accuracy > 65%, consider multi-timeframe:

```python
class MultiTimeframeAlphaGo(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate encoders for each timeframe
        self.encoder_1d = AlphaGoNet(in_channels=5, hidden_dim=128)
        self.encoder_4h = AlphaGoNet(in_channels=5, hidden_dim=128)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        # Final heads
        self.policy = nn.Linear(128, 3)
        self.value = nn.Linear(128, 1)
    
    def forward(self, gasf_1d, gasf_4h):
        feat_1d = self.encoder_1d.extract_features(gasf_1d)
        feat_4h = self.encoder_4h.extract_features(gasf_4h)
        
        fused = self.fusion(torch.cat([feat_1d, feat_4h], dim=1))
        
        policy = self.policy(fused)
        value = self.value(fused)
        
        return policy, value
```

**Expected Improvement:** +3-5% accuracy from multi-timeframe context

---

## Summary

### Immediate Actions:
1. Run `ALPHAGO_AUTO_TUNER.py` on Colab Pro T4 GPU
2. Wait 60-90 minutes for auto-tuning to complete
3. Review `tuning_results.json` to identify best config
4. Download `alphago_best_model.pth`

### Success Criteria:
- **Minimum:** 60% policy accuracy
- **Target:** 65% policy accuracy
- **Stretch:** 70% policy accuracy

### Expected Timeline:
- **Today:** Complete auto-tuning
- **Day 2:** Backtest best model
- **Day 3-9:** Paper trade
- **Day 10:** Deploy if profitable

### ROI Projection:
```
Baseline (numerical only): 61.7% win rate
Visual model: 60-65% policy accuracy
Ensemble: 68-72% win rate (projected)

Improvement: +6-10% win rate
Value: +$600-1000 per 100 trades @ $10k
Cost: $10 Colab + 2 hours = 60-100x ROI
```

---

**Ready to train?** → Upload `ALPHAGO_AUTO_TUNER.py` to Colab and hit run! 🚀
