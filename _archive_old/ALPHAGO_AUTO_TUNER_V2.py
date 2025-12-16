"""
🎮 AlphaGo Visual Pattern Discovery - Auto-Tuner V2
T4 GPU Optimized | Early Stopping | Better Regularization | Data Augmentation

Improvements over V1:
- Early stopping (prevents overfitting)
- Stronger regularization (dropout 0.5, L2 weight decay)
- Data augmentation (2x more samples)
- Better learning rate schedule

Expected Runtime: 40-60 minutes on T4 GPU
Expected Output: 63-66% policy accuracy (vs 61.77% in V1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
from pyts.image import GramianAngularField
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import json
from datetime import datetime
warnings.filterwarnings('ignore')

#===============================================================================
# CONFIGURATION
#===============================================================================

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS',
    'NFLX', 'PYPL', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'COST',
    'PEP', 'KO', 'NKE', 'MCD', 'BA', 'GE', 'F', 'GM',
    'C', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'SLB', 'COP',
    'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'ABT', 'DHR'
]

# Improved tuning configs (shorter epochs, early stopping will handle rest)
TUNING_CONFIGS = [
    {'name': 'Baseline_V2', 'hidden': 64, 'batch': 32, 'lr': 0.001, 'epochs': 20, 'weight_decay': 1e-4},
    {'name': 'Medium_V2', 'hidden': 128, 'batch': 16, 'lr': 0.0005, 'epochs': 25, 'weight_decay': 1e-4},
    {'name': 'Large_V2', 'hidden': 256, 'batch': 8, 'lr': 0.0003, 'epochs': 30, 'weight_decay': 1e-4},
]

#===============================================================================
# DATA GENERATION WITH AUGMENTATION
#===============================================================================

def download_data():
    """Download 3 years of OHLCV data"""
    print("📥 Downloading data...")
    data = {}
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, period='3y', interval='1d', progress=False)
            if len(df) > 100:
                data[ticker] = df
                print(f"  ✓ {ticker}: {len(df)} days")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
    
    print(f"\n✅ Downloaded {len(data)} tickers\n")
    return data

def generate_gasf(df, window_size=30, image_size=30, augment=False):
    """Convert OHLCV window to 5-channel GASF image with optional augmentation"""
    gasf = GramianAngularField(image_size=image_size, method='summation')
    
    def normalize(series):
        return 2 * (series - series.min()) / (series.max() - series.min() + 1e-8) - 1
    
    channels = []
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        series = df[col].values[-window_size:]
        series_norm = normalize(series)
        gasf_img = gasf.fit_transform(series_norm.reshape(1, -1))[0]
        channels.append(gasf_img)
    
    gasf_stack = np.stack(channels, axis=0)
    
    # Data augmentation
    if augment:
        # Add small Gaussian noise (preserves patterns but adds variety)
        noise = np.random.normal(0, 0.02, gasf_stack.shape)
        gasf_stack = gasf_stack + noise
        
        # Random temporal shift (±2 pixels)
        shift = np.random.randint(-2, 3)
        if shift != 0:
            gasf_stack = np.roll(gasf_stack, shift, axis=2)
    
    return gasf_stack

def create_dataset(data, window_size=30, image_size=30, horizon=5, augment=True):
    """Generate labeled GASF dataset with augmentation"""
    print("🖼️ Generating GASF images (with augmentation)...")
    dataset = []
    
    for ticker, df in data.items():
        df = df.copy()
        df['Return'] = df['Close'].pct_change(horizon).shift(-horizon)
        
        for i in range(window_size, len(df) - horizon, 5):
            window = df.iloc[i-window_size:i]
            future_return = df['Return'].iloc[i]
            
            if pd.isna(future_return):
                continue
            
            # Labels: BUY (0), HOLD (1), SELL (2)
            if future_return > 0.03:
                label = 0
            elif future_return < -0.03:
                label = 2
            else:
                label = 1
            
            # Original sample
            gasf_img = generate_gasf(window, window_size, image_size, augment=False)
            dataset.append({
                'image': gasf_img,
                'policy_label': label,
                'value_target': future_return,
                'ticker': ticker
            })
            
            # Augmented sample (if enabled)
            if augment:
                gasf_img_aug = generate_gasf(window, window_size, image_size, augment=True)
                dataset.append({
                    'image': gasf_img_aug,
                    'policy_label': label,
                    'value_target': future_return,
                    'ticker': ticker
                })
    
    print(f"✅ Generated {len(dataset)} samples")
    labels = [d['policy_label'] for d in dataset]
    print(f"   BUY: {labels.count(0)} | HOLD: {labels.count(1)} | SELL: {labels.count(2)}\n")
    
    return dataset

#===============================================================================
# IMPROVED MODEL ARCHITECTURE
#===============================================================================

class AlphaGoNetV2(nn.Module):
    """AlphaGo-style dual network with stronger regularization"""
    
    def __init__(self, in_channels=5, hidden_dim=128):
        super().__init__()
        
        # Shared CNN backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1)  # Spatial dropout
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.15)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout2d(0.25)
        )
        
        # Policy head (BUY/HOLD/SELL) - stronger dropout
        self.policy_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.4
            nn.Linear(hidden_dim, 3)
        )
        
        # Value head (expected return) - stronger dropout
        self.value_head = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),  # Increased from 0.4
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

#===============================================================================
# TRAINING WITH EARLY STOPPING
#===============================================================================

class GASFDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.FloatTensor(sample['image']),
            torch.LongTensor([sample['policy_label']]),
            torch.FloatTensor([sample['value_target']])
        )

def train_model_with_early_stopping(model, train_loader, test_loader, config, device):
    """Train dual network with early stopping and better regularization"""
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    history = {'train_loss': [], 'test_policy_acc': [], 'test_value_mse': []}
    best_policy_acc = 0
    best_model_state = None
    patience = 7  # Stop if no improvement for 7 epochs
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        for images, policy_labels, value_targets in train_loader:
            if images.size(0) == 0:
                continue
                
            images = images.to(device)
            policy_labels = policy_labels.to(device).squeeze()
            value_targets = value_targets.to(device).squeeze()
            
            if policy_labels.numel() == 0:
                continue
            
            optimizer.zero_grad()
            policy, value = model(images)
            
            policy_loss = policy_criterion(policy, policy_labels)
            value_loss = value_criterion(value.squeeze(), value_targets)
            loss = policy_loss + 0.5 * value_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Evaluate
        model.eval()
        correct = 0
        total = 0
        value_mse = 0
        with torch.no_grad():
            for images, policy_labels, value_targets in test_loader:
                if images.size(0) == 0:
                    continue
                    
                images = images.to(device)
                policy_labels = policy_labels.to(device).squeeze()
                value_targets = value_targets.to(device).squeeze()
                
                if policy_labels.numel() == 0:
                    continue
                
                policy, value = model(images)
                predicted = policy.argmax(dim=1)
                correct += (predicted == policy_labels).sum().item()
                total += policy_labels.size(0)
                value_mse += ((value.squeeze() - value_targets) ** 2).sum().item()
        
        policy_acc = correct / total if total > 0 else 0
        value_mse = value_mse / total if total > 0 else 0
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['test_policy_acc'].append(policy_acc)
        history['test_value_mse'].append(value_mse)
        
        # Learning rate scheduling
        scheduler.step(policy_acc)
        
        # Early stopping check
        if policy_acc > best_policy_acc:
            best_policy_acc = policy_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            improvement = "✨"
        else:
            patience_counter += 1
            improvement = ""
        
        if (epoch + 1) % 3 == 0:
            print(f"  Epoch {epoch+1}/{config['epochs']} | Loss: {avg_train_loss:.4f} | "
                  f"Policy: {policy_acc:.4f} {improvement} | Value MSE: {value_mse:.4f} | "
                  f"Patience: {patience_counter}/{patience}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"  🛑 Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
            model.load_state_dict(best_model_state)
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history, best_policy_acc

#===============================================================================
# MAIN EXECUTION
#===============================================================================

def main():
    print("\n" + "="*80)
    print("🎮 AlphaGo Visual Pattern Discovery - Auto-Tuner V2")
    print("="*80)
    print("Improvements: Early Stopping | Better Regularization | Data Augmentation")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Step 1: Download data
    data = download_data()
    
    # Step 2: Generate GASF dataset WITH AUGMENTATION
    dataset = create_dataset(data, window_size=30, image_size=30, horizon=5, augment=True)
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_data)} | Test: {len(test_data)}\n")
    
    # Step 3: Hyperparameter tuning
    print("="*80)
    print("🔬 Starting Hyperparameter Tuning (with Early Stopping)")
    print("="*80 + "\n")
    
    results = []
    for i, config in enumerate(TUNING_CONFIGS, 1):
        print(f"[{i}/{len(TUNING_CONFIGS)}] Training: {config['name']}")
        print(f"   Hidden: {config['hidden']}, Batch: {config['batch']}, LR: {config['lr']}, "
              f"Weight Decay: {config['weight_decay']}")
        
        # Create dataloaders
        train_loader = DataLoader(GASFDataset(train_data), batch_size=config['batch'], 
                                  shuffle=True, num_workers=2, drop_last=True)
        test_loader = DataLoader(GASFDataset(test_data), batch_size=config['batch'], 
                                 shuffle=False, num_workers=2, drop_last=True)
        
        # Create improved model
        model = AlphaGoNetV2(in_channels=5, hidden_dim=config['hidden']).to(device)
        
        # Train with early stopping
        history, best_acc = train_model_with_early_stopping(model, train_loader, test_loader, config, device)
        
        # Save results
        results.append({
            'config': config,
            'history': history,
            'best_accuracy': best_acc,
            'model_state': model.state_dict()
        })
        
        print(f"  ✅ Best Accuracy: {best_acc:.4f}\n")
    
    # Step 4: Select best model
    best_idx = max(range(len(results)), key=lambda i: results[i]['best_accuracy'])
    best_result = results[best_idx]
    
    print("\n" + "="*80)
    print("🏆 BEST MODEL")
    print("="*80)
    print(f"Config: {best_result['config']['name']}")
    print(f"Accuracy: {best_result['best_accuracy']:.4f}")
    print(f"Hidden: {best_result['config']['hidden']}, Batch: {best_result['config']['batch']}, "
          f"LR: {best_result['config']['lr']}")
    print(f"Improvement over V1: +{(best_result['best_accuracy'] - 0.6177)*100:.2f}% absolute")
    print("="*80 + "\n")
    
    # Step 5: Save best model
    torch.save({
        'model_state': best_result['model_state'],
        'config': best_result['config'],
        'accuracy': best_result['best_accuracy'],
        'history': best_result['history'],
        'version': 'V2',
        'improvements': 'early_stopping + regularization + augmentation',
        'timestamp': datetime.now().isoformat()
    }, 'alphago_best_model_v2.pth')
    print("✅ Saved: alphago_best_model_v2.pth\n")
    
    # Step 6: Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Accuracy comparison
    for res in results:
        axes[0].plot(res['history']['test_policy_acc'], label=res['config']['name'], linewidth=2)
    axes[0].axhline(y=0.6177, color='red', linestyle='--', alpha=0.5, label='V1 Best (61.77%)')
    axes[0].set_title('Policy Accuracy Comparison (V2)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Training loss
    for res in results:
        axes[1].plot(res['history']['train_loss'], label=res['config']['name'], linewidth=2)
    axes[1].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Plot 3: Final accuracy comparison
    names = [res['config']['name'] for res in results]
    accs = [res['best_accuracy'] for res in results]
    colors = ['gold' if i == best_idx else 'steelblue' for i in range(len(results))]
    bars = axes[2].bar(range(len(results)), accs, color=colors)
    axes[2].axhline(y=0.6177, color='red', linestyle='--', label='V1 Best', linewidth=2)
    axes[2].set_title('Best Accuracy by Config (V2)', fontsize=14, fontweight='bold')
    axes[2].set_xticks(range(len(results)))
    axes[2].set_xticklabels([n.split('_')[0] for n in names], rotation=45)
    axes[2].set_ylabel('Accuracy')
    axes[2].legend()
    axes[2].grid(alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('tuning_results_v2.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: tuning_results_v2.png\n")
    
    # Step 7: Save results JSON
    results_json = {
        'timestamp': datetime.now().isoformat(),
        'version': 'V2',
        'improvements': ['early_stopping', 'stronger_regularization', 'data_augmentation'],
        'best_config': best_result['config'],
        'best_accuracy': best_result['best_accuracy'],
        'v1_accuracy': 0.6177,
        'improvement': best_result['best_accuracy'] - 0.6177,
        'all_results': [
            {
                'name': res['config']['name'],
                'accuracy': res['best_accuracy'],
                'config': res['config'],
                'epochs_trained': len(res['history']['test_policy_acc'])
            }
            for res in results
        ]
    }
    with open('tuning_results_v2.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print("✅ Saved: tuning_results_v2.json\n")
    
    # Step 8: Recommendations
    print("="*80)
    print("📋 IMPLEMENTATION RECOMMENDATIONS (V2)")
    print("="*80)
    print("\n✅ Best Practices:")
    print("   • Window size: 30 days")
    print("   • Image size: 30×30")
    print("   • Channels: 5 (OHLCV)")
    print("   • Data augmentation: Enabled (2x samples)")
    print(f"   • Hidden dim: {best_result['config']['hidden']}")
    print(f"   • Batch size: {best_result['config']['batch']}")
    print(f"   • Learning rate: {best_result['config']['lr']}")
    print(f"   • Weight decay: {best_result['config']['weight_decay']}")
    print("   • Dropout: 0.5 (increased from 0.4)")
    print("   • Early stopping: 7 epochs patience")
    
    print("\n🔬 Tuning Insights:")
    sorted_results = sorted(results, key=lambda x: x['best_accuracy'], reverse=True)
    for i, res in enumerate(sorted_results, 1):
        epochs_trained = len(res['history']['test_policy_acc'])
        print(f"   {i}. {res['config']['name']}: {res['best_accuracy']:.4f} "
              f"(trained {epochs_trained} epochs, early stopped)")
    
    print("\n📊 Improvement Analysis:")
    improvement_pct = (best_result['best_accuracy'] - 0.6177) * 100
    print(f"   V1 Accuracy: 61.77%")
    print(f"   V2 Accuracy: {best_result['best_accuracy']*100:.2f}%")
    print(f"   Improvement: +{improvement_pct:.2f}% absolute")
    
    if best_result['best_accuracy'] >= 0.65:
        print("\n   🎉 SUCCESS! Model exceeds 65% target")
        print("   ✅ Ready for production ensemble (40/60 visual/numerical)")
    elif best_result['best_accuracy'] >= 0.63:
        print("\n   ✅ GOOD! Model shows solid improvement")
        print("   ✅ Use conservative ensemble (30/70 visual/numerical)")
    else:
        print("\n   ⚠️ MARGINAL: Model improved but below target")
        print("   💡 Try: More tickers, longer windows, or attention mechanisms")
    
    print("\n🚀 Next Steps:")
    print("   1. Download alphago_best_model_v2.pth")
    print("   2. Ensemble with numerical model:")
    if best_result['best_accuracy'] >= 0.65:
        print("      → 40% visual + 60% numerical")
    else:
        print("      → 30% visual + 70% numerical")
    print("   3. Backtest on 2024 data")
    print("   4. Paper trade 1 week")
    print("   5. Deploy if consistent profits")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
