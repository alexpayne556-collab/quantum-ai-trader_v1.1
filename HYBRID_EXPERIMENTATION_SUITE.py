"""
🧪 Hybrid Experimentation Suite - Complete Testing Framework
Systematically test ALL approaches to find what works best

Philosophy: Test everything, keep what works, iterate on winners
- Visual alone vs Numerical alone vs Hybrid
- Different fusion strategies (concat, attention, gated)
- Multiple ensemble methods (voting, stacking, weighted)
- Pattern discovery via GA (learn from humans, then improve)

Checkpointed: Resume from any failed experiment
Results tracked: JSON logs for every configuration
Runtime: 8-12 hours full suite, or run individual experiments

Expected outcome: Definitive answer on best architecture (68-75% win rate target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
from pyts.image import GramianAngularField
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import json
import pickle
import warnings
from datetime import datetime
from pathlib import Path
import random
warnings.filterwarnings('ignore')

#===============================================================================
# GLOBAL CONFIGURATION
#===============================================================================

TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'BRK-B',
    'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'HD', 'DIS',
    'NFLX', 'PYPL', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM', 'COST',
    'PEP', 'KO', 'NKE', 'MCD', 'BA', 'GE', 'F', 'GM',
    'C', 'BAC', 'WFC', 'GS', 'XOM', 'CVX', 'SLB', 'COP',
    'UNH', 'PFE', 'ABBV', 'TMO', 'LLY', 'MRK', 'ABT', 'DHR'
]

CHECKPOINT_DIR = Path('checkpoints')
RESULTS_DIR = Path('experiment_results')
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

#===============================================================================
# EXPERIMENT MANAGER - Track Everything
#===============================================================================

class ExperimentTracker:
    """Tracks all experiments, saves checkpoints, enables resume"""
    
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.results_file = RESULTS_DIR / f'{experiment_name}_results.json'
        self.checkpoint_file = CHECKPOINT_DIR / f'{experiment_name}_checkpoint.pkl'
        self.results = self._load_results()
        
    def _load_results(self):
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  ⚠️  Corrupted results file detected, backing up and starting fresh...")
                # Backup corrupted file
                backup_path = self.results_file.with_suffix('.json.backup')
                self.results_file.rename(backup_path)
                print(f"  💾 Backed up to: {backup_path}")
        return {'experiments': [], 'best_config': None, 'best_score': 0}
    
    def save_result(self, config, metrics):
        """Save experiment result"""
        # Create JSON-serializable config (remove non-serializable items)
        safe_config = {k: v for k, v in config.items() 
                      if not callable(v) and k not in ['model_class', 'model_args']}
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'config': safe_config,
            'metrics': metrics
        }
        self.results['experiments'].append(result)
        
        # Update best
        if metrics.get('accuracy', 0) > self.results['best_score']:
            self.results['best_score'] = metrics['accuracy']
            self.results['best_config'] = safe_config
        
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"  💾 Saved: {safe_config.get('name', 'unknown')} → {metrics.get('accuracy', 0):.4f}")
    
    def save_checkpoint(self, state):
        """Save checkpoint for resume"""
        with open(self.checkpoint_file, 'wb') as f:
            pickle.dump(state, f)
        print(f"  ✓ Checkpoint saved")
    
    def load_checkpoint(self):
        """Load checkpoint to resume"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def get_summary(self):
        """Generate summary report"""
        if not self.results['experiments']:
            return "No experiments completed yet"
        
        sorted_exp = sorted(self.results['experiments'], 
                           key=lambda x: x['metrics']['accuracy'], 
                           reverse=True)
        
        report = f"\n{'='*80}\n"
        report += f"📊 {self.experiment_name.upper()} - SUMMARY\n"
        report += f"{'='*80}\n"
        report += f"Total Experiments: {len(self.results['experiments'])}\n"
        report += f"Best Accuracy: {self.results['best_score']:.4f}\n"
        report += f"Best Config: {self.results['best_config']['name']}\n\n"
        report += "Top 5 Performers:\n"
        for i, exp in enumerate(sorted_exp[:5], 1):
            report += f"  {i}. {exp['config']['name']}: {exp['metrics']['accuracy']:.4f}\n"
        report += f"{'='*80}\n"
        
        return report

#===============================================================================
# DATA GENERATION (Shared across all experiments)
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
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
    print(f"✅ Downloaded {len(data)} tickers\n")
    return data

def generate_gasf(df, window_size=30, image_size=30, augment=False):
    """Convert OHLCV window to 5-channel GASF image"""
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
    
    if augment:
        noise = np.random.normal(0, 0.02, gasf_stack.shape)
        gasf_stack = gasf_stack + noise
        shift = np.random.randint(-2, 3)
        if shift != 0:
            gasf_stack = np.roll(gasf_stack, shift, axis=2)
    
    return gasf_stack

def generate_numerical_features(df, window_size=30):
    """Generate 15-D numerical features (same as v1.1)"""
    features = []
    
    # Price-based - flatten to ensure scalars
    close_prices = df['Close'].values[-window_size:].flatten()
    features.append(float(np.mean(close_prices)))  # Mean price
    features.append(float(np.std(close_prices)))   # Volatility
    features.append(float((close_prices[-1] - close_prices[0]) / (close_prices[0] + 1e-8)))  # Return
    
    # Volume-based
    volumes = df['Volume'].values[-window_size:].flatten()
    features.append(float(np.mean(volumes)))
    features.append(float(np.std(volumes)))
    
    # Technical indicators (simplified)
    mean_price = float(np.mean(close_prices))
    features.append(float(close_prices[-1] / (mean_price + 1e-8) - 1))  # Price vs MA
    features.append(float(np.max(close_prices) / (close_prices[-1] + 1e-8) - 1))   # Distance from high
    features.append(float(close_prices[-1] / (np.min(close_prices) + 1e-8) - 1))   # Distance from low
    
    # Momentum
    features.append(float(close_prices[-1] / (close_prices[-5] + 1e-8) - 1) if len(close_prices) >= 5 else 0.0)
    features.append(float(close_prices[-1] / (close_prices[-10] + 1e-8) - 1) if len(close_prices) >= 10 else 0.0)
    
    # Trend
    features.append(1.0 if close_prices[-1] > close_prices[-5] else 0.0)
    features.append(1.0 if close_prices[-1] > mean_price else 0.0)
    
    # Range
    features.append(float((np.max(close_prices) - np.min(close_prices)) / (mean_price + 1e-8)))
    
    # Pad to 15 features
    while len(features) < 15:
        features.append(0.0)
    
    return np.array(features[:15], dtype=np.float32)

def create_hybrid_dataset(data, window_size=30, image_size=30, horizon=5, augment=True):
    """Generate dataset with BOTH visual and numerical features"""
    print("🖼️ Generating hybrid dataset (visual + numerical)...")
    dataset = []
    
    for ticker, df in data.items():
        df = df.copy()
        df['Return'] = df['Close'].pct_change(horizon).shift(-horizon)
        
        for i in range(window_size, len(df) - horizon, 5):
            window = df.iloc[i-window_size:i]
            future_return = df['Return'].iloc[i]
            
            if pd.isna(future_return):
                continue
            
            # Labels
            if future_return > 0.03:
                label = 0  # BUY
            elif future_return < -0.03:
                label = 2  # SELL
            else:
                label = 1  # HOLD
            
            # Visual features
            gasf_img = generate_gasf(window, window_size, image_size, augment=False)
            
            # Numerical features
            numerical = generate_numerical_features(window, window_size)
            
            dataset.append({
                'image': gasf_img,
                'numerical': numerical,
                'label': label,
                'return': future_return,
                'ticker': ticker
            })
            
            # Augmented sample
            if augment:
                gasf_img_aug = generate_gasf(window, window_size, image_size, augment=True)
                dataset.append({
                    'image': gasf_img_aug,
                    'numerical': numerical,  # Same numerical features
                    'label': label,
                    'return': future_return,
                    'ticker': ticker
                })
    
    print(f"✅ Generated {len(dataset)} samples")
    labels = [d['label'] for d in dataset]
    print(f"   BUY: {labels.count(0)} | HOLD: {labels.count(1)} | SELL: {labels.count(2)}\n")
    
    return dataset

#===============================================================================
# MODEL ARCHITECTURES
#===============================================================================

class VisualOnlyCNN(nn.Module):
    """Pure visual model (baseline)"""
    
    def __init__(self, in_channels=5, hidden_dim=128):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2)
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
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, 3)
        )
    
    def forward(self, x, numerical=None):
        # numerical ignored (for compatibility)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
    
    def get_features(self, x):
        """Extract 256-D visual features"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x.view(x.size(0), -1)

class NumericalOnlyMLP(nn.Module):
    """Pure numerical model (baseline)"""
    
    def __init__(self, input_dim=15, hidden_dim=128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, image=None, numerical=None):
        # image ignored (for compatibility)
        return self.network(numerical)

class SimpleConcatFusion(nn.Module):
    """Simple concatenation fusion (Late-Fusion)"""
    
    def __init__(self, visual_backbone, numerical_dim=15, hidden_dim=128):
        super().__init__()
        self.visual_backbone = visual_backbone
        self.visual_dim = 256  # From CNN
        
        self.fusion = nn.Sequential(
            nn.Linear(self.visual_dim + numerical_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, image, numerical):
        visual_feat = self.visual_backbone.get_features(image)
        fused = torch.cat([visual_feat, numerical], dim=1)
        return self.fusion(fused)

class AttentionFusion(nn.Module):
    """Attention-weighted fusion (learns which modality to trust)"""
    
    def __init__(self, visual_backbone, numerical_dim=15, hidden_dim=128):
        super().__init__()
        self.visual_backbone = visual_backbone
        self.visual_dim = 256
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(self.visual_dim + numerical_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 2),  # 2 attention weights (visual, numerical)
            nn.Softmax(dim=1)
        )
        
        self.visual_proj = nn.Linear(self.visual_dim, hidden_dim)
        self.numerical_proj = nn.Linear(numerical_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, image, numerical):
        visual_feat = self.visual_backbone.get_features(image)
        
        # Compute attention weights
        concat = torch.cat([visual_feat, numerical], dim=1)
        attn_weights = self.attention(concat)  # [batch, 2]
        
        # Project to same dimension
        visual_proj = self.visual_proj(visual_feat)
        numerical_proj = self.numerical_proj(numerical)
        
        # Weighted combination
        fused = attn_weights[:, 0:1] * visual_proj + attn_weights[:, 1:2] * numerical_proj
        
        return self.classifier(fused)

class GatedFusion(nn.Module):
    """Gated fusion (learns when to use each modality)"""
    
    def __init__(self, visual_backbone, numerical_dim=15, hidden_dim=128):
        super().__init__()
        self.visual_backbone = visual_backbone
        self.visual_dim = 256
        
        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(self.visual_dim + numerical_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.Sigmoid()  # Gate values in [0, 1]
        )
        
        self.visual_proj = nn.Linear(self.visual_dim, hidden_dim)
        self.numerical_proj = nn.Linear(numerical_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 3)
        )
    
    def forward(self, image, numerical):
        visual_feat = self.visual_backbone.get_features(image)
        
        # Compute gate
        concat = torch.cat([visual_feat, numerical], dim=1)
        gate = self.gate(concat)
        
        # Project and gate
        visual_proj = self.visual_proj(visual_feat)
        numerical_proj = self.numerical_proj(numerical)
        
        fused = gate * visual_proj + (1 - gate) * numerical_proj
        
        return self.classifier(fused)

#===============================================================================
# DATASET WRAPPER
#===============================================================================

class HybridDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return (
            torch.FloatTensor(sample['image']),
            torch.FloatTensor(sample['numerical']),
            torch.LongTensor([sample['label']]),
            torch.FloatTensor([sample['return']])
        )

#===============================================================================
# TRAINING FUNCTIONS
#===============================================================================

def train_model(model, train_loader, val_loader, config, device):
    """Train any model with early stopping"""
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config.get('weight_decay', 1e-4))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    best_state = None
    patience = config.get('patience', 7)
    patience_counter = 0
    
    history = {'train_loss': [], 'val_acc': []}
    
    for epoch in range(config['epochs']):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0
        
        for images, numerical, labels, returns in train_loader:
            if images.size(0) == 0:
                continue
            
            images = images.to(device)
            numerical = numerical.to(device)
            labels = labels.to(device).squeeze()
            
            optimizer.zero_grad()
            
            if config['model_type'] == 'visual_only':
                outputs = model(images)
            elif config['model_type'] == 'numerical_only':
                outputs = model(numerical=numerical)
            else:
                outputs = model(images, numerical)
            
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Validate
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, numerical, labels, returns in val_loader:
                if images.size(0) == 0:
                    continue
                
                images = images.to(device)
                numerical = numerical.to(device)
                labels = labels.to(device).squeeze()
                
                if config['model_type'] == 'visual_only':
                    outputs = model(images)
                elif config['model_type'] == 'numerical_only':
                    outputs = model(numerical=numerical)
                else:
                    outputs = model(images, numerical)
                
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        val_acc = correct / total if total > 0 else 0
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{config['epochs']} | Loss: {avg_train_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | Patience: {patience_counter}/{patience}")
        
        if patience_counter >= patience:
            print(f"    🛑 Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_acc, history

#===============================================================================
# ENSEMBLE METHODS
#===============================================================================

def train_ensemble_voting(models, train_loader, val_loader, device):
    """Learn optimal ensemble weights via logistic regression"""
    print("  🗳️ Training ensemble voting...")
    
    # Collect predictions from all models
    all_preds = []
    all_labels = []
    
    for model in models:
        model.eval()
        preds = []
        
        with torch.no_grad():
            for images, numerical, labels, returns in val_loader:
                images = images.to(device)
                numerical = numerical.to(device)
                
                if hasattr(model, 'visual_backbone'):
                    outputs = model(images, numerical)
                elif isinstance(model, NumericalOnlyMLP):
                    outputs = model(numerical=numerical)
                else:
                    outputs = model(images)
                
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds.append(probs)
        
        all_preds.append(np.vstack(preds))
    
    # Get labels
    with torch.no_grad():
        for images, numerical, labels, returns in val_loader:
            all_labels.extend(labels.squeeze().cpu().numpy())
    
    all_labels = np.array(all_labels)
    
    # Stack predictions as features: [N, num_models * 3]
    X = np.hstack(all_preds)
    
    # Train logistic regression to learn weights
    lr = LogisticRegression(max_iter=1000, multi_class='multinomial')
    lr.fit(X, all_labels)
    
    # Get accuracy
    ensemble_acc = lr.score(X, all_labels)
    
    print(f"    ✓ Ensemble accuracy: {ensemble_acc:.4f}")
    
    return lr, ensemble_acc

#===============================================================================
# GENETIC ALGORITHM FOR PATTERN DISCOVERY
#===============================================================================

class TradingRule:
    """Trading rule with evolvable parameters"""
    
    def __init__(self):
        self.genes = {
            'rsi_period': random.choice([7, 10, 14, 21, 28]),
            'rsi_buy': random.uniform(20, 40),
            'rsi_sell': random.uniform(60, 80),
            'macd_threshold': random.uniform(0, 0.5),
            'volume_mult': random.uniform(0.8, 1.5),
            'use_visual': random.choice([True, False]),
            'use_numerical': random.choice([True, False]),
            'confidence_threshold': random.uniform(0.5, 0.9)
        }
    
    def mutate(self):
        """Randomly mutate one gene"""
        gene = random.choice(list(self.genes.keys()))
        
        if gene == 'rsi_period':
            self.genes[gene] = random.choice([7, 10, 14, 21, 28])
        elif gene in ['rsi_buy', 'rsi_sell']:
            self.genes[gene] += random.gauss(0, 5)
            self.genes[gene] = np.clip(self.genes[gene], 20, 80)
        elif gene == 'macd_threshold':
            self.genes[gene] += random.gauss(0, 0.1)
            self.genes[gene] = np.clip(self.genes[gene], 0, 1)
        elif gene == 'volume_mult':
            self.genes[gene] += random.gauss(0, 0.1)
            self.genes[gene] = np.clip(self.genes[gene], 0.5, 2.0)
        elif gene == 'confidence_threshold':
            self.genes[gene] += random.gauss(0, 0.05)
            self.genes[gene] = np.clip(self.genes[gene], 0.3, 0.95)
        else:
            self.genes[gene] = not self.genes[gene]
    
    def crossover(self, other):
        """Create child from two parents"""
        child = TradingRule()
        for key in self.genes:
            child.genes[key] = random.choice([self.genes[key], other.genes[key]])
        return child

def evaluate_rule_fitness(rule, dataset, models, device):
    """Evaluate trading rule on validation data"""
    correct = 0
    total = 0
    
    for sample in dataset[:500]:  # Use subset for speed
        image = torch.FloatTensor(sample['image']).unsqueeze(0).to(device)
        numerical = torch.FloatTensor(sample['numerical']).unsqueeze(0).to(device)
        true_label = sample['label']
        
        # Get model predictions based on rule
        predictions = []
        
        if rule.genes['use_visual'] and len(models) > 0:
            visual_model = models[0]
            visual_model.eval()
            with torch.no_grad():
                visual_out = visual_model(image)
                visual_probs = F.softmax(visual_out, dim=1)[0]
                if visual_probs.max().item() > rule.genes['confidence_threshold']:
                    predictions.append(visual_probs.argmax().item())
        
        if rule.genes['use_numerical'] and len(models) > 1:
            numerical_model = models[1]
            numerical_model.eval()
            with torch.no_grad():
                num_out = numerical_model(numerical=numerical)
                num_probs = F.softmax(num_out, dim=1)[0]
                if num_probs.max().item() > rule.genes['confidence_threshold']:
                    predictions.append(num_probs.argmax().item())
        
        # Voting
        if predictions:
            pred = max(set(predictions), key=predictions.count)
            if pred == true_label:
                correct += 1
        total += 1
    
    return correct / total if total > 0 else 0

def run_genetic_algorithm(models, dataset, device, generations=30, population_size=20):
    """Run GA to discover best trading rules"""
    print("  🧬 Running Genetic Algorithm for pattern discovery...")
    
    population = [TradingRule() for _ in range(population_size)]
    best_rule = None
    best_fitness = 0
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [evaluate_rule_fitness(rule, dataset, models, device) 
                         for rule in population]
        
        # Track best
        max_idx = np.argmax(fitness_scores)
        if fitness_scores[max_idx] > best_fitness:
            best_fitness = fitness_scores[max_idx]
            best_rule = population[max_idx]
        
        if (gen + 1) % 5 == 0:
            print(f"    Gen {gen+1}/{generations} | Best fitness: {best_fitness:.4f}")
        
        # Selection
        sorted_idx = np.argsort(fitness_scores)[::-1]
        survivors = [population[i] for i in sorted_idx[:population_size // 5]]
        
        # Reproduction
        offspring = []
        for _ in range(population_size - len(survivors)):
            parent1 = random.choice(survivors)
            parent2 = random.choice(survivors)
            child = parent1.crossover(parent2)
            if random.random() < 0.8:
                child.mutate()
            offspring.append(child)
        
        population = survivors + offspring
    
    print(f"    ✓ Best rule fitness: {best_fitness:.4f}")
    print(f"    ✓ Best genes: {best_rule.genes}")
    
    return best_rule, best_fitness

#===============================================================================
# MAIN EXPERIMENT SUITE
#===============================================================================

def run_experiment_suite():
    """Run complete experimentation suite"""
    
    print("\n" + "="*80)
    print("🧪 HYBRID EXPERIMENTATION SUITE - Complete Testing Framework")
    print("="*80)
    print("Testing: Visual | Numerical | Simple Fusion | Attention | Gated | Ensemble | GA")
    print("="*80 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}\n")
    
    # Initialize trackers
    tracker = ExperimentTracker('hybrid_experiments')
    
    # Check for checkpoint
    checkpoint = tracker.load_checkpoint()
    if checkpoint:
        print("📂 Found checkpoint, resuming from previous run...\n")
        data = checkpoint['data']
        train_data = checkpoint['train_data']
        val_data = checkpoint['val_data']
        test_data = checkpoint['test_data']
        completed_experiments = checkpoint['completed_experiments']
    else:
        print("🆕 Starting fresh experiment suite...\n")
        # Download data
        raw_data = download_data()
        
        # Generate hybrid dataset
        data = create_hybrid_dataset(raw_data, augment=True)
        
        # Split data
        train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        print(f"📊 Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}\n")
        
        completed_experiments = []
    
    # Create dataloaders
    train_loader = DataLoader(HybridDataset(train_data), batch_size=32, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(HybridDataset(val_data), batch_size=32, shuffle=False, num_workers=2, drop_last=True)
    test_loader = DataLoader(HybridDataset(test_data), batch_size=32, shuffle=False, num_workers=2, drop_last=True)
    
    # Define experiments
    experiments = [
        {
            'name': 'Visual_Only_Baseline',
            'model_class': VisualOnlyCNN,
            'model_args': {'in_channels': 5, 'hidden_dim': 128},
            'model_type': 'visual_only',
            'lr': 0.001,
            'epochs': 20,
            'weight_decay': 1e-4,
            'patience': 7
        },
        {
            'name': 'Numerical_Only_Baseline',
            'model_class': NumericalOnlyMLP,
            'model_args': {'input_dim': 15, 'hidden_dim': 128},
            'model_type': 'numerical_only',
            'lr': 0.001,
            'epochs': 20,
            'weight_decay': 1e-4,
            'patience': 7
        },
        {
            'name': 'Simple_Concat_Fusion',
            'model_class': SimpleConcatFusion,
            'requires_visual_backbone': True,
            'model_type': 'hybrid',
            'lr': 0.001,
            'epochs': 20,
            'weight_decay': 1e-4,
            'patience': 7
        },
        {
            'name': 'Attention_Fusion',
            'model_class': AttentionFusion,
            'requires_visual_backbone': True,
            'model_type': 'hybrid',
            'lr': 0.001,
            'epochs': 20,
            'weight_decay': 1e-4,
            'patience': 7
        },
        {
            'name': 'Gated_Fusion',
            'model_class': GatedFusion,
            'requires_visual_backbone': True,
            'model_type': 'hybrid',
            'lr': 0.001,
            'epochs': 20,
            'weight_decay': 1e-4,
            'patience': 7
        }
    ]
    
    trained_models = []
    
    # Run experiments
    print("="*80)
    print("🔬 PHASE 1: Individual Model Training")
    print("="*80 + "\n")
    
    for exp in experiments:
        if exp['name'] in completed_experiments:
            print(f"⏭️  Skipping {exp['name']} (already completed)\n")
            continue
        
        print(f"🧪 Experiment: {exp['name']}")
        print(f"   Type: {exp['model_type']}")
        
        # Create model
        if exp.get('requires_visual_backbone'):
            visual_backbone = VisualOnlyCNN(in_channels=5, hidden_dim=128).to(device)
            # Pre-train visual backbone
            print("   Pre-training visual backbone...")
            visual_config = {
                'model_type': 'visual_only',
                'lr': 0.001,
                'epochs': 15,
                'weight_decay': 1e-4,
                'patience': 5
            }
            visual_backbone, _, _ = train_model(visual_backbone, train_loader, val_loader, visual_config, device)
            model = exp['model_class'](visual_backbone, numerical_dim=15, hidden_dim=128).to(device)
        else:
            model = exp['model_class'](**exp['model_args']).to(device)
        
        # Train model
        model, best_acc, history = train_model(model, train_loader, val_loader, exp, device)
        
        # Save results
        metrics = {
            'accuracy': best_acc,
            'final_train_loss': history['train_loss'][-1],
            'epochs_trained': len(history['train_loss'])
        }
        tracker.save_result(exp, metrics)
        
        # Save model
        model_path = CHECKPOINT_DIR / f"{exp['name']}_model.pth"
        torch.save(model.state_dict(), model_path)
        
        trained_models.append((exp['name'], model, exp['model_type']))
        completed_experiments.append(exp['name'])
        
        # Save checkpoint
        tracker.save_checkpoint({
            'data': data,
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'completed_experiments': completed_experiments
        })
        
        print(f"  ✅ Completed: {exp['name']} → {best_acc:.4f}\n")
    
    # Phase 2: Ensemble methods
    print("\n" + "="*80)
    print("🔬 PHASE 2: Ensemble Methods")
    print("="*80 + "\n")
    
    if 'ensemble_voting' not in completed_experiments and len(trained_models) >= 2:
        print("🧪 Experiment: Ensemble_Voting")
        models_for_ensemble = [m[1] for m in trained_models[:3]]  # Top 3 models
        ensemble_model, ensemble_acc = train_ensemble_voting(models_for_ensemble, train_loader, val_loader, device)
        
        metrics = {'accuracy': ensemble_acc, 'num_models': len(models_for_ensemble)}
        tracker.save_result({'name': 'Ensemble_Voting', 'models': [m[0] for m in trained_models[:3]]}, metrics)
        
        # Save ensemble
        ensemble_path = CHECKPOINT_DIR / "ensemble_voting_model.pkl"
        with open(ensemble_path, 'wb') as f:
            pickle.dump(ensemble_model, f)
        
        completed_experiments.append('ensemble_voting')
        print(f"  ✅ Completed: Ensemble_Voting → {ensemble_acc:.4f}\n")
    
    # Phase 3: Genetic Algorithm
    print("\n" + "="*80)
    print("🔬 PHASE 3: Genetic Algorithm Pattern Discovery")
    print("="*80 + "\n")
    
    if 'genetic_algorithm' not in completed_experiments:
        print("🧪 Experiment: Genetic_Algorithm")
        models_for_ga = [m[1] for m in trained_models[:2]]
        best_rule, ga_fitness = run_genetic_algorithm(models_for_ga, val_data, device, generations=30, population_size=20)
        
        metrics = {'fitness': ga_fitness, 'genes': best_rule.genes}
        tracker.save_result({'name': 'Genetic_Algorithm'}, metrics)
        
        # Save best rule
        rule_path = CHECKPOINT_DIR / "best_trading_rule.json"
        with open(rule_path, 'w') as f:
            json.dump(best_rule.genes, f, indent=2)
        
        completed_experiments.append('genetic_algorithm')
        print(f"  ✅ Completed: Genetic_Algorithm → {ga_fitness:.4f}\n")
    
    # Final summary
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    print(tracker.get_summary())
    
    # Save final report
    report_path = RESULTS_DIR / 'FINAL_REPORT.txt'
    with open(report_path, 'w') as f:
        f.write(tracker.get_summary())
    
    print(f"\n✅ Complete experiment suite finished!")
    print(f"📁 Results saved to: {RESULTS_DIR}")
    print(f"💾 Checkpoints saved to: {CHECKPOINT_DIR}")
    print("="*80 + "\n")

if __name__ == "__main__":
    run_experiment_suite()
