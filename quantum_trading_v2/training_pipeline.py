"""
TRAINING PIPELINE FOR QUANTUM 14-DAY FORECASTER
===============================================
Complete training workflow with:
- Multi-ticker portfolio training
- Quantile loss optimization
- Early stopping & checkpointing
- Performance monitoring
- Walk-forward validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from quantum_forecaster_14day import (
    QuantumForecaster14Day, 
    QuantumForecastConfig,
    QuantileLoss,
    initialize_weights,
    count_parameters
)
from feature_engineering import QuantumFeatureEngineer


# ==================== DATA SCALER ====================

class FeatureScaler:
    """Multi-modal feature scaling"""
    
    def __init__(self):
        self.scalers = {
            'price': StandardScaler(),
            'micro': StandardScaler(),
            'alt': StandardScaler(),
            'sent': StandardScaler()
        }
        
    def fit(self, X_price, X_micro, X_alt, X_sent):
        """Fit scalers on training data"""
        # Reshape to 2D for fitting
        self.scalers['price'].fit(X_price.reshape(-1, X_price.shape[-1]))
        self.scalers['micro'].fit(X_micro.reshape(-1, X_micro.shape[-1]))
        self.scalers['alt'].fit(X_alt.reshape(-1, X_alt.shape[-1]))
        self.scalers['sent'].fit(X_sent.reshape(-1, X_sent.shape[-1]))
        return self
    
    def transform(self, X_price, X_micro, X_alt, X_sent):
        """Transform features"""
        batch_size, seq_len, _ = X_price.shape
        
        X_price_scaled = self.scalers['price'].transform(
            X_price.reshape(-1, X_price.shape[-1])
        ).reshape(batch_size, seq_len, -1)
        
        X_micro_scaled = self.scalers['micro'].transform(
            X_micro.reshape(-1, X_micro.shape[-1])
        ).reshape(batch_size, seq_len, -1)
        
        X_alt_scaled = self.scalers['alt'].transform(
            X_alt.reshape(-1, X_alt.shape[-1])
        ).reshape(batch_size, seq_len, -1)
        def __init__(self, config: QuantumForecastConfig, auto_tune: bool = False):
        X_sent_scaled = self.scalers['sent'].transform(
            X_sent.reshape(-1, X_sent.shape[-1])
        ).reshape(batch_size, seq_len, -1)
        
        return X_price_scaled, X_micro_scaled, X_alt_scaled, X_sent_scaled
            self.tuner = AutoTuner(self.config) if auto_tune else None
    def fit_transform(self, X_price, X_micro, X_alt, X_sent):
        """Fit and transform"""
        self.fit(X_price, X_micro, X_alt, X_sent)
        return self.transform(X_price, X_micro, X_alt, X_sent)


# ==================== TRAINING UTILITIES ====================

class EarlyStoppingWithPatience:
    """Early stopping with model checkpointing"""
    
    def __init__(self, patience: int = 15, delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if should stop training"""
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
            if self.verbose:
                print(f'  ✓ Validation loss improved to {val_loss:.6f}')
        
        return self.early_stop


class CosineWarmupScheduler:
    """Learning rate scheduler with warmup"""
    
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, 
                 min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.defaults['lr']
        self.current_epoch = 0
        
    def step(self):
        """Update learning rate"""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        self.current_epoch += 1
        return lr


# ==================== TRAINING PIPELINE ====================

class QuantumForecasterTrainer:
    """Complete training pipeline"""
    
    def __init__(self, config: QuantumForecastConfig, device: str = 'cuda'):
        self.config = config
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"🔧 Initializing trainer on device: {self.device}")
        
        # Initialize model
        self.model = QuantumForecaster14Day(config).to(self.device)
        initialize_weights(self.model)
        
        print(f"📊 Model parameters: {count_parameters(self.model):,}")
        
        # Loss function
        self.criterion = QuantileLoss(config.quantiles)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Feature scaler
        self.scaler = FeatureScaler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
    def prepare_data(self, 
                    tickers: List[str],
                    test_size: float = 0.2,
                    val_size: float = 0.1) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data for all tickers
        
        Args:
            tickers: List of stock symbols
            test_size: Fraction for test set
            val_size: Fraction for validation set
            
        Returns:
            Train, validation, test DataLoaders
        """
        print(f"\n📊 Preparing data for {len(tickers)} tickers...")
        
        engineer = QuantumFeatureEngineer()
        
        all_X_price, all_X_micro, all_X_alt, all_X_sent, all_y = [], [], [], [], []
        
        for ticker in tickers:
            try:
                print(f"  Processing {ticker}...")
                features = engineer.engineer_all_features(ticker, period='5y')
                X_price, X_micro, X_alt, X_sent, y = engineer.prepare_model_inputs(
                    features,
                    sequence_length=self.config.sequence_length,
                    prediction_horizon=self.config.prediction_horizon
                )
                
                all_X_price.append(X_price)
                all_X_micro.append(X_micro)
                all_X_alt.append(X_alt)
                all_X_sent.append(X_sent)
                all_y.append(y)
                
            except Exception as e:
                print(f"  ⚠️  Skipping {ticker}: {str(e)}")
                continue
        
        # Concatenate all tickers
        X_price = np.vstack(all_X_price)
        X_micro = np.vstack(all_X_micro)
        X_alt = np.vstack(all_X_alt)
        X_sent = np.vstack(all_X_sent)
        y = np.concatenate(all_y)
        
        print(f"\n✅ Combined dataset:")
        print(f"  - Total samples: {len(y)}")
        print(f"  - Price features: {X_price.shape}")
        print(f"  - Microstructure: {X_micro.shape}")
        print(f"  - Alternative: {X_alt.shape}")
        print(f"  - Sentiment: {X_sent.shape}")
        
        # Split data (time-series aware - no shuffle)
        n_samples = len(y)
        n_test = int(n_samples * test_size)
        n_val = int(n_samples * val_size)
        n_train = n_samples - n_test - n_val
        
        # Training set
        X_price_train = X_price[:n_train]
        X_micro_train = X_micro[:n_train]
        X_alt_train = X_alt[:n_train]
        X_sent_train = X_sent[:n_train]
        y_train = y[:n_train]
        
        # Validation set
        X_price_val = X_price[n_train:n_train+n_val]
        X_micro_val = X_micro[n_train:n_train+n_val]
        X_alt_val = X_alt[n_train:n_train+n_val]
        X_sent_val = X_sent[n_train:n_train+n_val]
        y_val = y[n_train:n_train+n_val]
        
        # Test set
        X_price_test = X_price[n_train+n_val:]
        X_micro_test = X_micro[n_train+n_val:]
        X_alt_test = X_alt[n_train+n_val:]
        X_sent_test = X_sent[n_train+n_val:]
        y_test = y[n_train+n_val:]
        
        # Fit scaler on training data and transform all sets
        print(f"\n🔧 Fitting scalers...")
        X_price_train, X_micro_train, X_alt_train, X_sent_train = self.scaler.fit_transform(
            X_price_train, X_micro_train, X_alt_train, X_sent_train
        )
        X_price_val, X_micro_val, X_alt_val, X_sent_val = self.scaler.transform(
            X_price_val, X_micro_val, X_alt_val, X_sent_val
        )
        X_price_test, X_micro_test, X_alt_test, X_sent_test = self.scaler.transform(
            X_price_test, X_micro_test, X_alt_test, X_sent_test
        )
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_price_train),
            torch.FloatTensor(X_micro_train),
            torch.FloatTensor(X_alt_train),
            torch.FloatTensor(X_sent_train),
            torch.FloatTensor(y_train).unsqueeze(1).repeat(1, self.config.prediction_horizon)
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_price_val),
            torch.FloatTensor(X_micro_val),
            torch.FloatTensor(X_alt_val),
            torch.FloatTensor(X_sent_val),
            torch.FloatTensor(y_val).unsqueeze(1).repeat(1, self.config.prediction_horizon)
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_price_test),
            torch.FloatTensor(X_micro_test),
            torch.FloatTensor(X_alt_test),
            torch.FloatTensor(X_sent_test),
            torch.FloatTensor(y_test).unsqueeze(1).repeat(1, self.config.prediction_horizon)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        print(f"\n✅ DataLoaders created:")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (X_price, X_micro, X_alt, X_sent, y) in enumerate(train_loader):
            X_price = X_price.to(self.device)
            X_micro = X_micro.to(self.device)
            X_alt = X_alt.to(self.device)
            X_sent = X_sent.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(X_price, X_micro, X_alt, X_sent)
            
            # Compute loss
            loss = self.criterion(output['quantile_predictions'], y)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X_price, X_micro, X_alt, X_sent, y in val_loader:
                X_price = X_price.to(self.device)
                X_micro = X_micro.to(self.device)
                X_alt = X_alt.to(self.device)
                X_sent = X_sent.to(self.device)
                y = y.to(self.device)
                
                output = self.model(X_price, X_micro, X_alt, X_sent)
                loss = self.criterion(output['quantile_predictions'], y)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: Optional[int] = None,
             patience: int = 15) -> Dict:
        """
        Complete training loop
        
        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"\n🚀 Starting training for {num_epochs} epochs...")
        print(f"  - Device: {self.device}")
        print(f"  - Batch size: {self.config.batch_size}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Patience: {patience}")
        
        scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_epochs=self.config.warmup_epochs,
            total_epochs=num_epochs
        )
        
        early_stopping = EarlyStoppingWithPatience(patience=patience)
        
        for epoch in range(num_epochs):
            epoch_start = datetime.now()
            
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update learning rate
             train_loader: DataLoader,
             val_loader: DataLoader,
             num_epochs: Optional[int] = None,
             patience: int = 15) -> Dict:
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s)")
                print(f"  Train Loss: {train_loss:.6f}")
                print(f"  Val Loss:   {val_loss:.6f}")
                print(f"  LR:         {current_lr:.2e}")
            
            # Early stopping check
            if early_stopping(val_loss, self.model):
                print(f"\n⏹️  Early stopping triggered at epoch {epoch+1}")
                break
        
        # Restore best model
        self.model.load_state_dict(early_stopping.best_model_state)
        
        print(f"\n✅ Training complete!")
        print(f"  Best val loss: {early_stopping.best_loss:.6f}")
        
        return self.history
    
    def evaluate(self, test_loader: DataLoader) -> Dict:
        """Evaluate on test set"""
        print(f"\n📊 Evaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for X_price, X_micro, X_alt, X_sent, y in test_loader:
                X_price = X_price.to(self.device)
                X_micro = X_micro.to(self.device)
                X_alt = X_alt.to(self.device)
                X_sent = X_sent.to(self.device)
                
                output = self.model(X_price, X_micro, X_alt, X_sent)
                
                all_predictions.append(output['median_forecast'].cpu().numpy())
                all_targets.append(y.cpu().numpy())
                all_uncertainties.append(output['uncertainty'].cpu().numpy())
        
        predictions = np.vstack(all_predictions)
        targets = np.vstack(all_targets)
        uncertainties = np.vstack(all_uncertainties)
        
        # Calculate metrics
        mae = np.abs(predictions - targets).mean()
        rmse = np.sqrt(((predictions - targets) ** 2).mean())
        
        # Directional accuracy
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        directional_acc = (pred_direction == true_direction).mean()
        
        results = {
            'mae': float(mae),
            'rmse': float(rmse),
            'directional_accuracy': float(directional_acc),
            'mean_uncertainty': float(uncertainties.mean()),
            'predictions': predictions,
            'targets': targets,
            'uncertainties': uncertainties
        }
        
        print(f"\n📈 Test Results:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Directional Accuracy: {directional_acc:.2%}")
        print(f"  Mean Uncertainty: {uncertainties.mean():.4f}")
        
        return results
    
    def save_model(self, save_path: str):
        """Save model and configuration"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'history': self.history
        }, save_path / 'model.pth')
        
        # Save scaler
        import pickle
        with open(save_path / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"\n💾 Model saved to {save_path}")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training & Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Learning rate
        axes[0, 1].plot(self.history['learning_rate'])
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Learning Rate')
        axes[0, 1].set_title('Learning Rate Schedule')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Epoch time
        axes[1, 0].plot(self.history['epoch_time'])
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].set_title('Training Time per Epoch')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Loss distribution
        axes[1, 1].hist(self.history['train_loss'], bins=30, alpha=0.5, label='Train')
        axes[1, 1].hist(self.history['val_loss'], bins=30, alpha=0.5, label='Val')
        axes[1, 1].set_xlabel('Loss')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Loss Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Training history saved to {save_path}")
        
        plt.show()


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("=" * 70)
    print("QUANTUM 14-DAY FORECASTER - TRAINING PIPELINE")
    print("=" * 70)
    
    # Configuration
    config = QuantumForecastConfig(
        d_model=256,
        nhead=8,
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=50,
        warmup_epochs=5
    )
    
    # Initialize trainer
    trainer = QuantumForecasterTrainer(config)
    
    # Portfolio tickers (start with 5 for testing)
    tickers = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    # Prepare data
    train_loader, val_loader, test_loader = trainer.prepare_data(
        tickers,
        test_size=0.2,
        val_size=0.1
    )
    
    # Train model
    history = trainer.train(
        train_loader,
        val_loader,
        num_epochs=50,
        patience=15
    )
    
    # Evaluate
    results = trainer.evaluate(test_loader)
    
    # Save model
    trainer.save_model('models/quantum_forecaster_v1')
    
    # Plot results
    trainer.plot_training_history('models/training_history.png')
    
    print("\n" + "=" * 70)
    print("✅ Training pipeline complete!")
    print("=" * 70)
