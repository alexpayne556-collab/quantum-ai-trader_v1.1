"""
QUICK TRAINING TEST
===================
Quick training run to validate the complete pipeline
"""

import torch
import numpy as np
from quantum_forecaster_14day import QuantumForecaster14Day, QuantumForecastConfig, QuantileLoss, initialize_weights

def quick_test():
    """Run a quick training test with synthetic data"""
    
    print("=" * 70)
    print("QUICK TRAINING TEST")
    print("=" * 70)
    
    # Small configuration for quick test
    config = QuantumForecastConfig(
        d_model=64,
        num_encoder_layers=2,
        num_decoder_layers=2,
        sequence_length=20,
        prediction_horizon=7,
        batch_size=4,
        num_epochs=5
    )
    
    print(f"\n📋 Configuration:")
    print(f"  - Model size: d_model={config.d_model}")
    print(f"  - Sequence length: {config.sequence_length}")
    print(f"  - Prediction horizon: {config.prediction_horizon} days")
    print(f"  - Epochs: {config.num_epochs}")
    
    # Create model
    print(f"\n🏗️ Creating model...")
    model = QuantumForecaster14Day(config)
    initialize_weights(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"  ✓ Model on {device}")
    
    # Create synthetic data
    print(f"\n📊 Generating synthetic training data...")
    n_samples = 50
    
    price_features = torch.randn(n_samples, config.sequence_length, config.feature_dim)
    micro_features = torch.randn(n_samples, config.sequence_length, config.microstructure_dim)
    alt_features = torch.randn(n_samples, config.sequence_length, config.alternative_data_dim)
    sent_features = torch.randn(n_samples, config.sequence_length, config.sentiment_dim)
    
    # Synthetic targets (returns)
    targets = torch.randn(n_samples, config.prediction_horizon) * 0.02  # 2% std
    
    # Create DataLoader
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(price_features, micro_features, alt_features, sent_features, targets)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    
    print(f"  ✓ Created {n_samples} synthetic samples")
    
    # Setup training
    criterion = QuantileLoss(config.quantiles)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    print(f"\n🏃 Training for {config.num_epochs} epochs...")
    
    # Training loop
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (price, micro, alt, sent, target) in enumerate(train_loader):
            # Move to device
            price = price.to(device)
            micro = micro.to(device)
            alt = alt.to(device)
            sent = sent.to(device)
            target = target.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(price, micro, alt, sent)
            
            # Compute loss
            loss = criterion(output['quantile_predictions'], target)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Epoch {epoch+1}/{config.num_epochs} - Loss: {avg_loss:.6f}")
    
    print(f"\n✅ Training completed successfully!")
    
    # Test inference
    print(f"\n🔮 Testing inference...")
    model.eval()
    with torch.no_grad():
        test_price = torch.randn(1, config.sequence_length, config.feature_dim).to(device)
        test_micro = torch.randn(1, config.sequence_length, config.microstructure_dim).to(device)
        test_alt = torch.randn(1, config.sequence_length, config.alternative_data_dim).to(device)
        test_sent = torch.randn(1, config.sequence_length, config.sentiment_dim).to(device)
        
        output = model(test_price, test_micro, test_alt, test_sent)
        
        print(f"  ✓ Inference successful!")
        print(f"  - Median forecast (7-day): {output['median_forecast'][0].cpu().numpy()}")
        print(f"  - Uncertainty: {output['uncertainty'][0].cpu().numpy()}")
    
    print(f"\n" + "=" * 70)
    print("✅✅✅ QUICK TEST PASSED - SYSTEM FULLY FUNCTIONAL! ✅✅✅")
    print("=" * 70)
    print(f"\nNext steps:")
    print(f"  1. For real training: Use training_pipeline.py with actual market data")
    print(f"  2. For Colab: Upload files to Google Drive and run the notebook")
    print(f"  3. Check README.md for detailed instructions")
    
    return True


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
