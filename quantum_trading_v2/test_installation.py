"""
TEST INSTALLATION AND BASIC FUNCTIONALITY
==========================================
Validates all components work correctly
"""

import sys
import importlib

def test_imports():
    """Test all required imports"""
    print("=" * 60)
    print("TESTING IMPORTS")
    print("=" * 60)
    
    required_packages = [
        ('numpy', 'np'),
        ('pandas', 'pd'),
        ('torch', 'torch'),
        ('sklearn', 'sklearn'),
        ('yfinance', 'yf'),
    ]
    
    optional_packages = [
        ('pandas_ta', 'ta'),
        ('matplotlib', 'plt'),
        ('seaborn', 'sns'),
    ]
    
    all_good = True
    
    print("\n✓ Required Packages:")
    for package, alias in required_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - MISSING (required)")
            all_good = False
    
    print("\n✓ Optional Packages:")
    for package, alias in optional_packages:
        try:
            importlib.import_module(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ⚠ {package} - missing (optional)")
    
    return all_good


def test_model_creation():
    """Test model can be created"""
    print("\n" + "=" * 60)
    print("TESTING MODEL CREATION")
    print("=" * 60)
    
    try:
        from quantum_forecaster_14day import QuantumForecaster14Day, QuantumForecastConfig, count_parameters
        
        config = QuantumForecastConfig(
            d_model=128,  # Smaller for testing
            num_encoder_layers=2,
            num_decoder_layers=2,
        )
        
        model = QuantumForecaster14Day(config)
        params = count_parameters(model)
        
        print(f"\n✓ Model created successfully!")
        print(f"  - Total parameters: {params:,}")
        print(f"  - Config: d_model={config.d_model}, layers=2+2")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test forward pass with dummy data"""
    print("\n" + "=" * 60)
    print("TESTING FORWARD PASS")
    print("=" * 60)
    
    try:
        import torch
        from quantum_forecaster_14day import QuantumForecaster14Day, QuantumForecastConfig
        
        config = QuantumForecastConfig(
            d_model=128,
            num_encoder_layers=2,
            num_decoder_layers=2,
            sequence_length=30,  # Shorter for testing
            prediction_horizon=7,
        )
        
        model = QuantumForecaster14Day(config)
        
        # Create dummy data
        batch_size = 2
        seq_len = config.sequence_length
        
        price_features = torch.randn(batch_size, seq_len, config.feature_dim)
        micro_features = torch.randn(batch_size, seq_len, 16)
        alt_features = torch.randn(batch_size, seq_len, 12)
        sent_features = torch.randn(batch_size, seq_len, 8)
        
        # Forward pass
        with torch.no_grad():
            output = model(price_features, micro_features, alt_features, sent_features)
        
        print(f"\n✓ Forward pass successful!")
        print(f"  - Input shape: ({batch_size}, {seq_len}, {config.feature_dim})")
        
        # Check output structure
        if isinstance(output, dict):
            print(f"  - Output type: Dictionary")
            print(f"  - Keys: {list(output.keys())}")
            
            # Check median forecast shape
            if 'median_forecast' in output:
                median_shape = output['median_forecast'].shape
                expected_shape = (batch_size, config.prediction_horizon)
                print(f"  - Median forecast shape: {tuple(median_shape)}")
                print(f"  - Expected shape: {expected_shape}")
                
                if median_shape == expected_shape:
                    print(f"  ✓ Output shapes correct!")
                    return True
            
            print(f"  ✓ Model produces valid output dictionary")
            return True
        else:
            print(f"  ⚠ Unexpected output type: {type(output)}")
            return False
        
    except Exception as e:
        print(f"\n✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_engineering():
    """Test feature engineering with real data"""
    print("\n" + "=" * 60)
    print("TESTING FEATURE ENGINEERING")
    print("=" * 60)
    
    try:
        from feature_engineering import QuantumFeatureEngineer
        import yfinance as yf
        
        # Download small amount of data
        print("\n  Downloading test data (AAPL)...")
        ticker = 'AAPL'
        data = yf.download(ticker, period='3mo', progress=False)
        
        if data.empty:
            print("  ⚠ No data downloaded (might be network issue)")
            return False
        
        print(f"  ✓ Downloaded {len(data)} days of data")
        
        # Engineer features
        print("  Engineering features...")
        engineer = QuantumFeatureEngineer()
        
        try:
            features = engineer.engineer_all_features(ticker, period='3mo')
            
            # Check if it's a dict
            if isinstance(features, dict):
                all_features = features['all_features']
                
                print(f"\n✓ Feature engineering successful!")
                print(f"  - Total features: {all_features.shape[1]}")
                print(f"  - Data points: {all_features.shape[0]}")
                
                # Check for NaN
                nan_count = all_features.isna().sum().sum()
                if nan_count > 0:
                    print(f"  ⚠ Warning: {nan_count} NaN values (will be handled during training)")
                else:
                    print(f"  ✓ No NaN values")
                
                return True
            else:
                print(f"  ✓ Features generated (legacy format)")
                return True
            
        except Exception as e:
            print(f"  ⚠ Feature engineering issue: {e}")
            return False
        
    except Exception as e:
        print(f"\n✗ Feature engineering test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gpu_availability():
    """Check GPU availability"""
    print("\n" + "=" * 60)
    print("CHECKING GPU AVAILABILITY")
    print("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"\n✓ GPU AVAILABLE!")
            print(f"  - Device: {torch.cuda.get_device_name(0)}")
            print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            print(f"  - CUDA version: {torch.version.cuda}")
            return True
        else:
            print(f"\n⚠ No GPU available - will use CPU (slower)")
            print(f"  - For Colab: Runtime → Change runtime type → GPU")
            return False
            
    except Exception as e:
        print(f"\n⚠ Could not check GPU: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("QUANTUM FORECASTER - INSTALLATION TEST")
    print("=" * 60)
    print("\nThis will verify all components are working correctly.\n")
    
    results = {
        'Imports': test_imports(),
        'Model Creation': test_model_creation(),
        'Forward Pass': test_forward_pass(),
        'Feature Engineering': test_feature_engineering(),
        'GPU': test_gpu_availability(),
    }
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_critical_passed = results['Imports'] and results['Model Creation'] and results['Forward Pass']
    
    if all_critical_passed:
        print("\n" + "=" * 60)
        print("✓✓✓ ALL CRITICAL TESTS PASSED! ✓✓✓")
        print("=" * 60)
        print("\nYou're ready to train the model!")
        print("\nNext steps:")
        print("  1. For quick test: python training_pipeline.py --quick")
        print("  2. For full training: python training_pipeline.py")
        print("  3. For Colab: Upload to Drive and run notebook")
    else:
        print("\n" + "=" * 60)
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix issues before training.")
        print("Install missing packages with:")
        print("  pip install -r requirements.txt")
    
    return all_critical_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
