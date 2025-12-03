"""
🔧 INSTALL DARTS PROPERLY
==========================

DARTS needs specific versions to work with Colab.
This installs everything correctly.
"""

print("📦 Installing DARTS and dependencies...")

# Uninstall conflicting packages
!pip uninstall -y pytorch-lightning lightning-fabric darts 2>/dev/null

# Install PyTorch (use latest available for Colab)
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Lightning (compatible with latest)
!pip install -q pytorch-lightning

# Install DARTS with all dependencies
!pip install -q darts==0.27.0

# Install missing DARTS dependencies
!pip install -q pmdarima>=1.8.0 statsforecast>=1.4 tbats>=1.1.0 \
  nfoursid>=1.0.0 pyod>=0.9.5 tensorboardX>=2.1

print("✅ DARTS installed!")

# Test import
try:
    from darts import TimeSeries
    from darts.models import NBEATSModel
    print("✅ DARTS import successful!")
except ImportError as e:
    print(f"❌ DARTS import failed: {e}")
    print("   Try: Runtime → Restart runtime, then rerun this cell")
