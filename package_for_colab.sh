#!/bin/bash
# Package local models for upload to Google Colab

echo "📦 Packaging local models for Google Colab upload..."
echo ""

cd /workspaces/quantum-ai-trader_v1.1

# Create temporary directory
rm -rf /tmp/quantum_trader_package
mkdir -p /tmp/quantum_trader_package

# Copy trained models
if [ -d "trained_models" ]; then
    echo "✅ Copying trained_models/ ($(du -sh trained_models | cut -f1))"
    cp -r trained_models /tmp/quantum_trader_package/
fi

# Copy training results
if [ -d "training_results" ]; then
    echo "✅ Copying training_results/ ($(du -sh training_results | cut -f1))"
    cp -r training_results /tmp/quantum_trader_package/
fi

# Create ZIP file
echo ""
echo "🗜️  Creating quantum_trader_local.zip..."
cd /tmp/quantum_trader_package
zip -r /workspaces/quantum-ai-trader_v1.1/quantum_trader_local.zip . -q

cd /workspaces/quantum-ai-trader_v1.1
SIZE=$(du -h quantum_trader_local.zip | cut -f1)
echo ""
echo "✅ Package created: quantum_trader_local.zip ($SIZE)"
echo ""
echo "📤 NEXT STEPS:"
echo "1. Download quantum_trader_local.zip from VS Code"
echo "2. Open COLAB_PRO_HYPERPARAMETER_OPTIMIZATION.ipynb in Google Colab"
echo "3. In Step 1.5, uncomment the upload code and run the cell"
echo "4. Upload quantum_trader_local.zip when prompted"
echo ""
echo "OR simply train from scratch (recommended for best results!)"
