#!/bin/bash
# AI Council Lab Setup Script
# Run this first: bash setup.sh

echo "========================================="
echo "AI COUNCIL LAB - ENVIRONMENT SETUP"
echo "========================================="

# Create venv if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Download spaCy model for NLP
echo "Downloading spaCy English model..."
python -m spacy download en_core_web_sm

echo ""
echo "========================================="
echo "✅ SETUP COMPLETE!"
echo "========================================="
echo ""
echo "To activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "To start Jupyter Lab:"
echo "  jupyter lab"
echo ""
echo "Open: AI_COUNCIL_MASTER_NOTEBOOK.ipynb"
echo "========================================="
