# PowerShell script to set up Python venv and install all dependencies
# Usage: Open PowerShell, navigate to quantum_trading_v2, then run: .\setup_venv.ps1

Write-Host "Creating Python virtual environment..."
python -m venv venv
Write-Host "Activating venv..."
.\venv\Scripts\Activate.ps1
Write-Host "Upgrading pip..."
pip install --upgrade pip
Write-Host "Installing required packages..."
pip install -r requirements.txt
Write-Host "Setup complete! To activate venv later, run: .\venv\Scripts\Activate.ps1"
