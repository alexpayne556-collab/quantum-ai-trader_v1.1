# Shadow PC venv full setup script
# Run in PowerShell as Administrator or with Developer Mode enabled
# Usage: Right-click -> Run with PowerShell (or execute from an opened PS window)

param(
    [string]$EnvName = "shadow_ai",
    [string]$Python = "py -3.11"
)

$ErrorActionPreference = "Stop"

Write-Host "== Creating venv '$EnvName' on Desktop =="
Set-Location "$HOME\Desktop"
& $Python -m venv $EnvName

Write-Host "== Activating venv =="
& ."$HOME\Desktop\$EnvName\Scripts\Activate"

Write-Host "== Upgrading pip =="
python -m pip install -U pip

Write-Host "== Installing GPU PyTorch (CUDA 12.1) =="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Write-Host "== Installing NLP + HF tooling =="
pip install "huggingface_hub[hf_xet]" safetensors accelerate transformers datasets sentencepiece

Write-Host "== Installing data + APIs =="
pip install pandas requests python-dotenv beautifulsoup4 finnhub-python fredapi sec-edgar-api schedule

Write-Host "== Installing Jupyter + registering kernel =="
pip install jupyter jupyterlab ipykernel
python -m ipykernel install --name $EnvName --display-name "Python ($EnvName)"

Write-Host "== Verifying CUDA + GPU =="
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU'); print('Torch:', torch.__version__)"

Write-Host "== Launching JupyterLab (no token) =="
jupyter lab --no-browser --NotebookApp.token='' --NotebookApp.password=''
