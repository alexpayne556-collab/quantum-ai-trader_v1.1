# ===== AI COUNCIL - READY TO GO =====
# Just run this file in PowerShell

cd C:\Users\alexj\Desktop\shadow_ai\quantum-ai-trader_v1.1
.\shadow_ai\Scripts\Activate.ps1
pip install --quiet yfinance plotly hmmlearn
Write-Host "`n✅ Environment ready! Starting Jupyter Lab...`n" -ForegroundColor Green
jupyter lab
