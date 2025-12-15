# 🔄 Sync Files to Shadow PC (Local Setup)

Since you **own Shadow PC**, it's best to have all files locally for speed and reliability.

## Option 1: Git Clone/Pull (RECOMMENDED) ✅

```bash
# On Shadow PC, open PowerShell or Git Bash
cd C:\Users\Shadow
git clone https://github.com/alexpayne556-collab/quantum-ai-trader_v1.1.git

# Or if already cloned, update:
cd C:\Users\Shadow\quantum-ai-trader_v1.1
git pull origin main
```

## Option 2: GitHub Desktop (Easiest for Windows)

1. Install [GitHub Desktop](https://desktop.github.com/)
2. Clone repository to `C:\Users\Shadow\quantum-ai-trader_v1.1`
3. Click "Fetch origin" to sync latest changes

## Option 3: Direct File Sync

If repo is already somewhere else on Shadow PC:
```powershell
# Copy entire folder
xcopy /E /I /Y "C:\Users\alexj\Desktop\shadow_ai\quantum-ai-trader_v1.1" "C:\Users\Shadow\quantum-ai-trader_v1.1"
```

## After Syncing:

1. **Verify ticker file exists:**
   ```powershell
   dir "C:\Users\Shadow\quantum-ai-trader_v1.1\data\ticker_universe_300.csv"
   ```
   Should show: **353 lines** (1 header + 352 data rows + blank)

2. **Open notebook in JupyterLab:**
   - Navigate to: `C:\Users\Shadow\quantum-ai-trader_v1.1\notebooks\02_systematic_research_engine.ipynb`
   - Run Cell 3 (Configuration)
   - Should show: `✅ Workspace: C:\Users\Shadow\quantum-ai-trader_v1.1`
   - Should show: `✅ File exists: True`

3. **Run cells 1-7** to validate setup

## Why Local is Better:

✅ **Speed:** No network latency for file operations  
✅ **GPU:** All data processing happens locally on your RTX GPU  
✅ **Reliable:** Works even if internet drops  
✅ **Ownership:** You control all the data  
✅ **Cost:** No ongoing cloud storage costs  

## What Gets Synced:

- ✅ 353-ticker universe CSV
- ✅ Research notebook (11 phases, 20+ cells)
- ✅ All watchlists (alpha_76, merged, small_caps, etc.)
- ✅ Existing Python modules (config, pattern extractors, etc.)
- ✅ All documentation and guides

## Next Steps:

1. Clone/sync repo to `C:\Users\Shadow\quantum-ai-trader_v1.1`
2. Verify ticker file: 353 tickers
3. Open notebook, run cells 1-7
4. You're ready to process all 353 stocks! 🚀
