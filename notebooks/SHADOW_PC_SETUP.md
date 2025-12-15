# Shadow PC Setup Instructions

## If You Get "File Not Found" Error

The notebook is looking for the ticker universe file. Here's how to fix it:

### Option 1: Copy from Codespace to Shadow PC

1. In Codespace, run:
```bash
cat /workspaces/quantum-ai-trader_v1.1/data/ticker_universe_300.csv
```

2. Copy the output

3. On Shadow PC, create file at:
```
C:\Users\alexj\Desktop\shadow_ai\quantum-ai-trader_v1.1\data\ticker_universe_300.csv
```

4. Paste the content and save

### Option 2: Auto-Download in Notebook

The notebook (cell 6) will automatically try to:
1. Download from GitHub
2. Or create a sample file if download fails

Just run the cells in order and it will handle it!

### Option 3: Use Git Pull

On Shadow PC:
```bash
cd C:\Users\alexj\Desktop\shadow_ai\quantum-ai-trader_v1.1
git pull origin main
```

This will sync the latest ticker_universe_300.csv (353 tickers) from the repo.

## Verify It Worked

After setup, you should see:
```
✅ Universe file found
✅ Contains 353 tickers
```

## Current File Stats
- **353 total tickers**
- 204 Small Cap
- 61 Mid Cap  
- 49 Large Cap
- 39 Micro Cap

All sectors covered: Technology, Healthcare, Aerospace, Energy, Fintech, Crypto, etc.
