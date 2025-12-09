# 🆓 FREE API Keys Setup Guide - Complete Stack

## ✅ What You Already Have (ACTIVE)
- ✅ Alpha Vantage: `9OS7LP4D495FW43S`
- ✅ Polygon: `iRXh2jGpwhcJxGWfW4ZRVn2C4s_v4ghr`
- ✅ Finnhub: `d3qj8p9r01quv7kb49igd3qj8p9r01quv7kb49j0`
- ✅ FMP: `15zYYtksuJnQsTBODSNs3MrfEedOSd3i`
- ✅ EODHD: `68f5419033db54.61168020`
- ✅ Perplexity: `your_perplexity_api_key_hereSugdX6yxqiIorS526CYof8aqlcySXisRbIoNf84BBQ7szSOl`
- ✅ OpenAI: Active (truncated in display)

---

## 🎯 CRITICAL: Get These 3 FREE Keys NOW

### 1. **Alpaca Paper Trading** (CRITICAL - FREE)
**What**: Commission-free paper trading + real-time market data  
**Cost**: 100% FREE (no credit card needed)  
**Limits**: Unlimited paper trading, real-time quotes  
**Signup**: https://alpaca.markets/  

**Steps**:
1. Go to https://alpaca.markets/
2. Click "Get Started Free"
3. Sign up (email verification)
4. Go to Dashboard → API Keys
5. Generate Paper Trading keys
6. Copy both keys:
   - `ALPACA_API_KEY` = `PKxxxxxx`
   - `ALPACA_SECRET_KEY` = `xxxxx`

**Why Critical**: Execute paper trades, real-time quotes, portfolio management

---

### 2. **FRED (Federal Reserve Economic Data)** (CRITICAL - FREE)
**What**: VIX, yield curve, macro indicators for regime detection  
**Cost**: 100% FREE  
**Limits**: Unlimited historical data  
**Signup**: https://fred.stlouisfed.org/docs/api/api_key.html  

**Steps**:
1. Go to https://fred.stlouisfed.org/docs/api/api_key.html
2. Click "Request an API Key"
3. Fill out simple form (name, email, reason)
4. Receive key instantly via email
5. Copy: `FRED_API_KEY` = `xxxxx`

**Why Critical**: Essential for regime classifier (VIX, yield curve, SPY returns)

---

### 3. **Twelve Data** (HIGH PRIORITY - FREE)
**What**: 800 requests/day, 8/minute - excellent backup data source  
**Cost**: 100% FREE  
**Limits**: 800/day, 8/minute (more than enough)  
**Signup**: https://twelvedata.com/pricing  

**Steps**:
1. Go to https://twelvedata.com/pricing
2. Click "Sign Up Free"
3. Confirm email
4. Dashboard → API Key
5. Copy: `TWELVE_DATA_API_KEY` = `xxxxx`

**Why High Priority**: Excellent fallback when Finnhub/Polygon hit limits

---

## 🚀 HIGHLY RECOMMENDED: Get These 5 FREE Keys

### 4. **NewsAPI** (FREE - 100/day)
**What**: News headlines for sentiment analysis  
**Cost**: 100% FREE (100 requests/day)  
**Limits**: 100/day (sufficient for daily checks)  
**Signup**: https://newsapi.org/register  

**Steps**:
1. Go to https://newsapi.org/register
2. Click "Get API Key"
3. Fill form
4. Instant key in dashboard
5. Copy: `NEWS_API_KEY` = `xxxxx`

**Use Case**: Sentiment feature for model training

---

### 5. **CoinGecko** (FREE - NO KEY NEEDED!)
**What**: Bitcoin/crypto data as SPY correlation proxy  
**Cost**: 100% FREE (no signup!)  
**Limits**: 10-30 calls/minute  
**Endpoint**: https://api.coingecko.com/api/v3  

**No signup needed** - just use the public endpoint:
```python
import requests
btc = requests.get('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd')
```

**Use Case**: Bitcoin leads SPY by 6-12 hours sometimes, useful regime indicator

---

### 6. **Quandl (Nasdaq Data Link)** (FREE - Limited)
**What**: Alternative economic data, futures, commodities  
**Cost**: FREE tier available  
**Limits**: 50 calls/day (free tier)  
**Signup**: https://data.nasdaq.com/sign-up  

**Steps**:
1. Go to https://data.nasdaq.com/sign-up
2. Free account signup
3. Account Settings → API Key
4. Copy: `QUANDL_API_KEY` = `xxxxx`

**Use Case**: Commodity prices (oil, gold) for sector-specific features

---

### 7. **IEX Cloud** (FREE SANDBOX)
**What**: Real-time and historical US stock data  
**Cost**: FREE (sandbox mode)  
**Limits**: Sandbox = unlimited, but 15-min delayed  
**Signup**: https://iexcloud.io/cloud-login#/register  

**Steps**:
1. Go to https://iexcloud.io/cloud-login#/register
2. Sign up free
3. Choose "Start" plan (sandbox)
4. Dashboard → API Tokens
5. Copy Sandbox token: `IEX_SANDBOX_KEY` = `xxxxx`

**Use Case**: Another fallback data source, fundamentals

---

### 8. **Tiingo** (FREE - 500/hour)
**What**: High-quality historical data + news  
**Cost**: FREE (500 requests/hour)  
**Limits**: 500/hour (excellent for hourly bars)  
**Signup**: https://www.tiingo.com/account/api/token  

**Steps**:
1. Go to https://www.tiingo.com/
2. Click "Sign Up" (free)
3. Email verification
4. Dashboard → API Key
5. Copy: `TIINGO_API_KEY` = `xxxxx`

**Use Case**: High-quality alternative to yfinance for training data

---

## 📢 NICE-TO-HAVE: Alerts & Notifications (All FREE)

### 9. **Discord Webhook** (FREE - 0 cost)
**What**: Send trade alerts to Discord server  
**Cost**: 100% FREE  
**Limits**: ~30 messages/minute  

**Steps**:
1. Create Discord server (free)
2. Server Settings → Integrations → Webhooks
3. New Webhook → Copy URL
4. Paste: `DISCORD_WEBHOOK_URL` = `https://discord.com/api/webhooks/xxxxx`

**Use Case**: Real-time trade notifications

---

### 10. **Telegram Bot** (FREE - 0 cost)
**What**: Mobile push notifications for trades  
**Cost**: 100% FREE  
**Limits**: Unlimited  

**Steps**:
1. Open Telegram app
2. Search `@BotFather`
3. Send `/newbot`
4. Follow prompts, get token
5. Copy: `TELEGRAM_BOT_TOKEN` = `xxxxx`
6. Send message to bot, then get chat ID:
   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```
7. Copy chat ID from response

**Use Case**: SMS-like alerts on your phone

---

### 11. **Weights & Biases** (FREE for individuals)
**What**: ML experiment tracking, model versioning  
**Cost**: FREE for personal use  
**Limits**: Unlimited projects, 100GB storage  
**Signup**: https://wandb.ai/signup  

**Steps**:
1. Go to https://wandb.ai/signup
2. Sign up free
3. User Settings → API Keys
4. Copy: `WANDB_API_KEY` = `xxxxx`

**Use Case**: Track training metrics, compare models, A/B testing

---

## 📊 COMPLETE .ENV FILE (Copy This)

```bash
# ============================================================================
# CRITICAL KEYS (Get These First)
# ============================================================================

# Alpaca Paper Trading (FREE) - https://alpaca.markets
ALPACA_API_KEY=your_alpaca_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets

# FRED Economic Data (FREE) - https://fred.stlouisfed.org/docs/api/api_key.html
FRED_API_KEY=your_fred_key_here

# Twelve Data (FREE 800/day) - https://twelvedata.com/pricing
TWELVE_DATA_API_KEY=your_twelve_data_key_here

# ============================================================================
# RECOMMENDED KEYS (High Value)
# ============================================================================

# NewsAPI (FREE 100/day) - https://newsapi.org/register
NEWS_API_KEY=your_news_api_key_here

# Quandl/Nasdaq Data Link (FREE 50/day) - https://data.nasdaq.com/sign-up
QUANDL_API_KEY=your_quandl_key_here

# IEX Cloud Sandbox (FREE) - https://iexcloud.io/cloud-login#/register
IEX_SANDBOX_KEY=your_iex_sandbox_key_here

# Tiingo (FREE 500/hour) - https://www.tiingo.com/account/api/token
TIINGO_API_KEY=your_tiingo_key_here

# ============================================================================
# ALERTS (All FREE)
# ============================================================================

# Discord Webhook (FREE) - Create Discord server → Webhooks
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_here

# Telegram Bot (FREE) - @BotFather on Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# ============================================================================
# ML TRACKING (FREE for personal use)
# ============================================================================

# Weights & Biases (FREE) - https://wandb.ai/signup
WANDB_API_KEY=your_wandb_key_here

# ============================================================================
# CONFIGURATION
# ============================================================================

# Portfolio Settings
INITIAL_CAPITAL=100000
MAX_POSITION_SIZE=0.10
MAX_DRAWDOWN=0.20
STOP_LOSS_PERCENT=0.08

# Risk Management
TARGET_PRECISION=0.60
MIN_CONFIDENCE=0.65

# Paths
MODEL_PATH=models/underdog_v1/xgboost.pkl
SCALER_PATH=models/underdog_v1/scaler.pkl
DB_PATH=data/trading.db
LOG_PATH=logs/trading.log

# Feature Flags
ENABLE_PAPER_TRADING=true
ENABLE_DISCORD_ALERTS=false
ENABLE_TELEGRAM_ALERTS=false
ENABLE_WANDB_TRACKING=false
```

---

## 🎯 PRIORITY ORDER (Do This Today)

### **Next 30 Minutes** (CRITICAL):
1. ✅ Alpaca (5 min) - Paper trading execution
2. ✅ FRED (5 min) - Regime detection
3. ✅ Twelve Data (5 min) - Backup data source

### **Next Hour** (HIGH VALUE):
4. ✅ NewsAPI (5 min) - Sentiment features
5. ✅ Tiingo (5 min) - Training data quality
6. ✅ Discord Webhook (10 min) - Trade alerts

### **Optional** (NICE-TO-HAVE):
7. Telegram Bot (10 min) - Mobile alerts
8. Quandl (5 min) - Commodities data
9. IEX Cloud (5 min) - Alternative quotes
10. Weights & Biases (5 min) - Experiment tracking

---

## 📝 Quick Setup Script

Save this as `setup_api_keys.sh`:

```bash
#!/bin/bash

echo "🔑 API Keys Setup Wizard"
echo "========================"
echo ""

# Check if .env exists
if [ -f .env ]; then
    echo "⚠️  .env file already exists. Creating backup..."
    cp .env .env.backup
fi

echo "Let's add your API keys to .env file"
echo ""

# Alpaca
echo "1️⃣  ALPACA PAPER TRADING"
echo "   Sign up: https://alpaca.markets/"
read -p "   Enter ALPACA_API_KEY (or press Enter to skip): " alpaca_key
read -p "   Enter ALPACA_SECRET_KEY (or press Enter to skip): " alpaca_secret

# FRED
echo ""
echo "2️⃣  FRED ECONOMIC DATA"
echo "   Sign up: https://fred.stlouisfed.org/docs/api/api_key.html"
read -p "   Enter FRED_API_KEY (or press Enter to skip): " fred_key

# Twelve Data
echo ""
echo "3️⃣  TWELVE DATA"
echo "   Sign up: https://twelvedata.com/pricing"
read -p "   Enter TWELVE_DATA_API_KEY (or press Enter to skip): " twelve_key

# Write to .env
cat >> .env << EOF

# Added by setup_api_keys.sh on $(date)
ALPACA_API_KEY=${alpaca_key:-your_alpaca_key_here}
ALPACA_SECRET_KEY=${alpaca_secret:-your_alpaca_secret_here}
ALPACA_BASE_URL=https://paper-api.alpaca.markets

FRED_API_KEY=${fred_key:-your_fred_key_here}
TWELVE_DATA_API_KEY=${twelve_key:-your_twelve_data_key_here}
EOF

echo ""
echo "✅ Keys added to .env file!"
echo "⚠️  Remember to get the remaining keys later"
```

Make executable: `chmod +x setup_api_keys.sh`

---

## 🚨 SECURITY CHECKLIST

- [ ] Never commit `.env` to GitHub
- [ ] Add `.env` to `.gitignore`
- [ ] Use GitHub Secrets for CI/CD
- [ ] Rotate keys every 90 days
- [ ] Use separate keys for dev/prod
- [ ] Never share keys in chat/email

---

## ✅ Verification Script

Save as `verify_api_keys.py`:

```python
import os
from dotenv import load_dotenv
import requests

load_dotenv()

def test_alpaca():
    key = os.getenv('ALPACA_API_KEY')
    secret = os.getenv('ALPACA_SECRET_KEY')
    if not key or not secret:
        return "❌ Not configured"
    # Test connection
    try:
        import alpaca_trade_api as tradeapi
        api = tradeapi.REST(key, secret, 'https://paper-api.alpaca.markets')
        account = api.get_account()
        return f"✅ Connected (Balance: ${float(account.cash):,.2f})"
    except:
        return "❌ Invalid keys"

def test_fred():
    key = os.getenv('FRED_API_KEY')
    if not key:
        return "❌ Not configured"
    try:
        url = f"https://api.stlouisfed.org/fred/series?series_id=VIXCLS&api_key={key}&file_type=json"
        r = requests.get(url)
        return "✅ Connected" if r.status_code == 200 else "❌ Invalid key"
    except:
        return "❌ Connection failed"

def test_twelve_data():
    key = os.getenv('TWELVE_DATA_API_KEY')
    if not key:
        return "❌ Not configured"
    try:
        url = f"https://api.twelvedata.com/time_series?symbol=AAPL&interval=1day&apikey={key}"
        r = requests.get(url)
        return "✅ Connected" if r.status_code == 200 else "❌ Invalid key"
    except:
        return "❌ Connection failed"

print("🔍 API Keys Verification")
print("=" * 50)
print(f"Alpaca:      {test_alpaca()}")
print(f"FRED:        {test_fred()}")
print(f"Twelve Data: {test_twelve_data()}")
```

Run: `python verify_api_keys.py`

---

## 🎯 BOTTOM LINE

**Get these 3 CRITICAL keys in the next 30 minutes:**
1. **Alpaca** → Paper trading execution
2. **FRED** → Regime detection (VIX, yields)
3. **Twelve Data** → Backup data source

Everything else is optional but valuable. You already have Finnhub, Polygon, and Alpha Vantage working, so you're 70% there!

**Start with Alpaca** - that's the only blocker for paper trading in Week 3.
