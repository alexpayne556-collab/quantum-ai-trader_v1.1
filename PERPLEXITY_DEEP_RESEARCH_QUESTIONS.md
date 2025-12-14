# 🔬 PERPLEXITY PRO DEEP RESEARCH QUESTIONS
## Goal: Make AI Trading System Dangerously Accurate

---

## CONTEXT FOR PERPLEXITY:
```
I have built an AI trading system using LightGBM with these results:
- 84.1% win rate predicting 5% gains in 3 days
- +3.89% expected value per trade  
- 100+ technical indicators (EMA ribbons, RSI, MACD, Bollinger, volume)
- Genetic algorithm discovering new indicator combinations
- 50 high-volatility tickers (TSLA, NVDA, AMD, LUNR, LEU, APLD, etc.)
- Walk-forward validation to prevent overfitting

I want to push this further. What am I missing?
```

---

## 🎯 QUESTION SET 1: ADVANCED MACHINE LEARNING

### Q1.1 - Beyond LightGBM
```
What machine learning models are quantitative hedge funds like Renaissance Technologies, 
Two Sigma, and Citadel reportedly using in 2024-2025 for short-term stock prediction? 
Specifically interested in:
- Transformer architectures for time series
- Graph neural networks for cross-asset relationships  
- Ensemble methods beyond random forest/gradient boosting
- Any open-source implementations I can test
```

### Q1.2 - Feature Engineering Secrets
```
What unconventional technical indicators or feature engineering techniques have been 
published in academic papers (2020-2025) that show predictive power for 3-5 day 
stock returns? Looking for things beyond standard RSI/MACD/Bollinger that most 
retail traders don't know about.
```

### Q1.3 - Genetic Algorithm Alpha
```
Are there published research papers on using genetic programming or evolutionary 
algorithms to discover trading signals or indicator combinations? What fitness 
functions work best? Any examples of formulas that were discovered this way?
```

---

## 🎯 QUESTION SET 2: POSITION SIZING & RISK

### Q2.1 - Kelly Criterion Deep Dive
```
For a trading system with 84% win rate, average win of +5%, average loss of -2%:
1. What does the Kelly Criterion say about optimal position size?
2. What is "fractional Kelly" and why do professional traders use it?
3. With 6-7 simultaneous positions, how should I adjust Kelly?
4. What's the math on risk of ruin with these parameters?
```

### Q2.2 - Portfolio Heat Management
```
Professional prop trading firms talk about "portfolio heat" - the total risk 
exposure across all positions. How do firms like SMB Capital or prop shops 
manage heat when running 5-10 simultaneous swing trades? What's the maximum 
recommended portfolio heat for aggressive but sustainable trading?
```

### Q2.3 - Correlation Risk
```
When holding 6-7 positions simultaneously, how do I calculate and manage 
correlation risk? If NVDA and AMD both trigger signals, should I take both 
or only one? What's the math/formula for position sizing based on correlation?
```

---

## 🎯 QUESTION SET 3: MARKET MICROSTRUCTURE

### Q3.1 - Order Flow Prediction
```
How do institutional traders use order flow imbalance to predict short-term 
price movements? Is there a way to calculate order flow imbalance from publicly 
available data (not Level 2)? What academic research exists on this?
```

### Q3.2 - Dark Pool Signals
```
Can retail traders access any dark pool activity data? Are there patterns in 
dark pool prints that predict next-day moves? What does unusual dark pool 
activity signal about institutional intentions?
```

### Q3.3 - Short Interest Alpha
```
What is the relationship between changes in short interest and future stock 
returns? Is there a predictable pattern around short interest report dates? 
How do I incorporate short squeeze probability into my model?
```

---

## 🎯 QUESTION SET 4: VOLATILITY EDGE

### Q4.1 - Volatility Expansion Prediction
```
How can I predict BEFORE a stock's volatility expands? Looking for indicators 
that signal "quiet before the storm" - when a stock is about to make a big move 
but hasn't yet. Squeeze indicators? Implied vs realized volatility? What works?
```

### Q4.2 - VIX Regime Trading
```
How does the VIX level affect individual stock momentum strategies? Should my 
AI trade differently when VIX is below 15 vs above 25? What academic research 
exists on VIX regime-based strategy adjustment?
```

### Q4.3 - Options Flow as Signal
```
Can unusual options activity predict stock direction? What metrics from options 
flow are most predictive - put/call ratio, unusual volume, large block trades? 
Are there free data sources for this?
```

---

## 🎯 QUESTION SET 5: TIMING & SEASONALITY

### Q5.1 - Intraday Optimal Entry
```
What does academic research say about optimal entry times for swing trades?
- Is the first 30 minutes (opening range) predictive?
- Power hour (3-4pm) patterns?
- Does entry time affect 3-day return probability?
```

### Q5.2 - Day of Week Effects
```
Is there research on day-of-week effects for momentum stocks? Are Monday entries 
better than Friday entries for 3-5 day holds? What about month-end rebalancing effects?
```

### Q5.3 - Earnings Calendar Alpha
```
How can I systematically exploit:
1. Pre-earnings run-up patterns (how many days before?)
2. Post-earnings drift (how long does it last?)
3. Earnings surprise prediction from price/volume patterns
```

---

## 🎯 QUESTION SET 6: ALTERNATIVE DATA

### Q6.1 - Free Alternative Data Sources
```
What FREE alternative data sources have shown predictive power for stock returns?
- Social sentiment (Reddit, Twitter/X, StockTwits)
- Google Trends
- Web traffic data
- Job postings
- Satellite/geolocation (any free proxies?)
List specific APIs or data sources I can access without paying thousands.
```

### Q6.2 - Sentiment Alpha
```
What is the actual predictive power of social media sentiment for stock returns?
Cite specific academic papers. What's the optimal lookback period? Does sentiment 
predict better for small caps vs large caps?
```

### Q6.3 - News Flow
```
How do quantitative funds process news for trading signals? Is there research on 
the speed of price reaction to news? Can I gain edge by classifying news sentiment 
faster than the market prices it in?
```

---

## 🎯 QUESTION SET 7: CROSS-ASSET INTELLIGENCE

### Q7.1 - Sector Rotation Signals
```
What leading indicators predict sector rotation BEFORE it happens? If tech is 
about to outperform, what signals would show this 1-2 weeks early? Academic 
research on sector momentum timing.
```

### Q7.2 - Correlation Regime Changes
```
How do I detect when cross-stock correlations are about to change? In a 
correlation breakdown, which stocks typically lead? Is there a way to profit 
from correlation regime shifts?
```

### Q7.3 - Market Leader/Laggard
```
In momentum phases, certain stocks lead and others follow. How do I identify 
the leaders in real-time? Is there research on using leader stocks to predict 
laggard moves?
```

---

## 🎯 QUESTION SET 8: REGIME DETECTION

### Q8.1 - Market Regime Classification
```
What is the best way to classify market regimes (bull, bear, sideways, volatile) 
in real-time? Hidden Markov Models? Clustering? What features work best for 
regime detection? Should my trading strategy adapt to each regime?
```

### Q8.2 - Crash Prediction
```
Are there reliable early warning signals for market corrections? What indicators 
showed elevated risk before major drops (2020 COVID, 2022 bear market)? Can I 
incorporate crash probability into position sizing?
```

### Q8.3 - Recovery Detection
```
After a market drop, what signals reliably indicate the bottom or early recovery? 
Breadth indicators? Volume patterns? VIX term structure?
```

---

## 🎯 QUESTION SET 9: ADVANCED BACKTESTING

### Q9.1 - Avoiding Overfitting
```
Beyond walk-forward validation, what techniques do quantitative researchers use 
to ensure trading strategies aren't overfit? Combinatorial cross-validation? 
Multiple hypothesis testing correction? What's the state of the art?
```

### Q9.2 - Transaction Cost Modeling
```
How do professional quants model realistic transaction costs including:
- Slippage as a function of position size and volatility
- Market impact for entries/exits
- Bid-ask spread modeling
What's a realistic cost assumption for retail swing trading?
```

### Q9.3 - Survivorship Bias
```
My training data only includes currently listed stocks. How do I correct for 
survivorship bias? Where can I get delisted stock data? How much does this 
bias inflate backtested returns?
```

---

## 🎯 QUESTION SET 10: COMPOUNDING & SCALING

### Q10.1 - Realistic Return Expectations
```
With an 84% win rate, 5% average gain, 2% average loss, and 3.89% EV per trade:
- What is the realistic MONTHLY return with 6-7 positions?
- What is the realistic ANNUAL return?
- How does compounding work with this frequency of trading?
- What is the probability distribution of outcomes (best case, worst case, median)?
```

### Q10.2 - Scaling Challenges
```
As account size grows, what problems emerge for retail swing traders?
- At what size does market impact become an issue for small caps?
- How do prop firms scale successful strategies?
- Should I diversify into more tickers as I scale?
```

### Q10.3 - Tax Optimization
```
For a high-frequency swing trading strategy (holding 3-5 days), what are the 
tax implications? Short-term capital gains strategies? Should I consider 
trading in an IRA? Any legal tax optimization strategies?
```

---

## 🔥 BONUS QUESTIONS - CUTTING EDGE

### B1 - Transformer Models for Trading
```
Has anyone successfully applied GPT-style transformer models to stock prediction?
What about time series transformers like Temporal Fusion Transformer? Are there
open-source implementations that beat gradient boosting for this task?
```

### B2 - Reinforcement Learning
```
What is the state of reinforcement learning for trading as of 2024-2025? Has 
anyone achieved consistent profitability with RL? What are the main challenges?
Any open-source implementations worth testing?
```

### B3 - Quantum Computing
```
Is quantum computing being used for trading yet? Any quantum algorithms that 
show promise for portfolio optimization or pattern recognition? When might this
become relevant for retail traders?
```

### B4 - Synthetic Data
```
Can I use GANs or other generative models to create synthetic market data for 
training? Would this help with regime robustness? Any research on this?
```

---

## 📋 HOW TO USE THESE QUESTIONS

1. **Copy one question set at a time** into Perplexity Pro
2. **Ask for sources** - request specific papers, books, or implementations
3. **Follow up** on anything interesting with deeper questions
4. **Document findings** in a research log
5. **Implement and test** the most promising ideas

---

## 🎯 PRIORITY ORDER

**High Priority (Do First):**
- Q2.1 (Kelly Criterion) - Position sizing is critical
- Q4.1 (Volatility Expansion) - Predict big moves before they happen
- Q6.1 (Free Alternative Data) - Easy wins with new data
- Q3.3 (Short Interest) - Squeeze potential is huge for your tickers

**Medium Priority:**
- Q1.1 (Advanced ML) - Maybe better models exist
- Q5.3 (Earnings Calendar) - Systematic catalyst trading
- Q7.1 (Sector Rotation) - Timing the rotation

**Lower Priority (Research Later):**
- B1-B4 (Cutting Edge) - Interesting but may not be practical yet

---

*Generated for Perplexity Pro Deep Research - December 2024*
