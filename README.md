# Bitcoin Trader Performance vs Market Sentiment

Analysis of how Hyperliquid trader performance (211K+ trades) relates to Bitcoin
market sentiment, using the Fear & Greed Index.

## Objective

Explore the relationship between trader performance and market sentiment on
Hyperliquid, uncover hidden patterns, and deliver insights that can inform
smarter trading strategies.

## Datasets

| File | Description | Key columns |
|---|---|---|
| `historical_data.csv` | Hyperliquid trade execution history | `Account`, `Coin`, `Execution Price`, `Size Tokens`, `Size USD`, `Side`, `Timestamp IST`, `Start Position`, `Direction`, `Closed PnL`, `Transaction Hash`, `Order ID`, `Crossed`, `Fee`, `Trade ID`, `Timestamp` |
| `fear_greed_index.csv` | Daily Bitcoin market sentiment | `timestamp`, `value`, `classification`, `date` |

`classification` buckets: `Extreme Fear`, `Fear`, `Neutral`, `Greed`, `Extreme Greed`.

## Methodology

1. Parse `Timestamp IST` into a datetime and derive a calendar `date`.
2. Merge trades with sentiment on that date.
3. Aggregate PnL, win rate, average trade size, fees, and profit factor per
   sentiment bucket.
4. Break down BUY vs SELL behavior within each sentiment regime.
5. Roll up performance per account, per sentiment regime.
6. Correlate daily sentiment value against daily total PnL / volume.
7. Visualize results in a 4-panel dashboard.

See `task.py` (script) and `task1.ipynb` (notebook walkthrough) for the full
implementation.

## Key Findings

- **Extreme Greed is the most profitable regime**, driven almost entirely by
  SELL trades — consistent with "sell into euphoria" behavior. Highest win
  rate (46.5%) and profit factor (11.02) of any bucket, despite the smallest
  average trade size.
- **Extreme Fear is the weakest regime**, not the best — lowest total PnL,
  lowest profit factor (2.16), lowest win rate (37.1%). The "buy the extreme
  fear" narrative does not hold up in this dataset.
- **Fear (moderate, not extreme) is where buying has an edge** — BUY trades
  in Fear average higher PnL than SELL trades in Fear.
- **Sentiment value has weak same-day linear correlation with daily PnL**
  (-0.08), meaning the relationship shows up in bucketed averages rather than
  as a straight linear trend — same-day sentiment score alone isn't a strong
  predictor.
- **PnL is concentrated in a small number of accounts** across sentiment
  regimes, suggesting a mix of whale and retail trading behavior worth
  separating in further analysis.
- **Caveat:** a large share of trades are position-building fills with
  `Closed PnL = 0` (not wins or losses). Metrics that use only realized
  (closed) trades give a cleaner read on true win rate and profit factor —
  see "Next steps" below.

## Outputs

Running `task.py` produces:

- `summary_by_sentiment.csv` — PnL, win rate, avg trade size, fees, profit
  factor per sentiment bucket
- `summary_by_sentiment_and_side.csv` — BUY vs SELL breakdown per bucket
- `account_performance_by_sentiment.csv` — per-account PnL and win rate per
  sentiment bucket
- `daily_pnl_vs_sentiment.csv` — daily PnL, volume, and sentiment value
- `sentiment_performance_dashboard.png` — 4-panel summary chart

## How to run

```bash
pip install -r requirements.txt
python task.py
```

Update the file paths at the top of `task.py` if your CSVs live somewhere
other than the project root.

## Requirements

```
pandas
numpy
matplotlib
seaborn
```

## Next steps

- Recompute win rate / profit factor using only trades where
  `Closed PnL != 0` (realized trades), to remove the dilution from
  position-building fills.
- Add a statistical significance test across sentiment buckets, since sample
  sizes vary widely (21K–62K trades per bucket).
- Segment accounts into whale vs retail cohorts and re-run the sentiment
  breakdown per cohort.
- Test lagged sentiment (previous day's classification) against next-day
  performance, since same-day correlation is weak.

## Project structure

```
.
├── task.py                              # main analysis script
├── task1.ipynb                          # notebook version
├── historical_data.csv                  # raw trade data
├── fear_greed_index.csv                 # raw sentiment data
├── summary_by_sentiment.csv             # generated
├── summary_by_sentiment_and_side.csv    # generated
├── account_performance_by_sentiment.csv # generated
├── daily_pnl_vs_sentiment.csv           # generated
├── sentiment_performance_dashboard.png  # generated
└── README.md
```
