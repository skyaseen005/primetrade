"""
Trader Performance vs Bitcoin Market Sentiment Analysis
=========================================================
Datasets:
  1. historical_data.csv   -> Hyperliquid trader execution data
  2. fear_greed_index.csv  -> Bitcoin Fear & Greed Index

Run this on your machine (paths below match what you gave: D:\\dstask\\...)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# -------------------------------------------------------------------
# 1. FILE PATHS  (edit if your files are elsewhere)
# -------------------------------------------------------------------
HISTORICAL_PATH = r"D:\dstask\historical_data.csv"
FEAR_GREED_PATH = r"D:\dstask\fear_greed_index.csv"

# -------------------------------------------------------------------
# 2. LOAD DATA
# -------------------------------------------------------------------
trades = pd.read_csv(HISTORICAL_PATH)
sentiment = pd.read_csv(FEAR_GREED_PATH)

print("Historical data shape:", trades.shape)
print("Fear & Greed data shape:", sentiment.shape)
print("\nHistorical columns:", list(trades.columns))
print("Fear & Greed columns:", list(sentiment.columns))

# -------------------------------------------------------------------
# 3. CLEAN / PREP THE TRADER DATA
# -------------------------------------------------------------------
# Standardize column names (strip spaces just in case)
trades.columns = trades.columns.str.strip()

# Rename to safe, code-friendly names but keep meaning identical
trades = trades.rename(columns={
    "Account": "account",
    "Coin": "coin",
    "Execution Price": "execution_price",
    "Size Tokens": "size_tokens",
    "Size USD": "size_usd",
    "Side": "side",
    "Timestamp IST": "timestamp_ist",
    "Start Position": "start_position",
    "Direction": "direction",
    "Closed PnL": "closed_pnl",
    "Transaction Hash": "tx_hash",
    "Order ID": "order_id",
    "Crossed": "crossed",
    "Fee": "fee",
    "Trade ID": "trade_id",
    "Timestamp": "timestamp_raw",
})

# Parse the IST timestamp (format seen: "02-12-2024 22:50" -> DD-MM-YYYY HH:MM)
trades["timestamp_ist"] = pd.to_datetime(
    trades["timestamp_ist"], format="%d-%m-%Y %H:%M", errors="coerce"
)

# Fallback: if that format fails for some rows, let pandas infer
mask_bad = trades["timestamp_ist"].isna()
if mask_bad.any():
    trades.loc[mask_bad, "timestamp_ist"] = pd.to_datetime(
        trades.loc[mask_bad, "timestamp_ist"], errors="coerce", dayfirst=True
    )

# Date-only key for merging with sentiment data
trades["date"] = trades["timestamp_ist"].dt.date

# Make sure numeric columns are numeric
numeric_cols = ["execution_price", "size_tokens", "size_usd",
                 "start_position", "closed_pnl", "fee"]
for col in numeric_cols:
    trades[col] = pd.to_numeric(trades[col], errors="coerce")

# -------------------------------------------------------------------
# 4. CLEAN / PREP THE FEAR & GREED DATA
# -------------------------------------------------------------------
sentiment.columns = sentiment.columns.str.strip()
sentiment["date"] = pd.to_datetime(sentiment["date"]).dt.date
sentiment["value"] = pd.to_numeric(sentiment["value"], errors="coerce")
sentiment["classification"] = sentiment["classification"].str.strip()

# -------------------------------------------------------------------
# 5. MERGE TRADES WITH SENTIMENT (by calendar date)
# -------------------------------------------------------------------
merged = trades.merge(
    sentiment[["date", "value", "classification"]],
    on="date",
    how="left"
)

print("\nRows with no matching sentiment date:", merged["classification"].isna().sum())
merged = merged.dropna(subset=["classification"])

# -------------------------------------------------------------------
# 6. DERIVED METRICS
# -------------------------------------------------------------------
merged["is_win"] = merged["closed_pnl"] > 0
merged["is_loss"] = merged["closed_pnl"] < 0
merged["trade_value_usd"] = merged["size_usd"]

# -------------------------------------------------------------------
# 7. AGGREGATE PERFORMANCE BY SENTIMENT CLASSIFICATION
# -------------------------------------------------------------------
summary = merged.groupby("classification").agg(
    total_trades=("closed_pnl", "count"),
    total_pnl=("closed_pnl", "sum"),
    avg_pnl=("closed_pnl", "mean"),
    median_pnl=("closed_pnl", "median"),
    win_rate=("is_win", "mean"),
    avg_trade_size_usd=("trade_value_usd", "mean"),
    total_fees=("fee", "sum"),
).sort_values("total_pnl", ascending=False)

summary["win_rate"] = (summary["win_rate"] * 100).round(2)
print("\n=== PERFORMANCE BY SENTIMENT CLASSIFICATION ===")
print(summary)

# Profit factor: gross profit / abs(gross loss) per sentiment bucket
profit_factor = merged.groupby("classification").apply(
    lambda g: g.loc[g["closed_pnl"] > 0, "closed_pnl"].sum() /
              abs(g.loc[g["closed_pnl"] < 0, "closed_pnl"].sum())
              if abs(g.loc[g["closed_pnl"] < 0, "closed_pnl"].sum()) > 0 else np.nan
)
summary["profit_factor"] = profit_factor.round(2)
print("\nWith profit factor added:")
print(summary)

# -------------------------------------------------------------------
# 8. LONG vs SHORT (BUY/SELL) BEHAVIOR ACROSS SENTIMENT
# -------------------------------------------------------------------
side_summary = merged.groupby(["classification", "side"]).agg(
    trades=("closed_pnl", "count"),
    avg_pnl=("closed_pnl", "mean"),
    total_pnl=("closed_pnl", "sum"),
).reset_index()
print("\n=== BUY vs SELL BEHAVIOR BY SENTIMENT ===")
print(side_summary)

# -------------------------------------------------------------------
# 9. PER-ACCOUNT PERFORMANCE UNDER EACH SENTIMENT REGIME
# -------------------------------------------------------------------
account_summary = merged.groupby(["account", "classification"]).agg(
    trades=("closed_pnl", "count"),
    total_pnl=("closed_pnl", "sum"),
    win_rate=("is_win", "mean"),
).reset_index()
account_summary["win_rate"] = (account_summary["win_rate"] * 100).round(2)

# Top 10 most profitable account/sentiment combos
top_accounts = account_summary.sort_values("total_pnl", ascending=False).head(10)
print("\n=== TOP 10 ACCOUNT x SENTIMENT COMBINATIONS BY PNL ===")
print(top_accounts)

# -------------------------------------------------------------------
# 10. DAILY SENTIMENT VALUE vs DAILY TOTAL PNL (correlation)
# -------------------------------------------------------------------
daily = merged.groupby("date").agg(
    daily_pnl=("closed_pnl", "sum"),
    daily_volume_usd=("trade_value_usd", "sum"),
    sentiment_value=("value", "mean"),
).reset_index()

corr = daily[["sentiment_value", "daily_pnl", "daily_volume_usd"]].corr()
print("\n=== CORRELATION: SENTIMENT VALUE vs DAILY PNL / VOLUME ===")
print(corr)

# -------------------------------------------------------------------
# 11. VISUALIZATIONS
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 11a. Total PnL by sentiment classification
sns.barplot(
    data=summary.reset_index(),
    x="classification", y="total_pnl", hue="classification",
    ax=axes[0, 0], palette="RdYlGn", legend=False
)
axes[0, 0].set_title("Total Closed PnL by Market Sentiment")
axes[0, 0].tick_params(axis="x", rotation=30)

# 11b. Win rate by sentiment classification
sns.barplot(
    data=summary.reset_index(),
    x="classification", y="win_rate", hue="classification",
    ax=axes[0, 1], palette="RdYlGn", legend=False
)
axes[0, 1].set_title("Win Rate (%) by Market Sentiment")
axes[0, 1].tick_params(axis="x", rotation=30)

# 11c. Sentiment value vs daily PnL scatter
sns.scatterplot(
    data=daily, x="sentiment_value", y="daily_pnl", ax=axes[1, 0]
)
axes[1, 0].axhline(0, color="gray", linestyle="--")
axes[1, 0].set_title("Fear & Greed Value vs Daily Total PnL")

# 11d. Average trade size by sentiment
sns.barplot(
    data=summary.reset_index(),
    x="classification", y="avg_trade_size_usd", hue="classification",
    ax=axes[1, 1], palette="Blues_d", legend=False
)
axes[1, 1].set_title("Average Trade Size (USD) by Market Sentiment")
axes[1, 1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("sentiment_performance_dashboard.png", dpi=150)
print("\nSaved chart: sentiment_performance_dashboard.png")

# -------------------------------------------------------------------
# 12. EXPORT SUMMARY TABLES FOR REPORTING
# -------------------------------------------------------------------
summary.to_csv("summary_by_sentiment.csv")
side_summary.to_csv("summary_by_sentiment_and_side.csv", index=False)
account_summary.to_csv("account_performance_by_sentiment.csv", index=False)
daily.to_csv("daily_pnl_vs_sentiment.csv", index=False)

print("\nExported: summary_by_sentiment.csv, summary_by_sentiment_and_side.csv, "
      "account_performance_by_sentiment.csv, daily_pnl_vs_sentiment.csv")

print("\nDone.")