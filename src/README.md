# 📊 Average Price Analysis Script (`Average_price.py`)

This script performs an **average price seasonality analysis** on financial assets using daily historical data from **Yahoo Finance**.
It normalizes each year of prices to a base value of 100, builds seasonal paths, and computes statistical aggregates (mean, median, 10th–90th percentiles).
The result is a **visual comparison of historical yearly trends vs. the most recent year**.

---

## 🚀 Features
- Download historical daily prices via [`yfinance`](https://pypi.org/project/yfinance/)
- Normalize yearly price paths (base 100 at the first trading day)
- Compute seasonality statistics:
  - Mean path
  - Median path
  - 10th and 90th percentile band
- Plot the normalized seasonal trend with monthly labels
- Optionally exclude the most recent year to highlight its divergence

---

## 📂 File Overview
- **`Average_price.py`** → main script for interactive seasonality analysis
- Functions:
  - `download_data()` → fetch and prepare Yahoo Finance data
  - `build_normalized_paths()` → normalize yearly paths
  - `seasonality_stats()` → compute mean/median/quantiles
  - `plot_seasonality()` → visualize seasonal behavior
  - `main()` → user input pipeline

---

## 🛠️ Requirements
Install dependencies:
```bash
pip install pandas numpy matplotlib yfinance