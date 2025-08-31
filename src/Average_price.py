# file_due.py - interactive seasonality analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Optional


# -----------------------------
# Download & prepare data
# -----------------------------
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    # Download daily price data from Yahoo Finance
    df = yf.download(
        ticker, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False,
        group_by="column"
    )
    if df.empty:
        raise ValueError(f"No data downloaded for {ticker} ({start}->{end}).")

    # Remove timezone info if present
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # --- Normalize and extract the "Close" column ---
    close = None
    if isinstance(df.columns, pd.MultiIndex):
        # Handle case when Yahoo Finance returns multi-index columns
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                if ticker in close.columns:
                    close = close[ticker]
                else:
                    close = close.iloc[:, 0]
        else:
            # Try to flatten multi-index columns if structure is different
            try:
                close = df[("Close", ticker)]
            except Exception:
                flat_cols = ["_".join([str(x) for x in col if str(x) != ""]).lower() for col in df.columns]
                df_flat = df.copy()
                df_flat.columns = flat_cols
                candidates = [c for c in df_flat.columns if c.startswith("close")]
                if not candidates:
                    raise ValueError(f"'Close' column not found: {list(df.columns)}")
                close = df_flat[candidates[0]]
    else:
        # Standard single-index dataframe
        cols_lower = {c.lower(): c for c in df.columns}
        if "close" in cols_lower:
            close = df[cols_lower["close"]]
        else:
            # Fallback: assume 4th column is "Close"
            if df.shape[1] >= 4:
                close = df.iloc[:, 3]
            else:
                raise ValueError(f"'Close' column not found: {list(df.columns)}")

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # Final dataframe with closing prices + year + day-of-year (DOY)
    out = pd.DataFrame({"Close": close}).dropna()
    out["Year"] = out.index.year
    # Remove February 29 to align all years to 365 days
    leap_29 = (out.index.month == 2) & (out.index.day == 29)
    out = out[~leap_29].copy()
    out["DOY"] = out.index.dayofyear
    return out


# -----------------------------
# Build normalized yearly paths
# -----------------------------
def build_normalized_paths(df: pd.DataFrame, exclude_year: Optional[int] = None) -> pd.DataFrame:
    cols = []
    for year, g in df.groupby("Year"):
        if exclude_year is not None and year == exclude_year:
            continue
        g = g.sort_index()
        if g.empty:
            continue
        base = float(g["Close"].iloc[0])
        if base == 0.0 or np.isnan(base):
            continue
        # Normalize each year to 100 at the first trading day
        s = (g["Close"] / base * 100.0)
        s.index = g["DOY"].values
        # If multiple rows share the same DOY, keep the last one
        s = s.groupby(level=0).last()
        s = s.reindex(range(1, 366))
        s.name = year
        cols.append(s)
    if not cols:
        raise ValueError("No valid paths built. Check ticker/period.")
    df_paths = pd.concat(cols, axis=1)
    return df_paths


# -----------------------------
# Compute seasonality statistics
# -----------------------------
def seasonality_stats(df_paths: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Mean":   df_paths.mean(axis=1, skipna=True),
        "Median": df_paths.median(axis=1, skipna=True),
        "Q10":    df_paths.quantile(0.10, axis=1),
        "Q90":    df_paths.quantile(0.90, axis=1),
    })


# -----------------------------
# Plot normalized seasonality
# -----------------------------
def plot_seasonality(df: pd.DataFrame, df_paths: pd.DataFrame, seas: pd.DataFrame,
                     ticker: str, exclude_year: Optional[int] = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    days = df_paths.index.values

    # Plot mean, median and quantile band
    ax.plot(days, seas["Mean"], color="blue", label="Seasonal Mean", linewidth=2)
    ax.plot(days, seas["Median"], color="green", linestyle="--", label="Seasonal Median")
    ax.fill_between(days, seas["Q10"], seas["Q90"], color="lightblue", alpha=0.35, label="10th–90th Percentile Band")

    # Plot excluded year as benchmark (e.g. last year)
    if exclude_year is not None:
        g = df[df["Year"] == exclude_year].sort_index()
        if not g.empty:
            base = float(g["Close"].iloc[0])
            norm = g["Close"] / base * 100.0
            ax.plot(g["DOY"], norm, color="orange", label=f"Close {exclude_year}", linewidth=2)

    # Format X-axis with months
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    start_month_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.set_xticks(start_month_doy)
    ax.set_xticklabels(months)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Month")
    ax.set_ylabel("Normalized Index (base 100)")
    ax.set_title(f"Annual Seasonality {ticker}")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN: user inputs ticker and start year
# -----------------------------
def main():
    # Input: ticker symbol (Yahoo Finance format)
    ticker = input("Enter the Yahoo ticker (e.g. ALC, TOL, AAPL): ").strip().upper()

    # Input: start year (defaults to 2010)
    start_year_raw = input("Enter start year [default 2010]: ").strip()
    if start_year_raw:
        if not (start_year_raw.isdigit() and len(start_year_raw) == 4):
            raise ValueError("Invalid start year. Please provide a 4-digit year, e.g. 2015.")
        start = f"{start_year_raw}-01-01"
    else:
        start = "2010-01-01"

    # Fixed end date
    end = "2025-12-31"

    # Exclude last year from normalization (for comparison)
    exclude_year = 2025

    # Full pipeline: download → build paths → compute stats → plot
    df = download_data(ticker, start, end)
    df_paths = build_normalized_paths(df, exclude_year=exclude_year)
    seas = seasonality_stats(df_paths)
    plot_seasonality(df, df_paths, seas, ticker, exclude_year=exclude_year)


if __name__ == "__main__":
    main()