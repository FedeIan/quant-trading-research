# file_due.py - analisi stagionalità interattiva
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Optional


# -----------------------------
# Download & prep
# -----------------------------
def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end,
        interval="1d", auto_adjust=True, progress=False,
        group_by="column"
    )
    if df.empty:
        raise ValueError(f"Nessun dato scaricato per {ticker} ({start}->{end}).")

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    # --- Normalizza colonne ---
    close = None
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" in df.columns.get_level_values(0):
            close = df["Close"]
            if isinstance(close, pd.DataFrame):
                if ticker in close.columns:
                    close = close[ticker]
                else:
                    close = close.iloc[:, 0]
        else:
            try:
                close = df[("Close", ticker)]
            except Exception:
                flat_cols = ["_".join([str(x) for x in col if str(x) != ""]).lower() for col in df.columns]
                df_flat = df.copy()
                df_flat.columns = flat_cols
                candidates = [c for c in df_flat.columns if c.startswith("close")]
                if not candidates:
                    raise ValueError(f"Colonna 'Close' non trovata: {list(df.columns)}")
                close = df_flat[candidates[0]]
    else:
        cols_lower = {c.lower(): c for c in df.columns}
        if "close" in cols_lower:
            close = df[cols_lower["close"]]
        else:
            if df.shape[1] >= 4:
                close = df.iloc[:, 3]
            else:
                raise ValueError(f"Colonna 'Close' non trovata: {list(df.columns)}")

    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    out = pd.DataFrame({"Close": close}).dropna()
    out["Year"] = out.index.year
    # Rimuovi il 29 febbraio per allineare i DOY su 365 giorni
    leap_29 = (out.index.month == 2) & (out.index.day == 29)
    out = out[~leap_29].copy()
    out["DOY"] = out.index.dayofyear
    return out


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
        s = (g["Close"] / base * 100.0)
        s.index = g["DOY"].values
        # Se più barre cadono nello stesso DOY, tieni l'ultima (tipico per dati rettificati)
        s = s.groupby(level=0).last()
        s = s.reindex(range(1, 366))
        s.name = year
        cols.append(s)
    if not cols:
        raise ValueError("Nessun percorso valido costruito. Controlla ticker/periodo.")
    df_paths = pd.concat(cols, axis=1)
    return df_paths


def seasonality_stats(df_paths: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "Mean":   df_paths.mean(axis=1, skipna=True),
        "Median": df_paths.median(axis=1, skipna=True),
        "Q10":    df_paths.quantile(0.10, axis=1),
        "Q90":    df_paths.quantile(0.90, axis=1),
    })


def plot_seasonality(df: pd.DataFrame, df_paths: pd.DataFrame, seas: pd.DataFrame,
                     ticker: str, exclude_year: Optional[int] = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    days = df_paths.index.values

    ax.plot(days, seas["Mean"], color="blue", label="Media stagionale", linewidth=2)
    ax.plot(days, seas["Median"], color="green", linestyle="--", label="Mediana stagionale")
    ax.fill_between(days, seas["Q10"], seas["Q90"], color="lightblue", alpha=0.35, label="Banda 10°–90°")

    if exclude_year is not None:
        g = df[df["Year"] == exclude_year].sort_index()
        if not g.empty:
            base = float(g["Close"].iloc[0])
            norm = g["Close"] / base * 100.0
            ax.plot(g["DOY"], norm, color="orange", label=f"Close {exclude_year}", linewidth=2)

    mesi = ["Gen","Feb","Mar","Apr","Mag","Giu","Lug","Ago","Set","Ott","Nov","Dic"]
    start_mese_doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    ax.set_xticks(start_mese_doy)
    ax.set_xticklabels(mesi)
    ax.set_xlim(1, 365)
    ax.set_xlabel("Mese")
    ax.set_ylabel("Indice normalizzato (base 100)")
    ax.set_title(f"Stagionalità annuale {ticker}")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


# -----------------------------
# MAIN: input per ticker e ANNO di inizio
# -----------------------------
def main():
    # Ticker da input
    ticker = input("Inserisci il ticker Yahoo (es. ALC, TOL, AAPL): ").strip().upper()

    # Anno di inizio da input (solo anno), mese/giorno fissati a 01-01
    start_year_raw = input("Inserisci l'anno di inizio [default 2010]: ").strip()
    if start_year_raw:
        if not (start_year_raw.isdigit() and len(start_year_raw) == 4):
            raise ValueError("Anno di inizio non valido. Inserisci un anno a 4 cifre, es. 2015.")
        start = f"{start_year_raw}-01-01"
    else:
        start = "2010-01-01"

    # Data fine fissa
    end = "2025-12-31"

    # Anno da escludere fisso
    exclude_year = 2025

    # Pipelinef
    df = download_data(ticker, start, end)
    df_paths = build_normalized_paths(df, exclude_year=exclude_year)
    seas = seasonality_stats(df_paths)
    plot_seasonality(df, df_paths, seas, ticker, exclude_year=exclude_year)


if __name__ == "__main__":
    main()
