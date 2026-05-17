"""
Stock Price Analysis Tool
Author: Devaditya Raghuvanshi

A Python-based analysis pipeline using Pandas and Matplotlib to process
and visualize historical stock data with technical indicators.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta


# ── Configuration ────────────────────────────────────────────────────────────

TICKER = "RELIANCE.NS"        # Change to any ticker: AAPL, TCS.NS, INFY.NS etc.
PERIOD_YEARS = 2              # How many years of historical data
MA_SHORT = 20                 # Short moving average window
MA_LONG = 50                  # Long moving average window
SIP_MONTHLY_AMOUNT = 5000     # Monthly SIP investment (in currency units)
LUMP_SUM_AMOUNT = 120000      # Lump sum investment (equivalent to 2yr SIP total)


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_data(ticker: str, years: int) -> pd.DataFrame:
    """Fetch historical OHLCV data using yfinance."""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    print(f"Fetching data for {ticker} from {start.date()} to {end.date()}...")
    df = yf.download(ticker, start=start, end=end, progress=False)
    df.dropna(inplace=True)
    print(f"  Loaded {len(df)} trading days of data.\n")
    return df


# ── Technical Indicators ──────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA20, MA50, and daily returns."""
    df = df.copy()
    close = df["Close"].squeeze()
    df["MA20"] = close.rolling(window=MA_SHORT).mean()
    df["MA50"] = close.rolling(window=MA_LONG).mean()
    df["Daily_Return"] = close.pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
    return df


# ── Investment Strategy Comparison ───────────────────────────────────────────

def compute_sip(df: pd.DataFrame, monthly_amount: float) -> pd.Series:
    """
    Simulate SIP: invest a fixed amount on the first trading day of each month.
    Returns a Series of portfolio value over time.
    """
    close = df["Close"].squeeze()
    units_held = 0.0
    portfolio_values = pd.Series(index=df.index, dtype=float)

    # Identify first trading day of each month
    monthly_first = df.resample("MS").first().index  # month start
    first_trading_days = [
        df.index[df.index >= d][0] for d in monthly_first if any(df.index >= d)
    ]

    for date in df.index:
        if date in first_trading_days:
            price = float(close.loc[date])
            units_held += monthly_amount / price
        portfolio_values.loc[date] = units_held * float(close.loc[date])

    return portfolio_values


def compute_lump_sum(df: pd.DataFrame, amount: float) -> pd.Series:
    """
    Simulate lump sum: invest full amount on day 1.
    Returns a Series of portfolio value over time.
    """
    close = df["Close"].squeeze()
    initial_price = float(close.iloc[0])
    units = amount / initial_price
    return (close * units).squeeze()


def strategy_summary(sip_series: pd.Series, lump_series: pd.Series,
                      sip_invested: float, lump_invested: float) -> dict:
    """Compute final returns and CAGR for both strategies."""
    def cagr(final, invested, years):
        return ((final / invested) ** (1 / years) - 1) * 100

    sip_final = sip_series.iloc[-1]
    lump_final = lump_series.iloc[-1]
    years = PERIOD_YEARS

    return {
        "SIP": {
            "invested": sip_invested,
            "final_value": sip_final,
            "absolute_return": sip_final - sip_invested,
            "return_pct": (sip_final - sip_invested) / sip_invested * 100,
            "cagr": cagr(sip_final, sip_invested, years),
        },
        "Lump Sum": {
            "invested": lump_invested,
            "final_value": lump_final,
            "absolute_return": lump_final - lump_invested,
            "return_pct": (lump_final - lump_invested) / lump_invested * 100,
            "cagr": cagr(lump_final, lump_invested, years),
        },
    }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, sip_series: pd.Series, lump_series: pd.Series,
             summary: dict, ticker: str):
    """Generate a 3-panel dashboard."""
    close = df["Close"].squeeze()

    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor("#0f0f0f")
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    ACCENT = "#00e5ff"
    GREEN  = "#00e676"
    ORANGE = "#ff9100"
    GRID_COLOR = "#2a2a2a"
    TEXT_COLOR = "#e0e0e0"

    def style_ax(ax, title):
        ax.set_facecolor("#1a1a1a")
        ax.set_title(title, color=TEXT_COLOR, fontsize=11, fontweight="bold", pad=8)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(color=GRID_COLOR, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.xaxis.label.set_color(TEXT_COLOR)
        ax.yaxis.label.set_color(TEXT_COLOR)

    # ── Panel 1: Price + Moving Averages (full width) ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(close.index, close, color=ACCENT, linewidth=1.2, label="Close Price")
    ax1.plot(df.index, df["MA20"], color=GREEN, linewidth=1.0,
             linestyle="--", label=f"MA{MA_SHORT}")
    ax1.plot(df.index, df["MA50"], color=ORANGE, linewidth=1.0,
             linestyle="--", label=f"MA{MA_LONG}")
    ax1.fill_between(close.index, close, alpha=0.05, color=ACCENT)
    ax1.legend(facecolor="#1a1a1a", edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    style_ax(ax1, f"{ticker} — Price & Moving Averages (MA{MA_SHORT}/MA{MA_LONG})")
    ax1.set_ylabel("Price")

    # ── Panel 2: Daily Returns distribution ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    returns = df["Daily_Return"].dropna() * 100
    ax2.hist(returns, bins=60, color=ACCENT, alpha=0.7, edgecolor="#0f0f0f")
    ax2.axvline(0, color="white", linewidth=0.8, linestyle="--")
    ax2.axvline(returns.mean(), color=GREEN, linewidth=1.2,
                linestyle="-", label=f"Mean: {returns.mean():.2f}%")
    ax2.legend(facecolor="#1a1a1a", edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=8)
    style_ax(ax2, "Daily Return Distribution")
    ax2.set_xlabel("Daily Return (%)")
    ax2.set_ylabel("Frequency")

    # ── Panel 3: Cumulative return ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    cumret = df["Cumulative_Return"] * 100
    ax3.plot(cumret.index, cumret, color=ORANGE, linewidth=1.2)
    ax3.fill_between(cumret.index, cumret, alpha=0.1, color=ORANGE)
    ax3.axhline(0, color="white", linewidth=0.6, linestyle="--")
    style_ax(ax3, "Cumulative Return (%)")
    ax3.set_ylabel("Return (%)")

    # ── Panel 4: SIP vs Lump Sum portfolio value ─────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.plot(sip_series.index, sip_series, color=GREEN,
             linewidth=1.4, label="SIP Strategy")
    ax4.plot(lump_series.index, lump_series, color=ORANGE,
             linewidth=1.4, label="Lump Sum Strategy")
    ax4.legend(facecolor="#1a1a1a", edgecolor=GRID_COLOR,
               labelcolor=TEXT_COLOR, fontsize=9)
    style_ax(ax4, "Investment Strategy Comparison: SIP vs Lump Sum")
    ax4.set_ylabel("Portfolio Value")

    # ── Summary text box ─────────────────────────────────────────────────────
    s = summary
    txt = (
        f"SIP  →  Invested: {s['SIP']['invested']:,.0f}  |  "
        f"Final: {s['SIP']['final_value']:,.0f}  |  "
        f"Return: {s['SIP']['return_pct']:.1f}%  |  "
        f"CAGR: {s['SIP']['cagr']:.1f}%\n"
        f"Lump →  Invested: {s['Lump Sum']['invested']:,.0f}  |  "
        f"Final: {s['Lump Sum']['final_value']:,.0f}  |  "
        f"Return: {s['Lump Sum']['return_pct']:.1f}%  |  "
        f"CAGR: {s['Lump Sum']['cagr']:.1f}%"
    )
    fig.text(0.5, 0.01, txt, ha="center", va="bottom", fontsize=8.5,
             color=TEXT_COLOR, family="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a1a",
                       edgecolor=GRID_COLOR, alpha=0.9))

    plt.suptitle(f"Stock Analysis Dashboard — {ticker}",
                 color=TEXT_COLOR, fontsize=14, fontweight="bold", y=1.01)

    plt.savefig("stock_analysis.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print("Chart saved as stock_analysis.png")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = fetch_data(TICKER, PERIOD_YEARS)
    df = compute_indicators(df)

    # SIP: invest monthly, total invested = months * monthly amount
    months = PERIOD_YEARS * 12
    sip_total_invested = months * SIP_MONTHLY_AMOUNT

    sip_series  = compute_sip(df, SIP_MONTHLY_AMOUNT)
    lump_series = compute_lump_sum(df, LUMP_SUM_AMOUNT)
    summary     = strategy_summary(sip_series, lump_series,
                                   sip_total_invested, LUMP_SUM_AMOUNT)

    print("=" * 55)
    print(f"  PERFORMANCE SUMMARY — {TICKER}")
    print("=" * 55)
    for strategy, metrics in summary.items():
        print(f"\n  {strategy}")
        print(f"    Invested       : {metrics['invested']:>12,.2f}")
        print(f"    Final Value    : {metrics['final_value']:>12,.2f}")
        print(f"    Absolute Gain  : {metrics['absolute_return']:>12,.2f}")
        print(f"    Return         : {metrics['return_pct']:>11.2f}%")
        print(f"    CAGR           : {metrics['cagr']:>11.2f}%")
    print("\n" + "=" * 55)

    plot_all(df, sip_series, lump_series, summary, TICKER)


if __name__ == "__main__":
    main()
