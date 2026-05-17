# Stock Price Analysis Tool

A Python-based analysis pipeline to fetch, process, and visualize historical stock data using **Pandas**, **Matplotlib**, and **yfinance**.

## Features

- Fetches live historical stock data for any ticker (NSE/BSE/US markets)
- Computes technical indicators — **MA20** and **MA50** — for trend identification
- Compares two investment strategies: **SIP (Systematic Investment Plan)** vs **Lump Sum**
- Calculates absolute returns, percentage returns, and **CAGR** for both strategies
- Generates a 4-panel dark-themed dashboard saved as `stock_analysis.png`

## Demo

![Stock Analysis Dashboard](stock_analysis.png)

## Installation

```bash
pip install yfinance pandas matplotlib numpy
```

## Usage

```bash
python3 stock_analysis.py
```

By default the script runs on `RELIANCE.NS`. To change the stock, edit the config at the top of the file:

```python
TICKER = "AAPL"        # Any valid ticker: TCS.NS, INFY.NS, AAPL, MSFT etc.
PERIOD_YEARS = 2       # Years of historical data
SIP_MONTHLY_AMOUNT = 5000
LUMP_SUM_AMOUNT = 120000
```

## Output

**Terminal summary:**
```
SIP
  Invested       :      120,000.00
  Final Value    :      135,200.00
  Absolute Gain  :       15,200.00
  Return         :          12.67%
  CAGR           :           6.12%

Lump Sum
  Invested       :      120,000.00
  Final Value    :      141,800.00
  ...
```

**Chart:** A 4-panel dashboard with price + moving averages, daily return distribution, cumulative return, and strategy comparison.

## Tech Stack

- `yfinance` — live market data
- `pandas` — data processing and time-series analysis
- `matplotlib` — visualization
- `numpy` — numerical computations

## Author

Devaditya Raghuvanshi — DTU, Maths & Computing
