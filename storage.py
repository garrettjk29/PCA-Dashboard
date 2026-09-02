"""Local persistence for the dashboard's saved tickers.

Keeps the last-fetched price data on disk as a plain CSV file (instead of a
pickle) so:
  - you can open portfolio_prices.csv directly in Excel/a text editor to see
    the data, and
  - a PCA script/notebook can load it straight into a pandas DataFrame ready
    for manual PCA (rows = dates/observations, columns = tickers/variables).

Usage from another file:

    from storage import load_portfolio

    prices = load_portfolio()
    # prices: dict[ticker -> pandas Series of daily closes, indexed by date]

To load the raw table for PCA yourself:

    import pandas as pd
    df = pd.read_csv("portfolio_prices.csv", index_col=0, parse_dates=True)
"""

from pathlib import Path

import pandas as pd

PRICES_FILE = Path(__file__).resolve().parent / "portfolio_prices.csv"


def save_portfolio(prices: dict) -> None:
    """Persist the current ticker -> price data to disk."""
    if prices:
        df = pd.DataFrame(prices)
        df.index.name = "Date"
        df.to_csv(PRICES_FILE)
    elif PRICES_FILE.exists():
        PRICES_FILE.unlink()


def load_portfolio() -> dict:
    """Load previously saved ticker -> price data. Returns {} if none exists."""
    prices = {}
    if PRICES_FILE.exists():
        df = pd.read_csv(PRICES_FILE, index_col=0, parse_dates=True)
        prices = {ticker: df[ticker].dropna() for ticker in df.columns}

    return prices
