import yfinance as yf
import pandas as pd
from typing import Sequence, Union

# Maps human-friendly timeframe strings to yfinance's `period` values.
_TIMEFRAME_MAP = {
    "1 week": "5d",     # yfinance has no native "1wk" period; 5d = last week of trading days
    "1 month": "1mo",
    "6 months": "6mo",
    "1 year": "1y",
    "5 years": "5y",
    "10 years": "10y",
    "all": "max",
}


def get_daily_prices(
    tickers: Union[str, Sequence[str]], timeframe: str
) -> Union[pd.Series, pd.DataFrame]:
    """
    Pull daily end-of-day (adjusted close) prices for one or more stock/ETF tickers
    over a given timeframe.

    Fetches all tickers in a single batched, multi-threaded request (via
    yf.download) instead of one request per ticker, and skips dividend/split
    data we don't use -- both cut retrieval time substantially when pulling
    more than one symbol.

    Parameters
    ----------
    tickers : str or sequence of str
        Stock or ETF ticker symbol(s), e.g. "AAPL" or ["AAPL", "MSFT", "SPY"].
    timeframe : str
        One of: "1 week", "1 month", "6 months", "1 year",
        "5 years", "10 years", "all".

    Returns
    -------
    pd.Series or pd.DataFrame
        Daily EOD close prices, indexed by date. A single ticker returns a
        Series named "{ticker}_close"; multiple tickers return a DataFrame
        with one column per ticker (uppercased).

    Raises
    ------
    ValueError
        If timeframe isn't recognized or no data is returned for the ticker(s).
    """
    key = timeframe.strip().lower()
    if key not in _TIMEFRAME_MAP:
        valid = ", ".join(_TIMEFRAME_MAP.keys())
        raise ValueError(f"Invalid timeframe '{timeframe}'. Must be one of: {valid}")

    period = _TIMEFRAME_MAP[key]

    single = isinstance(tickers, str)
    symbols = [tickers] if single else list(tickers)

    data = yf.download(
        symbols,
        period=period,
        interval="1d",
        auto_adjust=True,
        actions=False,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    if data.empty:
        raise ValueError(f"No data returned for ticker(s) {symbols}. Check the symbol(s).")

    if single:
        prices = data[symbols[0]]["Close"].dropna()
        prices.name = f"{symbols[0].upper()}_close"
        return prices

    closes = pd.concat(
        {sym.upper(): data[sym]["Close"] for sym in symbols if sym in data.columns.get_level_values(0)},
        axis=1,
    ).dropna(how="all")
    return closes


if __name__ == "__main__":
    prices = get_daily_prices("SPY", "5 years")
    print(prices)
    print(f"\nMean close: {prices.mean():.2f}")
    print(f"Std dev: {prices.std():.2f}")
