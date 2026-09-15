"""Market data loading with an on-disk cache.

The notebooks in this repository should be re-runnable without a network
connection and should produce the same numbers on every run.  ``load_prices``
downloads a price series once, caches it as CSV under ``data/``, and reads from
that cache on every subsequent call.  Pass ``refresh=True`` to re-download.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def load_prices(
    ticker: str = "^SPX",
    period: str = "max",
    interval: str = "1d",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
) -> pd.Series:
    """Return a cached close-price series for ``ticker``.

    Parameters
    ----------
    ticker
        Yahoo Finance symbol, e.g. ``"^SPX"``.
    period, interval
        Passed through to ``yfinance.download`` on a cache miss.
    cache_dir
        Directory holding the CSV cache.  Created if absent.
    refresh
        Ignore any cached file and re-download.

    Returns
    -------
    pandas.Series
        Adjusted close prices indexed by date, name set to ``ticker``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ticker.lstrip('^')}_{interval}_{period}.csv"

    if cache.exists() and not refresh:
        frame = pd.read_csv(cache, index_col=0, parse_dates=True)
        series = frame.iloc[:, 0]
    else:
        import yfinance as yf  # imported lazily so a cache hit needs no network

        raw = yf.download(
            tickers=ticker,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=True,
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError(
                f"yfinance returned no data for {ticker!r}. Check the symbol and "
                f"your network connection, or supply a cached CSV at {cache}."
            )
        close = raw["Close"]
        series = close[ticker] if isinstance(close, pd.DataFrame) else close
        series = series.dropna()
        series.to_csv(cache, header=True)

    series.name = ticker
    series.index.name = "Date"
    return series.astype(float)


def simple_returns(prices: pd.Series) -> pd.Series:
    """Daily simple returns, first observation dropped."""
    return prices.pct_change().dropna()


def levered_series(prices: pd.Series, leverage: float = 1.0) -> pd.Series:
    """Daily-rebalanced levered index built from ``prices``, starting at 1.0.

    This is a gross-of-cost proxy: it captures volatility drag but not the
    financing cost of the leverage or an ETF expense ratio, so realised levered
    returns are strictly worse than this series.
    """
    return (simple_returns(prices) * leverage + 1.0).cumprod()
