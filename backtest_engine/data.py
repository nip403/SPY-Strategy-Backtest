import pandas as pd
from datetime import datetime
import os

def request(*, ticker: str = "SPY", config: dict, start: datetime = datetime(2017, 1, 1), end: datetime = datetime(2026, 6, 20), use_cache: bool = True) -> pd.DataFrame:
    """
    Request and lightly preprocess intraday market data from the Alpaca API.

    Loads cached data when available or fetches minute bars from Alpaca before applying preprocessing and saving the result locally.

    ticker : str = "SPY"
        Equity symbol to request. Could be expanded to support multiple tickers in future.
    config : dict
        API credentials required for Alpaca access. Attrs {"key", "secret"}.
    start : datetime = datetime(2017, 1, 1)
        Start timestamp for requested market data.
    end : datetime = datetime(2026, 6, 20)
        End timestamp for requested market data.
    use_cache : bool = True
        Whether to use locally cached data when available.

    Returns pd.DataFrame
        Preprocessed intraday market data.
    """
    
    cache_filename = f"cache_{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
    
    if use_cache and os.path.exists(cache_filename):
        print(f"Loading data from local cache: {cache_filename}")
        return pd.read_parquet(cache_filename)

    # module-level import deferred to cache misses only
    from alpaca.data.historical.stock import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    print("Cache not found. Fetching from Alpaca API...")
    client = StockHistoricalDataClient(
        api_key=config["key"], 
        secret_key=config["secret"]
    )

    bars = client.get_stock_bars(
        StockBarsRequest(
            symbol_or_symbols=[ticker],
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
    )
    
    df = _preprocess(bars.df)
    
    print(f"Saving fetched data to local cache: {cache_filename}")
    df.to_parquet(cache_filename)
    
    return df

def _preprocess(alpaca_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw Alpaca bars into strategy-ready intraday market data.
    Hardcoded compatibility with US equities.

    Converts timestamps and filter to regular US equity trading hours.

    alpaca_df : pd.DataFrame
        Raw minute bar data returned by Alpaca.

    Returns pd.DataFrame
        Cleaned intraday OHLCV data.
    """
    
    df = alpaca_df.tz_convert("America/New_York", level="timestamp").droplevel("symbol")
    df = df[df.index.dayofweek < 5].between_time("09:30", "15:59")[["open", "high", "low", "close", "volume"]].sort_index()
    
    # minutes since midnight instead of raw datetime.time - ~6x less memory and much faster for downstream groupby (datetime.time interface never used)
    df["time"] = (df.index.hour * 60 + df.index.minute).astype("int16")

    return df.dropna()