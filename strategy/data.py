from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime
import os

def request(*, ticker: str = "SPY", config: dict, start: datetime = datetime(2017, 1, 1), end: datetime = datetime(2026, 6, 20), use_cache: bool = True) -> pd.DataFrame:
    cache_filename = f"cache_{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
    
    if use_cache and os.path.exists(cache_filename):
        print(f"Loading data from local cache: {cache_filename}")
        return pd.read_parquet(cache_filename)
        
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
    US equity intraday minute data
    """
    
    df = alpaca_df.tz_convert("America/New_York", level="timestamp").droplevel("symbol")
    df = df[df.index.dayofweek < 5].between_time("09:30", "15:59")[["open", "high", "low", "close", "volume"]].sort_index()

    pv = df["volume"] * (df["high"] + df["low"] + df["close"]) / 3 
    df["vwap"] = pv.groupby(df.index.date).cumsum() / df["volume"].groupby(df.index.date).cumsum()

    df["daily_open"] = df.groupby(pd.Grouper(freq="D"))["open"].transform("first")

    closes = df.groupby(df.index.date)["close"].last()
    df["prev_close"] = pd.Series(df.index.date, index=df.index).map(closes.shift(1))

    df["time"] = df.index.time
    
    return df.dropna()