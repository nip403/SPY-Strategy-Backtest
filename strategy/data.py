from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime

def request(*, ticker: str = "SPY", config: dict, start: datetime = datetime(2017, 1, 1), end: datetime = datetime(2026, 6, 20)) -> pd.DataFrame:
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
    
    return preprocess(bars.df)

def preprocess(alpaca_df: pd.DataFrame) -> pd.DataFrame:
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

def main() -> None:
    pass

if __name__ == "__main__":
    main()