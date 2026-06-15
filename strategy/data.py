import yfinance as yf
import numpy as np
import pandas as pd

from pathlib import Path

def main() -> None:
    #ticker = yf.Ticker("SPY")
    
    #data = ticker.history(period="max")
    
    data = pd.read_csv(Path(__file__).parent / "1_min_SPY_2008-2021.csv")
    print(data.head(100))

if __name__ == "__main__":
    main()