import yfinance
import pandas as pd


class FetchData:
    def __init__(self, symbol:str) -> None:
        self.symbol = symbol

    def fetch(self):
        data = yfinance.download(tickers=[self.symbol], start='2024-07-01', end='2024-08-01', period='1d')
        data = data.reset_index()
        return data


