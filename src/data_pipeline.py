import yfinance as yf
import pandas as pd
import os

class EurozoneDataPipeline:
    def __init__(self, start_date="2008-01-01", end_date="2024-01-01"):
        print("🌍 Initializing Global Macro-Finance Pipeline...")
        self.start_date = start_date
        self.end_date = end_date
        self.data_dir = "../data"
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
        # Swapped VSTOXX for VIX to ensure a stable, reliable data feed
        self.tickers = {
            "Euro_Stoxx_50": "^STOXX50E",
            "EUR_USD": "EURUSD=X",
            "VIX": "^VIX"
        }

    def fetch_data(self):
        """Downloads historical data from Yahoo Finance and cleans it."""
        print(f"📡 Fetching data from {self.start_date} to {self.end_date}...")
        raw_data = pd.DataFrame()
        
        for name, ticker in self.tickers.items():
            print(f"   -> Downloading {name} ({ticker})")
            # Using standard 'Close' price to avoid Adjusted Close errors on currencies/indices
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)['Close']
            raw_data[name] = df
            
        # FIX: Updated to the modern pandas syntax for forward-filling data
        clean_data = raw_data.ffill().dropna()
        return clean_data

    def calculate_returns(self, df):
        """Calculates the daily percentage returns."""
        print("🧮 Calculating daily logarithmic returns...")
        returns_df = pd.DataFrame()
        returns_df['Stoxx_Return'] = df['Euro_Stoxx_50'].pct_change()
        returns_df['EURUSD_Return'] = df['EUR_USD'].pct_change()
        returns_df['VIX_Change'] = df['VIX'].diff()
        
        return returns_df.dropna()

    def save_to_csv(self, df, filename="eurozone_macro_data.csv"):
        """Saves the cleaned dataset to the data folder."""
        filepath = os.path.join(os.path.dirname(__file__), '..', 'data', filename)
        df.to_csv(filepath)
        print(f"✅ Data successfully saved to {filepath}")


if __name__ == "__main__":
    pipeline = EurozoneDataPipeline(start_date="2010-01-01", end_date="2026-01-01")
    macro_prices = pipeline.fetch_data()
    macro_returns = pipeline.calculate_returns(macro_prices)
    pipeline.save_to_csv(macro_returns)