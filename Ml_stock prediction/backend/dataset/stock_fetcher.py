import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

DEMO_CSV_PATH = os.path.join(os.path.dirname(__file__), "demo_stock_data.csv")

def generate_realistic_stock_dataset(ticker="AAPL", start_date="2022-01-01", end_date="2024-01-01", base_price=175.0):
    """
    Generates realistic historical daily OHLCV stock dataset for offline demo mode
    using geometric Brownian motion with mean-reversion and market cycles.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate business trading days
    date_range = pd.date_range(start=start, end=end, freq='B')
    n = len(date_range)
    if n == 0:
        n = 500
        date_range = pd.date_range(end=pd.Timestamp.today(), periods=n, freq='B')

    np.random.seed(sum(ord(c) for c in ticker) + 42)
    
    # Base price map for common tickers
    ticker_bases = {
        "AAPL": 175.0,
        "MSFT": 380.0,
        "GOOGL": 140.0,
        "AMZN": 155.0,
        "TSLA": 210.0,
        "NVDA": 480.0,
        "META": 320.0
    }
    price = ticker_bases.get(ticker.upper(), base_price)
    
    # Geometric Brownian Motion simulation
    dt = 1 / 252.0
    mu = 0.12 # 12% annual drift
    sigma = 0.28 # 28% volatility
    
    prices = [price]
    for i in range(1, n):
        # Add cyclical market components (quarterly earnings cycles + macro waves)
        cycle = 0.003 * np.sin(i / 15.0) + 0.002 * np.cos(i / 40.0)
        drift = (mu - 0.5 * sigma**2) * dt + cycle
        shock = sigma * np.sqrt(dt) * np.random.normal()
        new_p = prices[-1] * np.exp(drift + shock)
        prices.append(max(10.0, new_p))
        
    data = []
    for d, p in zip(date_range, prices):
        close_p = round(p, 2)
        daily_spread = p * (0.01 + 0.015 * np.random.rand())
        high_p = round(close_p + abs(daily_spread * np.random.uniform(0.3, 1.0)), 2)
        low_p = round(close_p - abs(daily_spread * np.random.uniform(0.3, 1.0)), 2)
        open_p = round(np.random.uniform(low_p, high_p), 2)
        vol = int(np.random.normal(55_000_000, 15_000_000))
        vol = max(5_000_000, vol)
        
        data.append({
            "Date": d.strftime("%Y-%m-%d"),
            "Open": open_p,
            "High": high_p,
            "Low": low_p,
            "Close": close_p,
            "Volume": vol
        })
        
    df = pd.DataFrame(data)
    return df

class StockFetcher:
    def __init__(self, demo_file_path=DEMO_CSV_PATH):
        self.demo_file_path = demo_file_path
        self._ensure_demo_dataset()

    def _ensure_demo_dataset(self):
        """Creates the default AAPL demo dataset if it does not already exist"""
        if not os.path.exists(self.demo_file_path):
            df = generate_realistic_stock_dataset("AAPL", "2022-01-01", "2024-01-01", base_price=175.0)
            df.to_csv(self.demo_file_path, index=False)

    def fetch_stock_data(self, ticker="AAPL", start_date="2022-01-01", end_date="2024-01-01"):
        """
        Fetches stock data using yfinance.
        If Yahoo Finance is unreachable (e.g. offline, rate limit, SSL proxy),
        smoothly falls back to authentic demo mode and marks is_demo: True.
        """
        ticker = ticker.strip().upper() if ticker else "AAPL"
        is_demo = False
        source = "Yahoo Finance (Live API)"
        
        try:
            # Attempt real yfinance download
            raw_df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
                timeout=5
            )
            
            if raw_df is not None and not raw_df.empty and len(raw_df) > 10:
                # Format dataframe
                # In modern yfinance multi-index columns may appear
                if isinstance(raw_df.columns, pd.MultiIndex):
                    raw_df.columns = raw_df.columns.get_level_values(0)
                    
                df = raw_df.reset_index()
                
                # Check column names
                col_map = {}
                for c in df.columns:
                    if str(c).lower() in ['date', 'datetime']:
                        col_map[c] = 'Date'
                    elif str(c).lower() == 'open':
                        col_map[c] = 'Open'
                    elif str(c).lower() == 'high':
                        col_map[c] = 'High'
                    elif str(c).lower() == 'low':
                        col_map[c] = 'Low'
                    elif str(c).lower() == 'close':
                        col_map[c] = 'Close'
                    elif str(c).lower() == 'volume':
                        col_map[c] = 'Volume'
                        
                df = df.rename(columns=col_map)
                
                # Verify required columns exist
                req = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
                if all(c in df.columns for c in req):
                    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
                    df = df[req].round(2)
                    return df, is_demo, source
        except Exception as e:
            # Fall through to demo fallback
            pass

        # Demo Fallback
        is_demo = True
        source = f"Demo Mode (Offline Market Simulator: {ticker})"
        df = generate_realistic_stock_dataset(ticker, start_date, end_date)
        return df, is_demo, source

    def parse_uploaded_csv(self, file_storage_or_path):
        """
        Parses and validates user-uploaded CSV dataset.
        Expected format: Date,Open,High,Low,Close,Volume
        """
        if isinstance(file_storage_or_path, str):
            df = pd.read_csv(file_storage_or_path)
        else:
            df = pd.read_csv(file_storage_or_path)

        # Standardize column headers
        df.columns = [c.strip().capitalize() for c in df.columns]
        required = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Uploaded CSV must contain header '{col}'. Found headers: {list(df.columns)}")

        # Format date
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        for num_col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')

        df = df.dropna(subset=required).sort_values('Date').reset_index(drop=True)
        return df, False, "User Uploaded CSV"
