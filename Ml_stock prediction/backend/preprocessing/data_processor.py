import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class DataProcessor:
    def __init__(self, sequence_length=60, train_split=0.8):
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.is_fitted = False
        self.feature_columns = ['Close']
        self.target_column = 'Close'

    def clean_and_prepare(self, df: pd.DataFrame):
        """
        Cleans raw stock dataframe:
        - Checks required columns: Date, Open, High, Low, Close, Volume
        - Ensures Date is datetime and sorts chronologically
        - Handles missing values using forward fill then backward fill
        """
        data = df.copy()
        
        # Standardize column names (strip whitespace, capitalize)
        data.columns = [c.strip().capitalize() for c in data.columns]
        
        required = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required:
            if col not in data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Date column handling
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
        elif not isinstance(data.index, pd.DatetimeIndex):
            # If Date is index or absent
            if 'Date' in df.columns:
                data['Date'] = pd.to_datetime(df['Date'])
            else:
                data['Date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(data), freq='B')

        # Convert numeric columns
        for col in required:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Record missing value count before fixing
        missing_count = int(data[required].isna().sum().sum())
        
        # Fix missing values: forward fill followed by backward fill
        data = data.ffill().bfill()
        
        # Drop any remaining NaN rows if dataset was empty
        data = data.dropna(subset=required).reset_index(drop=True)
        
        return data, missing_count

    def get_dataset_stats(self, df: pd.DataFrame):
        """Returns statistical summary for the UI dataset tab"""
        stats = {
            "total_rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df['Date'].min())[:10] if 'Date' in df.columns else "N/A",
                "end": str(df['Date'].max())[:10] if 'Date' in df.columns else "N/A"
            },
            "close_avg": float(round(df['Close'].mean(), 2)) if len(df) > 0 else 0,
            "close_max": float(round(df['Close'].max(), 2)) if len(df) > 0 else 0,
            "close_min": float(round(df['Close'].min(), 2)) if len(df) > 0 else 0,
            "volume_avg": float(round(df['Volume'].mean(), 0)) if len(df) > 0 else 0
        }
        return stats

    def create_sequences(self, data_series, seq_len):
        """
        Creates sliding window sequences (X) and targets (y)
        X shape: (samples, seq_len, 1)
        y shape: (samples,)
        """
        X, y = [], []
        for i in range(seq_len, len(data_series)):
            X.append(data_series[i - seq_len:i])
            y.append(data_series[i, 0])
        return np.array(X), np.array(y)

    def process_for_training(self, df: pd.DataFrame, seq_len=None):
        """
        Full preprocessing pipeline:
        1. Clean and sort
        2. Normalize Close prices using MinMaxScaler
        3. Split chronologically into Train and Test (80% / 20%)
        4. Generate sliding-window time-series sequences
        """
        if seq_len is not None:
            self.sequence_length = seq_len
            
        data, missing_count = self.clean_and_prepare(df)
        
        close_prices = data[['Close']].values
        
        # Chronological train/test split index
        train_size = int(len(close_prices) * self.train_split)
        if train_size <= self.sequence_length:
            raise ValueError(
                f"Dataset size ({len(close_prices)}) too small for sequence length {self.sequence_length}. "
                f"Please provide at least {self.sequence_length * 2} data points."
            )
            
        train_data = close_prices[:train_size]
        test_data = close_prices[train_size - self.sequence_length:] # Include overlap for test sequences
        
        # Fit scaler ONLY on train data to prevent data leakage!
        scaled_train = self.scaler.fit_transform(train_data)
        scaled_test = self.scaler.transform(test_data)
        self.is_fitted = True
        
        # Create sequences
        X_train, y_train = self.create_sequences(scaled_train, self.sequence_length)
        X_test, y_test = self.create_sequences(scaled_test, self.sequence_length)
        
        # Actual unscaled test prices for evaluation
        actual_test_prices = close_prices[train_size:].flatten()
        test_dates = [str(d)[:10] for d in data['Date'].iloc[train_size:]] if 'Date' in data.columns else []

        info = {
            "total_records": len(data),
            "train_records": len(X_train),
            "test_records": len(X_test),
            "sequence_length": self.sequence_length,
            "missing_values_handled": missing_count,
            "normalization": "MinMaxScaler (0, 1)",
            "target_feature": "Close",
            "features_selected": ["Close"]
        }
        
        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "actual_test_prices": actual_test_prices,
            "test_dates": test_dates,
            "info": info,
            "cleaned_df": data
        }

    def inverse_transform(self, scaled_values):
        """Inverse transform scaled price back to actual dollar price"""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet.")
        arr = np.array(scaled_values).reshape(-1, 1)
        return self.scaler.inverse_transform(arr).flatten()

    def transform(self, values):
        """Transform raw prices to scaled values"""
        if not self.is_fitted:
            raise ValueError("Scaler is not fitted yet.")
        arr = np.array(values).reshape(-1, 1)
        return self.scaler.transform(arr)

    def save_scaler(self, filepath):
        """Save fitted scaler to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.scaler, filepath)

    def load_scaler(self, filepath):
        """Load fitted scaler from disk"""
        if os.path.exists(filepath):
            self.scaler = joblib.load(filepath)
            self.is_fitted = True
            return True
        return False
