import os
import numpy as np
import pandas as pd
from datetime import timedelta
import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error

class ModelPredictor:
    def __init__(self, models_dir="backend/models", data_processor=None):
        self.models_dir = models_dir
        self.data_processor = data_processor

    def load_model(self, model_type):
        """Loads saved .keras model from disk"""
        path = os.path.join(self.models_dir, f"{model_type.lower()}_model.keras")
        if not os.path.exists(path):
            return None
        return keras.models.load_model(path)

    def evaluate_model(self, model, X_test, actual_prices):
        """
        Runs inference on test set and computes real evaluation metrics:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error)
        - MAE (Mean Absolute Error)
        - MAPE (Mean Absolute Percentage Error)
        - Directional Accuracy (%)
        """
        # Model predictions (scaled)
        scaled_preds = model.predict(X_test, verbose=0)
        
        # Inverse transform to original stock price ($)
        pred_prices = self.data_processor.inverse_transform(scaled_preds)
        
        # Ensure identical array length
        min_len = min(len(pred_prices), len(actual_prices))
        pred_sub = pred_prices[:min_len]
        act_sub = actual_prices[:min_len]
        
        # Compute exact sklearn metrics
        mse = float(mean_squared_error(act_sub, pred_sub))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(act_sub, pred_sub))
        
        # Mean absolute percentage error
        mape = float(np.mean(np.abs((act_sub - pred_sub) / (act_sub + 1e-8))) * 100)
        
        # Directional Accuracy (trend prediction)
        if min_len > 1:
            act_diff = np.diff(act_sub)
            pred_diff = np.diff(pred_sub)
            directional_accuracy = float(np.mean((act_diff > 0) == (pred_diff > 0)) * 100)
        else:
            directional_accuracy = 100.0

        return {
            "predicted_prices": [round(float(p), 2) for p in pred_sub],
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "mape": round(mape, 2),
            "directional_accuracy": round(directional_accuracy, 1)
        }

    def evaluate_both(self, X_test, actual_prices):
        """Evaluates both LSTM and GRU, and automatically identifies the better performer"""
        lstm_model = self.load_model("lstm")
        gru_model = self.load_model("gru")
        
        results = {
            "lstm": None,
            "gru": None,
            "best_model": None,
            "comparison_reason": None
        }
        
        if lstm_model is not None:
            results["lstm"] = self.evaluate_model(lstm_model, X_test, actual_prices)
        if gru_model is not None:
            results["gru"] = self.evaluate_model(gru_model, X_test, actual_prices)
            
        if results["lstm"] and results["gru"]:
            lstm_rmse = results["lstm"]["rmse"]
            gru_rmse = results["gru"]["rmse"]
            if lstm_rmse <= gru_rmse:
                results["best_model"] = "LSTM"
                diff = round(gru_rmse - lstm_rmse, 4)
                results["comparison_reason"] = f"LSTM achieved lower RMSE by {diff} ({lstm_rmse} vs {gru_rmse})"
            else:
                results["best_model"] = "GRU"
                diff = round(lstm_rmse - gru_rmse, 4)
                results["comparison_reason"] = f"GRU achieved lower RMSE by {diff} ({gru_rmse} vs {lstm_rmse})"
        elif results["lstm"]:
            results["best_model"] = "LSTM"
            results["comparison_reason"] = "LSTM evaluated successfully"
        elif results["gru"]:
            results["best_model"] = "GRU"
            results["comparison_reason"] = "GRU evaluated successfully"
            
        return results

    def predict_future(self, model_type, last_sequence, days=30, last_date_str=None):
        """
        Performs recursive multi-step future forecasting:
        Takes the last sequence_length historical points, iteratively predicts next day,
        updates the rolling window, and predicts day t+1, t+2, ..., t+N.
        """
        model = self.load_model(model_type)
        if model is None:
            raise ValueError(f"Model {model_type} has not been trained yet.")
            
        current_seq = np.copy(last_sequence) # Shape: (seq_len, 1)
        future_scaled = []
        
        for _ in range(days):
            # Input to model: (1, seq_len, 1)
            model_input = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
            pred = model.predict(model_input, verbose=0) # Shape: (1, 1)
            pred_val = pred[0, 0]
            future_scaled.append(pred_val)
            
            # Roll sequence: drop oldest, append newly predicted scaled price
            current_seq = np.append(current_seq[1:], [[pred_val]], axis=0)

        # Invert scaling
        future_prices = self.data_processor.inverse_transform(future_scaled)
        
        # Generate future business trading dates
        start_date = pd.to_datetime(last_date_str) if last_date_str else pd.Timestamp.today()
        future_dates = []
        curr = start_date
        while len(future_dates) < days:
            curr += timedelta(days=1)
            # Skip Saturday (5) and Sunday (6) for financial markets
            if curr.weekday() < 5:
                future_dates.append(curr.strftime("%Y-%m-%d"))

        return [
            {"date": d, "predicted_price": round(float(p), 2)}
            for d, p in zip(future_dates, future_prices)
        ]
