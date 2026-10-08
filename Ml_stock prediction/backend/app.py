import os
import sys

# Ensure backend directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Prefer torch if tensorflow not present
if "KERAS_BACKEND" not in os.environ:
    try:
        import tensorflow
        os.environ["KERAS_BACKEND"] = "tensorflow"
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np

from preprocessing.data_processor import DataProcessor
from training.trainer import ModelTrainer
from prediction.predictor import ModelPredictor
from dataset.stock_fetcher import StockFetcher

# Directories
FRONTEND_DIR = os.path.join(parent_dir, "frontend")
if not os.path.exists(FRONTEND_DIR):
    FRONTEND_DIR = parent_dir # Fallback to root if frontend files are in root

MODELS_DIR = os.path.join(current_dir, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# Global Application State
stock_fetcher = StockFetcher()
data_processor = DataProcessor(sequence_length=60)
trainer = ModelTrainer(models_dir=MODELS_DIR)
predictor = ModelPredictor(models_dir=MODELS_DIR, data_processor=data_processor)

app_data = {
    "current_ticker": "AAPL",
    "raw_df": None,
    "processed_data": None,
    "is_demo": False,
    "data_source": "None",
    "last_evaluation": None
}

def load_initial_dataset():
    """Initializes with default stock dataset on server startup instantly"""
    try:
        if os.path.exists(stock_fetcher.demo_file_path):
            df = pd.read_csv(stock_fetcher.demo_file_path)
            is_demo = True
            source = "Demo Mode (Bundled Dataset: AAPL)"
        else:
            df, is_demo, source = stock_fetcher.fetch_stock_data("AAPL", "2022-01-01", "2024-01-01")
            
        app_data["raw_df"] = df
        app_data["is_demo"] = is_demo
        app_data["data_source"] = source
        app_data["processed_data"] = data_processor.process_for_training(df, seq_len=60)
        print(f"[*] Initialized dataset: {len(df)} records. Source: {source}")
    except Exception as e:
        print(f"[!] Warning during initial load: {e}")

# Frontend Routes
@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(FRONTEND_DIR, path)

# REST API Endpoints

@app.route("/api/stock-data", methods=["GET"])
def get_stock_data():
    """Fetches stock data for a given ticker and date range"""
    ticker = request.args.get("ticker", "AAPL").strip().upper()
    start_date = request.args.get("start", "2022-01-01")
    end_date = request.args.get("end", "2024-01-01")
    
    try:
        df, is_demo, source = stock_fetcher.fetch_stock_data(ticker, start_date, end_date)
        app_data["current_ticker"] = ticker
        app_data["raw_df"] = df
        app_data["is_demo"] = is_demo
        app_data["data_source"] = source
        
        # Reprocess with current sequence length
        seq_len = data_processor.sequence_length
        app_data["processed_data"] = data_processor.process_for_training(df, seq_len=seq_len)
        
        stats = data_processor.get_dataset_stats(df)
        records = df.to_dict(orient="records")
        
        return jsonify({
            "status": "success",
            "ticker": ticker,
            "is_demo": is_demo,
            "data_source": source,
            "stats": stats,
            "preprocessing_info": app_data["processed_data"]["info"],
            "total_records": len(records),
            "data": records[:150], # Sample preview for speed
            "all_dates": [r["Date"] for r in records],
            "all_closes": [r["Close"] for r in records]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/upload-dataset", methods=["POST"])
def upload_dataset():
    """Allows uploading custom CSV dataset"""
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "Empty filename"}), 400

    try:
        df, is_demo, source = stock_fetcher.parse_uploaded_csv(file)
        app_data["current_ticker"] = os.path.splitext(file.filename)[0].upper()
        app_data["raw_df"] = df
        app_data["is_demo"] = is_demo
        app_data["data_source"] = source
        
        seq_len = data_processor.sequence_length
        app_data["processed_data"] = data_processor.process_for_training(df, seq_len=seq_len)
        
        stats = data_processor.get_dataset_stats(df)
        records = df.to_dict(orient="records")
        
        return jsonify({
            "status": "success",
            "ticker": app_data["current_ticker"],
            "is_demo": is_demo,
            "data_source": source,
            "stats": stats,
            "preprocessing_info": app_data["processed_data"]["info"],
            "total_records": len(records),
            "data": records[:150],
            "all_dates": [r["Date"] for r in records],
            "all_closes": [r["Close"] for r in records]
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/preprocess", methods=["POST"])
def preprocess_dataset():
    """Configures preprocessing pipeline and sequence length (20, 30, 60, 90)"""
    if app_data["raw_df"] is None:
        return jsonify({"status": "error", "message": "No dataset loaded. Please load or download data first."}), 400
        
    payload = request.get_json(silent=True) or {}
    seq_len = int(payload.get("sequence_length", 60))
    
    try:
        app_data["processed_data"] = data_processor.process_for_training(app_data["raw_df"], seq_len=seq_len)
        return jsonify({
            "status": "success",
            "message": f"Dataset preprocessed with sequence length {seq_len}",
            "info": app_data["processed_data"]["info"],
            "stats": data_processor.get_dataset_stats(app_data["raw_df"])
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/train/lstm", methods=["POST"])
def train_lstm():
    """Initiates asynchronous LSTM training"""
    if app_data["processed_data"] is None:
        return jsonify({"status": "error", "message": "Please preprocess dataset before training."}), 400

    config = request.get_json(silent=True) or {}
    pdata = app_data["processed_data"]
    
    def on_done():
        app_data["last_evaluation"] = predictor.evaluate_both(
            pdata["X_test"], pdata["actual_test_prices"]
        )

    success, msg = trainer.train_async(
        "lstm", pdata["X_train"], pdata["y_train"], config, on_complete=on_done
    )
    return jsonify({"status": "success" if success else "error", "message": msg})

@app.route("/api/train/gru", methods=["POST"])
def train_gru():
    """Initiates asynchronous GRU training"""
    if app_data["processed_data"] is None:
        return jsonify({"status": "error", "message": "Please preprocess dataset before training."}), 400

    config = request.get_json(silent=True) or {}
    pdata = app_data["processed_data"]
    
    def on_done():
        app_data["last_evaluation"] = predictor.evaluate_both(
            pdata["X_test"], pdata["actual_test_prices"]
        )

    success, msg = trainer.train_async(
        "gru", pdata["X_train"], pdata["y_train"], config, on_complete=on_done
    )
    return jsonify({"status": "success" if success else "error", "message": msg})

@app.route("/api/train/both", methods=["POST"])
def train_both():
    """Initiates sequential training for both LSTM and GRU"""
    if app_data["processed_data"] is None:
        return jsonify({"status": "error", "message": "Please preprocess dataset before training."}), 400

    config = request.get_json(silent=True) or {}
    pdata = app_data["processed_data"]
    
    def on_done():
        app_data["last_evaluation"] = predictor.evaluate_both(
            pdata["X_test"], pdata["actual_test_prices"]
        )

    success, msg = trainer.train_async(
        "both", pdata["X_train"], pdata["y_train"], config, on_complete=on_done
    )
    return jsonify({"status": "success" if success else "error", "message": msg})

@app.route("/api/train/status", methods=["GET"])
def get_training_status():
    """Polls real-time training progress (epoch, loss, val_loss, percent)"""
    state = trainer.get_state()
    return jsonify(state)

@app.route("/api/results", methods=["GET"])
def get_results():
    """Returns evaluation metrics, actual test prices, and model predictions"""
    if app_data["processed_data"] is None:
        return jsonify({"status": "error", "message": "No dataset processed"}), 400

    pdata = app_data["processed_data"]
    eval_res = predictor.evaluate_both(pdata["X_test"], pdata["actual_test_prices"])
    app_data["last_evaluation"] = eval_res

    return jsonify({
        "status": "success",
        "ticker": app_data["current_ticker"],
        "is_demo": app_data["is_demo"],
        "test_dates": pdata["test_dates"],
        "actual_prices": [round(float(p), 2) for p in pdata["actual_test_prices"]],
        "evaluation": eval_res,
        "preprocessing_info": pdata["info"]
    })

@app.route("/api/predict", methods=["POST"])
@app.route("/api/predict-future", methods=["POST"])
def predict_future():
    """Generates multi-step future stock price forecasts recursively (1, 5, 7, 30 days)"""
    if app_data["processed_data"] is None:
        return jsonify({"status": "error", "message": "Dataset not ready"}), 400

    payload = request.get_json(silent=True) or {}
    days = int(payload.get("days", 30))
    model_type = payload.get("model", "lstm").lower()
    
    pdata = app_data["processed_data"]
    last_seq = pdata["X_test"][-1]
    last_date = pdata["test_dates"][-1] if pdata["test_dates"] else None

    try:
        future_forecast = predictor.predict_future(
            model_type=model_type,
            last_sequence=last_seq,
            days=days,
            last_date_str=last_date
        )
        return jsonify({
            "status": "success",
            "model": model_type.upper(),
            "days": days,
            "forecast": future_forecast
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/api/training-history", methods=["GET"])
def get_training_history():
    """Returns epoch-by-epoch loss and validation loss for LSTM and GRU"""
    state = trainer.get_state()
    return jsonify({
        "status": "success",
        "history": state["history"]
    })

if __name__ == "__main__":
    load_initial_dataset()
    print("================================================================")
    print("  AI Stock Price Predictor (LSTM & GRU) - Server Starting")
    print("  Local URL: http://127.0.0.1:5000")
    print("================================================================")
    app.run(host="0.0.0.0", port=5000, debug=False)
