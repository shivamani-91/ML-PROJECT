import os
import time
import threading
import keras
from keras import callbacks
from .lstm_model import build_lstm_model
from .gru_model import build_gru_model

class ProgressCallback(callbacks.Callback):
    def __init__(self, model_name, total_epochs, state_tracker):
        super().__init__()
        self.model_name = model_name
        self.total_epochs = total_epochs
        self.state_tracker = state_tracker

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current_epoch = epoch + 1
        loss = float(logs.get("loss", 0.0))
        val_loss = float(logs.get("val_loss", 0.0))
        mae = float(logs.get("mae", 0.0))
        val_mae = float(logs.get("val_mae", 0.0))
        
        # Update progress in thread-safe dictionary
        self.state_tracker["current_epoch"] = current_epoch
        self.state_tracker["total_epochs"] = self.total_epochs
        self.state_tracker["percent"] = int((current_epoch / self.total_epochs) * 100)
        self.state_tracker["loss"] = round(loss, 6)
        self.state_tracker["val_loss"] = round(val_loss, 6)
        self.state_tracker["history"]["epochs"].append(current_epoch)
        self.state_tracker["history"]["loss"].append(round(loss, 6))
        self.state_tracker["history"]["val_loss"].append(round(val_loss, 6))
        self.state_tracker["status_text"] = f"Epoch {current_epoch}/{self.total_epochs} - Loss: {loss:.5f} - Val Loss: {val_loss:.5f}"

class ModelTrainer:
    def __init__(self, models_dir="backend/models"):
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        self.active_training = False
        self.training_thread = None
        self.state = {
            "status": "idle", # idle, training, completed, error
            "current_model": None, # 'lstm', 'gru', 'both'
            "active_model_name": "",
            "current_epoch": 0,
            "total_epochs": 0,
            "percent": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "status_text": "Ready to train",
            "history": {
                "lstm": {"epochs": [], "loss": [], "val_loss": []},
                "gru": {"epochs": [], "loss": [], "val_loss": []}
            },
            "error": None
        }

    def get_state(self):
        return self.state

    def reset_state(self, model_type):
        self.state["status"] = "training"
        self.state["current_model"] = model_type
        self.state["percent"] = 0
        self.state["error"] = None
        self.state["status_text"] = f"Initializing training for {model_type.upper()}..."

    def train_single_model(self, model_type, X_train, y_train, epochs=20, batch_size=32, units=50, layers=2, dropout=0.2, learning_rate=0.001):
        """Trains a single LSTM or GRU model synchronously"""
        input_shape = (X_train.shape[1], X_train.shape[2])
        
        if model_type.lower() == "lstm":
            model = build_lstm_model(input_shape=input_shape, units=units, num_layers=layers, dropout=dropout, learning_rate=learning_rate)
            save_path = os.path.join(self.models_dir, "lstm_model.keras")
        elif model_type.lower() == "gru":
            model = build_gru_model(input_shape=input_shape, units=units, num_layers=layers, dropout=dropout, learning_rate=learning_rate)
            save_path = os.path.join(self.models_dir, "gru_model.keras")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Initialize history tracking for this model
        tracker = {
            "current_epoch": 0,
            "total_epochs": epochs,
            "percent": 0,
            "loss": 0.0,
            "val_loss": 0.0,
            "status_text": f"Starting {model_type.upper()}...",
            "history": {"epochs": [], "loss": [], "val_loss": []}
        }
        
        self.state["active_model_name"] = model_type.upper()
        
        # Link tracker to global state
        callback = ProgressCallback(model_type, epochs, tracker)
        
        # Monitor thread periodically updates global state
        def sync_tracker():
            while self.active_training and tracker["current_epoch"] < epochs:
                self.state["current_epoch"] = tracker["current_epoch"]
                self.state["total_epochs"] = tracker["total_epochs"]
                self.state["percent"] = tracker["percent"]
                self.state["loss"] = tracker["loss"]
                self.state["val_loss"] = tracker["val_loss"]
                self.state["status_text"] = tracker["status_text"]
                self.state["history"][model_type.lower()] = tracker["history"]
                time.sleep(0.1)

        sync_t = threading.Thread(target=sync_tracker, daemon=True)
        sync_t.start()

        # Fit model
        history = model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[callback],
            verbose=0
        )
        
        # Final update
        self.state["history"][model_type.lower()] = tracker["history"]
        self.state["percent"] = 100
        
        # Save model
        model.save(save_path)
        return model, history

    def train_async(self, model_type, X_train, y_train, config, on_complete=None):
        """Runs training in background thread so API doesn't hang"""
        if self.active_training:
            return False, "Training is already in progress."

        def worker():
            self.active_training = True
            self.reset_state(model_type)
            try:
                epochs = int(config.get("epochs", 15))
                batch_size = int(config.get("batch_size", 32))
                units = int(config.get("units", 50))
                num_layers = int(config.get("layers", 2))
                dropout = float(config.get("dropout", 0.2))
                learning_rate = float(config.get("learning_rate", 0.001))

                if model_type.lower() in ["lstm", "gru"]:
                    self.train_single_model(
                        model_type.lower(), X_train, y_train,
                        epochs=epochs, batch_size=batch_size,
                        units=units, layers=num_layers,
                        dropout=dropout, learning_rate=learning_rate
                    )
                elif model_type.lower() == "both":
                    # First train LSTM
                    self.state["status_text"] = "Training LSTM model (1/2)..."
                    self.train_single_model(
                        "lstm", X_train, y_train,
                        epochs=epochs, batch_size=batch_size,
                        units=units, layers=num_layers,
                        dropout=dropout, learning_rate=learning_rate
                    )
                    # Next train GRU
                    self.state["status_text"] = "Training GRU model (2/2)..."
                    self.train_single_model(
                        "gru", X_train, y_train,
                        epochs=epochs, batch_size=batch_size,
                        units=units, layers=num_layers,
                        dropout=dropout, learning_rate=learning_rate
                    )

                self.state["status"] = "completed"
                self.state["status_text"] = f"Training completed successfully for {model_type.upper()}!"
                if on_complete:
                    on_complete()

            except Exception as e:
                self.state["status"] = "error"
                self.state["error"] = str(e)
                self.state["status_text"] = f"Training error: {str(e)}"
            finally:
                self.active_training = False

        self.training_thread = threading.Thread(target=worker, daemon=True)
        self.training_thread.start()
        return True, "Training started in background."
