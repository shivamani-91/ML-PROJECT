import os
if "KERAS_BACKEND" not in os.environ:
    try:
        import tensorflow
        os.environ["KERAS_BACKEND"] = "tensorflow"
    except ImportError:
        os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, models, optimizers

def build_gru_model(input_shape=(60, 1), units=50, num_layers=2, dropout=0.2, learning_rate=0.001):
    """
    Builds a deep GRU (Gated Recurrent Unit) neural network for time-series prediction.
    Architecture:
    Input -> GRU -> Dropout -> GRU -> Dropout -> Dense(25) -> Dense(1)
    """
    model = models.Sequential(name="GRU_Stock_Predictor")
    model.add(layers.Input(shape=input_shape))
    
    # Layer 1
    return_seq = num_layers > 1
    model.add(layers.GRU(units=units, return_sequences=return_seq))
    model.add(layers.Dropout(rate=dropout))
    
    # Intermediate layers if num_layers > 2
    for i in range(1, num_layers - 1):
        model.add(layers.GRU(units=units, return_sequences=True))
        model.add(layers.Dropout(rate=dropout))
        
    # Last recurrent layer if num_layers >= 2
    if num_layers >= 2:
        model.add(layers.GRU(units=units, return_sequences=False))
        model.add(layers.Dropout(rate=dropout))
        
    # Dense regression head
    model.add(layers.Dense(units=25, activation="relu"))
    model.add(layers.Dense(units=1))
    
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mean_squared_error", metrics=["mae"])
    return model
