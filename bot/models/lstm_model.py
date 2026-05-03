"""
LSTM neural network for sequence-based price direction prediction.
Architecture: Stacked LSTM → Dropout → Dense → Sigmoid
"""
import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers

    HAS_TF = True
except ImportError:
    HAS_TF = False

from bot.config import config
from bot.logger import logger

MODEL_PATH = os.path.join("saved_models", "lstm_model.keras")


class LSTMModel:
    def __init__(self) -> None:
        if not HAS_TF:
            raise ImportError("TensorFlow is required for LSTMModel. Install with: pip install tensorflow")
        self.model: Optional[keras.Model] = None
        self.lookback = config.lstm_lookback
        self.units = config.lstm_units
        self.dropout = config.lstm_dropout
        self._is_trained = False

    def build(self, n_features: int) -> None:
        inp = keras.Input(shape=(self.lookback, n_features))

        # First LSTM layer — return sequences for stacking
        x = layers.LSTM(
            self.units,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4),
        )(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)

        # Second LSTM layer
        x = layers.LSTM(
            self.units // 2,
            return_sequences=True,
            kernel_regularizer=regularizers.l2(1e-4),
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)

        # Third LSTM layer
        x = layers.LSTM(self.units // 4, return_sequences=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.dropout)(x)

        # Dense head
        x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(self.dropout / 2)(x)
        x = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid")(x)

        self.model = keras.Model(inputs=inp, outputs=out)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )
        logger.info("LSTM built: lookback=%d, features=%d, units=%d", self.lookback, n_features, self.units)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> dict:
        if self.model is None:
            self.build(X_train.shape[2])

        # Compute class weights to handle imbalance
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        class_weight = {0: 1.0, 1: n_neg / (n_pos + 1e-10)}

        cb_list = [
            callbacks.EarlyStopping(
                monitor="val_auc", patience=config.early_stopping_patience,
                restore_best_weights=True, mode="max"
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, min_lr=1e-6, verbose=0
            ),
            callbacks.ModelCheckpoint(
                MODEL_PATH, monitor="val_auc", save_best_only=True, mode="max", verbose=0
            ),
        ]

        os.makedirs("saved_models", exist_ok=True)
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=config.epochs,
            batch_size=config.batch_size,
            class_weight=class_weight,
            callbacks=cb_list,
            verbose=0,
        )
        self._is_trained = True

        best_val_auc = max(history.history.get("val_auc", [0]))
        best_val_acc = max(history.history.get("val_accuracy", [0]))
        logger.info("LSTM trained | best val_auc=%.4f | best val_acc=%.4f", best_val_auc, best_val_acc)
        return history.history

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None or not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        probs = self.model.predict(X, verbose=0).flatten()
        return probs

    def save(self, path: str = MODEL_PATH) -> None:
        if self.model:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.model.save(path)
            logger.info("LSTM saved to %s", path)

    def load(self, path: str = MODEL_PATH) -> bool:
        if os.path.exists(path):
            self.model = keras.models.load_model(path)
            self._is_trained = True
            logger.info("LSTM loaded from %s", path)
            return True
        return False

    @property
    def is_trained(self) -> bool:
        return self._is_trained


def make_sequences(
    features: np.ndarray, labels: np.ndarray, lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts flat feature array into (samples, lookback, features) sequences."""
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback : i])
        y.append(labels[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
