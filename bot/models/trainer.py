"""
Orchestrates model training across all configured trading pairs.
Handles data fetching, feature engineering, labeling, and ensemble training.
"""
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd

from bot.config import config
from bot.data_collector import DataCollector
from bot.feature_engineer import compute_features, build_labels, align_multi_timeframe
from bot.logger import logger
from bot.models.ensemble_model import EnsembleModel


class Trainer:
    def __init__(self, data_collector: DataCollector) -> None:
        self.collector = data_collector
        self.models: Dict[str, EnsembleModel] = {}

    def train_pair(self, product_id: str, force_retrain: bool = False) -> Optional[EnsembleModel]:
        """
        Trains (or loads) an ensemble model for a given trading pair.
        Returns the trained model or None on failure.
        """
        model = EnsembleModel()

        # Try to load existing model unless forced
        if not force_retrain and model.load():
            self.models[product_id] = model
            logger.info("Loaded existing model for %s", product_id)
            return model

        logger.info("Training new model for %s...", product_id)

        # Fetch multi-timeframe data
        tf_data = self.collector.fetch_multi_timeframe(product_id)
        if "1h" not in tf_data or tf_data["1h"].empty:
            logger.error("Insufficient data to train for %s", product_id)
            return None

        base_df = tf_data["1h"]
        if len(base_df) < config.lstm_lookback + 50:
            logger.error("Not enough candles (%d) to train LSTM for %s", len(base_df), product_id)
            return None

        # Build feature matrix
        features = compute_features(base_df, prefix="1h_")

        # Merge higher-timeframe features (15m and 5m forward-filled)
        for label, df_tf in tf_data.items():
            if label != "1h" and not df_tf.empty:
                htf_feats = align_multi_timeframe(base_df, df_tf, prefix=f"{label}_")
                features = features.join(htf_feats, how="left")

        # Drop ATR raw cols (used for TP/SL but not ML features)
        atr_cols = [c for c in features.columns if "atr_raw" in c]
        features = features.drop(columns=atr_cols, errors="ignore")

        # Build labels: 1 if price rises ≥ 0.5% over next 3 candles
        labels = build_labels(base_df, lookahead=3, threshold=0.005)

        # Align and drop NaNs
        combined = features.join(labels.rename("label"), how="inner")
        combined = combined.dropna()

        if len(combined) < config.lstm_lookback + 100:
            logger.error("Not enough clean rows (%d) after NaN drop for %s", len(combined), product_id)
            return None

        X = combined.drop(columns=["label"])
        y = combined["label"]

        pos_rate = y.mean()
        logger.info(
            "Training data for %s: %d samples, %.1f%% positive labels",
            product_id, len(y), pos_rate * 100
        )

        try:
            metrics = model.train(X, y)
            self.models[product_id] = model
            logger.info("Model ready for %s | metrics=%s", product_id, metrics)
            return model
        except Exception as e:
            logger.error("Training failed for %s: %s", product_id, e)
            return None

    def train_all(self, force_retrain: bool = False) -> None:
        for pair in config.trading_pairs:
            logger.info("=" * 60)
            logger.info("Training %s", pair)
            self.train_pair(pair, force_retrain=force_retrain)
            time.sleep(1)  # brief pause between pairs

    def get_model(self, product_id: str) -> Optional[EnsembleModel]:
        return self.models.get(product_id)
