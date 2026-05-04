"""
Ensemble model combining:
  1. Stacked LSTM neural network (sequence model)
  2. XGBoost gradient boosting (tabular)
  3. LightGBM gradient boosting (tabular)
  4. Random Forest (tabular, diversity)

Final signal = weighted soft-voting across all models.
"""
import os
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from bot.config import config
from bot.logger import logger
from bot.models.lstm_model import LSTMModel, make_sequences

SCALER_PATH = os.path.join("saved_models", "scaler.pkl")
XGB_PATH = os.path.join("saved_models", "xgb_model.pkl")
LGB_PATH = os.path.join("saved_models", "lgb_model.pkl")
RF_PATH = os.path.join("saved_models", "rf_model.pkl")
META_PATH = os.path.join("saved_models", "ensemble_meta.pkl")

# Model weights for soft voting (higher = more trust)
MODEL_WEIGHTS = {
    "lstm": 0.40,
    "xgb": 0.25,
    "lgb": 0.20,
    "rf": 0.15,
}


class EnsembleModel:
    def __init__(self) -> None:
        self.scaler = RobustScaler()
        try:
            self.lstm = LSTMModel()
        except Exception:
            self.lstm = None  # type: ignore[assignment]
        self.xgb_clf = None
        self.lgb_clf = None
        self.rf_clf = None
        self._trained = False
        self._feature_cols: List[str] = []

        self._init_tabular_models()

    def _init_tabular_models(self) -> None:
        if HAS_XGB:
            self.xgb_clf = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )

        if HAS_LGB:
            self.lgb_clf = lgb.LGBMClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            )

        self.rf_clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
    ) -> Dict[str, float]:
        """
        Train all ensemble members.
        features_df: full feature DataFrame (will be split internally)
        labels: binary 0/1 series aligned with features_df
        """
        # Drop rows with NaN labels or features
        valid_mask = labels.notna() & features_df.notna().all(axis=1)
        features_df = features_df[valid_mask]
        labels = labels[valid_mask]

        self._feature_cols = features_df.columns.tolist()
        X = features_df.values.astype(np.float32)
        y = labels.values.astype(np.float32)

        # Train/val split (time-ordered — no shuffling)
        split = int(len(X) * config.train_test_split)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        metrics: Dict[str, float] = {}

        # --- Train LSTM (optional — skipped if TensorFlow not installed) ---
        if self.lstm is not None:
            try:
                X_seq_train, y_seq_train = make_sequences(X_train_scaled, y_train, config.lstm_lookback)
                X_seq_val, y_seq_val = make_sequences(X_val_scaled, y_val, config.lstm_lookback)
                if len(X_seq_train) > config.lstm_lookback:
                    self.lstm.train(X_seq_train, y_seq_train, X_seq_val, y_seq_val)
                    lstm_preds = self.lstm.predict_proba(X_seq_val)
                    lstm_acc = ((lstm_preds > 0.5) == y_seq_val).mean()
                    metrics["lstm_val_acc"] = float(lstm_acc)
                    logger.info("LSTM val accuracy: %.4f", lstm_acc)
            except Exception as e:
                logger.warning("LSTM training skipped: %s", e)

        # --- Train tabular models ---
        if self.xgb_clf is not None:
            try:
                self.xgb_clf.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                    verbose=False,
                )
                xgb_acc = (self.xgb_clf.predict(X_val_scaled) == y_val).mean()
                metrics["xgb_val_acc"] = float(xgb_acc)
                logger.info("XGBoost val accuracy: %.4f", xgb_acc)
            except Exception as e:
                logger.error("XGBoost training failed: %s", e)

        if self.lgb_clf is not None:
            try:
                self.lgb_clf.fit(
                    X_train_scaled, y_train,
                    eval_set=[(X_val_scaled, y_val)],
                )
                lgb_acc = (self.lgb_clf.predict(X_val_scaled) == y_val).mean()
                metrics["lgb_val_acc"] = float(lgb_acc)
                logger.info("LightGBM val accuracy: %.4f", lgb_acc)
            except Exception as e:
                logger.error("LightGBM training failed: %s", e)

        try:
            self.rf_clf.fit(X_train_scaled, y_train)
            rf_acc = (self.rf_clf.predict(X_val_scaled) == y_val).mean()
            metrics["rf_val_acc"] = float(rf_acc)
            logger.info("RandomForest val accuracy: %.4f", rf_acc)
        except Exception as e:
            logger.error("RandomForest training failed: %s", e)

        self._trained = True
        self.save()
        logger.info("Ensemble training complete. Metrics: %s", metrics)
        return metrics

    def predict(self, features_df: pd.DataFrame) -> Tuple[float, float]:
        """
        Returns (signal, confidence).
        signal: 1.0 = buy, 0.0 = hold/sell
        confidence: 0.0 – 1.0 (probability of positive class)
        """
        if not self._trained:
            raise RuntimeError("Ensemble not trained. Call train() first.")

        # Align columns
        if self._feature_cols:
            missing = [c for c in self._feature_cols if c not in features_df.columns]
            if missing:
                for col in missing:
                    features_df[col] = 0.0
            features_df = features_df[self._feature_cols]

        X = features_df.values.astype(np.float32)
        X_scaled = self.scaler.transform(X)

        probs: Dict[str, float] = {}

        # LSTM — needs sequence of lookback candles
        if self.lstm is not None and self.lstm.is_trained and len(X_scaled) >= config.lstm_lookback:
            seq = X_scaled[-config.lstm_lookback :][np.newaxis, ...]  # (1, lookback, features)
            try:
                probs["lstm"] = float(self.lstm.predict_proba(seq)[0])
            except Exception as e:
                logger.warning("LSTM inference failed: %s", e)

        # Tabular models use only the latest candle
        latest = X_scaled[-1:, :]

        if self.xgb_clf is not None:
            try:
                probs["xgb"] = float(self.xgb_clf.predict_proba(latest)[0, 1])
            except Exception:
                pass

        if self.lgb_clf is not None:
            try:
                probs["lgb"] = float(self.lgb_clf.predict_proba(latest)[0, 1])
            except Exception:
                pass

        if self.rf_clf is not None:
            try:
                probs["rf"] = float(self.rf_clf.predict_proba(latest)[0, 1])
            except Exception:
                pass

        if not probs:
            return 0.0, 0.0

        # Weighted average
        total_weight = sum(MODEL_WEIGHTS[k] for k in probs)
        confidence = sum(probs[k] * MODEL_WEIGHTS[k] for k in probs) / total_weight

        signal = 1.0 if confidence >= config.min_confidence else 0.0
        logger.debug(
            "Ensemble probs=%s → confidence=%.4f signal=%.0f",
            {k: f"{v:.3f}" for k, v in probs.items()}, confidence, signal
        )
        return signal, confidence

    def save(self) -> None:
        os.makedirs("saved_models", exist_ok=True)
        joblib.dump(self.scaler, SCALER_PATH)
        if self.xgb_clf is not None:
            joblib.dump(self.xgb_clf, XGB_PATH)
        if self.lgb_clf is not None:
            joblib.dump(self.lgb_clf, LGB_PATH)
        if self.rf_clf is not None:
            joblib.dump(self.rf_clf, RF_PATH)
        meta = {
            "trained": self._trained,
            "feature_cols": self._feature_cols,
        }
        joblib.dump(meta, META_PATH)
        if self.lstm is not None:
            self.lstm.save()
        logger.info("Ensemble saved to saved_models/")

    def load(self) -> bool:
        if not os.path.exists(META_PATH):
            return False
        try:
            self.scaler = joblib.load(SCALER_PATH)
            if os.path.exists(XGB_PATH):
                self.xgb_clf = joblib.load(XGB_PATH)
            if os.path.exists(LGB_PATH):
                self.lgb_clf = joblib.load(LGB_PATH)
            if os.path.exists(RF_PATH):
                self.rf_clf = joblib.load(RF_PATH)
            meta = joblib.load(META_PATH)
            self._trained = meta.get("trained", False)
            self._feature_cols = meta.get("feature_cols", [])
            if self.lstm is not None:
                self.lstm.load()
            logger.info("Ensemble loaded from saved_models/")
            return True
        except Exception as e:
            logger.error("Failed to load ensemble: %s", e)
            return False

    @property
    def is_trained(self) -> bool:
        return self._trained
