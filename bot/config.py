import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    api_key: str = field(default_factory=lambda: os.getenv("COINBASE_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.getenv("COINBASE_API_SECRET", ""))

    trading_pairs: List[str] = field(
        default_factory=lambda: [
            p.strip()
            for p in os.getenv("TRADING_PAIRS", "BTC-USD,ETH-USD,SOL-USD").split(",")
        ]
    )

    risk_per_trade: float = field(
        default_factory=lambda: float(os.getenv("RISK_PER_TRADE", "0.02"))
    )
    max_open_positions: int = field(
        default_factory=lambda: int(os.getenv("MAX_OPEN_POSITIONS", "3"))
    )
    min_confidence: float = field(
        default_factory=lambda: float(os.getenv("MIN_CONFIDENCE", "0.62"))
    )
    paper_trading: bool = field(
        default_factory=lambda: os.getenv("PAPER_TRADING", "true").lower() == "true"
    )
    retrain_interval_hours: int = field(
        default_factory=lambda: int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
    )

    # Model hyperparameters
    lstm_lookback: int = 60          # candles fed into LSTM
    lstm_units: int = 128
    lstm_dropout: float = 0.2
    ensemble_vote_threshold: float = 0.6   # fraction of models that must agree

    # Feature engineering
    candle_granularity: int = 3600   # 1h candles in seconds
    history_candles: int = 500       # candles fetched per symbol for training

    # Training
    train_test_split: float = 0.8
    batch_size: int = 32
    epochs: int = 50
    early_stopping_patience: int = 8

    # Risk management multipliers (ATR-based)
    sl_atr_multiplier: float = 1.5
    tp_atr_multiplier: float = 2.5

    def validate(self) -> None:
        if not self.api_key or not self.api_secret:
            raise ValueError("COINBASE_API_KEY and COINBASE_API_SECRET must be set in .env")
        if not (0 < self.risk_per_trade <= 0.1):
            raise ValueError("RISK_PER_TRADE must be between 0 and 0.1 (10%)")


config = Config()
