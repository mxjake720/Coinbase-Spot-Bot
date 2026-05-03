"""
Fetches and caches OHLCV candle data from Coinbase Advanced API.
Supports multiple timeframes for multi-timeframe analysis.
"""
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from bot.coinbase_client import CoinbaseClient
from bot.config import config
from bot.logger import logger

# Timeframes used for multi-timeframe feature engineering
TIMEFRAMES = {
    "1h": 3600,
    "15m": 900,
    "5m": 300,
}

CANDLE_LIMIT_PER_REQUEST = 300  # Coinbase API max per call


class DataCollector:
    def __init__(self, client: CoinbaseClient) -> None:
        self.client = client
        self._cache: Dict[str, pd.DataFrame] = {}

    def fetch_candles(
        self, product_id: str, granularity: int = 3600, n_candles: int = 500
    ) -> pd.DataFrame:
        """
        Fetches up to n_candles OHLCV candles by making multiple API calls if needed.
        Returns a DataFrame with columns: [open, high, low, close, volume] indexed by datetime.
        """
        all_candles: List[Dict] = []
        end_time = int(time.time())
        remaining = n_candles

        while remaining > 0:
            batch = min(remaining, CANDLE_LIMIT_PER_REQUEST)
            start_time = end_time - granularity * batch
            try:
                raw = self.client.get_candles(product_id, granularity=granularity, limit=batch)
                if not raw:
                    break
                all_candles = raw + all_candles
                end_time = start_time - granularity
                remaining -= batch
                if len(raw) < batch:
                    break
                time.sleep(0.2)  # rate-limit courtesy delay
            except Exception as e:
                logger.warning("Failed to fetch candles for %s: %s", product_id, e)
                break

        if not all_candles:
            logger.error("No candle data returned for %s", product_id)
            return pd.DataFrame()

        df = _candles_to_df(all_candles)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        logger.info("Fetched %d candles for %s (granularity=%ds)", len(df), product_id, granularity)
        return df

    def fetch_multi_timeframe(self, product_id: str) -> Dict[str, pd.DataFrame]:
        """Returns dict of {timeframe_label: DataFrame} for multi-timeframe analysis."""
        result = {}
        for label, gran in TIMEFRAMES.items():
            n = config.history_candles if gran >= 3600 else config.history_candles * 4
            df = self.fetch_candles(product_id, granularity=gran, n_candles=n)
            if not df.empty:
                result[label] = df
        return result

    def get_current_price(self, product_id: str) -> Optional[float]:
        try:
            book = self.client.get_best_bid_ask(product_id)
            bids = book.get("bids", [])
            asks = book.get("asks", [])
            if bids and asks:
                return (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
        except Exception as e:
            logger.warning("Could not fetch price for %s: %s", product_id, e)
        return None

    def get_cached(self, key: str) -> Optional[pd.DataFrame]:
        return self._cache.get(key)

    def update_cache(self, key: str, df: pd.DataFrame) -> None:
        self._cache[key] = df


def _candles_to_df(candles: List[Dict]) -> pd.DataFrame:
    rows = []
    for c in candles:
        rows.append({
            "timestamp": pd.to_datetime(int(c["start"]), unit="s", utc=True),
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
            "volume": float(c["volume"]),
        })
    df = pd.DataFrame(rows).set_index("timestamp")
    return df
