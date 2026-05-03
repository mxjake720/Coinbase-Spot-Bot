"""
Risk manager: determines position size, stop loss, and take profit levels.
All parameters are ML/ATR-derived — no hard-coded dollar amounts.
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from bot.config import config
from bot.feature_engineer import get_atr
from bot.logger import logger


@dataclass
class TradeParams:
    product_id: str
    side: str                  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_quote: float  # USD amount to spend
    risk_reward: float
    confidence: float
    atr: float


class RiskManager:
    def __init__(self) -> None:
        self.open_positions: dict = {}  # product_id → TradeParams

    def compute_trade_params(
        self,
        product_id: str,
        side: str,
        entry_price: float,
        confidence: float,
        df: pd.DataFrame,
        portfolio_value_usd: float,
    ) -> Optional[TradeParams]:
        """
        Computes ATR-based TP/SL and risk-scaled position size.
        Returns None if trade doesn't meet risk criteria.
        """
        if len(df) < 20:
            logger.warning("Not enough candles to compute ATR for %s", product_id)
            return None

        atr = get_atr(df).iloc[-1]
        if np.isnan(atr) or atr <= 0:
            logger.warning("Invalid ATR for %s", product_id)
            return None

        # Scale TP/SL multipliers by confidence (higher confidence → wider TP)
        confidence_scale = 0.8 + 0.4 * confidence  # range: 0.8 – 1.2
        sl_mult = config.sl_atr_multiplier
        tp_mult = config.tp_atr_multiplier * confidence_scale

        if side.upper() == "BUY":
            stop_loss = entry_price - atr * sl_mult
            take_profit = entry_price + atr * tp_mult
        else:
            stop_loss = entry_price + atr * sl_mult
            take_profit = entry_price - atr * tp_mult

        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit <= 0:
            return None

        risk_reward = abs(take_profit - entry_price) / risk_per_unit

        # Minimum R:R of 1.5 to proceed
        if risk_reward < 1.5:
            logger.info(
                "%s R:R=%.2f below minimum 1.5 — skipping", product_id, risk_reward
            )
            return None

        # Position sizing: risk a fixed fraction of portfolio
        risk_usd = portfolio_value_usd * config.risk_per_trade
        # Adjust by confidence: scale up slightly for high-confidence trades
        risk_usd *= 0.5 + confidence
        position_size_quote = risk_usd / (risk_per_unit / entry_price)

        # Cap at 20% of portfolio per trade
        max_position = portfolio_value_usd * 0.20
        position_size_quote = min(position_size_quote, max_position)

        # Minimum $10 trade
        if position_size_quote < 10:
            logger.info("Position too small (%.2f) for %s — skipping", position_size_quote, product_id)
            return None

        params = TradeParams(
            product_id=product_id,
            side=side.upper(),
            entry_price=entry_price,
            stop_loss=round(stop_loss, 6),
            take_profit=round(take_profit, 6),
            position_size_quote=round(position_size_quote, 2),
            risk_reward=round(risk_reward, 3),
            confidence=round(confidence, 4),
            atr=round(atr, 6),
        )
        logger.info(
            "[RISK] %s %s | entry=%.4f SL=%.4f TP=%.4f | size=$%.2f R:R=%.2f conf=%.3f",
            side.upper(), product_id, entry_price, stop_loss, take_profit,
            position_size_quote, risk_reward, confidence,
        )
        return params

    def register_position(self, params: TradeParams) -> None:
        self.open_positions[params.product_id] = params

    def close_position(self, product_id: str) -> None:
        self.open_positions.pop(product_id, None)

    def has_open_position(self, product_id: str) -> bool:
        return product_id in self.open_positions

    def count_open(self) -> int:
        return len(self.open_positions)

    def should_stop_loss(self, product_id: str, current_price: float) -> bool:
        p = self.open_positions.get(product_id)
        if not p:
            return False
        if p.side == "BUY":
            return current_price <= p.stop_loss
        return current_price >= p.stop_loss

    def should_take_profit(self, product_id: str, current_price: float) -> bool:
        p = self.open_positions.get(product_id)
        if not p:
            return False
        if p.side == "BUY":
            return current_price >= p.take_profit
        return current_price <= p.take_profit

    def get_position(self, product_id: str) -> Optional[TradeParams]:
        return self.open_positions.get(product_id)
