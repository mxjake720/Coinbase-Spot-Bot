"""
Trading engine: polls market data, runs inference, manages orders.
Handles entry, exit, stop-loss, take-profit, and paper trading mode.
"""
import time
from typing import Dict, List, Optional

import pandas as pd

from bot.coinbase_client import CoinbaseClient
from bot.config import config
from bot.data_collector import DataCollector
from bot.feature_engineer import compute_features, align_multi_timeframe
from bot.logger import logger
from bot.models.ensemble_model import EnsembleModel
from bot.models.trainer import Trainer
from bot.risk_manager import RiskManager, TradeParams

POLL_INTERVAL = 60  # seconds between signal checks


class PaperPosition:
    """Simulated position for paper trading."""
    def __init__(self, params: TradeParams) -> None:
        self.params = params
        self.open_time = time.time()


class TradingEngine:
    def __init__(self) -> None:
        self.client = CoinbaseClient()
        self.collector = DataCollector(self.client)
        self.risk_manager = RiskManager()
        self.trainer = Trainer(self.collector)
        self.paper_positions: Dict[str, PaperPosition] = {}
        self._running = False
        self._last_retrain: float = 0
        self._trade_log: List[Dict] = []
        self.gui = None  # set by main.py when running with GUI
        self._last_prices: Dict[str, float] = {}
        self._total_fees: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        logger.info("=" * 60)
        logger.info("Coinbase Spot Bot starting")
        logger.info("  Pairs: %s", config.trading_pairs)
        logger.info("  Paper trading: %s", config.paper_trading)
        logger.info("  Min confidence: %.2f", config.min_confidence)
        logger.info("  Risk per trade: %.1f%%", config.risk_per_trade * 100)
        logger.info("=" * 60)

        if not config.paper_trading:
            config.validate()

        # Initial training
        logger.info("Starting initial model training...")
        self._gui_log("Training ML models — please wait...", "WARNING")
        self._gui_ml_status("Training...", "Fetching candles and fitting ensemble")
        self.trainer.train_all(force_retrain=False)
        self._last_retrain = time.time()
        self._gui_ml_status("Active", self._ml_detail_str())

        self._running = True
        self._loop()

    def stop(self) -> None:
        logger.info("Bot stopping — closing all positions...")
        self._running = False
        self._close_all_positions()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                self._tick()
                self._maybe_retrain()
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt — stopping")
                self.stop()
                break
            except Exception as e:
                logger.error("Tick error: %s", e, exc_info=True)

            self._sleep(POLL_INTERVAL)

    def _tick(self) -> None:
        portfolio_usd = self._get_portfolio_value()
        logger.info(
            "--- Tick | portfolio=$%.2f | open_positions=%d ---",
            portfolio_usd, self.risk_manager.count_open()
        )

        # Refresh prices for GUI ticker
        prices: Dict[str, float] = {}
        for pair in config.trading_pairs:
            p = self.collector.get_current_price(pair)
            if p:
                prices[pair] = p
        if prices:
            self._last_prices = prices
            if self.gui:
                self.gui.update_prices(prices)

        for pair in config.trading_pairs:
            try:
                self._process_pair(pair, portfolio_usd)
            except Exception as e:
                logger.error("Error processing %s: %s", pair, e)

        # Push stats to GUI
        self._push_gui_stats(portfolio_usd)

    def _process_pair(self, product_id: str, portfolio_usd: float) -> None:
        # 1. Monitor existing position for TP/SL
        if self.risk_manager.has_open_position(product_id):
            self._check_exit(product_id)
            return

        # 2. Skip if max positions reached
        if self.risk_manager.count_open() >= config.max_open_positions:
            return

        model = self.trainer.get_model(product_id)
        if model is None or not model.is_trained:
            logger.info("No trained model for %s — skipping", product_id)
            return

        # 3. Fetch latest data and generate features
        tf_data = self.collector.fetch_multi_timeframe(product_id)
        if "1h" not in tf_data or tf_data["1h"].empty:
            return

        base_df = tf_data["1h"]
        features = compute_features(base_df, prefix="1h_")

        for label, df_tf in tf_data.items():
            if label != "1h" and not df_tf.empty:
                htf_feats = align_multi_timeframe(base_df, df_tf, prefix=f"{label}_")
                features = features.join(htf_feats, how="left")

        # Remove ATR raw helper cols
        atr_cols = [c for c in features.columns if "atr_raw" in c]
        features = features.drop(columns=atr_cols, errors="ignore")
        features = features.ffill().bfill()

        if features.empty or features.isna().all().all():
            return

        # 4. Run ensemble inference
        try:
            signal, confidence = model.predict(features)
        except Exception as e:
            logger.error("Inference error for %s: %s", product_id, e)
            return

        logger.info("%s | signal=%.0f confidence=%.3f", product_id, signal, confidence)

        if signal != 1.0:
            return

        # 5. Get current price
        current_price = self.collector.get_current_price(product_id)
        if current_price is None:
            return

        # 6. Compute risk params
        trade_params = self.risk_manager.compute_trade_params(
            product_id=product_id,
            side="BUY",
            entry_price=current_price,
            confidence=confidence,
            df=base_df,
            portfolio_value_usd=portfolio_usd,
        )
        if trade_params is None:
            return

        # 7. Execute entry
        self._enter_trade(trade_params)

    # ------------------------------------------------------------------
    # Entry / Exit
    # ------------------------------------------------------------------

    def _enter_trade(self, params: TradeParams) -> None:
        logger.info(
            "[ENTRY] %s %s | price=%.4f SL=%.4f TP=%.4f | $%.2f conf=%.3f",
            params.side, params.product_id, params.entry_price,
            params.stop_loss, params.take_profit,
            params.position_size_quote, params.confidence,
        )

        if config.paper_trading:
            self.paper_positions[params.product_id] = PaperPosition(params)
            self.risk_manager.register_position(params)
            self._log_trade("ENTER", params, params.entry_price, "paper")
            self._gui_log(
                f"[ENTRY] {params.side} {params.product_id} @ {params.entry_price:.4f} "
                f"SL={params.stop_loss:.4f} TP={params.take_profit:.4f} "
                f"${params.position_size_quote:.2f} conf={params.confidence:.2%}", "ENTRY"
            )
            if self.gui:
                self.gui.update_positions()
            return

        try:
            resp = self.client.place_market_order(
                product_id=params.product_id,
                side=params.side,
                quote_size=params.position_size_quote,
            )
            order_id = resp.get("order_id", "unknown")
            logger.info("Order placed: %s", order_id)
            self.risk_manager.register_position(params)
            self._log_trade("ENTER", params, params.entry_price, order_id)
        except Exception as e:
            logger.error("Failed to place order for %s: %s", params.product_id, e)

    def _check_exit(self, product_id: str) -> None:
        current_price = self.collector.get_current_price(product_id)
        if current_price is None:
            return

        position = self.risk_manager.get_position(product_id)
        if position is None:
            return

        exit_reason = None
        if self.risk_manager.should_stop_loss(product_id, current_price):
            exit_reason = "STOP_LOSS"
        elif self.risk_manager.should_take_profit(product_id, current_price):
            exit_reason = "TAKE_PROFIT"

        if exit_reason:
            self._exit_trade(product_id, current_price, exit_reason)

    def _exit_trade(self, product_id: str, current_price: float, reason: str) -> None:
        position = self.risk_manager.get_position(product_id)
        if position is None:
            return

        pnl_pct = (current_price - position.entry_price) / position.entry_price * 100
        if position.side == "SELL":
            pnl_pct = -pnl_pct

        logger.info(
            "[EXIT] %s | reason=%s | entry=%.4f exit=%.4f pnl=%.2f%%",
            product_id, reason, position.entry_price, current_price, pnl_pct,
        )

        if config.paper_trading:
            self.paper_positions.pop(product_id, None)
            self.risk_manager.close_position(product_id)
            self._log_trade("EXIT", position, current_price, reason, pnl_pct=pnl_pct)
            self._gui_log(
                f"[EXIT] {product_id} {reason} @ {current_price:.4f}  P&L={pnl_pct:+.2f}%", "EXIT"
            )
            if self.gui:
                self.gui.update_positions()
                self.gui.update_history()
            return

        # Calculate base size to sell
        base_size = position.position_size_quote / position.entry_price
        try:
            resp = self.client.place_market_order(
                product_id=product_id,
                side="SELL" if position.side == "BUY" else "BUY",
                base_size=base_size,
            )
            order_id = resp.get("order_id", "unknown")
            logger.info("Exit order placed: %s", order_id)
            self.risk_manager.close_position(product_id)
            self._log_trade("EXIT", position, current_price, order_id, pnl_pct=pnl_pct)
        except Exception as e:
            logger.error("Failed to close position for %s: %s", product_id, e)

    def _close_all_positions(self) -> None:
        for product_id in list(self.risk_manager.open_positions.keys()):
            price = self.collector.get_current_price(product_id)
            if price:
                self._exit_trade(product_id, price, "BOT_STOPPED")

    # ------------------------------------------------------------------
    # Portfolio / Account
    # ------------------------------------------------------------------

    def _get_portfolio_value(self) -> float:
        if config.paper_trading:
            return 1000.0  # default paper trading portfolio

        try:
            return self.client.get_account_balance("USD")
        except Exception as e:
            logger.warning("Could not fetch portfolio value: %s", e)
            return 1000.0

    # ------------------------------------------------------------------
    # Retraining scheduler
    # ------------------------------------------------------------------

    def _maybe_retrain(self) -> None:
        if config.retrain_interval_hours <= 0:
            return
        elapsed_hours = (time.time() - self._last_retrain) / 3600
        if elapsed_hours >= config.retrain_interval_hours:
            logger.info("Scheduled retrain triggered (%.1f hours elapsed)", elapsed_hours)
            self.trainer.train_all(force_retrain=True)
            self._last_retrain = time.time()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sleep(self, seconds: int) -> None:
        for _ in range(seconds):
            if not self._running:
                break
            time.sleep(1)

    def _log_trade(
        self,
        action: str,
        params: TradeParams,
        price: float,
        order_id: str,
        pnl_pct: Optional[float] = None,
    ) -> None:
        record = {
            "action": action,
            "product_id": params.product_id,
            "side": params.side,
            "price": price,
            "stop_loss": params.stop_loss,
            "take_profit": params.take_profit,
            "size_usd": params.position_size_quote,
            "confidence": params.confidence,
            "order_id": order_id,
            "pnl_pct": pnl_pct,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        self._trade_log.append(record)

    def print_stats(self) -> None:
        if not self._trade_log:
            logger.info("No trades recorded yet.")
            return
        exits = [t for t in self._trade_log if t["action"] == "EXIT" and t["pnl_pct"] is not None]
        if not exits:
            logger.info("No closed trades yet.")
            return
        wins = [t for t in exits if t["pnl_pct"] > 0]
        losses = [t for t in exits if t["pnl_pct"] <= 0]
        win_rate = len(wins) / len(exits) * 100
        avg_win = sum(t["pnl_pct"] for t in wins) / max(len(wins), 1)
        avg_loss = sum(t["pnl_pct"] for t in losses) / max(len(losses), 1)
        logger.info(
            "Stats: %d trades | win_rate=%.1f%% | avg_win=%.2f%% | avg_loss=%.2f%%",
            len(exits), win_rate, avg_win, avg_loss,
        )

    # ------------------------------------------------------------------
    # GUI bridge helpers
    # ------------------------------------------------------------------

    def _gui_log(self, msg: str, level: str = "INFO") -> None:
        logger.info(msg)
        if self.gui:
            self.gui.append_log(msg, level)

    def _gui_ml_status(self, status: str, detail: str = "") -> None:
        if self.gui:
            self.gui.update_ml_status(status, detail)

    def _push_gui_stats(self, portfolio_usd: float) -> None:
        if not self.gui:
            return
        exits = [t for t in self._trade_log if t["action"] == "EXIT" and t.get("pnl_pct") is not None]
        wins = [t for t in exits if t["pnl_pct"] > 0]
        losses = [t for t in exits if t["pnl_pct"] <= 0]
        win_rate = len(wins) / max(len(exits), 1) * 100
        total_pnl = sum(t["pnl_pct"] / 100 * t.get("size_usd", 0) for t in exits)
        avg_win = sum(t["pnl_pct"] for t in wins) / max(len(wins), 1)
        avg_loss = sum(t["pnl_pct"] for t in losses) / max(len(losses), 1)
        profit_factor = (
            abs(sum(t["pnl_pct"] for t in wins)) / max(abs(sum(t["pnl_pct"] for t in losses)), 0.001)
        )
        self.gui.update_stats(
            balance=portfolio_usd + total_pnl,
            pnl=total_pnl,
            win_rate=win_rate,
            trades=len(exits),
            fees=self._total_fees,
        )
        self.gui.update_perf({
            "_perf_trades":  str(len(exits)),
            "_perf_winrate": f"{win_rate:.1f}%",
            "_perf_avg_win": f"{avg_win:+.2f}%",
            "_perf_avg_loss": f"{avg_loss:+.2f}%",
            "_perf_pf":      f"{profit_factor:.2f}",
            "_perf_sharpe":  "—",
            "_perf_mdd":     "—",
            "_perf_exp":     f"${total_pnl / max(len(exits), 1):.2f}",
        })
        self.gui.update_learn_status(len(wins), len(losses), 0)

    def _ml_detail_str(self) -> str:
        parts = []
        for pair, model in self.trainer.models.items():
            short = pair.replace("-USD", "")
            parts.append(f"{short}: {'✓' if model and model.is_trained else '…'}")
        return " | ".join(parts) if parts else "No models loaded"
