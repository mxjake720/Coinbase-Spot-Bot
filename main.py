#!/usr/bin/env python3
"""
Coinbase ML Spot Trading Bot
Autonomous trading using Coinbase Advanced Trade API + ensemble ML.

Usage:
    python main.py             # Start with GUI (default, paper trading)
    python main.py --nogui     # Headless / CLI mode
    python main.py --train     # Force retrain models then start
    python main.py --live      # Live trading (prompts confirmation)
    python main.py --stats     # Print trade statistics only

Setup:
    1. Copy .env.example → .env and fill in your API keys
    2. pip install -r requirements.txt
    3. python main.py
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coinbase ML Spot Trading Bot")
    parser.add_argument("--nogui",  action="store_true", help="Run headless without GUI")
    parser.add_argument("--train",  action="store_true", help="Force retrain all models")
    parser.add_argument("--live",   action="store_true", help="Enable live trading")
    parser.add_argument("--stats",  action="store_true", help="Print stats and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.live:
        os.environ.setdefault("PAPER_TRADING", "true")

    from bot.config import config
    from bot.logger import logger
    from bot.trading_engine import TradingEngine

    logger.info("Coinbase ML Spot Bot starting | paper=%s", config.paper_trading)

    if not config.paper_trading:
        if not config.api_key or not config.api_secret:
            logger.error("API credentials not set. Copy .env.example → .env and add your keys.")
            sys.exit(1)
        logger.warning("LIVE TRADING MODE — real orders will be placed!")
        confirm = input("Type 'yes' to confirm live trading: ").strip().lower()
        if confirm != "yes":
            logger.info("Aborted.")
            sys.exit(0)

    engine = TradingEngine()

    if args.stats:
        engine.print_stats()
        return

    if args.nogui:
        # ── Headless CLI mode ────────────────────────────────────────────
        if args.train:
            engine.trainer.train_all(force_retrain=True)
        try:
            engine.start()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            engine.stop()
            engine.print_stats()
        return

    # ── GUI mode (default) ───────────────────────────────────────────────
    try:
        import tkinter as tk
    except ImportError:
        logger.error("tkinter not available — run with --nogui for headless mode.")
        sys.exit(1)

    from bot.gui import BotGUI

    gui = BotGUI(engine)
    engine.gui = gui

    # Redirect logger output to GUI log panel
    _patch_logger_to_gui(gui)

    if args.train:
        import threading
        threading.Thread(
            target=engine.trainer.train_all,
            kwargs={"force_retrain": True},
            daemon=True,
        ).start()

    gui.run()


def _patch_logger_to_gui(gui) -> None:
    """Add a handler that mirrors log records into the GUI log panel."""
    import logging

    class GUIHandler(logging.Handler):
        _LEVEL_TAG = {
            logging.DEBUG:   "INFO",
            logging.INFO:    "INFO",
            logging.WARNING: "WARNING",
            logging.ERROR:   "ERROR",
        }

        def emit(self, record: logging.LogRecord) -> None:
            try:
                msg = self.format(record)
                tag = self._LEVEL_TAG.get(record.levelno, "INFO")
                if "[ENTRY]" in msg:
                    tag = "ENTRY"
                elif "[EXIT]" in msg:
                    tag = "EXIT"
                gui.append_log(msg, tag)
            except Exception:
                pass

    from bot.logger import logger as bot_logger
    handler = GUIHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                                           datefmt="%H:%M:%S"))
    bot_logger.addHandler(handler)


if __name__ == "__main__":
    main()
