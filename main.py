#!/usr/bin/env python3
"""
Coinbase Spot Trading Bot
ML-powered autonomous trading using Coinbase Advanced Trade API.

Usage:
    python main.py             # Start bot (reads .env for config)
    python main.py --train     # Force retrain models then start
    python main.py --stats     # Print trade statistics only

Setup:
    1. Copy .env.example → .env
    2. Add your Coinbase Advanced Trade API key and secret
    3. pip install -r requirements.txt
    4. python main.py
"""
import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coinbase ML Spot Trading Bot")
    parser.add_argument("--train", action="store_true", help="Force retrain all models before starting")
    parser.add_argument("--paper", action="store_true", help="Override to paper trading mode")
    parser.add_argument("--stats", action="store_true", help="Print trade statistics and exit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.paper:
        os.environ["PAPER_TRADING"] = "true"

    # Import after env is loaded
    from bot.config import config
    from bot.logger import logger
    from bot.trading_engine import TradingEngine

    logger.info("Coinbase ML Spot Bot initializing")
    logger.info("Paper trading: %s", config.paper_trading)

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

    if args.train:
        logger.info("Forcing model retrain...")
        engine.trainer.train_all(force_retrain=True)

    try:
        engine.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        engine.stop()
        engine.print_stats()


if __name__ == "__main__":
    main()
