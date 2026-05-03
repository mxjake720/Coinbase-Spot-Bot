"""
Coinbase Advanced Trade REST API client.
Supports both JWT (EC key) and HMAC (legacy) authentication.
"""
import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

import requests

from bot.config import config
from bot.logger import logger

BASE_URL = "https://api.coinbase.com"


class CoinbaseClient:
    def __init__(self) -> None:
        self.api_key = config.api_key
        self.api_secret = config.api_secret
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Auth helpers
    # ------------------------------------------------------------------

    def _sign_request(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time()))
        message = timestamp + method.upper() + path + body
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            message.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).hexdigest()
        return {
            "CB-ACCESS-KEY": self.api_key,
            "CB-ACCESS-SIGN": signature,
            "CB-ACCESS-TIMESTAMP": timestamp,
        }

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Any:
        url = BASE_URL + path
        body = json.dumps(data) if data else ""
        query = ("?" + urlencode(params)) if params else ""
        headers = self._sign_request(method, path + query, body)

        try:
            resp = self.session.request(
                method,
                url,
                headers=headers,
                params=params,
                data=body if body else None,
                timeout=15,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP %s %s → %s: %s", method, path, e.response.status_code, e.response.text[:300])
            raise
        except requests.exceptions.RequestException as e:
            logger.error("Request failed %s %s: %s", method, path, e)
            raise

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_candles(
        self, product_id: str, granularity: int = 3600, limit: int = 300
    ) -> List[Dict]:
        """
        Returns OHLCV candles sorted oldest→newest.
        granularity in seconds: 60, 300, 900, 3600, 21600, 86400
        """
        end_time = int(time.time())
        start_time = end_time - granularity * limit
        params = {
            "start": str(start_time),
            "end": str(end_time),
            "granularity": _granularity_to_str(granularity),
        }
        resp = self._request("GET", f"/api/v3/brokerage/products/{product_id}/candles", params=params)
        candles = resp.get("candles", [])
        # API returns newest-first; reverse to oldest-first
        candles.reverse()
        return candles

    def get_best_bid_ask(self, product_id: str) -> Dict:
        resp = self._request("GET", f"/api/v3/brokerage/best_bid_ask", params={"product_ids": product_id})
        pricebooks = resp.get("pricebooks", [])
        if pricebooks:
            return pricebooks[0]
        return {}

    def get_product(self, product_id: str) -> Dict:
        return self._request("GET", f"/api/v3/brokerage/products/{product_id}")

    def get_products(self) -> List[Dict]:
        resp = self._request("GET", "/api/v3/brokerage/products")
        return resp.get("products", [])

    # ------------------------------------------------------------------
    # Account / portfolio
    # ------------------------------------------------------------------

    def get_accounts(self) -> List[Dict]:
        resp = self._request("GET", "/api/v3/brokerage/accounts")
        return resp.get("accounts", [])

    def get_account_balance(self, currency: str) -> float:
        accounts = self.get_accounts()
        for acc in accounts:
            if acc.get("currency") == currency:
                return float(acc.get("available_balance", {}).get("value", 0))
        return 0.0

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def place_market_order(
        self, product_id: str, side: str, quote_size: Optional[float] = None, base_size: Optional[float] = None
    ) -> Dict:
        order_config: Dict[str, Any] = {}
        if side.upper() == "BUY":
            if quote_size is not None:
                order_config = {"market_market_ioc": {"quote_size": f"{quote_size:.2f}"}}
            else:
                order_config = {"market_market_ioc": {"base_size": f"{base_size:.8f}"}}
        else:
            order_config = {"market_market_ioc": {"base_size": f"{base_size:.8f}"}}

        payload = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": order_config,
        }
        logger.info("[ORDER] %s %s | config=%s", side.upper(), product_id, order_config)
        return self._request("POST", "/api/v3/brokerage/orders", data=payload)

    def place_limit_order(
        self, product_id: str, side: str, base_size: float, limit_price: float, post_only: bool = False
    ) -> Dict:
        payload = {
            "client_order_id": str(uuid.uuid4()),
            "product_id": product_id,
            "side": side.upper(),
            "order_configuration": {
                "limit_limit_gtc": {
                    "base_size": f"{base_size:.8f}",
                    "limit_price": f"{limit_price:.2f}",
                    "post_only": post_only,
                }
            },
        }
        logger.info("[LIMIT ORDER] %s %s @ %.4f (size=%.8f)", side.upper(), product_id, limit_price, base_size)
        return self._request("POST", "/api/v3/brokerage/orders", data=payload)

    def cancel_orders(self, order_ids: List[str]) -> Dict:
        payload = {"order_ids": order_ids}
        return self._request("POST", "/api/v3/brokerage/orders/batch_cancel", data=payload)

    def get_order(self, order_id: str) -> Dict:
        return self._request("GET", f"/api/v3/brokerage/orders/historical/{order_id}")

    def list_open_orders(self, product_id: Optional[str] = None) -> List[Dict]:
        params: Dict[str, Any] = {"order_status": "OPEN"}
        if product_id:
            params["product_id"] = product_id
        resp = self._request("GET", "/api/v3/brokerage/orders/historical/batch", params=params)
        return resp.get("orders", [])


def _granularity_to_str(granularity: int) -> str:
    mapping = {
        60: "ONE_MINUTE",
        300: "FIVE_MINUTE",
        900: "FIFTEEN_MINUTE",
        3600: "ONE_HOUR",
        21600: "SIX_HOUR",
        86400: "ONE_DAY",
    }
    return mapping.get(granularity, "ONE_HOUR")
