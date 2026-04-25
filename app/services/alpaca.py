from __future__ import annotations

from datetime import datetime, timedelta, timezone
import math

import httpx

from app.config import Settings
from app.models import NewsItem, PositionSnapshot, PriceBar, TradeIntent
from app.sentiment import score_text


class AlpacaService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.headers = {
            "APCA-API-KEY-ID": settings.alpaca_api_key,
            "APCA-API-SECRET-KEY": settings.alpaca_secret_key,
        }

    def _request(self, base_url: str, method: str, path: str, *, params: dict | None = None, payload: dict | None = None):
        if not self.settings.trading_configured:
            raise RuntimeError("Alpaca credentials are not configured yet.")
        with httpx.Client(base_url=base_url, timeout=20.0, follow_redirects=True) as client:
            response = client.request(method, path, headers=self.headers, params=params, json=payload)
        response.raise_for_status()
        return response.json()

    def get_account(self) -> dict:
        return self._request(self.settings.alpaca_trading_base_url, "GET", "/v2/account")

    def get_positions(self) -> list[PositionSnapshot]:
        rows = self._request(self.settings.alpaca_trading_base_url, "GET", "/v2/positions")
        positions: list[PositionSnapshot] = []
        for row in rows:
            positions.append(
                PositionSnapshot(
                    symbol=row.get("symbol", ""),
                    qty=float(row.get("qty") or 0),
                    avg_entry_price=float(row.get("avg_entry_price") or 0),
                    market_value=float(row.get("market_value") or 0),
                    unrealized_plpc=float(row.get("unrealized_plpc") or 0),
                    side=row.get("side", "long"),
                )
            )
        return positions

    def get_orders(self, limit: int = 10) -> list[dict]:
        return self._request(
            self.settings.alpaca_trading_base_url,
            "GET",
            "/v2/orders",
            params={"status": "all", "direction": "desc", "limit": limit},
        )

    def get_daily_bars(self, symbol: str, lookback_days: int) -> list[PriceBar]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=max(lookback_days * 2, 90))
        payload = self._request(
            self.settings.alpaca_data_base_url,
            "GET",
            "/v2/stocks/bars",
            params={
                "symbols": symbol,
                "timeframe": "1Day",
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": lookback_days,
                "adjustment": "split",
                "feed": "iex",
            },
        )
        bars = payload.get("bars", {}).get(symbol, [])
        parsed: list[PriceBar] = []
        for item in bars:
            parsed.append(
                PriceBar(
                    symbol=symbol,
                    timestamp=item.get("t", ""),
                    open=float(item.get("o") or 0),
                    high=float(item.get("h") or 0),
                    low=float(item.get("l") or 0),
                    close=float(item.get("c") or 0),
                    volume=float(item.get("v") or 0),
                )
            )
        return parsed

    def get_news(self, symbol: str, days: int, limit: int = 8) -> list[NewsItem]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        payload = self._request(
            self.settings.alpaca_data_base_url,
            "GET",
            "/v1beta1/news",
            params={
                "symbols": symbol,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "sort": "desc",
                "limit": limit,
                "include_content": "true",
            },
        )
        raw_items = payload.get("news", payload if isinstance(payload, list) else [])
        items: list[NewsItem] = []
        for raw_item in raw_items:
            headline = raw_item.get("headline", "")
            summary = raw_item.get("summary", "")
            content = raw_item.get("content", "") or raw_item.get("body", "")
            published_at = raw_item.get("created_at") or raw_item.get("updated_at") or ""
            items.append(
                NewsItem(
                    symbol=symbol,
                    headline=headline,
                    summary=summary,
                    content=content,
                    source=raw_item.get("source", ""),
                    url=raw_item.get("url", ""),
                    published_at=published_at,
                    sentiment=score_text(f"{headline}. {summary}. {content}"),
                )
            )
        return items

    def submit_market_order(self, symbol: str, qty: int, side: str, reason: str) -> TradeIntent:
        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side,
            "type": "market",
            "time_in_force": "day",
        }
        response = self._request(self.settings.alpaca_trading_base_url, "POST", "/v2/orders", payload=payload)
        return TradeIntent(
            symbol=symbol,
            side=side,
            qty=qty,
            notional=float(response.get("filled_avg_price") or 0) * qty,
            reason=reason,
            status=response.get("status", "submitted"),
            broker_order_id=response.get("id", ""),
        )

    def close_position(self, symbol: str) -> dict:
        return self._request(self.settings.alpaca_trading_base_url, "DELETE", f"/v2/positions/{symbol}")
