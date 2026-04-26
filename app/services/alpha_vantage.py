from __future__ import annotations

from datetime import datetime, timedelta, timezone
import csv
import io
import json
import time

import httpx

from app.config import Settings
from app.db import cache_get, cache_set
from app.models import EarningsBundle
from app.sentiment import score_text


_AV_ERROR_KEYS = {"Information", "Note", "Error Message"}


def _check_av_response(payload: dict, function: str) -> None:
    """Raise if Alpha Vantage returned an API-level error inside the JSON body."""
    for key in _AV_ERROR_KEYS:
        if key in payload:
            raise RuntimeError(f"Alpha Vantage {function}: {payload[key]}")


class AlphaVantageService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_request_at = 0.0

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        minimum_gap = 60.0 / max(1, self.settings.alpha_vantage_requests_per_minute)
        if elapsed < minimum_gap:
            time.sleep(minimum_gap - elapsed)
        self._last_request_at = time.monotonic()

    def _request_json(self, function: str, params: dict, conn, cache_hours: int) -> dict:
        if not self.settings.earnings_configured:
            raise RuntimeError("Alpha Vantage key is not configured yet.")
        cache_key = f"av:json:{function}:{json.dumps(params, sort_keys=True)}"
        cached = cache_get(conn, cache_key)
        if cached:
            return json.loads(cached)
        self._rate_limit()
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(
                "https://www.alphavantage.co/query",
                params={"function": function, "apikey": self.settings.alpha_vantage_api_key, **params},
            )
        response.raise_for_status()
        payload = response.json()
        _check_av_response(payload, function)
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=cache_hours)).replace(microsecond=0).isoformat()
        cache_set(conn, cache_key, payload, expires_at)
        return payload

    def _request_text(self, function: str, params: dict, conn, cache_hours: int) -> str:
        if not self.settings.earnings_configured:
            raise RuntimeError("Alpha Vantage key is not configured yet.")
        cache_key = f"av:text:{function}:{json.dumps(params, sort_keys=True)}"
        cached = cache_get(conn, cache_key)
        if cached:
            return cached
        self._rate_limit()
        with httpx.Client(timeout=20.0, follow_redirects=True) as client:
            response = client.get(
                "https://www.alphavantage.co/query",
                params={"function": function, "apikey": self.settings.alpha_vantage_api_key, **params},
            )
        response.raise_for_status()
        payload = response.text
        expires_at = (datetime.now(timezone.utc) + timedelta(hours=cache_hours)).replace(microsecond=0).isoformat()
        cache_set(conn, cache_key, payload, expires_at)
        return payload

    @staticmethod
    def _safe_float(value) -> float | None:
        if value in {None, "", "None"}:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _fiscal_date_to_quarter(fiscal_date: str | None) -> str | None:
        if not fiscal_date:
            return None
        year, month, _ = fiscal_date.split("-", 2)
        quarter = ((int(month) - 1) // 3) + 1
        return f"{year}Q{quarter}"

    def get_earnings_bundle(self, symbol: str, conn, *, include_calendar: bool = False) -> EarningsBundle | None:
        payload = self._request_json("EARNINGS", {"symbol": symbol}, conn, cache_hours=24)
        rows = payload.get("quarterlyEarnings") or []
        if not rows:
            return None
        latest = rows[0]
        bundle = EarningsBundle(
            symbol=symbol,
            fiscal_date_ending=latest.get("fiscalDateEnding"),
            reported_date=latest.get("reportedDate"),
            reported_eps=self._safe_float(latest.get("reportedEPS")),
            estimated_eps=self._safe_float(latest.get("estimatedEPS")),
            surprise=self._safe_float(latest.get("surprise")),
            surprise_pct=self._safe_float(latest.get("surprisePercentage")),
            report_time=latest.get("reportTime"),
        )
        quarter = self._fiscal_date_to_quarter(bundle.fiscal_date_ending)
        if quarter:
            transcript_payload = self._request_json(
                "EARNINGS_CALL_TRANSCRIPT",
                {"symbol": symbol, "quarter": quarter},
                conn,
                cache_hours=24 * 90,
            )
            transcript_rows = transcript_payload.get("transcript") or []
            important_sections = []
            for row in transcript_rows:
                title = (row.get("title") or "").lower()
                if "chief executive" in title or "chief financial" in title or "ceo" in title or "cfo" in title:
                    important_sections.append(row.get("content", ""))
            if not important_sections:
                important_sections = [row.get("content", "") for row in transcript_rows[:8]]
            transcript_text = " ".join(part for part in important_sections if part).strip()
            bundle.transcript_sentiment = score_text(transcript_text[:18_000])
            bundle.transcript_excerpt = transcript_text[:320]
        if include_calendar:
            bundle.upcoming_report_date = self.get_upcoming_earnings_date(symbol, conn)
        return bundle

    def get_quarterly_earnings_history(self, symbol: str, conn) -> list[EarningsBundle]:
        payload = self._request_json("EARNINGS", {"symbol": symbol}, conn, cache_hours=24)
        rows = payload.get("quarterlyEarnings") or []
        history: list[EarningsBundle] = []
        for row in rows:
            history.append(
                EarningsBundle(
                    symbol=symbol,
                    fiscal_date_ending=row.get("fiscalDateEnding"),
                    reported_date=row.get("reportedDate"),
                    reported_eps=self._safe_float(row.get("reportedEPS")),
                    estimated_eps=self._safe_float(row.get("estimatedEPS")),
                    surprise=self._safe_float(row.get("surprise")),
                    surprise_pct=self._safe_float(row.get("surprisePercentage")),
                    report_time=row.get("reportTime"),
                )
            )
        history.sort(key=lambda item: item.reported_date or "")
        return history

    def get_upcoming_earnings_date(self, symbol: str, conn) -> str | None:
        payload = self._request_text(
            "EARNINGS_CALENDAR",
            {"symbol": symbol, "horizon": "12month"},
            conn,
            cache_hours=12,
        )
        reader = csv.DictReader(io.StringIO(payload))
        today = datetime.now(timezone.utc).date()
        for row in reader:
            report_date = row.get("reportDate", "")
            if not report_date:
                continue
            try:
                parsed = datetime.fromisoformat(report_date).date()
            except ValueError:
                continue
            if parsed >= today:
                return parsed.isoformat()
        return None
