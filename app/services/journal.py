from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import sqlite3

from app.config import Settings
from app.db import fetch_signal_history, get_connection, initialize_database
from app.models import (
    JournalClosedTrade,
    JournalFactorSummary,
    JournalOpenPosition,
    JournalOrderActivity,
    PaperJournalResult,
)
from app.services.alpaca import AlpacaService


MAX_SIGNAL_AGE_DAYS = 14


@dataclass(slots=True)
class _SignalContext:
    dominant_factor: str
    composite_score: float
    momentum_score: float
    sentiment_score: float
    earnings_score: float
    rationale: str
    headline: str


@dataclass(slots=True)
class _OpenLot:
    qty: int
    entry_price: float
    entry_at: datetime
    signal_context: _SignalContext


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _timestamp_label(value: datetime | None) -> str:
    if value is None:
        return ""
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _days_between(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(0, (end.date() - start.date()).days)


def _safe_float(value) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value) -> int:
    try:
        return int(round(float(value or 0.0)))
    except (TypeError, ValueError):
        return 0


def _dominant_factor_from_signal(signal_row: sqlite3.Row | None) -> _SignalContext:
    if signal_row is None:
        return _SignalContext(
            dominant_factor="unknown",
            composite_score=0.0,
            momentum_score=0.0,
            sentiment_score=0.0,
            earnings_score=0.0,
            rationale="No stored signal snapshot was found close to entry.",
            headline="",
        )
    factor_scores = {
        "momentum": _safe_float(signal_row["momentum_score"]),
        "sentiment": _safe_float(signal_row["sentiment_score"]),
        "earnings": _safe_float(signal_row["earnings_score"]),
    }
    dominant_factor = max(factor_scores, key=factor_scores.get)
    return _SignalContext(
        dominant_factor=dominant_factor,
        composite_score=_safe_float(signal_row["composite_score"]),
        momentum_score=factor_scores["momentum"],
        sentiment_score=factor_scores["sentiment"],
        earnings_score=factor_scores["earnings"],
        rationale=signal_row["rationale"] or "",
        headline=signal_row["headline"] or "",
    )


def _signals_by_symbol(rows: list[sqlite3.Row]) -> dict[str, list[tuple[datetime, sqlite3.Row]]]:
    output: dict[str, list[tuple[datetime, sqlite3.Row]]] = defaultdict(list)
    for row in rows:
        parsed = _parse_timestamp(row["created_at"])
        if parsed is None:
            continue
        output[row["symbol"]].append((parsed, row))
    for symbol_rows in output.values():
        symbol_rows.sort(key=lambda item: item[0])
    return output


def _find_signal_context(
    signal_map: dict[str, list[tuple[datetime, sqlite3.Row]]],
    symbol: str,
    entry_at: datetime,
) -> _SignalContext:
    matches = signal_map.get(symbol, [])
    for created_at, row in reversed(matches):
        if created_at > entry_at:
            continue
        if (entry_at - created_at).days > MAX_SIGNAL_AGE_DAYS:
            break
        return _dominant_factor_from_signal(row)
    return _dominant_factor_from_signal(None)


def _normalized_filled_orders(raw_orders: list[dict]) -> list[dict]:
    normalized: list[dict] = []
    for raw_order in raw_orders:
        side = (raw_order.get("side") or "").lower()
        if side not in {"buy", "sell"}:
            continue
        qty = _safe_int(raw_order.get("filled_qty") or raw_order.get("qty"))
        price = _safe_float(raw_order.get("filled_avg_price"))
        timestamp = _parse_timestamp(
            raw_order.get("filled_at")
            or raw_order.get("updated_at")
            or raw_order.get("submitted_at")
            or raw_order.get("created_at")
        )
        if qty < 1 or price <= 0 or timestamp is None:
            continue
        normalized.append(
            {
                "symbol": (raw_order.get("symbol") or "").upper(),
                "side": side,
                "qty": qty,
                "price": price,
                "status": raw_order.get("status", ""),
                "submitted_at": _parse_timestamp(raw_order.get("submitted_at") or raw_order.get("created_at")),
                "filled_at": timestamp,
            }
        )
    normalized.sort(key=lambda item: (item["filled_at"], item["symbol"], item["side"]))
    return normalized


def _build_factor_summaries(trades: list[JournalClosedTrade]) -> list[JournalFactorSummary]:
    grouped: dict[str, list[JournalClosedTrade]] = defaultdict(list)
    for trade in trades:
        grouped[trade.dominant_factor].append(trade)
    summaries: list[JournalFactorSummary] = []
    for factor_name, factor_trades in grouped.items():
        wins = [trade for trade in factor_trades if trade.pnl > 0]
        summaries.append(
            JournalFactorSummary(
                factor_name=factor_name,
                total_trades=len(factor_trades),
                win_rate_pct=round(len(wins) / len(factor_trades), 4) if factor_trades else 0.0,
                average_return_pct=round(sum(trade.return_pct for trade in factor_trades) / len(factor_trades), 4) if factor_trades else 0.0,
                total_pnl=round(sum(trade.pnl for trade in factor_trades), 2),
            )
        )
    summaries.sort(key=lambda item: (item.total_pnl, item.win_rate_pct, item.total_trades), reverse=True)
    return summaries


class PaperJournalService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.alpaca = AlpacaService(settings)
        initialize_database(settings)

    def run(self, order_limit: int = 200) -> PaperJournalResult:
        if not self.settings.trading_configured:
            raise RuntimeError("Add your Alpaca keys before opening the paper-trade journal.")

        raw_orders = self.alpaca.get_orders(limit=order_limit)
        positions = self.alpaca.get_positions()
        symbols = sorted(
            {
                *((raw_order.get("symbol") or "").upper() for raw_order in raw_orders if raw_order.get("symbol")),
                *(position.symbol for position in positions if position.symbol),
            }
        )
        with get_connection(self.settings) as conn:
            signal_rows = fetch_signal_history(conn, symbols=symbols, limit=4000)
        signal_map = _signals_by_symbol(signal_rows)
        filled_orders = _normalized_filled_orders(raw_orders)
        open_lots: dict[str, list[_OpenLot]] = defaultdict(list)
        closed_trades: list[JournalClosedTrade] = []

        for order in filled_orders:
            symbol = order["symbol"]
            if order["side"] == "buy":
                open_lots[symbol].append(
                    _OpenLot(
                        qty=order["qty"],
                        entry_price=order["price"],
                        entry_at=order["filled_at"],
                        signal_context=_find_signal_context(signal_map, symbol, order["filled_at"]),
                    )
                )
                continue

            remaining_qty = order["qty"]
            while remaining_qty > 0 and open_lots[symbol]:
                lot = open_lots[symbol][0]
                matched_qty = min(remaining_qty, lot.qty)
                pnl = (order["price"] - lot.entry_price) * matched_qty
                capital = lot.entry_price * matched_qty
                closed_trades.append(
                    JournalClosedTrade(
                        symbol=symbol,
                        qty=matched_qty,
                        entry_at=_timestamp_label(lot.entry_at),
                        exit_at=_timestamp_label(order["filled_at"]),
                        entry_price=round(lot.entry_price, 2),
                        exit_price=round(order["price"], 2),
                        pnl=round(pnl, 2),
                        return_pct=round((pnl / capital) if capital else 0.0, 4),
                        hold_days=_days_between(lot.entry_at, order["filled_at"]),
                        dominant_factor=lot.signal_context.dominant_factor,
                        entry_composite_score=round(lot.signal_context.composite_score, 4),
                        entry_momentum_score=round(lot.signal_context.momentum_score, 4),
                        entry_sentiment_score=round(lot.signal_context.sentiment_score, 4),
                        entry_earnings_score=round(lot.signal_context.earnings_score, 4),
                        entry_rationale=lot.signal_context.rationale,
                        entry_headline=lot.signal_context.headline,
                    )
                )
                lot.qty -= matched_qty
                remaining_qty -= matched_qty
                if lot.qty == 0:
                    open_lots[symbol].pop(0)

        now = datetime.now(timezone.utc)
        journal_positions: list[JournalOpenPosition] = []
        for position in positions:
            qty = abs(position.qty)
            current_price = (position.market_value / qty) if qty else 0.0
            open_symbol_lots = open_lots.get(position.symbol, [])
            reference_lot = open_symbol_lots[-1] if open_symbol_lots else None
            oldest_lot_time = open_symbol_lots[0].entry_at if open_symbol_lots else None
            signal_context = reference_lot.signal_context if reference_lot is not None else _dominant_factor_from_signal(None)
            journal_positions.append(
                JournalOpenPosition(
                    symbol=position.symbol,
                    qty=position.qty,
                    avg_entry_price=round(position.avg_entry_price, 2),
                    current_price=round(current_price, 2),
                    market_value=round(position.market_value, 2),
                    unrealized_pnl=round(position.market_value - (position.avg_entry_price * qty), 2),
                    unrealized_plpc=round(position.unrealized_plpc, 4),
                    hold_days=_days_between(oldest_lot_time, now),
                    dominant_factor=signal_context.dominant_factor,
                    entry_composite_score=round(signal_context.composite_score, 4),
                    entry_momentum_score=round(signal_context.momentum_score, 4),
                    entry_sentiment_score=round(signal_context.sentiment_score, 4),
                    entry_earnings_score=round(signal_context.earnings_score, 4),
                    entry_rationale=signal_context.rationale,
                    entry_headline=signal_context.headline,
                )
            )

        closed_trades.sort(key=lambda item: item.exit_at, reverse=True)
        realized_pnl = round(sum(trade.pnl for trade in closed_trades), 2)
        unrealized_pnl = round(sum(position.unrealized_pnl for position in journal_positions), 2)
        wins = [trade for trade in closed_trades if trade.pnl > 0]
        average_return_pct = round(sum(trade.return_pct for trade in closed_trades) / len(closed_trades), 4) if closed_trades else 0.0
        best_trade_pnl = max((trade.pnl for trade in closed_trades), default=0.0)
        worst_trade_pnl = min((trade.pnl for trade in closed_trades), default=0.0)
        recent_orders = [
            JournalOrderActivity(
                symbol=order["symbol"],
                side=order["side"],
                qty=order["qty"],
                filled_price=round(order["price"], 2),
                status=order["status"],
                submitted_at=_timestamp_label(order["submitted_at"]),
                filled_at=_timestamp_label(order["filled_at"]),
            )
            for order in filled_orders[::-1][:20]
        ]

        summary = (
            f"Tracked {len(closed_trades)} closed paper trades and {len(journal_positions)} open positions "
            f"from Alpaca fills, then matched them to recent signal snapshots when available."
        )
        notes = [
            "Closed trades are reconstructed with simple FIFO matching from Alpaca paper-order fills.",
            "Dominant factor is based on the strongest stored raw factor score near entry, not a full causal attribution model yet.",
            "If a position was opened before the saved signal history window, the entry context will show as unknown.",
        ]
        return PaperJournalResult(
            status="ok",
            summary=summary,
            total_closed_trades=len(closed_trades),
            win_rate_pct=round(len(wins) / len(closed_trades), 4) if closed_trades else 0.0,
            realized_pnl=realized_pnl,
            average_return_pct=average_return_pct,
            open_positions_count=len(journal_positions),
            unrealized_pnl=unrealized_pnl,
            realized_plus_unrealized_pnl=round(realized_pnl + unrealized_pnl, 2),
            best_trade_pnl=round(best_trade_pnl, 2),
            worst_trade_pnl=round(worst_trade_pnl, 2),
            closed_trades=closed_trades,
            open_positions=sorted(journal_positions, key=lambda item: item.market_value, reverse=True),
            factor_summaries=_build_factor_summaries(closed_trades),
            recent_orders=recent_orders,
            notes=notes,
        )
