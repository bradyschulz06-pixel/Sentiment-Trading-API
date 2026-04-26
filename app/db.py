from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

from app.config import Settings
from app.models import EngineRunResult, NewsItem, PositionSnapshot, SignalScore, TradeIntent


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@contextmanager
def get_connection(settings: Settings):
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize_database(settings: Settings) -> None:
    with get_connection(settings) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trigger TEXT NOT NULL,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                warnings_json TEXT NOT NULL DEFAULT '[]',
                started_at TEXT NOT NULL,
                completed_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                momentum_score REAL NOT NULL,
                sentiment_score REAL NOT NULL,
                earnings_score REAL NOT NULL,
                composite_score REAL NOT NULL,
                decision TEXT NOT NULL,
                rationale TEXT NOT NULL,
                stop_price REAL NOT NULL,
                target_price REAL NOT NULL,
                next_earnings_date TEXT,
                headline TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS news_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                headline TEXT NOT NULL,
                summary TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                url TEXT NOT NULL,
                published_at TEXT NOT NULL,
                sentiment REAL NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS positions_snapshot (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                qty REAL NOT NULL,
                avg_entry_price REAL NOT NULL,
                market_value REAL NOT NULL,
                unrealized_plpc REAL NOT NULL,
                side TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                qty INTEGER NOT NULL,
                notional REAL NOT NULL,
                reason TEXT NOT NULL,
                status TEXT NOT NULL,
                broker_order_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                payload_json TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_signals_run_id ON signals(run_id);
            CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_created ON signals(symbol, created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_news_items_run_id ON news_items(run_id);
            CREATE INDEX IF NOT EXISTS idx_trades_run_id ON trades(run_id);
            CREATE INDEX IF NOT EXISTS idx_positions_run_id ON positions_snapshot(run_id);
            CREATE INDEX IF NOT EXISTS idx_cache_expires_at ON cache(expires_at);
            """
        )


def cache_get(conn: sqlite3.Connection, cache_key: str) -> str | None:
    row = conn.execute(
        "SELECT payload_json, expires_at FROM cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()
    if row is None:
        return None
    if row["expires_at"] <= utc_now_iso():
        conn.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
        return None
    return row["payload_json"]


def cache_set(conn: sqlite3.Connection, cache_key: str, payload: dict | list | str, expires_at: str) -> None:
    payload_json = payload if isinstance(payload, str) else json.dumps(payload)
    conn.execute(
        """
        INSERT INTO cache (cache_key, payload_json, expires_at, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(cache_key) DO UPDATE SET
            payload_json = excluded.payload_json,
            expires_at = excluded.expires_at,
            updated_at = excluded.updated_at
        """,
        (cache_key, payload_json, expires_at, utc_now_iso()),
    )


def save_run(conn: sqlite3.Connection, result: EngineRunResult) -> int:
    cursor = conn.execute(
        """
        INSERT INTO runs (trigger, status, summary, error, warnings_json, started_at, completed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            result.trigger,
            result.status,
            result.summary,
            result.error,
            json.dumps(result.warnings),
            result.started_at,
            result.completed_at,
        ),
    )
    run_id = int(cursor.lastrowid)
    created_at = utc_now_iso()
    for signal in result.signals:
        conn.execute(
            """
            INSERT INTO signals (
                run_id, symbol, price, momentum_score, sentiment_score, earnings_score,
                composite_score, decision, rationale, stop_price, target_price,
                next_earnings_date, headline, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                signal.symbol,
                signal.price,
                signal.momentum_score,
                signal.sentiment_score,
                signal.earnings_score,
                signal.composite_score,
                signal.decision,
                signal.rationale,
                signal.stop_price,
                signal.target_price,
                signal.next_earnings_date,
                signal.headline,
                created_at,
            ),
        )
    for item in result.news_items:
        conn.execute(
            """
            INSERT INTO news_items (
                run_id, symbol, headline, summary, content, source, url, published_at, sentiment
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                item.symbol,
                item.headline,
                item.summary,
                item.content,
                item.source,
                item.url,
                item.published_at,
                item.sentiment,
            ),
        )
    for position in result.positions:
        conn.execute(
            """
            INSERT INTO positions_snapshot (
                run_id, symbol, qty, avg_entry_price, market_value, unrealized_plpc, side
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                position.symbol,
                position.qty,
                position.avg_entry_price,
                position.market_value,
                position.unrealized_plpc,
                position.side,
            ),
        )
    for trade in result.trades:
        conn.execute(
            """
            INSERT INTO trades (
                run_id, symbol, side, qty, notional, reason, status, broker_order_id, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                trade.symbol,
                trade.side,
                trade.qty,
                trade.notional,
                trade.reason,
                trade.status,
                trade.broker_order_id,
                created_at,
            ),
        )
    return run_id


def get_last_regime_label(conn: sqlite3.Connection) -> str:
    """Return the regime label from the most recent successful run, or 'unknown'."""
    row = conn.execute(
        "SELECT warnings_json FROM runs WHERE status = 'ok' ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return "unknown"
    try:
        warnings = json.loads(row["warnings_json"])
        prefix = "MARKET_REGIME::"
        for w in warnings:
            if isinstance(w, str) and w.startswith(prefix):
                payload = json.loads(w[len(prefix):])
                return payload.get("label", "unknown")
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def fetch_latest_run(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()


def fetch_recent_runs(conn: sqlite3.Connection, limit: int = 10) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall())


def fetch_signals_for_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT * FROM signals WHERE run_id = ? ORDER BY composite_score DESC, symbol ASC",
            (run_id,),
        ).fetchall()
    )


def fetch_signal_by_symbol(conn: sqlite3.Connection, symbol: str) -> sqlite3.Row | None:
    return conn.execute(
        """
        SELECT * FROM signals
        WHERE symbol = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()


def fetch_signal_history(conn: sqlite3.Connection, symbols: list[str] | None = None, limit: int = 1000) -> list[sqlite3.Row]:
    if symbols:
        placeholders = ", ".join("?" for _ in symbols)
        query = f"""
            SELECT *
            FROM signals
            WHERE symbol IN ({placeholders})
            ORDER BY created_at DESC, id DESC
            LIMIT ?
        """
        params: tuple = (*symbols, limit)
        return list(conn.execute(query, params).fetchall())
    return list(
        conn.execute(
            """
            SELECT *
            FROM signals
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    )


def fetch_news_for_run(conn: sqlite3.Connection, run_id: int, limit: int = 25) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT * FROM news_items
            WHERE run_id = ?
            ORDER BY published_at DESC
            LIMIT ?
            """,
            (run_id, limit),
        ).fetchall()
    )


def fetch_positions_for_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT * FROM positions_snapshot
            WHERE run_id = ?
            ORDER BY market_value DESC, symbol ASC
            """,
            (run_id,),
        ).fetchall()
    )


def fetch_trades_for_run(conn: sqlite3.Connection, run_id: int) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            """
            SELECT * FROM trades
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        ).fetchall()
    )
