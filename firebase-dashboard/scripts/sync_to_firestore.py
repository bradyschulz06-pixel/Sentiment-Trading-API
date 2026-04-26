#!/usr/bin/env python3
"""
sync_to_firestore.py
====================
Reads data from the trading system's SQLite database and writes it
to Firestore so the Firebase dashboard stays current.

HOW TO RUN
----------
# After an engine run (push latest state):
  python scripts/sync_to_firestore.py

# Also sync journal closed trades + performance summary:
  python scripts/sync_to_firestore.py --journal

# Sync a specific number of runs to history:
  python scripts/sync_to_firestore.py --history 10

SETUP
-----
1. pip install -r scripts/requirements.txt
2. Create a Firebase service account:
     Firebase Console → Project Settings → Service Accounts → Generate new private key
3. Save the downloaded JSON as:  scripts/serviceAccountKey.json
4. Run this script from the project root:
     PYTHONPATH=. python scripts/sync_to_firestore.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ── Bootstrap: make sure app/ is importable ──────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Firebase admin SDK ────────────────────────────────────────────
try:
    import firebase_admin
    from firebase_admin import credentials, firestore as fb_firestore
except ImportError:
    sys.exit(
        "ERROR: firebase-admin is not installed.\n"
        "Run:  pip install firebase-admin"
    )

# ── App imports ───────────────────────────────────────────────────
from app.config import get_settings
from app.db import get_connection, fetch_latest_run, fetch_signals_for_run, \
    fetch_positions_for_run, fetch_trades_for_run, fetch_recent_runs
from app.services.journal import PaperJournalService
from app.services.market_regime import parse_regime_warning


# ═══════════════════════════════════════════════════════════════════
#  FIREBASE INIT
# ═══════════════════════════════════════════════════════════════════

SERVICE_ACCOUNT_PATH = Path(__file__).parent / "serviceAccountKey.json"


def init_firebase() -> fb_firestore.Client:
    """Initialize Firebase Admin SDK and return a Firestore client."""
    if not SERVICE_ACCOUNT_PATH.exists():
        sys.exit(
            f"ERROR: Service account key not found at:\n  {SERVICE_ACCOUNT_PATH}\n\n"
            "Download it from:\n"
            "  Firebase Console → Project Settings → Service Accounts "
            "→ Generate new private key\n"
            "Then save it as scripts/serviceAccountKey.json"
        )
    cred = credentials.Certificate(str(SERVICE_ACCOUNT_PATH))
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    return fb_firestore.client()


# ═══════════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════════

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_regime_from_warnings(warnings_json: str) -> dict | None:
    """Extract the market regime dict from a run's warnings JSON string."""
    try:
        warnings = json.loads(warnings_json or "[]")
    except json.JSONDecodeError:
        return None
    for w in warnings:
        parsed = parse_regime_warning(w)
        if parsed is not None:
            return parsed
    return None


def user_warnings(warnings_json: str) -> list[str]:
    """Return warnings that are NOT the regime JSON blob."""
    try:
        warnings = json.loads(warnings_json or "[]")
    except json.JSONDecodeError:
        return []
    return [w for w in warnings if not w.startswith("MARKET_REGIME::")]


# ═══════════════════════════════════════════════════════════════════
#  SYNC: LATEST STATE  (/state/latest)
# ═══════════════════════════════════════════════════════════════════

def sync_latest_state(db: fb_firestore.Client, settings) -> None:
    """
    Write (or overwrite) the /state/latest document.
    This is the primary document the dashboard reads in real time.
    """
    print("Syncing /state/latest …")

    with get_connection(settings) as conn:
        run = fetch_latest_run(conn)
        if run is None:
            print("  No runs found in database — nothing to sync.")
            return

        signals   = fetch_signals_for_run(conn, run["id"])
        positions = fetch_positions_for_run(conn, run["id"])
        trades    = fetch_trades_for_run(conn, run["id"])

    regime   = parse_regime_from_warnings(run["warnings_json"])
    warnings = user_warnings(run["warnings_json"])

    # Build signal documents (all symbols scored this run)
    signals_list = [
        {
            "symbol":           s["symbol"],
            "price":            s["price"],
            "momentum_score":   s["momentum_score"],
            "sentiment_score":  s["sentiment_score"],
            "earnings_score":   s["earnings_score"],
            "composite_score":  s["composite_score"],
            "decision":         s["decision"],
            "rationale":        s["rationale"],
            "stop_price":       s["stop_price"],
            "target_price":     s["target_price"],
            "next_earnings_date": s["next_earnings_date"],
            "headline":         s["headline"] or "",
            "created_at":       s["created_at"],
        }
        for s in signals
    ]
    # Sort: buy first, then by composite_score descending
    signals_list.sort(
        key=lambda x: (x["decision"] != "buy", -x["composite_score"])
    )

    # Build positions
    positions_list = [
        {
            "symbol":           p["symbol"],
            "qty":              p["qty"],
            "avg_entry_price":  p["avg_entry_price"],
            "market_value":     p["market_value"],
            "unrealized_plpc":  p["unrealized_plpc"],
            "side":             p["side"],
        }
        for p in positions
    ]

    # Build trade intents (planned/executed by this run)
    intents_list = [
        {
            "symbol":          t["symbol"],
            "side":            t["side"],
            "qty":             t["qty"],
            "notional":        t["notional"],
            "reason":          t["reason"],
            "status":          t["status"],
            "broker_order_id": t["broker_order_id"] or "",
            "created_at":      t["created_at"],
        }
        for t in trades
    ]

    doc = {
        "updated_at": now_iso(),
        "run": {
            "run_id":           str(run["id"]),
            "trigger":          run["trigger"],
            "status":           run["status"],
            "summary":          run["summary"],
            "error":            run["error"] or "",
            "started_at":       run["started_at"],
            "completed_at":     run["completed_at"],
            "signal_threshold": settings.signal_threshold,
        },
        "regime":         regime,
        "warnings":       warnings,
        "signals":        signals_list,
        "positions":      positions_list,
        "trade_intents":  intents_list,
        "config": {
            "auto_trade_enabled":      settings.auto_trade_enabled,
            "market_regime_filter":    settings.market_regime_filter_enabled,
            "max_positions":           settings.max_positions,
            "stop_loss_pct":           settings.stop_loss_pct,
            "signal_threshold":        settings.signal_threshold,
            "universe_preset":         settings.universe_preset,
        },
    }

    db.collection("state").document("latest").set(doc)
    buy_count = sum(1 for s in signals_list if s["decision"] == "buy")
    print(f"  ✓ Wrote {len(signals_list)} signals ({buy_count} buys), "
          f"{len(positions_list)} positions, {len(intents_list)} trade intents.")


# ═══════════════════════════════════════════════════════════════════
#  SYNC: RUN HISTORY  (/runs/{run_id})
# ═══════════════════════════════════════════════════════════════════

def sync_run_history(db: fb_firestore.Client, settings, limit: int = 20) -> None:
    """
    Write the last `limit` engine runs to the /runs collection.
    Each document is keyed by the SQLite run ID.
    """
    print(f"Syncing /runs (last {limit}) …")
    with get_connection(settings) as conn:
        runs = fetch_recent_runs(conn, limit=limit)

    batch = db.batch()
    for run in runs:
        regime = parse_regime_from_warnings(run["warnings_json"])

        # Fetch signal/position counts from embedded data if available
        doc = {
            "run_id":       str(run["id"]),
            "trigger":      run["trigger"],
            "status":       run["status"],
            "summary":      run["summary"],
            "error":        run["error"] or "",
            "started_at":   run["started_at"],
            "completed_at": run["completed_at"],
            "regime_label": regime["label"] if regime else None,
        }
        ref = db.collection("runs").document(str(run["id"]))
        batch.set(ref, doc, merge=True)

    batch.commit()
    print(f"  ✓ Wrote {len(runs)} run records.")


# ═══════════════════════════════════════════════════════════════════
#  SYNC: JOURNAL PERFORMANCE  (/performance/summary + /trades)
# ═══════════════════════════════════════════════════════════════════

def sync_journal(db: fb_firestore.Client, settings) -> None:
    """
    Pull closed trades and performance metrics from the paper journal
    (via Alpaca paper API) and write them to Firestore.
    """
    print("Syncing journal (requires Alpaca API credentials) …")

    if not settings.trading_configured:
        print("  ⚠ Alpaca keys not configured — skipping journal sync.")
        print("    Set ALPACA_API_KEY and ALPACA_SECRET_KEY in your .env file.")
        return

    try:
        service = PaperJournalService(settings)
        result  = service.run()
    except Exception as exc:
        print(f"  ✗ Journal sync failed: {exc}")
        return

    if result.status != "ok":
        print(f"  ✗ Journal returned status '{result.status}'")
        return

    # ── /performance/summary ──────────────────────────────────────
    perf_doc = {
        "updated_at":           now_iso(),
        "source":               "journal",
        "total_trades":         result.total_closed_trades,
        "win_rate_pct":         result.win_rate_pct,
        "realized_pnl":         result.realized_pnl,
        "average_trade_return_pct": result.average_return_pct,
        "unrealized_pnl":       result.unrealized_pnl,
        "best_trade_pnl":       result.best_trade_pnl,
        "worst_trade_pnl":      result.worst_trade_pnl,
        # wins/losses count for the donut chart
        "wins":  sum(1 for t in result.closed_trades if t.pnl >= 0),
        "losses": sum(1 for t in result.closed_trades if t.pnl < 0),
        # closed_trades array (for performance page trade table)
        "closed_trades": [
            {
                "symbol":               t.symbol,
                "qty":                  t.qty,
                "entry_at":             t.entry_at,
                "exit_at":              t.exit_at,
                "entry_price":          t.entry_price,
                "exit_price":           t.exit_price,
                "pnl":                  t.pnl,
                "return_pct":           t.return_pct,
                "hold_days":            t.hold_days,
                "dominant_factor":      t.dominant_factor,
                "entry_composite_score": t.entry_composite_score,
                "entry_earnings_score": t.entry_earnings_score,
                "entry_rationale":      t.entry_rationale,
            }
            for t in result.closed_trades
        ],
        # factor breakdown
        "factor_summaries": [
            {
                "factor_name":        f.factor_name,
                "total_trades":       f.total_trades,
                "win_rate_pct":       f.win_rate_pct,
                "average_return_pct": f.average_return_pct,
                "total_pnl":          f.total_pnl,
            }
            for f in result.factor_summaries
        ],
    }

    db.collection("performance").document("summary").set(perf_doc)
    print(f"  ✓ Performance summary: {result.total_closed_trades} trades, "
          f"win rate {result.win_rate_pct:.1f}%")

    # ── /trades/{auto_id}  (individual closed trade records) ─────
    # Only write trades that don't exist yet (keyed by symbol+entry_at)
    batch = db.batch()
    count = 0
    for t in result.closed_trades:
        # Use a stable document ID so re-runs don't create duplicates
        doc_id = f"{t.symbol}_{t.entry_at[:10]}_{t.exit_at[:10]}"
        ref    = db.collection("trades").document(doc_id)
        batch.set(ref, {
            "symbol":               t.symbol,
            "qty":                  t.qty,
            "entry_at":             t.entry_at,
            "exit_at":              t.exit_at,
            "entry_price":          t.entry_price,
            "exit_price":           t.exit_price,
            "pnl":                  t.pnl,
            "return_pct":           t.return_pct,
            "hold_days":            t.hold_days,
            "dominant_factor":      t.dominant_factor,
            "entry_composite_score": t.entry_composite_score,
            "entry_earnings_score": t.entry_earnings_score,
            "entry_rationale":      t.entry_rationale,
        }, merge=True)
        count += 1

    batch.commit()
    print(f"  ✓ Wrote {count} closed trade records to /trades.")

    # ── Update /performance/summary equity field ──────────────────
    # Add current account equity from open positions
    total_mkt_value = sum(p.market_value for p in result.open_positions)
    if total_mkt_value:
        db.collection("performance").document("summary").update({
            "equity":              total_mkt_value + result.realized_pnl + 100_000,
            "open_positions_count": result.open_positions_count,
        })


# ═══════════════════════════════════════════════════════════════════
#  SYNC: BACKTEST EQUITY CURVE  (/performance/summary)
# ═══════════════════════════════════════════════════════════════════

def sync_backtest_result(db: fb_firestore.Client, result) -> None:
    """
    Write a BacktestResult to /performance/summary.
    Call this from your own script after running a backtest.

    Example:
        from firebase_dashboard.scripts.sync_to_firestore import sync_backtest_result, init_firebase
        db = init_firebase()
        sync_backtest_result(db, backtest_result)
    """
    equity_curve = [
        {"date": pt.date, "equity": pt.equity, "benchmark_equity": pt.benchmark_equity}
        for pt in (result.daily_points or [])
    ]

    wins   = sum(1 for t in result.trades if t.pnl >= 0)
    losses = sum(1 for t in result.trades if t.pnl < 0)

    doc = {
        "updated_at":               now_iso(),
        "source":                   "backtest",
        "total_trades":             result.total_trades,
        "win_rate_pct":             result.win_rate_pct * 100,   # stored as 0–100
        "total_return_pct":         result.total_return_pct * 100,
        "benchmark_return_pct":     result.benchmark_return_pct * 100,
        "benchmark_symbol":         result.benchmark_symbol,
        "max_drawdown_pct":         result.max_drawdown_pct * 100,
        "sharpe_ratio":             result.sharpe_ratio,
        "average_trade_return_pct": result.average_trade_return_pct * 100,
        "wins":                     wins,
        "losses":                   losses,
        "best_trade_pnl":  max((t.pnl for t in result.trades), default=0.0),
        "worst_trade_pnl": min((t.pnl for t in result.trades), default=0.0),
        "equity_curve":    equity_curve,
        "closed_trades": [
            {
                "symbol":       t.symbol,
                "entry_at":     t.entry_date,
                "exit_at":      t.exit_date,
                "entry_price":  t.entry_price,
                "exit_price":   t.exit_price,
                "qty":          t.qty,
                "pnl":          t.pnl,
                "return_pct":   t.return_pct,
                "hold_days":    t.hold_days,
                "dominant_factor": "earnings",
            }
            for t in result.trades
        ],
    }
    db.collection("performance").document("summary").set(doc)
    print(f"  ✓ Backtest result synced: {result.total_return_pct*100:.2f}% return, "
          f"{result.total_trades} trades.")


# ═══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync trading system data from SQLite to Firebase Firestore."
    )
    parser.add_argument("--journal",  action="store_true",
                        help="Also sync paper journal trades and performance.")
    parser.add_argument("--history",  type=int, default=20,
                        help="How many historical runs to write to /runs (default 20).")
    args = parser.parse_args()

    settings = get_settings()
    db       = init_firebase()

    # Always sync the latest state
    sync_latest_state(db, settings)

    # Always sync run history
    sync_run_history(db, settings, limit=args.history)

    # Optionally sync journal performance data
    if args.journal:
        sync_journal(db, settings)

    print("\n✅ Sync complete. Open your Firebase dashboard to see the data.")


if __name__ == "__main__":
    main()
