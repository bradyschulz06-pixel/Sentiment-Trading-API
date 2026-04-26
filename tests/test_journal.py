from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.config import Settings
from app.db import get_connection, initialize_database
from app.models import PositionSnapshot
from app.services.journal import PaperJournalService


def _settings(db_path: Path) -> Settings:
    return Settings(
        app_name="Test",
        environment="test",
        host="127.0.0.1",
        port=8000,
        debug=False,
        admin_username="admin",
        admin_password="secret",
        admin_password_hash="",
        session_secret="secret",
        alpaca_api_key="key",
        alpaca_secret_key="secret",
        alpaca_trading_base_url="https://paper-api.alpaca.markets",
        alpaca_data_base_url="https://data.alpaca.markets",
        alpha_vantage_api_key="av",
        universe_preset="sector_leaders",
        universe_symbols="",
        lookback_days=180,
        news_window_days=7,
        earnings_lookup_limit=8,
        upcoming_earnings_buffer_days=2,
        max_positions=5,
        max_position_pct=0.10,
        stop_loss_pct=0.08,
        signal_threshold=0.30,
        factor_momentum_weight=0.40,
        factor_sentiment_weight=0.25,
        factor_earnings_weight=0.35,
        market_regime_filter_enabled=True,
        market_regime_cautious_threshold_boost=0.04,
        market_regime_cautious_max_positions_multiplier=0.50,
        auto_trade_enabled=False,
        engine_run_interval_minutes=0,
        backtest_benchmark_symbol="SPY",
        backtest_commission_per_order=1.00,
        backtest_slippage_bps=8.0,
        backtest_max_hold_days=18,
        backtest_min_hold_days=3,
        backtest_trailing_stop_pct=0.06,
        backtest_trailing_arm_pct=0.00,
        backtest_take_profit_pct=0.14,
        backtest_breakeven_arm_pct=0.03,
        backtest_breakeven_floor_pct=0.005,
        backtest_reentry_cooldown_days=3,
        conviction_sizing_enabled=True,
        conviction_sizing_min_scalar=0.75,
        conviction_sizing_max_scalar=1.25,
        alpha_vantage_requests_per_minute=5,
        max_daily_loss_pct=0.02,
        max_trades_per_symbol_per_day=1,
        max_positions_per_sector=2,
        backtest_min_bars=0,
        db_path=db_path,
    )


def test_paper_journal_reconstructs_closed_and_open_positions(tmp_path) -> None:
    settings = _settings(tmp_path / "journal.db")
    initialize_database(settings)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    buy_time = now - timedelta(days=5)
    sell_time = now - timedelta(days=1)
    msft_buy_time = now - timedelta(days=3)

    with get_connection(settings) as conn:
        run_id = conn.execute(
            """
            INSERT INTO runs (trigger, status, summary, error, warnings_json, started_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            ("manual", "ok", "test", "", "[]", (now - timedelta(days=6)).isoformat(), now.isoformat()),
        ).lastrowid
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
                "AAPL",
                190.0,
                0.62,
                0.18,
                0.25,
                0.41,
                "buy",
                "trend is above the 20/50-day moving averages. news tone is positive.",
                175.0,
                212.0,
                None,
                "Apple demand remains healthy",
                (buy_time - timedelta(hours=2)).isoformat(),
            ),
        )
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
                "MSFT",
                420.0,
                0.40,
                0.12,
                0.61,
                0.43,
                "buy",
                "recent earnings surprise is supportive.",
                386.0,
                470.0,
                None,
                "Azure momentum stays firm",
                (msft_buy_time - timedelta(hours=1)).isoformat(),
            ),
        )

    service = PaperJournalService(settings)
    service.alpaca.get_orders = lambda limit=200: [
        {
            "symbol": "AAPL",
            "side": "sell",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "205",
            "submitted_at": sell_time.isoformat(),
            "filled_at": sell_time.isoformat(),
        },
        {
            "symbol": "MSFT",
            "side": "buy",
            "status": "filled",
            "filled_qty": "5",
            "filled_avg_price": "420",
            "submitted_at": msft_buy_time.isoformat(),
            "filled_at": msft_buy_time.isoformat(),
        },
        {
            "symbol": "AAPL",
            "side": "buy",
            "status": "filled",
            "filled_qty": "10",
            "filled_avg_price": "190",
            "submitted_at": buy_time.isoformat(),
            "filled_at": buy_time.isoformat(),
        },
    ]
    service.alpaca.get_positions = lambda: [
        PositionSnapshot(
            symbol="MSFT",
            qty=5,
            avg_entry_price=420.0,
            market_value=2150.0,
            unrealized_plpc=(2150.0 - 2100.0) / 2100.0,
        )
    ]

    result = service.run()

    assert result.status == "ok"
    assert result.total_closed_trades == 1
    assert result.closed_trades[0].symbol == "AAPL"
    assert result.closed_trades[0].dominant_factor == "momentum"
    assert result.closed_trades[0].pnl == 150.0
    assert result.open_positions_count == 1
    assert result.open_positions[0].symbol == "MSFT"
    assert result.open_positions[0].dominant_factor == "earnings"
    assert result.factor_summaries[0].factor_name == "momentum"
