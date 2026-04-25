from pathlib import Path

from app.config import Settings
from app.models import EarningsBundle, PriceBar
from app.services.backtest import simulate_backtest


def _settings() -> Settings:
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
        universe_preset="balanced_quality",
        universe_symbols="AAA,BBB",
        lookback_days=180,
        news_window_days=7,
        earnings_lookup_limit=8,
        upcoming_earnings_buffer_days=2,
        max_positions=2,
        max_position_pct=0.50,
        stop_loss_pct=0.08,
        signal_threshold=0.20,
        factor_momentum_weight=0.45,
        factor_sentiment_weight=0.20,
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
        backtest_min_bars=0,
        db_path=Path("test.db"),
    )


def _bars(symbol: str, closes: list[float]) -> list[PriceBar]:
    output: list[PriceBar] = []
    for index, close in enumerate(closes, start=1):
        output.append(
            PriceBar(
                symbol=symbol,
                timestamp=f"2026-{((index - 1) // 28) + 1:02d}-{((index - 1) % 28) + 1:02d}T00:00:00Z",
                open=close - 1,
                high=close + 1,
                low=close - 2,
                close=close,
                volume=1_000_000,
            )
        )
    return output


def test_simulate_backtest_generates_trades_and_curve() -> None:
    settings = _settings()
    price_map = {
        "AAA": _bars("AAA", [100 + (i * 1.2) for i in range(180)]),
        "BBB": _bars("BBB", [120 + (i * 0.9) for i in range(180)]),
        "SPY": _bars("SPY", [400 + (i * 0.5) for i in range(180)]),
    }
    earnings_map = {
        "AAA": [EarningsBundle(symbol="AAA", reported_date="2026-03-15", surprise_pct=9.0)],
        "BBB": [EarningsBundle(symbol="BBB", reported_date="2026-03-15", surprise_pct=3.0)],
    }
    result = simulate_backtest(settings, price_map, earnings_map, "SPY", 100, 100_000.0)
    assert result.status == "ok"
    assert result.daily_points
    assert result.total_trades >= 1
    assert result.ending_equity > 0
    assert result.chart_points
    assert result.benchmark_chart_points
    assert result.commission_per_order == 1.0
    assert result.trailing_arm_pct == 0.0
    assert result.universe_preset == "balanced_quality"
    assert result.signal_threshold == 0.2


def test_simulate_backtest_blocks_new_longs_in_risk_off_regime() -> None:
    settings = _settings()
    price_map = {
        "AAA": _bars("AAA", [100 + (i * 1.0) for i in range(180)]),
        "BBB": _bars("BBB", [120 + (i * 0.8) for i in range(180)]),
        "SPY": _bars("SPY", [450 - (i * 1.1) for i in range(180)]),
    }
    earnings_map = {
        "AAA": [EarningsBundle(symbol="AAA", reported_date="2026-03-15", surprise_pct=12.0)],
        "BBB": [EarningsBundle(symbol="BBB", reported_date="2026-03-15", surprise_pct=10.0)],
    }
    result = simulate_backtest(settings, price_map, earnings_map, "SPY", 100, 100_000.0)
    assert result.total_trades == 0
    assert any("risk-off" in note.lower() for note in result.notes)
