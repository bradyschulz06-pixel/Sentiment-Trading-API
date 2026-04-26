from pathlib import Path

from app.config import Settings
from app.models import PriceBar
from app.services.market_regime import evaluate_market_regime


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
        universe_preset="sector_leaders",
        universe_symbols="",
        lookback_days=180,
        news_window_days=7,
        earnings_lookup_limit=8,
        upcoming_earnings_buffer_days=2,
        max_positions=4,
        max_position_pct=0.08,
        stop_loss_pct=0.07,
        signal_threshold=0.30,
        factor_momentum_weight=0.00,
        factor_sentiment_weight=0.00,
        factor_earnings_weight=1.00,
        market_regime_filter_enabled=True,
        market_regime_cautious_threshold_boost=0.04,
        market_regime_cautious_max_positions_multiplier=0.50,
        auto_trade_enabled=False,
        engine_run_interval_minutes=0,
        backtest_benchmark_symbol="SPY",
        backtest_commission_per_order=1.0,
        backtest_slippage_bps=20.0,
        backtest_max_hold_days=20,
        backtest_min_hold_days=3,
        backtest_trailing_stop_pct=0.00,
        backtest_trailing_arm_pct=0.00,
        backtest_take_profit_pct=0.00,
        backtest_breakeven_arm_pct=0.00,
        backtest_breakeven_floor_pct=0.005,
        backtest_reentry_cooldown_days=3,
        backtest_rolling_drawdown_window=10,
        backtest_rolling_drawdown_limit=0.05,
        conviction_sizing_enabled=True,
        conviction_sizing_min_scalar=0.75,
        conviction_sizing_max_scalar=1.25,
        alpha_vantage_requests_per_minute=5,
        max_daily_loss_pct=0.02,
        max_trades_per_symbol_per_day=1,
        max_positions_per_sector=2,
        backtest_min_bars=0,
        db_path=Path("test.db"),
    )


def _bars(symbol: str, closes: list[float]) -> list[PriceBar]:
    return [
        PriceBar(
            symbol=symbol,
            timestamp=f"2026-01-{(index % 28) + 1:02d}T00:00:00Z",
            open=close - 1,
            high=close + 1,
            low=close - 2,
            close=close,
            volume=1_000_000,
        )
        for index, close in enumerate(closes)
    ]


def test_risk_on_when_spy_above_sma50() -> None:
    """SPY trending up above its SMA50 → risk_on, new longs allowed."""
    settings = _settings()
    benchmark = _bars("SPY", [400 + (i * 1.0) for i in range(80)])
    regime = evaluate_market_regime(settings, "SPY", benchmark, {})
    assert regime.label == "risk_on"
    assert regime.allow_new_longs
    assert regime.threshold_boost == 0.0
    assert regime.max_positions_multiplier == 1.0


def test_risk_off_when_spy_below_sma50() -> None:
    """SPY declining below its SMA50 → risk_off, new longs blocked."""
    settings = _settings()
    benchmark = _bars("SPY", [500 - (i * 1.2) for i in range(80)])
    regime = evaluate_market_regime(settings, "SPY", benchmark, {})
    assert regime.label == "risk_off"
    assert not regime.allow_new_longs
    assert regime.effective_max_positions(settings.max_positions) == 0


def test_risk_off_when_benchmark_history_too_short() -> None:
    """Fewer than 50 bars → defaults to risk_off (conservative)."""
    settings = _settings()
    benchmark = _bars("SPY", [400.0] * 30)
    regime = evaluate_market_regime(settings, "SPY", benchmark, {})
    assert regime.label == "risk_off"
    assert not regime.allow_new_longs


def test_inactive_when_filter_disabled() -> None:
    """Filter disabled → inactive regime, longs always allowed."""
    settings_off = Settings(
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
        max_positions=4,
        max_position_pct=0.08,
        stop_loss_pct=0.07,
        signal_threshold=0.30,
        factor_momentum_weight=0.00,
        factor_sentiment_weight=0.00,
        factor_earnings_weight=1.00,
        market_regime_filter_enabled=False,
        market_regime_cautious_threshold_boost=0.04,
        market_regime_cautious_max_positions_multiplier=0.50,
        auto_trade_enabled=False,
        engine_run_interval_minutes=0,
        backtest_benchmark_symbol="SPY",
        backtest_commission_per_order=1.0,
        backtest_slippage_bps=20.0,
        backtest_max_hold_days=20,
        backtest_min_hold_days=3,
        backtest_trailing_stop_pct=0.00,
        backtest_trailing_arm_pct=0.00,
        backtest_take_profit_pct=0.00,
        backtest_breakeven_arm_pct=0.00,
        backtest_breakeven_floor_pct=0.005,
        backtest_reentry_cooldown_days=3,
        backtest_rolling_drawdown_window=10,
        backtest_rolling_drawdown_limit=0.05,
        conviction_sizing_enabled=True,
        conviction_sizing_min_scalar=0.75,
        conviction_sizing_max_scalar=1.25,
        alpha_vantage_requests_per_minute=5,
        max_daily_loss_pct=0.02,
        max_trades_per_symbol_per_day=1,
        max_positions_per_sector=2,
        backtest_min_bars=0,
        db_path=Path("test.db"),
    )
    benchmark = _bars("SPY", [400 + (i * 1.0) for i in range(80)])
    regime = evaluate_market_regime(settings_off, "SPY", benchmark, {})
    assert regime.label == "inactive"
    assert regime.allow_new_longs


def test_no_previous_label_parameter() -> None:
    """evaluate_market_regime no longer accepts previous_label — verify 2-state only."""
    settings = _settings()
    benchmark = _bars("SPY", [400 + (i * 1.0) for i in range(80)])
    regime = evaluate_market_regime(settings, "SPY", benchmark, {})
    assert regime.label in {"risk_on", "risk_off", "inactive"}
