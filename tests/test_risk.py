from pathlib import Path

from app.config import Settings
from app.services.risk import RiskGate, RiskState, SECTOR_MAP


def _settings(**overrides) -> Settings:
    base = dict(
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
    base.update(overrides)
    return Settings(**base)


def _gate(**overrides) -> tuple[RiskGate, RiskState]:
    return RiskGate(_settings(**overrides)), RiskState()


def test_risk_gate_approves_normal_buy() -> None:
    gate, state = _gate()
    verdict = gate.evaluate_buy("AAPL", daily_pnl_pct=0.0, state=state, current_sector_counts={})
    assert verdict.approved


def test_daily_loss_limit_blocks_buy() -> None:
    gate, state = _gate(max_daily_loss_pct=0.02)
    verdict = gate.evaluate_buy("AAPL", daily_pnl_pct=-0.03, state=state, current_sector_counts={})
    assert not verdict.approved
    assert "Daily loss limit" in verdict.reason


def test_circuit_breaker_blocks_after_three_stops() -> None:
    gate, state = _gate()
    state.stops_today = 3
    verdict = gate.evaluate_buy("AAPL", daily_pnl_pct=0.0, state=state, current_sector_counts={})
    assert not verdict.approved
    assert "Circuit breaker" in verdict.reason


def test_trade_frequency_blocks_second_trade_same_symbol() -> None:
    gate, state = _gate(max_trades_per_symbol_per_day=1)
    state.record_trade("AAPL")
    verdict = gate.evaluate_buy("AAPL", daily_pnl_pct=0.0, state=state, current_sector_counts={})
    assert not verdict.approved
    assert "Trade frequency" in verdict.reason


def test_sector_cap_blocks_third_position_in_same_sector() -> None:
    gate, state = _gate(max_positions_per_sector=2)
    # AAPL, MSFT both Technology → sector count = 2 → NVDA (also Technology) is blocked
    sector_counts = {"Technology": 2}
    verdict = gate.evaluate_buy("NVDA", daily_pnl_pct=0.0, state=state, current_sector_counts=sector_counts)
    assert not verdict.approved
    assert "Sector cap" in verdict.reason


def test_sector_cap_allows_buy_in_different_sector() -> None:
    gate, state = _gate(max_positions_per_sector=2)
    sector_counts = {"Technology": 2}
    # JPM is Financials — not capped
    verdict = gate.evaluate_buy("JPM", daily_pnl_pct=0.0, state=state, current_sector_counts=sector_counts)
    assert verdict.approved


def test_record_stop_increments_counter() -> None:
    state = RiskState()
    assert state.stops_today == 0
    state.record_stop()
    state.record_stop()
    assert state.stops_today == 2


def test_sector_map_has_expected_entries() -> None:
    assert SECTOR_MAP["AAPL"] == "Technology"
    assert SECTOR_MAP["JPM"] == "Financials"
    assert SECTOR_MAP["XOM"] == "Energy"
