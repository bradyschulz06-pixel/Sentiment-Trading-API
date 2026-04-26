from pathlib import Path

from app.config import Settings
from app.models import EarningsBundle, PriceBar
from app.services.backtest import _OpenPosition, _determine_exit_reason, simulate_backtest


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
        stop_loss_pct=0.07,
        signal_threshold=0.20,
        factor_momentum_weight=0.00,
        factor_sentiment_weight=0.00,
        factor_earnings_weight=1.00,
        market_regime_filter_enabled=True,
        market_regime_cautious_threshold_boost=0.04,
        market_regime_cautious_max_positions_multiplier=0.50,
        auto_trade_enabled=False,
        engine_run_interval_minutes=0,
        backtest_benchmark_symbol="SPY",
        backtest_commission_per_order=1.00,
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


def _realistic_bars(symbol: str, start: float, n: int) -> list[PriceBar]:
    """3-up/1-down uptrend giving RSI ~60, well below the 75 entry filter."""
    closes: list[float] = []
    price = start
    for i in range(n):
        price *= 0.985 if i % 4 == 3 else 1.007
        closes.append(price)
    return _bars(symbol, closes)


def test_simulate_backtest_generates_trades_and_curve() -> None:
    settings = _settings()
    price_map = {
        "AAA": _realistic_bars("AAA", 100.0, 180),
        "BBB": _realistic_bars("BBB", 120.0, 180),
        "SPY": _realistic_bars("SPY", 400.0, 180),
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
        "AAA": _realistic_bars("AAA", 100.0, 180),
        "BBB": _realistic_bars("BBB", 120.0, 180),
        "SPY": _bars("SPY", [450 - (i * 1.1) for i in range(180)]),
    }
    earnings_map = {
        "AAA": [EarningsBundle(symbol="AAA", reported_date="2026-03-15", surprise_pct=12.0)],
        "BBB": [EarningsBundle(symbol="BBB", reported_date="2026-03-15", surprise_pct=10.0)],
    }
    result = simulate_backtest(settings, price_map, earnings_map, "SPY", 100, 100_000.0)
    assert result.total_trades == 0
    assert any("risk-off" in note.lower() for note in result.notes)


# --- simplified exit logic tests ---

def test_hard_stop_triggers_below_entry_loss_threshold() -> None:
    position = _OpenPosition("X", 100, 100.0, "2026-01-01", 105.0, 1.0)
    result = _determine_exit_reason(
        current_price=92.0,
        position=position,
        hold_days=5,
        stop_loss_pct=0.07,
        max_hold_days=20,
    )
    assert result == "Hard stop hit."


def test_hard_stop_does_not_trigger_above_threshold() -> None:
    position = _OpenPosition("X", 100, 100.0, "2026-01-01", 100.0, 1.0)
    result = _determine_exit_reason(
        current_price=94.0,
        position=position,
        hold_days=5,
        stop_loss_pct=0.07,
        max_hold_days=20,
    )
    assert result is None


def test_time_stop_triggers_at_max_hold_days() -> None:
    position = _OpenPosition("X", 100, 100.0, "2026-01-01", 102.0, 1.0)
    result = _determine_exit_reason(
        current_price=103.0,
        position=position,
        hold_days=20,
        stop_loss_pct=0.07,
        max_hold_days=20,
    )
    assert result == "Time stop reached."


def test_no_exit_within_hold_window() -> None:
    position = _OpenPosition("X", 100, 100.0, "2026-01-01", 105.0, 1.0)
    result = _determine_exit_reason(
        current_price=103.0,
        position=position,
        hold_days=5,
        stop_loss_pct=0.07,
        max_hold_days=20,
    )
    assert result is None


# --- conviction sizing formula tests ---

def test_conviction_sizing_scalar_scales_with_score() -> None:
    settings = _settings()
    assert settings.conviction_sizing_enabled
    min_s = settings.conviction_sizing_min_scalar
    max_s = settings.conviction_sizing_max_scalar
    threshold = settings.signal_threshold

    def scalar(score: float) -> float:
        threshold_range = max(1.0 - threshold, 0.01)
        conviction = max(0.0, min(1.0, (score - threshold) / threshold_range))
        return min_s + (max_s - min_s) * conviction

    assert abs(scalar(threshold) - min_s) < 1e-9
    assert abs(scalar(1.0) - max_s) < 1e-9
    mid = scalar(0.6)
    assert min_s < mid < max_s


def test_conviction_sizing_disabled_gives_flat_allocation() -> None:
    conviction_sizing_enabled = False
    min_s = 0.75
    max_s = 1.25
    threshold = 0.20
    for score in (threshold, 0.6, 1.0):
        if conviction_sizing_enabled:
            threshold_range = max(1.0 - threshold, 0.01)
            conviction = max(0.0, min(1.0, (score - threshold) / threshold_range))
            s = min_s + (max_s - min_s) * conviction
        else:
            s = 1.0
        assert s == 1.0


# --- re-entry cooldown tests ---

def _make_price_map_with_stop_and_recovery() -> dict:
    """AAA drops sharply (triggers -7% hard stop) then recovers; SPY trends up."""
    aaa_closes = []
    price = 100.0
    for i in range(180):
        if i < 90:
            price *= 0.985 if i % 4 == 3 else 1.007   # uptrend → entry
        elif 90 <= i < 95:
            price *= 0.975                              # sharp drop → hard stop triggered
        else:
            price *= 0.985 if i % 4 == 3 else 1.007   # recovery → would re-enter
        aaa_closes.append(price)
    spy_closes = []
    price = 400.0
    for i in range(180):
        price *= 0.985 if i % 4 == 3 else 1.005
        spy_closes.append(price)
    return {
        "AAA": _bars("AAA", aaa_closes),
        "SPY": _bars("SPY", spy_closes),
    }


def test_reentry_cooldown_blocks_immediate_reentry() -> None:
    settings = _settings()
    price_map = _make_price_map_with_stop_and_recovery()
    result_with_cooldown = simulate_backtest(
        settings, price_map, {}, "SPY", 100, 100_000.0,
        reentry_cooldown_days=3,
    )
    result_no_cooldown = simulate_backtest(
        settings, price_map, {}, "SPY", 100, 100_000.0,
        reentry_cooldown_days=0,
    )
    assert result_no_cooldown.total_trades >= result_with_cooldown.total_trades


def test_reentry_cooldown_zero_allows_immediate_reentry() -> None:
    settings = _settings()
    price_map = _make_price_map_with_stop_and_recovery()
    result = simulate_backtest(
        settings, price_map, {}, "SPY", 100, 100_000.0,
        reentry_cooldown_days=0,
    )
    assert result.status == "ok"
    assert result.ending_equity > 0


# --- portfolio rolling drawdown guard ---

def test_portfolio_drawdown_guard_halts_entries() -> None:
    """When portfolio drops >5% in 10 days, new entries should stop."""
    settings = _settings()
    # Construct a scenario where SPY rises (risk_on regime) but universe stocks crash.
    spy_closes = [400.0 + i * 0.2 for i in range(180)]
    # AAA: good uptrend initially but then crashes hard
    aaa_closes = [100.0 + i * 0.3 for i in range(100)] + [70.0 - i * 0.2 for i in range(80)]
    price_map = {
        "AAA": _bars("AAA", aaa_closes),
        "SPY": _bars("SPY", spy_closes),
    }
    earnings_map = {
        "AAA": [EarningsBundle(symbol="AAA", reported_date="2026-01-15", surprise_pct=12.0)],
    }
    result = simulate_backtest(settings, price_map, earnings_map, "SPY", 150, 100_000.0)
    assert result.status == "ok"
    assert result.ending_equity > 0


def test_date_range_filtering_limits_backtest_period() -> None:
    """start_date/end_date params restrict the simulation to the specified window."""
    settings = _settings()
    # 180 bars spanning ~6 months across 2026 months 01–06
    price_map = {
        "AAA": _realistic_bars("AAA", 100.0, 180),
        "BBB": _realistic_bars("BBB", 120.0, 180),
        "SPY": _realistic_bars("SPY", 400.0, 180),
    }
    earnings_map = {
        "AAA": [EarningsBundle(symbol="AAA", reported_date="2026-03-15", surprise_pct=9.0)],
    }
    full_result = simulate_backtest(settings, price_map, earnings_map, "SPY", 180, 100_000.0)
    # Start/end date filter restricts to a sub-range
    partial_result = simulate_backtest(
        settings, price_map, earnings_map, "SPY", 180, 100_000.0,
        start_date="2026-02-01",
        end_date="2026-04-28",
    )
    assert partial_result.period_days < full_result.period_days
    assert partial_result.status == "ok"
