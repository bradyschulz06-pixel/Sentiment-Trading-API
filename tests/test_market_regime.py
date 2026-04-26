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
        backtest_commission_per_order=1.0,
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


def _rising_universe(n: int = 6) -> dict[str, list[PriceBar]]:
    """Generate `n` steadily rising symbols — all above their 20/50-day MAs."""
    bases = [100, 120, 140, 160, 80, 200, 250, 300]
    return {
        chr(65 + i) * 3: _bars(chr(65 + i) * 3, [bases[i % len(bases)] + (j * 0.8) for j in range(80)])
        for i in range(n)
    }


def test_market_regime_supportive_when_trend_and_breadth_align() -> None:
    settings = _settings()
    benchmark = _bars("SPY", [400 + (i * 1.0) for i in range(80)])
    universe = _rising_universe(6)
    regime = evaluate_market_regime(settings, "SPY", benchmark, universe)
    assert regime.label == "supportive"
    assert regime.allow_new_longs
    assert regime.threshold_boost == 0.0


def test_market_regime_risk_off_when_benchmark_breaks_down() -> None:
    settings = _settings()
    benchmark = _bars("SPY", [500 - (i * 1.2) for i in range(80)])
    universe = _rising_universe(6)
    regime = evaluate_market_regime(settings, "SPY", benchmark, universe)
    assert regime.label == "risk_off"
    assert not regime.allow_new_longs
    assert regime.effective_max_positions(settings.max_positions) == 0


def test_market_regime_cautious_when_breadth_is_sparse() -> None:
    """Fewer than the minimum sample symbols falls back to neutral breadth → cautious."""
    settings = _settings()
    benchmark = _bars("SPY", [400 + (i * 1.0) for i in range(80)])
    universe = {
        "AAA": _bars("AAA", [100 + (i * 0.8) for i in range(80)]),
        "BBB": _bars("BBB", [120 + (i * 0.9) for i in range(80)]),
    }
    regime = evaluate_market_regime(settings, "SPY", benchmark, universe)
    # Only 2 symbols — below _BREADTH_MIN_SAMPLE; breadth is neutral so can't be "supportive".
    assert regime.label in {"cautious", "risk_off"}
