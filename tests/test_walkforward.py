from pathlib import Path

import app.services.walkforward as walkforward_module
from app.config import Settings
from app.models import BacktestResult
from app.services.walkforward import WalkForwardService


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
        universe_symbols="",
        lookback_days=180,
        news_window_days=7,
        earnings_lookup_limit=8,
        upcoming_earnings_buffer_days=2,
        max_positions=2,
        max_position_pct=0.50,
        stop_loss_pct=0.08,
        signal_threshold=0.32,
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


def _fake_backtest_result(
    *,
    universe_preset: str,
    threshold: float,
    momentum_weight: float,
    sentiment_weight: float,
    earnings_weight: float,
    period_days: int,
) -> BacktestResult:
    base_return = 0.02 + (period_days / 10_000.0)
    if universe_preset == "balanced_quality":
        base_return += 0.015
    if earnings_weight > momentum_weight:
        base_return += 0.01
    if abs(threshold - 0.32) < 0.001:
        base_return += 0.008
    benchmark_return = 0.025
    outperformance = base_return - benchmark_return
    return BacktestResult(
        status="ok",
        summary="stub",
        start_date="2026-01-01",
        end_date="2026-04-01",
        period_days=period_days,
        starting_capital=100_000.0,
        ending_equity=100_000.0 * (1.0 + base_return),
        total_return_pct=round(base_return, 4),
        benchmark_symbol="SPY",
        benchmark_return_pct=benchmark_return,
        outperformance_pct=round(outperformance, 4),
        max_drawdown_pct=-0.02 if universe_preset == "balanced_quality" else -0.04,
        universe_preset=universe_preset,
        universe_label=universe_preset.replace("_", " ").title(),
        signal_threshold=threshold,
        factor_momentum_weight=momentum_weight,
        factor_sentiment_weight=sentiment_weight,
        factor_earnings_weight=earnings_weight,
        total_trades=8,
        win_rate_pct=0.5,
        average_trade_return_pct=0.01,
        sharpe_ratio=0.8,
        sortino_ratio=1.1,
        chart_points="",
        benchmark_chart_points="",
        benchmark_ending_equity=102_500.0,
        commission_per_order=1.0,
        slippage_bps=8.0,
        max_hold_days=18,
        min_hold_days=3,
        trailing_stop_pct=0.06,
        trailing_arm_pct=0.0,
        take_profit_pct=0.14,
    )


def test_walkforward_prefers_more_stable_candidates(monkeypatch) -> None:
    service = WalkForwardService(_settings())
    monkeypatch.setattr(
        service.backtests,
        "load_market_data",
        lambda symbols, fetch_days: ({symbol: [] for symbol in symbols}, {symbol: [] for symbol in symbols}),
    )

    def fake_simulate_backtest(*args, **kwargs):
        return _fake_backtest_result(
            universe_preset=kwargs["universe_preset"],
            threshold=kwargs["signal_threshold"],
            momentum_weight=kwargs["factor_momentum_weight"],
            sentiment_weight=kwargs["factor_sentiment_weight"],
            earnings_weight=kwargs["factor_earnings_weight"],
            period_days=kwargs["period_days"],
        )

    monkeypatch.setattr(walkforward_module, "simulate_backtest", fake_simulate_backtest)

    result = service.run(windows=[60, 90], thresholds=[0.32], starting_capital=100_000.0)

    assert result.status == "ok"
    assert result.best_candidate is not None
    assert result.best_candidate.universe_preset == "balanced_quality"
    assert result.best_candidate.signal_threshold == 0.32
    assert len(result.best_candidate.windows) == 2
