from pathlib import Path

import app.services.walkforward as walkforward_module
from app.config import Settings
from app.models import BacktestResult, ValidationFold, ValidationResult
from app.services.walkforward import WalkForwardService, _calmar


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
        stop_loss_pct=0.07,
        signal_threshold=0.32,
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


def _fake_backtest_result(
    *,
    universe_preset: str,
    threshold: float,
    period_days: int,
    start_date: str | None = None,
    end_date: str | None = None,
    **_kwargs,
) -> BacktestResult:
    base_return = 0.02 + (period_days / 10_000.0)
    if universe_preset == "balanced_quality":
        base_return += 0.015
    if abs(threshold - 0.32) < 0.001:
        base_return += 0.008
    benchmark_return = 0.025
    outperformance = base_return - benchmark_return
    return BacktestResult(
        status="ok",
        summary="stub",
        start_date=start_date or "2026-01-01",
        end_date=end_date or "2026-04-01",
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
        factor_momentum_weight=0.0,
        factor_sentiment_weight=0.0,
        factor_earnings_weight=1.0,
        total_trades=8,
        win_rate_pct=0.5,
        average_trade_return_pct=0.01,
        sharpe_ratio=0.8,
        sortino_ratio=1.1,
        chart_points="",
        benchmark_chart_points="",
        benchmark_ending_equity=102_500.0,
        commission_per_order=1.0,
        slippage_bps=20.0,
        max_hold_days=20,
        min_hold_days=3,
        trailing_stop_pct=0.0,
        trailing_arm_pct=0.0,
        take_profit_pct=0.0,
    )


def test_walkforward_run_returns_best_candidate(monkeypatch) -> None:
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
            period_days=kwargs["period_days"],
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
        )

    monkeypatch.setattr(walkforward_module, "simulate_backtest", fake_simulate_backtest)

    result = service.run(windows=[60, 90], thresholds=[0.32], starting_capital=100_000.0)

    assert result.status == "ok"
    assert result.best_candidate is not None
    assert result.best_candidate.universe_preset == "balanced_quality"
    assert result.best_candidate.signal_threshold == 0.32
    assert len(result.best_candidate.windows) == 2


def test_walkforward_run_includes_stage2_candidates(monkeypatch) -> None:
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
            period_days=kwargs["period_days"],
        )

    monkeypatch.setattr(walkforward_module, "simulate_backtest", fake_simulate_backtest)

    result = service.run(windows=[60, 90], thresholds=[0.32], starting_capital=100_000.0)

    assert result.best_candidate is not None
    assert hasattr(result.best_candidate, "trailing_arm_pct")
    assert hasattr(result.best_candidate, "max_hold_days")
    assert isinstance(result.best_candidate.trailing_arm_pct, float)
    assert isinstance(result.best_candidate.max_hold_days, int)
    stage2 = [c for c in result.candidates if "arm" in c.label]
    assert len(stage2) > 0, "Stage-2 candidates should appear in results"


def test_walkforward_candidate_exposes_average_sharpe_ratio(monkeypatch) -> None:
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
            period_days=kwargs["period_days"],
        )

    monkeypatch.setattr(walkforward_module, "simulate_backtest", fake_simulate_backtest)

    result = service.run(windows=[60, 90], thresholds=[0.32], starting_capital=100_000.0)

    assert result.best_candidate is not None
    assert hasattr(result.best_candidate, "average_sharpe_ratio")
    assert isinstance(result.best_candidate.average_sharpe_ratio, float)
    assert result.best_candidate.average_sharpe_ratio > 0.0


# --- Gate 1/2/3 validation tests ---

def test_run_validation_returns_gate1_result(monkeypatch) -> None:
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
            period_days=kwargs["period_days"],
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
        )

    monkeypatch.setattr(walkforward_module, "simulate_backtest", fake_simulate_backtest)

    result = service.run_validation(universe_preset="balanced_quality", starting_capital=100_000.0)

    assert isinstance(result, ValidationResult)
    assert result.status == "ok"
    assert result.gate1 is not None
    assert len(result.gate1.folds) == 9
    assert result.gate1.positive_period_count >= 0
    assert isinstance(result.gate1.average_sharpe, float)
    assert isinstance(result.gate1.average_trades_per_period, float)


def test_run_validation_gate1_folds_have_correct_structure(monkeypatch) -> None:
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
            period_days=kwargs["period_days"],
            start_date=kwargs.get("start_date"),
            end_date=kwargs.get("end_date"),
        )

    monkeypatch.setattr(walkforward_module, "simulate_backtest", fake_simulate_backtest)

    result = service.run_validation(starting_capital=100_000.0)

    assert result.gate1 is not None
    for i, fold in enumerate(result.gate1.folds):
        assert isinstance(fold, ValidationFold)
        assert fold.fold_index == i + 1
        assert fold.train_start.startswith("2017")
        assert fold.best_threshold in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]


def test_calmar_returns_negative_sentinel_for_no_trades() -> None:
    result = _fake_backtest_result(
        universe_preset="balanced_quality",
        threshold=0.30,
        period_days=100,
    )
    result_no_trades = type(result.__class__.__name__, (), {
        "total_trades": 0, "period_days": 100,
        "ending_equity": 100_000.0, "starting_capital": 100_000.0,
        "max_drawdown_pct": 0.0,
    })()
    assert _calmar(result_no_trades) == -999.0


def test_calmar_is_positive_for_profitable_low_drawdown_result() -> None:
    result = _fake_backtest_result(
        universe_preset="balanced_quality",
        threshold=0.30,
        period_days=252,
    )
    calmar = _calmar(result)
    assert calmar > 0.0
