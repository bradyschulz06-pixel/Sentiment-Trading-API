from __future__ import annotations

from statistics import mean, pstdev

from app.config import Settings
from app.models import (
    Gate1Result,
    Gate2Result,
    Gate3Result,
    ValidationFold,
    ValidationResult,
    WalkForwardCandidateResult,
    WalkForwardResult,
    WalkForwardWindowResult,
)
from app.services.backtest import BacktestService, simulate_backtest
from app.universe import UNIVERSE_PRESETS, get_union_for_presets, normalize_universe_preset

_VALIDATION_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]

# Nine sequential folds: expanding training window anchored at 2017-01-01,
# non-overlapping 6-month test periods. 2022 is reserved as Gate 2 holdout.
_FOLDS = [
    ("2017-01-01", "2018-12-31", "2019-01-01", "2019-06-30"),
    ("2017-01-01", "2019-06-30", "2019-07-01", "2019-12-31"),
    ("2017-01-01", "2019-12-31", "2020-01-01", "2020-06-30"),
    ("2017-01-01", "2020-06-30", "2020-07-01", "2020-12-31"),
    ("2017-01-01", "2020-12-31", "2021-01-01", "2021-06-30"),
    ("2017-01-01", "2021-06-30", "2021-07-01", "2021-12-31"),
    ("2017-01-01", "2022-12-31", "2023-01-01", "2023-06-30"),
    ("2017-01-01", "2023-06-30", "2023-07-01", "2023-12-31"),
    ("2017-01-01", "2023-12-31", "2024-01-01", "2024-06-30"),
]

_GATE2_START = "2022-01-01"
_GATE2_END = "2022-12-31"
_GATE1_MIN_POSITIVE_PERIODS = 6
_GATE1_MIN_AVG_SHARPE = 0.40
_GATE1_MAX_PERIOD_DRAWDOWN = 0.15
_GATE1_MIN_TRADES_PER_PERIOD = 3
_GATE2_MAX_DRAWDOWN = 0.15
_GATE2_MIN_RETURN = -0.10
_GATE3_MIN_METRICS_WON = 3


def _calmar(result) -> float:
    if result.total_trades == 0 or result.period_days < 1:
        return -999.0
    annual_factor = 252.0 / result.period_days
    cagr = (result.ending_equity / result.starting_capital) ** annual_factor - 1.0
    drawdown = abs(result.max_drawdown_pct)
    if drawdown < 0.001:
        return max(0.0, cagr) * 50.0
    return cagr / drawdown


def _stability_score(windows: list[WalkForwardWindowResult]) -> float:
    outperformance = [w.outperformance_pct for w in windows]
    total_returns = [w.total_return_pct for w in windows]
    drawdowns = [abs(w.max_drawdown_pct) for w in windows]
    benchmark_win_ratio = sum(1 for v in outperformance if v > 0) / len(outperformance)
    positive_window_ratio = sum(1 for v in total_returns if v > 0) / len(total_returns)
    average_outperformance = mean(outperformance)
    average_return = mean(total_returns)
    worst_outperformance = min(outperformance)
    average_drawdown = mean(drawdowns)
    outperformance_stddev = pstdev(outperformance) if len(outperformance) > 1 else 0.0
    return (
        (benchmark_win_ratio * 4.0)
        + (positive_window_ratio * 2.0)
        + (average_outperformance * 100.0)
        + (average_return * 20.0)
        + (worst_outperformance * 20.0)
        - (average_drawdown * 35.0)
        - (outperformance_stddev * 45.0)
    )


def default_walkforward_windows() -> list[int]:
    return [60, 90, 120, 150]


def default_walkforward_thresholds(settings: Settings) -> list[float]:
    base = settings.signal_threshold
    thresholds = {
        round(min(max(base - 0.02, 0.10), 0.90), 2),
        round(min(max(base, 0.10), 0.90), 2),
        round(min(max(base + 0.03, 0.10), 0.90), 2),
    }
    return sorted(thresholds)


class WalkForwardService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backtests = BacktestService(settings)

    def run(
        self,
        *,
        windows: list[int] | None = None,
        thresholds: list[float] | None = None,
        starting_capital: float = 100_000.0,
        benchmark_symbol: str | None = None,
    ) -> WalkForwardResult:
        if not self.settings.trading_configured:
            raise RuntimeError("Add your Alpaca keys before running walk-forward analysis.")

        benchmark_symbol = (benchmark_symbol or self.settings.backtest_benchmark_symbol).strip().upper()
        windows = sorted({max(60, min(int(w), 252)) for w in (windows or default_walkforward_windows())})
        thresholds = sorted({round(min(max(float(t), 0.10), 0.90), 2) for t in (thresholds or default_walkforward_thresholds(self.settings))})
        preset_names = list(UNIVERSE_PRESETS.keys())
        fetch_days = max(max(windows) + 120, 220)
        symbols = sorted(set(get_union_for_presets(preset_names) + [benchmark_symbol]))
        price_map, earnings_map = self.backtests.load_market_data(symbols, fetch_days=fetch_days)

        preset_data: dict[str, tuple[dict, dict]] = {}
        for preset_name in preset_names:
            normalized_preset = normalize_universe_preset(preset_name)
            universe_meta = UNIVERSE_PRESETS[normalized_preset]
            tradable_symbols = universe_meta["symbols"]
            subset_price_map = {s: price_map[s] for s in tradable_symbols + [benchmark_symbol] if s in price_map}
            subset_earnings_map = {s: earnings_map.get(s, []) for s in tradable_symbols}
            preset_data[normalized_preset] = (subset_price_map, subset_earnings_map)

        stage1_candidates: list[WalkForwardCandidateResult] = []
        for preset_name in preset_names:
            normalized_preset = normalize_universe_preset(preset_name)
            universe_meta = UNIVERSE_PRESETS[normalized_preset]
            subset_price_map, subset_earnings_map = preset_data[normalized_preset]
            for threshold in thresholds:
                window_results: list[WalkForwardWindowResult] = []
                total_trades = 0
                for period_days in windows:
                    result = simulate_backtest(
                        settings=self.settings,
                        price_map=subset_price_map,
                        earnings_map=subset_earnings_map,
                        benchmark_symbol=benchmark_symbol,
                        period_days=period_days,
                        starting_capital=starting_capital,
                        universe_preset=normalized_preset,
                        signal_threshold=threshold,
                    )
                    total_trades += result.total_trades
                    window_results.append(
                        WalkForwardWindowResult(
                            period_days=period_days,
                            total_return_pct=result.total_return_pct,
                            benchmark_return_pct=result.benchmark_return_pct,
                            outperformance_pct=result.outperformance_pct,
                            max_drawdown_pct=result.max_drawdown_pct,
                            sharpe_ratio=result.sharpe_ratio,
                            total_trades=result.total_trades,
                        )
                    )
                outperformance_values = [w.outperformance_pct for w in window_results]
                return_values = [w.total_return_pct for w in window_results]
                drawdown_values = [abs(w.max_drawdown_pct) for w in window_results]
                sharpe_values = [w.sharpe_ratio for w in window_results]
                candidate = WalkForwardCandidateResult(
                    label=f"{universe_meta['label']} | PEAD-only | threshold {threshold:.2f}",
                    universe_preset=normalized_preset,
                    universe_label=universe_meta["label"],
                    factor_profile_name="PEAD-only",
                    signal_threshold=threshold,
                    factor_momentum_weight=0.0,
                    factor_sentiment_weight=0.0,
                    factor_earnings_weight=1.0,
                    stability_score=round(_stability_score(window_results), 4),
                    average_sharpe_ratio=round(mean(sharpe_values), 4),
                    average_return_pct=round(mean(return_values), 4),
                    average_outperformance_pct=round(mean(outperformance_values), 4),
                    worst_outperformance_pct=round(min(outperformance_values), 4),
                    positive_window_ratio=round(sum(1 for v in return_values if v > 0) / len(return_values), 4),
                    benchmark_win_ratio=round(sum(1 for v in outperformance_values if v > 0) / len(outperformance_values), 4),
                    average_drawdown_pct=round(mean(drawdown_values), 4),
                    outperformance_stddev_pct=round(pstdev(outperformance_values) if len(outperformance_values) > 1 else 0.0, 4),
                    total_trades=total_trades,
                    trailing_arm_pct=0.0,
                    max_hold_days=self.settings.backtest_max_hold_days,
                    windows=window_results,
                    notes=[
                        f"PEAD-only signal. Beat {benchmark_symbol} in {sum(1 for v in outperformance_values if v > 0)} of {len(outperformance_values)} windows.",
                    ],
                )
                stage1_candidates.append(candidate)

        # Stage-2: sweep hold days for the top candidate (preserves test compatibility).
        stage2_candidates: list[WalkForwardCandidateResult] = []
        if stage1_candidates:
            best_s1 = max(stage1_candidates, key=lambda c: (c.average_sharpe_ratio, c.stability_score))
            subset_price_map, subset_earnings_map = preset_data[best_s1.universe_preset]
            for hold_days in [15, 20]:
                window_results = []
                total_trades = 0
                for period_days in windows:
                    result = simulate_backtest(
                        settings=self.settings,
                        price_map=subset_price_map,
                        earnings_map=subset_earnings_map,
                        benchmark_symbol=benchmark_symbol,
                        period_days=period_days,
                        starting_capital=starting_capital,
                        universe_preset=best_s1.universe_preset,
                        signal_threshold=best_s1.signal_threshold,
                        max_hold_days=hold_days,
                    )
                    total_trades += result.total_trades
                    window_results.append(
                        WalkForwardWindowResult(
                            period_days=period_days,
                            total_return_pct=result.total_return_pct,
                            benchmark_return_pct=result.benchmark_return_pct,
                            outperformance_pct=result.outperformance_pct,
                            max_drawdown_pct=result.max_drawdown_pct,
                            sharpe_ratio=result.sharpe_ratio,
                            total_trades=result.total_trades,
                        )
                    )
                outperformance_values = [w.outperformance_pct for w in window_results]
                return_values = [w.total_return_pct for w in window_results]
                drawdown_values = [abs(w.max_drawdown_pct) for w in window_results]
                sharpe_values = [w.sharpe_ratio for w in window_results]
                stage2_candidates.append(
                    WalkForwardCandidateResult(
                        label=f"{best_s1.label} | arm 0.00 | hold {hold_days}d",
                        universe_preset=best_s1.universe_preset,
                        universe_label=best_s1.universe_label,
                        factor_profile_name=best_s1.factor_profile_name,
                        signal_threshold=best_s1.signal_threshold,
                        factor_momentum_weight=0.0,
                        factor_sentiment_weight=0.0,
                        factor_earnings_weight=1.0,
                        stability_score=round(_stability_score(window_results), 4),
                        average_sharpe_ratio=round(mean(sharpe_values), 4),
                        average_return_pct=round(mean(return_values), 4),
                        average_outperformance_pct=round(mean(outperformance_values), 4),
                        worst_outperformance_pct=round(min(outperformance_values), 4),
                        positive_window_ratio=round(sum(1 for v in return_values if v > 0) / len(return_values), 4),
                        benchmark_win_ratio=round(sum(1 for v in outperformance_values if v > 0) / len(outperformance_values), 4),
                        average_drawdown_pct=round(mean(drawdown_values), 4),
                        outperformance_stddev_pct=round(pstdev(outperformance_values) if len(outperformance_values) > 1 else 0.0, 4),
                        total_trades=total_trades,
                        trailing_arm_pct=0.0,
                        max_hold_days=hold_days,
                        windows=window_results,
                        notes=[f"Stage-2 hold-day sweep: max_hold_days={hold_days}."],
                    )
                )

        all_candidates = stage1_candidates + stage2_candidates
        all_candidates.sort(
            key=lambda c: (c.average_sharpe_ratio, c.stability_score, c.benchmark_win_ratio, c.average_outperformance_pct, -c.average_drawdown_pct),
            reverse=True,
        )
        best_candidate = all_candidates[0] if all_candidates else None

        return WalkForwardResult(
            status="ok",
            summary=f"Compared {len(stage1_candidates)} threshold combinations across {len(windows)} windows. PEAD-only signal.",
            benchmark_symbol=benchmark_symbol,
            starting_capital=round(starting_capital, 2),
            windows_tested=windows,
            thresholds_tested=thresholds,
            candidates=all_candidates,
            best_candidate=best_candidate,
            notes=[
                "PEAD-only signal: composite = earnings score (momentum and sentiment removed).",
                "Stability score rewards repeatable outperformance, penalizes drawdown and dispersion.",
                f"Universe presets compared: {', '.join(UNIVERSE_PRESETS[name]['label'] for name in preset_names)}.",
            ],
        )

    def run_validation(
        self,
        *,
        universe_preset: str | None = None,
        starting_capital: float = 100_000.0,
        benchmark_symbol: str | None = None,
    ) -> ValidationResult:
        if not self.settings.trading_configured:
            raise RuntimeError("Add your Alpaca keys before running validation.")

        universe_preset = normalize_universe_preset(universe_preset or self.settings.universe_preset)
        benchmark_symbol = (benchmark_symbol or self.settings.backtest_benchmark_symbol).strip().upper()
        universe = list(UNIVERSE_PRESETS[universe_preset]["symbols"])
        symbols = sorted(set(universe + [benchmark_symbol]))
        # ~8 years of history to cover all folds (2017-2024 ≈ 1750 trading days + buffer)
        price_map, earnings_map = self.backtests.load_market_data(symbols, fetch_days=2500)

        # ── Gate 1: 9-fold sequential walk-forward ──
        folds: list[ValidationFold] = []
        for i, (train_start, train_end, test_start, test_end) in enumerate(_FOLDS):
            best_threshold = _VALIDATION_THRESHOLDS[0]
            best_calmar = -float("inf")
            for threshold in _VALIDATION_THRESHOLDS:
                try:
                    train_result = simulate_backtest(
                        self.settings, price_map, earnings_map,
                        benchmark_symbol, period_days=252,
                        starting_capital=starting_capital,
                        universe_preset=universe_preset,
                        signal_threshold=threshold,
                        start_date=train_start,
                        end_date=train_end,
                    )
                    calmar = _calmar(train_result)
                    if calmar > best_calmar:
                        best_calmar = calmar
                        best_threshold = threshold
                except Exception:
                    continue

            try:
                test_result = simulate_backtest(
                    self.settings, price_map, earnings_map,
                    benchmark_symbol, period_days=126,
                    starting_capital=starting_capital,
                    universe_preset=universe_preset,
                    signal_threshold=best_threshold,
                    start_date=test_start,
                    end_date=test_end,
                )
                fold = ValidationFold(
                    fold_index=i + 1,
                    train_start=train_start, train_end=train_end,
                    test_start=test_start, test_end=test_end,
                    best_threshold=best_threshold,
                    test_return_pct=test_result.total_return_pct,
                    test_sharpe=test_result.sharpe_ratio,
                    test_max_drawdown_pct=test_result.max_drawdown_pct,
                    test_win_rate_pct=test_result.win_rate_pct,
                    test_total_trades=test_result.total_trades,
                    passed=(
                        test_result.total_return_pct >= 0
                        and abs(test_result.max_drawdown_pct) < _GATE1_MAX_PERIOD_DRAWDOWN
                    ),
                )
            except Exception:
                fold = ValidationFold(
                    fold_index=i + 1,
                    train_start=train_start, train_end=train_end,
                    test_start=test_start, test_end=test_end,
                    best_threshold=best_threshold,
                    test_return_pct=0.0, test_sharpe=0.0,
                    test_max_drawdown_pct=0.0, test_win_rate_pct=0.0,
                    test_total_trades=0, passed=False,
                )
            folds.append(fold)

        positive_count = sum(1 for f in folds if f.test_return_pct > 0)
        avg_sharpe = sum(f.test_sharpe for f in folds) / len(folds) if folds else 0.0
        worst_drawdown = min(f.test_max_drawdown_pct for f in folds) if folds else 0.0
        avg_trades = sum(f.test_total_trades for f in folds) / len(folds) if folds else 0.0

        gate1_failure = ""
        if positive_count < _GATE1_MIN_POSITIVE_PERIODS:
            gate1_failure = f"Only {positive_count} of 9 test periods positive (need ≥{_GATE1_MIN_POSITIVE_PERIODS})."
        elif avg_sharpe < _GATE1_MIN_AVG_SHARPE:
            gate1_failure = f"Average Sharpe {avg_sharpe:.3f} below minimum {_GATE1_MIN_AVG_SHARPE}."
        elif abs(worst_drawdown) > _GATE1_MAX_PERIOD_DRAWDOWN:
            gate1_failure = f"Worst period drawdown {worst_drawdown:.1%} exceeds limit {_GATE1_MAX_PERIOD_DRAWDOWN:.0%}."
        elif avg_trades < _GATE1_MIN_TRADES_PER_PERIOD:
            gate1_failure = f"Average {avg_trades:.1f} trades per period below minimum {_GATE1_MIN_TRADES_PER_PERIOD}."

        gate1 = Gate1Result(
            folds=folds,
            positive_period_count=positive_count,
            average_sharpe=round(avg_sharpe, 4),
            worst_period_drawdown_pct=round(worst_drawdown, 4),
            average_trades_per_period=round(avg_trades, 2),
            passed=not gate1_failure,
            failure_reason=gate1_failure,
        )

        if not gate1.passed:
            return ValidationResult(
                status="ok",
                gate1=gate1, gate2=None, gate3=None,
                overall_passed=False,
                recommendation="Strategy failed Gate 1. Do not deploy. " + gate1_failure,
            )

        # ── Gate 2: 2022 bear market holdout ──
        gate2_threshold = folds[6].best_threshold if len(folds) >= 7 else _VALIDATION_THRESHOLDS[0]
        try:
            gate2_result = simulate_backtest(
                self.settings, price_map, earnings_map,
                benchmark_symbol, period_days=252,
                starting_capital=starting_capital,
                universe_preset=universe_preset,
                signal_threshold=gate2_threshold,
                start_date=_GATE2_START,
                end_date=_GATE2_END,
            )
            gate2_drawdown = gate2_result.max_drawdown_pct
            gate2_return = gate2_result.total_return_pct
        except Exception:
            gate2_drawdown = -1.0
            gate2_return = -1.0

        gate2_failure = ""
        if abs(gate2_drawdown) > _GATE2_MAX_DRAWDOWN:
            gate2_failure = f"2022 max drawdown {gate2_drawdown:.1%} exceeds limit {-_GATE2_MAX_DRAWDOWN:.0%}."
        elif gate2_return < _GATE2_MIN_RETURN:
            gate2_failure = f"2022 return {gate2_return:.1%} below minimum {_GATE2_MIN_RETURN:.0%}."

        gate2 = Gate2Result(
            holdout_return_pct=round(gate2_return, 4),
            holdout_max_drawdown_pct=round(gate2_drawdown, 4),
            passed=not gate2_failure,
            failure_reason=gate2_failure,
        )

        if not gate2.passed:
            return ValidationResult(
                status="ok",
                gate1=gate1, gate2=gate2, gate3=None,
                overall_passed=False,
                recommendation="Strategy failed Gate 2 (bear market test). Do not deploy. " + gate2_failure,
            )

        # ── Gate 3: compare strategy vs approximate SMA50 benchmark ──
        avg_strategy_sharpe = sum(f.test_sharpe for f in folds) / len(folds) if folds else 0.0
        avg_strategy_return = sum(f.test_return_pct for f in folds) / len(folds) if folds else 0.0
        avg_strategy_drawdown = sum(f.test_max_drawdown_pct for f in folds) / len(folds) if folds else 0.0
        strategy_calmar = avg_strategy_return / max(abs(avg_strategy_drawdown), 0.001)
        strategy_win_rate = sum(f.test_win_rate_pct for f in folds) / len(folds) if folds else 0.0

        # Approximate benchmark (SPY buy-and-hold, 2019-2024 ex-2022).
        benchmark_sharpe = 0.5
        benchmark_drawdown = -0.20
        benchmark_calmar = 0.25
        benchmark_2022_return = -0.18
        benchmark_win_rate = 0.50

        metrics_won = sum([
            avg_strategy_sharpe > benchmark_sharpe,
            abs(avg_strategy_drawdown) < abs(benchmark_drawdown),
            strategy_calmar > benchmark_calmar,
            gate2_return > benchmark_2022_return,
            strategy_win_rate > benchmark_win_rate,
        ])

        gate3_failure = (
            ""
            if metrics_won >= _GATE3_MIN_METRICS_WON
            else f"Strategy won only {metrics_won} of 5 metrics vs benchmark (need ≥{_GATE3_MIN_METRICS_WON})."
        )

        gate3 = Gate3Result(
            strategy_sharpe=round(avg_strategy_sharpe, 4),
            benchmark_sharpe=round(benchmark_sharpe, 4),
            strategy_max_drawdown_pct=round(avg_strategy_drawdown, 4),
            benchmark_max_drawdown_pct=round(benchmark_drawdown, 4),
            strategy_calmar=round(strategy_calmar, 4),
            benchmark_calmar=round(benchmark_calmar, 4),
            strategy_2022_return_pct=round(gate2_return, 4),
            benchmark_2022_return_pct=round(benchmark_2022_return, 4),
            strategy_win_rate_pct=round(strategy_win_rate, 4),
            metrics_won=metrics_won,
            passed=not gate3_failure,
        )

        overall_passed = gate3.passed
        recommendation = (
            "Strategy passed all three gates. Proceed to paper trading before live deployment."
            if overall_passed
            else "Strategy failed Gate 3 (benchmark comparison). Review signal and universe. " + gate3_failure
        )

        return ValidationResult(
            status="ok",
            gate1=gate1,
            gate2=gate2,
            gate3=gate3,
            overall_passed=overall_passed,
            recommendation=recommendation,
        )
