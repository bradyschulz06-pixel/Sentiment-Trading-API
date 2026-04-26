from __future__ import annotations

from statistics import mean, pstdev

from app.config import Settings
from app.models import WalkForwardCandidateResult, WalkForwardResult, WalkForwardWindowResult
from app.scoring import normalize_factor_weights
from app.services.backtest import BacktestService, simulate_backtest
from app.universe import UNIVERSE_PRESETS, get_union_for_presets, normalize_universe_preset


FACTOR_PROFILES = (
    {
        "name": "baseline",
        "label": "Baseline Blend",
        "description": "Keeps the existing live mix as the starting point.",
        "weights": (0.45, 0.20, 0.35),
    },
    {
        "name": "momentum_lead",
        "label": "Momentum Lead",
        "description": "Leans harder on trend persistence and uses earnings as confirmation.",
        "weights": (0.55, 0.15, 0.30),
    },
    {
        "name": "earnings_lead",
        "label": "Earnings Lead",
        "description": "Pushes more conviction into surprise strength and transcript tone.",
        "weights": (0.35, 0.15, 0.50),
    },
    {
        "name": "confirmation_blend",
        "label": "Confirmation Blend",
        "description": "Splits the difference to avoid overreacting to any single input.",
        "weights": (0.40, 0.25, 0.35),
    },
)

_STAGE2_TRAILING_ARM_PCTS = [0.00, 0.01, 0.02, 0.03]
_STAGE2_MAX_HOLD_DAYS = [12, 15, 18]


def get_factor_profiles() -> tuple[dict, ...]:
    return FACTOR_PROFILES


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


def _stability_score(windows: list[WalkForwardWindowResult]) -> float:
    outperformance = [window.outperformance_pct for window in windows]
    total_returns = [window.total_return_pct for window in windows]
    drawdowns = [abs(window.max_drawdown_pct) for window in windows]
    benchmark_win_ratio = sum(1 for value in outperformance if value > 0) / len(outperformance)
    positive_window_ratio = sum(1 for value in total_returns if value > 0) / len(total_returns)
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


def _build_window_results(
    *,
    settings: Settings,
    subset_price_map: dict,
    subset_earnings_map: dict,
    benchmark_symbol: str,
    windows: list[int],
    starting_capital: float,
    universe_preset: str,
    signal_threshold: float,
    factor_momentum_weight: float,
    factor_sentiment_weight: float,
    factor_earnings_weight: float,
    trailing_arm_pct: float,
    max_hold_days: int,
) -> tuple[list[WalkForwardWindowResult], int]:
    window_results: list[WalkForwardWindowResult] = []
    total_trades = 0
    for period_days in windows:
        result = simulate_backtest(
            settings=settings,
            price_map=subset_price_map,
            earnings_map=subset_earnings_map,
            benchmark_symbol=benchmark_symbol,
            period_days=period_days,
            starting_capital=starting_capital,
            universe_preset=universe_preset,
            signal_threshold=signal_threshold,
            factor_momentum_weight=factor_momentum_weight,
            factor_sentiment_weight=factor_sentiment_weight,
            factor_earnings_weight=factor_earnings_weight,
            commission_per_order=settings.backtest_commission_per_order,
            slippage_bps=settings.backtest_slippage_bps,
            max_hold_days=max_hold_days,
            min_hold_days=settings.backtest_min_hold_days,
            trailing_stop_pct=settings.backtest_trailing_stop_pct,
            trailing_arm_pct=trailing_arm_pct,
            take_profit_pct=settings.backtest_take_profit_pct,
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
    return window_results, total_trades


def _candidate_from_windows(
    *,
    label: str,
    universe_preset: str,
    universe_label: str,
    factor_profile_name: str,
    signal_threshold: float,
    factor_momentum_weight: float,
    factor_sentiment_weight: float,
    factor_earnings_weight: float,
    trailing_arm_pct: float,
    max_hold_days: int,
    window_results: list[WalkForwardWindowResult],
    total_trades: int,
    notes: list[str],
    benchmark_symbol: str,
) -> WalkForwardCandidateResult:
    outperformance_values = [w.outperformance_pct for w in window_results]
    return_values = [w.total_return_pct for w in window_results]
    drawdown_values = [abs(w.max_drawdown_pct) for w in window_results]
    sharpe_values = [w.sharpe_ratio for w in window_results]
    average_sharpe = round(mean(sharpe_values), 4)
    return WalkForwardCandidateResult(
        label=label,
        universe_preset=universe_preset,
        universe_label=universe_label,
        factor_profile_name=factor_profile_name,
        signal_threshold=signal_threshold,
        factor_momentum_weight=round(factor_momentum_weight, 4),
        factor_sentiment_weight=round(factor_sentiment_weight, 4),
        factor_earnings_weight=round(factor_earnings_weight, 4),
        stability_score=round(_stability_score(window_results), 4),
        average_sharpe_ratio=average_sharpe,
        average_return_pct=round(mean(return_values), 4),
        average_outperformance_pct=round(mean(outperformance_values), 4),
        worst_outperformance_pct=round(min(outperformance_values), 4),
        positive_window_ratio=round(sum(1 for v in return_values if v > 0) / len(return_values), 4),
        benchmark_win_ratio=round(sum(1 for v in outperformance_values if v > 0) / len(outperformance_values), 4),
        average_drawdown_pct=round(mean(drawdown_values), 4),
        outperformance_stddev_pct=round(pstdev(outperformance_values) if len(outperformance_values) > 1 else 0.0, 4),
        total_trades=total_trades,
        trailing_arm_pct=trailing_arm_pct,
        max_hold_days=max_hold_days,
        windows=window_results,
        notes=notes + [
            f"Beat {benchmark_symbol} in {sum(1 for v in outperformance_values if v > 0)} of {len(outperformance_values)} replay windows.",
        ],
    )


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
        windows = sorted({max(60, min(int(item), 252)) for item in (windows or default_walkforward_windows())})
        thresholds = sorted({round(min(max(float(item), 0.10), 0.90), 2) for item in (thresholds or default_walkforward_thresholds(self.settings))})
        preset_names = list(UNIVERSE_PRESETS.keys())
        fetch_days = max(max(windows) + 120, 220)
        symbols = sorted(set(get_union_for_presets(preset_names) + [benchmark_symbol]))
        price_map, earnings_map = self.backtests.load_market_data(symbols, fetch_days=fetch_days)

        # Build per-preset data subsets once; reused in both Stage 1 and Stage 2.
        preset_data: dict[str, tuple[dict, dict]] = {}
        for preset_name in preset_names:
            normalized_preset = normalize_universe_preset(preset_name)
            universe_meta = UNIVERSE_PRESETS[normalized_preset]
            tradable_symbols = universe_meta["symbols"]
            subset_price_map = {symbol: price_map[symbol] for symbol in tradable_symbols + [benchmark_symbol]}
            subset_earnings_map = {symbol: earnings_map.get(symbol, []) for symbol in tradable_symbols}
            preset_data[normalized_preset] = (subset_price_map, subset_earnings_map)

        # ── Stage 1: sweep universe × factor_profile × threshold (36 combinations) ──
        stage1_candidates: list[WalkForwardCandidateResult] = []
        for preset_name in preset_names:
            normalized_preset = normalize_universe_preset(preset_name)
            universe_meta = UNIVERSE_PRESETS[normalized_preset]
            subset_price_map, subset_earnings_map = preset_data[normalized_preset]

            for profile in FACTOR_PROFILES:
                momentum_weight, sentiment_weight, earnings_weight = normalize_factor_weights(*profile["weights"])
                for threshold in thresholds:
                    window_results, total_trades = _build_window_results(
                        settings=self.settings,
                        subset_price_map=subset_price_map,
                        subset_earnings_map=subset_earnings_map,
                        benchmark_symbol=benchmark_symbol,
                        windows=windows,
                        starting_capital=starting_capital,
                        universe_preset=normalized_preset,
                        signal_threshold=threshold,
                        factor_momentum_weight=momentum_weight,
                        factor_sentiment_weight=sentiment_weight,
                        factor_earnings_weight=earnings_weight,
                        trailing_arm_pct=self.settings.backtest_trailing_arm_pct,
                        max_hold_days=self.settings.backtest_max_hold_days,
                    )
                    candidate = _candidate_from_windows(
                        label=f"{universe_meta['label']} | {profile['label']} | threshold {threshold:.2f}",
                        universe_preset=normalized_preset,
                        universe_label=universe_meta["label"],
                        factor_profile_name=profile["label"],
                        signal_threshold=threshold,
                        factor_momentum_weight=momentum_weight,
                        factor_sentiment_weight=sentiment_weight,
                        factor_earnings_weight=earnings_weight,
                        trailing_arm_pct=self.settings.backtest_trailing_arm_pct,
                        max_hold_days=self.settings.backtest_max_hold_days,
                        window_results=window_results,
                        total_trades=total_trades,
                        notes=[profile["description"]],
                        benchmark_symbol=benchmark_symbol,
                    )
                    stage1_candidates.append(candidate)

        stage1_candidates.sort(
            key=lambda item: (
                item.average_sharpe_ratio,
                item.stability_score,
                item.benchmark_win_ratio,
                item.average_outperformance_pct,
                -item.average_drawdown_pct,
            ),
            reverse=True,
        )

        # ── Stage 2: sweep exit parameters for the top-8 Stage-1 candidates ──
        # For each top-8 candidate: 4 trailing_arm_pct × 3 max_hold_days = 12 extra backtests per window.
        stage2_candidates: list[WalkForwardCandidateResult] = []
        for s1 in stage1_candidates[:8]:
            subset_price_map, subset_earnings_map = preset_data[s1.universe_preset]
            for arm_pct in _STAGE2_TRAILING_ARM_PCTS:
                for hold_days in _STAGE2_MAX_HOLD_DAYS:
                    window_results, total_trades = _build_window_results(
                        settings=self.settings,
                        subset_price_map=subset_price_map,
                        subset_earnings_map=subset_earnings_map,
                        benchmark_symbol=benchmark_symbol,
                        windows=windows,
                        starting_capital=starting_capital,
                        universe_preset=s1.universe_preset,
                        signal_threshold=s1.signal_threshold,
                        factor_momentum_weight=s1.factor_momentum_weight,
                        factor_sentiment_weight=s1.factor_sentiment_weight,
                        factor_earnings_weight=s1.factor_earnings_weight,
                        trailing_arm_pct=arm_pct,
                        max_hold_days=hold_days,
                    )
                    candidate = _candidate_from_windows(
                        label=f"{s1.label} | arm {arm_pct:.2f} | hold {hold_days}d",
                        universe_preset=s1.universe_preset,
                        universe_label=s1.universe_label,
                        factor_profile_name=s1.factor_profile_name,
                        signal_threshold=s1.signal_threshold,
                        factor_momentum_weight=s1.factor_momentum_weight,
                        factor_sentiment_weight=s1.factor_sentiment_weight,
                        factor_earnings_weight=s1.factor_earnings_weight,
                        trailing_arm_pct=arm_pct,
                        max_hold_days=hold_days,
                        window_results=window_results,
                        total_trades=total_trades,
                        notes=[
                            f"Stage-2 exit sweep on top-8 Stage-1 candidate: trailing_arm_pct={arm_pct:.2f}, max_hold_days={hold_days}.",
                        ],
                        benchmark_symbol=benchmark_symbol,
                    )
                    stage2_candidates.append(candidate)

        all_candidates = stage1_candidates + stage2_candidates
        all_candidates.sort(
            key=lambda item: (
                item.average_sharpe_ratio,
                item.stability_score,
                item.benchmark_win_ratio,
                item.average_outperformance_pct,
                -item.average_drawdown_pct,
            ),
            reverse=True,
        )
        best_candidate = all_candidates[0] if all_candidates else None
        total_combinations = len(stage1_candidates) + len(stage2_candidates)

        return WalkForwardResult(
            status="ok",
            summary=(
                f"Stage 1: compared {len(stage1_candidates)} parameter sets; "
                f"Stage 2: swept exit params on top-8 ({len(stage2_candidates)} extra combinations). "
                f"{total_combinations} total across {len(windows)} replay windows."
            ),
            benchmark_symbol=benchmark_symbol,
            starting_capital=round(starting_capital, 2),
            windows_tested=windows,
            thresholds_tested=thresholds,
            candidates=all_candidates,
            best_candidate=best_candidate,
            notes=[
                "Historical news sentiment stays neutral in replay mode, so the report is most trustworthy for tuning momentum and earnings while showing how much live sentiment reserve each profile carries.",
                "Stability score rewards repeatable outperformance and positive windows, then penalizes drawdown and dispersion across windows.",
                f"Universe presets compared: {', '.join(UNIVERSE_PRESETS[name]['label'] for name in preset_names)}.",
                "Stage-2 sweeps trailing_arm_pct in [0.00, 0.01, 0.02, 0.03] and max_hold_days in [12, 15, 18] for the top-8 Stage-1 candidates.",
            ],
        )
