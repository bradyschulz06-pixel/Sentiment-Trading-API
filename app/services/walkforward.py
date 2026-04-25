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
        candidates: list[WalkForwardCandidateResult] = []

        for preset_name in preset_names:
            normalized_preset = normalize_universe_preset(preset_name)
            universe_meta = UNIVERSE_PRESETS[normalized_preset]
            tradable_symbols = universe_meta["symbols"]
            subset_price_map = {symbol: price_map[symbol] for symbol in tradable_symbols + [benchmark_symbol]}
            subset_earnings_map = {symbol: earnings_map.get(symbol, []) for symbol in tradable_symbols}

            for profile in FACTOR_PROFILES:
                momentum_weight, sentiment_weight, earnings_weight = normalize_factor_weights(*profile["weights"])
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
                            factor_momentum_weight=momentum_weight,
                            factor_sentiment_weight=sentiment_weight,
                            factor_earnings_weight=earnings_weight,
                            commission_per_order=self.settings.backtest_commission_per_order,
                            slippage_bps=self.settings.backtest_slippage_bps,
                            max_hold_days=self.settings.backtest_max_hold_days,
                            min_hold_days=self.settings.backtest_min_hold_days,
                            trailing_stop_pct=self.settings.backtest_trailing_stop_pct,
                            trailing_arm_pct=self.settings.backtest_trailing_arm_pct,
                            take_profit_pct=self.settings.backtest_take_profit_pct,
                        )
                        total_trades += result.total_trades
                        window_results.append(
                            WalkForwardWindowResult(
                                period_days=period_days,
                                total_return_pct=result.total_return_pct,
                                benchmark_return_pct=result.benchmark_return_pct,
                                outperformance_pct=result.outperformance_pct,
                                max_drawdown_pct=result.max_drawdown_pct,
                                total_trades=result.total_trades,
                            )
                        )

                    outperformance_values = [window.outperformance_pct for window in window_results]
                    return_values = [window.total_return_pct for window in window_results]
                    drawdown_values = [abs(window.max_drawdown_pct) for window in window_results]
                    candidate = WalkForwardCandidateResult(
                        label=f"{universe_meta['label']} | {profile['label']} | threshold {threshold:.2f}",
                        universe_preset=normalized_preset,
                        universe_label=universe_meta["label"],
                        factor_profile_name=profile["label"],
                        signal_threshold=threshold,
                        factor_momentum_weight=round(momentum_weight, 4),
                        factor_sentiment_weight=round(sentiment_weight, 4),
                        factor_earnings_weight=round(earnings_weight, 4),
                        stability_score=round(_stability_score(window_results), 4),
                        average_return_pct=round(mean(return_values), 4),
                        average_outperformance_pct=round(mean(outperformance_values), 4),
                        worst_outperformance_pct=round(min(outperformance_values), 4),
                        positive_window_ratio=round(sum(1 for value in return_values if value > 0) / len(return_values), 4),
                        benchmark_win_ratio=round(sum(1 for value in outperformance_values if value > 0) / len(outperformance_values), 4),
                        average_drawdown_pct=round(mean(drawdown_values), 4),
                        outperformance_stddev_pct=round(pstdev(outperformance_values) if len(outperformance_values) > 1 else 0.0, 4),
                        total_trades=total_trades,
                        windows=window_results,
                        notes=[
                            profile["description"],
                            f"Beat {benchmark_symbol} in {sum(1 for value in outperformance_values if value > 0)} of {len(outperformance_values)} replay windows.",
                        ],
                    )
                    candidates.append(candidate)

        candidates.sort(
            key=lambda item: (
                item.stability_score,
                item.benchmark_win_ratio,
                item.average_outperformance_pct,
                -item.average_drawdown_pct,
            ),
            reverse=True,
        )
        best_candidate = candidates[0] if candidates else None

        return WalkForwardResult(
            status="ok",
            summary=f"Compared {len(candidates)} parameter sets across {len(windows)} replay windows using the current exit model.",
            benchmark_symbol=benchmark_symbol,
            starting_capital=round(starting_capital, 2),
            windows_tested=windows,
            thresholds_tested=thresholds,
            candidates=candidates,
            best_candidate=best_candidate,
            notes=[
                "Historical news sentiment stays neutral in replay mode, so the report is most trustworthy for tuning momentum and earnings while showing how much live sentiment reserve each profile carries.",
                "Stability score rewards repeatable outperformance and positive windows, then penalizes drawdown and dispersion across windows.",
                f"Universe presets compared: {', '.join(UNIVERSE_PRESETS[name]['label'] for name in preset_names)}.",
            ],
        )
