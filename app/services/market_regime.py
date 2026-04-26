from __future__ import annotations

from dataclasses import dataclass
import json
import math

from app.config import Settings
from app.models import PriceBar, SignalScore
from app.scoring import compute_momentum_score
from app.regime_adaptive import (
    MarketRegime as AdaptiveMarketRegime,
    detect_market_regime,
    get_regime_parameters,
    apply_regime_adjustments
)


REGIME_WARNING_PREFIX = "MARKET_REGIME::"


@dataclass(slots=True)
class MarketRegime:
    benchmark_symbol: str
    label: str
    summary: str
    benchmark_price: float
    sma20: float
    sma50: float
    ret21: float
    breadth_above_20: float
    breadth_above_50: float
    threshold_boost: float
    max_positions_multiplier: float
    allow_new_longs: bool

    def effective_signal_threshold(self, base_threshold: float) -> float:
        return min(0.95, base_threshold + self.threshold_boost)

    def effective_max_positions(self, configured_max_positions: int) -> int:
        scaled = math.floor(configured_max_positions * self.max_positions_multiplier)
        if configured_max_positions <= 0:
            return 0
        if self.max_positions_multiplier > 0 and scaled == 0:
            scaled = 1
        return max(0, min(configured_max_positions, scaled))

    def to_warning(self) -> str:
        payload = {
            "benchmark_symbol": self.benchmark_symbol,
            "label": self.label,
            "summary": self.summary,
            "benchmark_price": round(self.benchmark_price, 2),
            "sma20": round(self.sma20, 2),
            "sma50": round(self.sma50, 2),
            "ret21": round(self.ret21, 4),
            "breadth_above_20": round(self.breadth_above_20, 4),
            "breadth_above_50": round(self.breadth_above_50, 4),
            "threshold_boost": round(self.threshold_boost, 4),
            "max_positions_multiplier": round(self.max_positions_multiplier, 4),
            "allow_new_longs": self.allow_new_longs,
        }
        return REGIME_WARNING_PREFIX + json.dumps(payload)


def evaluate_market_regime(
    settings: Settings,
    benchmark_symbol: str,
    benchmark_bars: list[PriceBar],
    universe_bars: dict[str, list[PriceBar]],
) -> MarketRegime:
    if not settings.market_regime_filter_enabled:
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            label="inactive",
            summary="Market regime filter is turned off.",
            benchmark_price=0.0,
            sma20=0.0,
            sma50=0.0,
            ret21=0.0,
            breadth_above_20=0.0,
            breadth_above_50=0.0,
            threshold_boost=0.0,
            max_positions_multiplier=1.0,
            allow_new_longs=True,
        )

    if len(benchmark_bars) < 50:
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            label="risk_off",
            summary="Benchmark history too short to classify — defaulting to risk_off.",
            benchmark_price=benchmark_bars[-1].close if benchmark_bars else 0.0,
            sma20=0.0,
            sma50=0.0,
            ret21=0.0,
            breadth_above_20=0.0,
            breadth_above_50=0.0,
            threshold_boost=0.0,
            max_positions_multiplier=0.0,
            allow_new_longs=False,
        )

    _, benchmark_metrics = compute_momentum_score(benchmark_bars)
    benchmark_price = benchmark_metrics["current"]
    sma20 = benchmark_metrics["sma20"]
    sma50 = benchmark_metrics["sma50"]
    ret21 = benchmark_metrics["ret21"]

    if benchmark_price > sma50:
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            label="risk_on",
            summary=f"{benchmark_symbol} is above its 50-day SMA — new long entries are allowed.",
            benchmark_price=benchmark_price,
            sma20=sma20,
            sma50=sma50,
            ret21=ret21,
            breadth_above_20=0.0,
            breadth_above_50=0.0,
            threshold_boost=0.0,
            max_positions_multiplier=1.0,
            allow_new_longs=True,
        )

    return MarketRegime(
        benchmark_symbol=benchmark_symbol,
        label="risk_off",
        summary=f"{benchmark_symbol} is at or below its 50-day SMA — new long entries are blocked.",
        benchmark_price=benchmark_price,
        sma20=sma20,
        sma50=sma50,
        ret21=ret21,
        breadth_above_20=0.0,
        breadth_above_50=0.0,
        threshold_boost=0.0,
        max_positions_multiplier=0.0,
        allow_new_longs=False,
    )


def apply_market_regime_to_signal(signal: SignalScore, regime: MarketRegime, base_threshold: float) -> SignalScore:
    if signal.decision == "buy" and not regime.allow_new_longs:
        signal.decision = "watch"
        signal.rationale = f"{signal.rationale}. Market regime is risk-off, so new longs are blocked."
    return signal


def parse_regime_warning(raw_warning: str) -> dict | None:
    if not raw_warning.startswith(REGIME_WARNING_PREFIX):
        return None
    try:
        return json.loads(raw_warning[len(REGIME_WARNING_PREFIX):])
    except json.JSONDecodeError:
        return None


def evaluate_adaptive_regime(
    settings: Settings,
    benchmark_symbol: str,
    benchmark_bars: list[PriceBar],
    universe_bars: dict[str, list[PriceBar]],
    vix_level: float = 20.0,
    recent_drawdown: float = 0.0
) -> tuple[MarketRegime, AdaptiveMarketRegime]:
    """
    Evaluate both traditional and adaptive market regimes.

    Returns the traditional MarketRegime for backward compatibility
    and the new AdaptiveMarketRegime for enhanced parameter adjustment.
    """
    # Get traditional regime
    traditional_regime = evaluate_market_regime(
        settings, benchmark_symbol, benchmark_bars, universe_bars
    )

    # Calculate adaptive regime inputs
    _, benchmark_metrics = compute_momentum_score(benchmark_bars)
    spy_trend = benchmark_metrics.get("ret21", 0.0)

    # Calculate volatility (simplified)
    if len(benchmark_bars) >= 21:
        returns = [
            (benchmark_bars[i].close - benchmark_bars[i-1].close) / benchmark_bars[i-1].close
            for i in range(1, min(22, len(benchmark_bars)))
        ]
        volatility = (sum(r**2 for r in returns) / len(returns)) ** 0.5 * (252**0.5)
    else:
        volatility = 0.20

    # Calculate breadth
    above_20 = sum(
        1 for bars in universe_bars.values()
        if len(bars) >= 20 and bars[-1].close > bars[-20].close
    )
    above_50 = sum(
        1 for bars in universe_bars.values()
        if len(bars) >= 50 and bars[-1].close > bars[-50].close
    )
    breadth = above_20 / max(1, len(universe_bars))

    # Detect adaptive regime
    adaptive_regime = detect_market_regime(
        spy_trend=spy_trend,
        spy_volatility=volatility,
        vix_level=vix_level,
        breadth=breadth,
        recent_drawdown=recent_drawdown
    )

    return traditional_regime, adaptive_regime


def apply_adaptive_parameters(
    settings: Settings,
    adaptive_regime: AdaptiveMarketRegime
) -> dict:
    """
    Apply adaptive regime parameters to base settings.

    Returns a dictionary of adjusted parameters.
    """
    regime_params = get_regime_parameters(adaptive_regime)

    # Apply regime adjustments
    adjusted_threshold, adjusted_max_positions, adjusted_stop_loss, adjusted_target = apply_regime_adjustments(
        signal_threshold=settings.signal_threshold,
        max_positions=settings.max_positions,
        stop_loss_pct=settings.stop_loss_pct,
        target_multiplier=1.0,
        regime_params=regime_params
    )

    return {
        "signal_threshold": adjusted_threshold,
        "max_positions": adjusted_max_positions,
        "stop_loss_pct": adjusted_stop_loss,
        "target_multiplier": adjusted_target,
        "momentum_weight": regime_params.momentum_weight,
        "earnings_weight": regime_params.earnings_weight,
        "trailing_stop_enabled": regime_params.trailing_stop_enabled,
        "trailing_stop_pct": regime_params.trailing_stop_pct,
        "regime_description": regime_params.description,
        "position_size_multiplier": regime_params.position_size_multiplier
    }
