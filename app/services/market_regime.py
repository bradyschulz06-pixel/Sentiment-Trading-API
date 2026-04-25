from __future__ import annotations

from dataclasses import dataclass
import json
import math

from app.config import Settings
from app.models import PriceBar, SignalScore
from app.scoring import compute_momentum_score


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


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _realized_vol(bars: list[PriceBar], days: int = 21) -> float:
    """Annualized realized volatility from daily close returns over `days` bars."""
    closes = [bar.close for bar in bars]
    if len(closes) < days + 1:
        return 0.15
    rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - days, len(closes))]
    mean = sum(rets) / len(rets)
    variance = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    return max(0.01, math.sqrt(variance * 252))


_BREADTH_MIN_SAMPLE = 5


def _breadth_metrics(universe_bars: dict[str, list[PriceBar]]) -> tuple[float, float]:
    above_20 = 0
    above_50 = 0
    sample_size = 0
    for bars in universe_bars.values():
        if len(bars) < 50:
            continue
        closes = [bar.close for bar in bars]
        current = closes[-1]
        sma20 = _average(closes[-20:])
        sma50 = _average(closes[-50:])
        sample_size += 1
        if current > sma20:
            above_20 += 1
        if current > sma50:
            above_50 += 1
    if sample_size < _BREADTH_MIN_SAMPLE:
        # Too few symbols to calculate meaningful breadth; return neutral so
        # the regime falls through to the cautious/risk-off path naturally.
        return 0.5, 0.5
    return above_20 / sample_size, above_50 / sample_size


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
            summary="Market regime filter is turned off, so the model is trading without an index and breadth gate.",
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
            label="cautious",
            summary="Benchmark history is too short to classify the tape cleanly, so the model is using a cautious stance.",
            benchmark_price=benchmark_bars[-1].close if benchmark_bars else 0.0,
            sma20=0.0,
            sma50=0.0,
            ret21=0.0,
            breadth_above_20=0.5,
            breadth_above_50=0.5,
            threshold_boost=settings.market_regime_cautious_threshold_boost,
            max_positions_multiplier=settings.market_regime_cautious_max_positions_multiplier,
            allow_new_longs=True,
        )

    _, benchmark_metrics = compute_momentum_score(benchmark_bars)
    benchmark_price = benchmark_metrics["current"]
    sma20 = benchmark_metrics["sma20"]
    sma50 = benchmark_metrics["sma50"]
    ret21 = benchmark_metrics["ret21"]
    breadth_above_20, breadth_above_50 = _breadth_metrics(universe_bars)

    realized_vol = _realized_vol(benchmark_bars)
    vol_spike = realized_vol >= 0.40      # panic-level volatility: always cautious regardless of trend
    vol_elevated = realized_vol >= 0.25   # elevated vol: downgrade supportive → cautious
    vol_note = f" Realized vol is {realized_vol * 100:.0f}% annualized."

    supportive = (
        benchmark_price > sma20 > sma50
        and ret21 >= 0.0
        and breadth_above_20 >= 0.55
        and breadth_above_50 >= 0.50
        and not vol_spike
    )
    risk_off = (
        benchmark_price < sma50
        or (benchmark_price < sma20 and sma20 < sma50)
        or ret21 <= -0.03
        or breadth_above_20 < 0.40
        or breadth_above_50 < 0.35
    )

    if supportive and not vol_elevated:
        summary = (
            f"{benchmark_symbol} is above its 20/50-day trend stack and breadth is supportive "
            f"({breadth_above_20 * 100:.0f}% above 20-day, {breadth_above_50 * 100:.0f}% above 50-day)."
            f"{vol_note}"
        )
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            label="supportive",
            summary=summary,
            benchmark_price=benchmark_price,
            sma20=sma20,
            sma50=sma50,
            ret21=ret21,
            breadth_above_20=breadth_above_20,
            breadth_above_50=breadth_above_50,
            threshold_boost=0.0,
            max_positions_multiplier=1.0,
            allow_new_longs=True,
        )

    if risk_off:
        summary = (
            f"{benchmark_symbol} trend or breadth is weak "
            f"({breadth_above_20 * 100:.0f}% above 20-day, {breadth_above_50 * 100:.0f}% above 50-day), "
            f"so new long entries are blocked.{vol_note}"
        )
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            label="risk_off",
            summary=summary,
            benchmark_price=benchmark_price,
            sma20=sma20,
            sma50=sma50,
            ret21=ret21,
            breadth_above_20=breadth_above_20,
            breadth_above_50=breadth_above_50,
            threshold_boost=0.10,
            max_positions_multiplier=0.0,
            allow_new_longs=False,
        )

    if vol_spike:
        summary = (
            f"{benchmark_symbol} trend looks acceptable but volatility is extremely elevated "
            f"({realized_vol * 100:.0f}% annualized), so the model is cutting position capacity "
            f"and raising the entry bar to protect against gap risk."
        )
        return MarketRegime(
            benchmark_symbol=benchmark_symbol,
            label="cautious",
            summary=summary,
            benchmark_price=benchmark_price,
            sma20=sma20,
            sma50=sma50,
            ret21=ret21,
            breadth_above_20=breadth_above_20,
            breadth_above_50=breadth_above_50,
            threshold_boost=settings.market_regime_cautious_threshold_boost + 0.04,
            max_positions_multiplier=max(0.25, settings.market_regime_cautious_max_positions_multiplier - 0.25),
            allow_new_longs=True,
        )

    vol_suffix = f" Elevated vol ({realized_vol * 100:.0f}%) reduces conviction." if vol_elevated else ""
    summary = (
        f"{benchmark_symbol} is mixed, so the model is trading cautiously "
        f"with a tighter entry gate and fewer open slots "
        f"({breadth_above_20 * 100:.0f}% above 20-day, {breadth_above_50 * 100:.0f}% above 50-day)."
        f"{vol_suffix}"
    )
    return MarketRegime(
        benchmark_symbol=benchmark_symbol,
        label="cautious",
        summary=summary,
        benchmark_price=benchmark_price,
        sma20=sma20,
        sma50=sma50,
        ret21=ret21,
        breadth_above_20=breadth_above_20,
        breadth_above_50=breadth_above_50,
        threshold_boost=settings.market_regime_cautious_threshold_boost,
        max_positions_multiplier=settings.market_regime_cautious_max_positions_multiplier,
        allow_new_longs=True,
    )


def apply_market_regime_to_signal(signal: SignalScore, regime: MarketRegime, base_threshold: float) -> SignalScore:
    if signal.decision == "buy" and not regime.allow_new_longs:
        signal.decision = "watch"
        signal.rationale = f"{signal.rationale}. Market regime is risk-off, so new longs are blocked."
        return signal

    if regime.label == "cautious" and signal.decision == "buy":
        signal.rationale = f"{signal.rationale}. Market regime is cautious, so position count is reduced."
        return signal

    if (
        regime.label == "cautious"
        and signal.decision == "watch"
        and signal.composite_score >= base_threshold
        and signal.composite_score < regime.effective_signal_threshold(base_threshold)
    ):
        signal.rationale = (
            f"{signal.rationale}. Market regime is cautious, so the live buy threshold is temporarily higher."
        )
    return signal


def parse_regime_warning(raw_warning: str) -> dict | None:
    if not raw_warning.startswith(REGIME_WARNING_PREFIX):
        return None
    try:
        return json.loads(raw_warning[len(REGIME_WARNING_PREFIX):])
    except json.JSONDecodeError:
        return None
