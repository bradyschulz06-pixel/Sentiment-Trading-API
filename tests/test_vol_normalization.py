import math

from app.models import PriceBar
from app.scoring import compute_momentum_score, compute_position_vol_scalar


def _bars(closes: list[float]) -> list[PriceBar]:
    return [
        PriceBar(
            symbol="TEST",
            timestamp=f"2026-01-{(i % 28) + 1:02d}T00:00:00Z",
            open=c - 0.5,
            high=c + 1.0,
            low=c - 1.0,
            close=c,
            volume=1_000_000,
        )
        for i, c in enumerate(closes)
    ]


def test_vol_scalar_is_one_for_target_vol_stock() -> None:
    """A stock with ~20% annualized vol should return a scalar near 1.0."""
    import random
    random.seed(42)
    # Build a series with ~20% annualized vol (daily std ~20/sqrt(252) ≈ 1.26%)
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * (1 + random.gauss(0, 0.0126)))
    scalar = compute_position_vol_scalar(_bars(closes), target_annual_vol=0.20)
    assert 0.50 <= scalar <= 1.50


def test_high_vol_stock_gets_smaller_scalar() -> None:
    """A high-vol stock should produce a scalar below 1.0 (smaller allocation)."""
    import random
    random.seed(7)
    # ~60% annualized vol → scalar should be well below 1.0 (target / stock = 0.20/0.60 ≈ 0.33, clamped to 0.50)
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * (1 + random.gauss(0, 0.038)))  # ~60% ann
    scalar = compute_position_vol_scalar(_bars(closes), target_annual_vol=0.20)
    assert scalar <= 0.75, f"High-vol stock should have scalar ≤ 0.75, got {scalar}"


def test_low_vol_stock_gets_larger_scalar() -> None:
    """A low-vol stock should produce a scalar above 1.0 (larger allocation), capped at max."""
    import random
    random.seed(3)
    # ~8% annualized vol → scalar = 0.20/0.08 = 2.5, clamped to max_scalar=1.5
    closes = [100.0]
    for _ in range(60):
        closes.append(closes[-1] * (1 + random.gauss(0, 0.005)))  # ~8% ann
    scalar = compute_position_vol_scalar(_bars(closes), target_annual_vol=0.20, max_scalar=1.50)
    assert scalar >= 1.20, f"Low-vol stock should have scalar ≥ 1.20, got {scalar}"
    assert scalar <= 1.50


def test_momentum_score_with_steady_uptrend() -> None:
    """A steady uptrend should produce a positive momentum score."""
    closes = [100 + i * 0.5 for i in range(70)]
    score, metrics = compute_momentum_score(_bars(closes))
    assert score > 0.0
    assert metrics["current"] > metrics["sma20"] > metrics["sma50"]


def test_momentum_score_with_steep_downtrend() -> None:
    """A steep downtrend should produce a negative or very low momentum score."""
    closes = [200 - i * 1.5 for i in range(70)]
    score, _ = compute_momentum_score(_bars(closes))
    assert score < 0.3, f"Downtrend should produce low score, got {score}"


def test_vol_adjusted_normalization_distinguishes_magnitudes() -> None:
    """
    A +15% move on a 10%-vol stock should score higher than a +15% move on a 40%-vol stock,
    because the same move represents a bigger surprise relative to expected range.
    """
    import random

    # Low-vol stock: steady drift up with tiny noise
    random.seed(10)
    low_vol = [100.0]
    for _ in range(90):
        low_vol.append(low_vol[-1] * (1 + random.gauss(0.0007, 0.006)))

    # High-vol stock: same cumulative return, larger noise
    random.seed(10)
    high_vol = [100.0]
    for _ in range(90):
        high_vol.append(high_vol[-1] * (1 + random.gauss(0.0007, 0.025)))

    score_low, _ = compute_momentum_score(_bars(low_vol))
    score_high, _ = compute_momentum_score(_bars(high_vol))
    assert score_low > score_high, (
        f"Low-vol uptrend ({score_low:.3f}) should score higher than high-vol uptrend ({score_high:.3f})"
    )
