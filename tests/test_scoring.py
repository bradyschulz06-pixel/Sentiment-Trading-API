from __future__ import annotations

from datetime import date

from app.models import EarningsBundle, PriceBar
from app.models import PositionSnapshot
from app.scoring import _compute_macd, _compute_rsi, _volume_ratio, build_signal, compute_momentum_score


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


def _realistic_uptrend_bars(symbol: str = "TEST", n: int = 70) -> list[PriceBar]:
    """Uptrend with 3-up/1-down rhythm giving RSI ~58-65 — no overbought dampening."""
    closes = []
    price = 100.0
    for i in range(n):
        price *= 0.985 if i % 4 == 3 else 1.007
        closes.append(price)
    return _bars(symbol, closes)


def test_earnings_only_composite() -> None:
    """Composite score combines momentum and earnings for better profit potential."""
    bars = _realistic_uptrend_bars()
    bundle = EarningsBundle(
        symbol="TEST",
        reported_date="2026-04-18",
        fiscal_date_ending="2026-03-31",
        surprise_pct=8.5,
        transcript_sentiment=0.45,
        upcoming_report_date="2026-06-20",
    )
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        bundle=bundle,
        threshold=0.01,
        stop_loss_pct=0.07,
        upcoming_earnings_buffer_days=2,
        today=date(2026, 4, 19),
    )
    # Enhanced composite: 40% momentum + 60% earnings
    expected_composite = (signal.momentum_score * 0.40) + (signal.earnings_score * 0.60)
    assert abs(signal.composite_score - expected_composite) < 0.01
    assert signal.sentiment_score == 0.0


def test_buy_signal_when_earnings_aligned() -> None:
    bars = _realistic_uptrend_bars()
    bundle = EarningsBundle(
        symbol="TEST",
        reported_date="2026-04-18",
        fiscal_date_ending="2026-03-31",
        surprise_pct=8.5,
        transcript_sentiment=0.45,
        upcoming_report_date="2026-06-20",
    )
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        bundle=bundle,
        threshold=0.32,
        stop_loss_pct=0.07,
        upcoming_earnings_buffer_days=2,
        today=date(2026, 4, 26),
    )
    assert signal.decision == "buy"
    assert signal.composite_score > 0.32


def test_watch_signal_when_earnings_are_too_close() -> None:
    bars = _realistic_uptrend_bars()
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        bundle=EarningsBundle(
            symbol="TEST",
            reported_date="2026-04-15",
            fiscal_date_ending="2026-03-31",
            surprise_pct=10.0,
            transcript_sentiment=0.3,
            upcoming_report_date="2026-04-27",
        ),
        threshold=0.32,
        stop_loss_pct=0.07,
        upcoming_earnings_buffer_days=2,
        today=date(2026, 4, 26),
    )
    assert signal.decision == "watch"


def test_rsi_is_included_in_momentum_metrics() -> None:
    bars = _realistic_uptrend_bars()
    _, metrics = compute_momentum_score(bars)
    assert "rsi" in metrics
    assert 0.0 <= metrics["rsi"] <= 100.0


def test_rsi_is_elevated_for_linear_uptrend() -> None:
    linear_bars = _bars("LINEAR", [100 + i for i in range(80)])
    _, metrics = compute_momentum_score(linear_bars)
    assert metrics["rsi"] > 80, f"Perfectly linear uptrend should have RSI > 80, got {metrics['rsi']:.1f}"


def test_overbought_rsi_blocks_new_buy() -> None:
    linear_bars = _bars("LINEAR", [100 + i for i in range(80)])
    _, metrics = compute_momentum_score(linear_bars)
    assert metrics["rsi"] > 75, "Precondition: RSI must be overbought for this test"
    signal = build_signal(
        symbol="LINEAR",
        bars=linear_bars,
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.07,
        upcoming_earnings_buffer_days=2,
    )
    assert signal.decision == "watch", f"Overbought RSI ({metrics['rsi']:.0f}) should block buy; got {signal.decision}"
    assert "RSI" in signal.rationale


def test_target_price_matches_risk_reward_ratio() -> None:
    """Enhanced target price uses volatility-adjusted risk-reward with momentum bonus."""
    bars = _realistic_uptrend_bars()
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.07,
        upcoming_earnings_buffer_days=2,
    )
    # Enhanced target: 2x risk minimum, with momentum bonus for strong trends
    volatility_adjusted_target = 0.07 * 2.0  # 2:1 risk-reward minimum
    # If momentum is strong (>0.3), target should be extended
    if signal.momentum_score > 0.3:
        volatility_adjusted_target *= 1.3  # 30% extension for strong momentum

    expected = signal.price * (1.0 + volatility_adjusted_target)
    # Allow small rounding difference
    assert abs(signal.target_price - expected) < 0.01


def test_volume_ratio_returns_one_with_insufficient_data() -> None:
    short_bars = _bars("X", [100.0] * 5)
    assert _volume_ratio(short_bars) == 1.0


def test_rsi_returns_neutral_with_insufficient_data() -> None:
    assert _compute_rsi([100.0, 101.0]) == 50.0


def test_macd_histogram_present_in_metrics() -> None:
    bars = _realistic_uptrend_bars(n=70)
    _, metrics = compute_momentum_score(bars)
    assert "macd_histogram" in metrics
    assert isinstance(metrics["macd_histogram"], float)


def test_macd_histogram_positive_for_accelerating_uptrend() -> None:
    closes = [100.0 * (1.008 ** i) for i in range(80)]
    bars = _bars("UP", closes)
    _, metrics = compute_momentum_score(bars)
    assert metrics["macd_histogram"] > 0, f"Accelerating uptrend should produce positive MACD histogram, got {metrics['macd_histogram']}"


def test_macd_histogram_negative_after_sharp_reversal() -> None:
    up = [100.0 * (1.005 ** i) for i in range(60)]
    peak = up[-1]
    down = [peak * (0.985 ** i) for i in range(1, 21)]
    bars = _bars("REV", up + down)
    _, metrics = compute_momentum_score(bars)
    assert metrics["macd_histogram"] < 0, f"Post-reversal MACD histogram should be negative, got {metrics['macd_histogram']}"


def test_macd_returns_zero_tuple_with_insufficient_data() -> None:
    macd, signal, hist = _compute_macd([100.0] * 30)
    assert macd == 0.0 and signal == 0.0 and hist == 0.0


def test_sma200_present_in_metrics_with_enough_bars() -> None:
    bars = _bars("X", [100 + i * 0.1 for i in range(220)])
    _, metrics = compute_momentum_score(bars)
    assert "sma200" in metrics
    assert metrics["sma200"] is not None
    assert metrics["sma200"] > 0


def test_sma200_none_with_fewer_than_200_bars() -> None:
    bars = _bars("X", [100.0] * 80)
    _, metrics = compute_momentum_score(bars)
    assert metrics.get("sma200") is None


def test_hold_signal_when_position_is_profitable() -> None:
    bars = _realistic_uptrend_bars()
    bundle = EarningsBundle(
        symbol="TEST",
        reported_date="2026-04-15",
        fiscal_date_ending="2026-03-31",
        surprise_pct=8.0,
        transcript_sentiment=0.3,
    )
    position = PositionSnapshot(
        symbol="TEST",
        qty=10,
        avg_entry_price=90.0,
        market_value=1_100.0,
        unrealized_plpc=0.10,
    )
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        bundle=bundle,
        threshold=0.01,
        stop_loss_pct=0.07,
        upcoming_earnings_buffer_days=2,
        position=position,
        today=date(2026, 4, 20),
    )
    assert signal.decision == "hold"
