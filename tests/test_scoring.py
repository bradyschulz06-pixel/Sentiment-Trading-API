from __future__ import annotations

from datetime import date

from app.models import EarningsBundle, NewsItem, PriceBar
from app.scoring import _compute_rsi, _volume_ratio, build_signal, compute_momentum_score, normalize_factor_weights


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


def test_buy_signal_when_trend_news_and_earnings_align() -> None:
    bars = _realistic_uptrend_bars()
    news = [
        NewsItem(
            symbol="TEST",
            headline="Company beats expectations and raises guidance",
            summary="Demand remains strong and margins are expanding.",
            content="Record revenue and strong demand drove another upside quarter.",
            source="ExampleWire",
            url="https://example.com",
            published_at="2026-04-20T12:00:00+00:00",
        )
    ]
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
        news_items=news,
        bundle=bundle,
        threshold=0.32,
        stop_loss_pct=0.08,
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
        news_items=[],
        bundle=EarningsBundle(
            symbol="TEST",
            reported_date="2026-04-15",
            fiscal_date_ending="2026-03-31",
            surprise_pct=10.0,
            transcript_sentiment=0.3,
            upcoming_report_date="2026-04-27",
        ),
        threshold=0.32,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
        today=date(2026, 4, 26),
    )
    assert signal.decision == "watch"


def test_factor_weights_normalize_when_inputs_do_not_sum_to_one() -> None:
    momentum_weight, sentiment_weight, earnings_weight = normalize_factor_weights(4.0, 2.0, 4.0)
    assert round(momentum_weight + sentiment_weight + earnings_weight, 6) == 1.0
    assert momentum_weight == 0.4
    assert sentiment_weight == 0.2
    assert earnings_weight == 0.4


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
        news_items=[],
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
    )
    assert signal.decision == "watch", f"Overbought RSI ({metrics['rsi']:.0f}) should block buy; got {signal.decision}"
    assert "RSI" in signal.rationale


def test_target_price_matches_risk_reward_ratio() -> None:
    bars = _realistic_uptrend_bars()
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        news_items=[],
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
    )
    expected = round(signal.price * (1.0 + 0.08 * 1.75), 2)
    assert abs(signal.target_price - expected) < 0.02


def test_volume_ratio_returns_one_with_insufficient_data() -> None:
    short_bars = _bars("X", [100.0] * 5)
    assert _volume_ratio(short_bars) == 1.0


def test_rsi_returns_neutral_with_insufficient_data() -> None:
    assert _compute_rsi([100.0, 101.0]) == 50.0
