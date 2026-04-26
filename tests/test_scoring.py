from __future__ import annotations

from datetime import date

from app.models import EarningsBundle, NewsItem, PriceBar
from app.models import PositionSnapshot
from app.scoring import _compute_macd, _compute_rsi, _volume_ratio, build_signal, compute_momentum_score, normalize_factor_weights


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


def test_macd_histogram_present_in_metrics() -> None:
    bars = _realistic_uptrend_bars(n=70)
    _, metrics = compute_momentum_score(bars)
    assert "macd_histogram" in metrics
    assert isinstance(metrics["macd_histogram"], float)


def test_macd_histogram_positive_for_accelerating_uptrend() -> None:
    # Exponential growth: fast EMA tracks the acceleration faster than slow EMA → positive histogram.
    closes = [100.0 * (1.008 ** i) for i in range(80)]
    bars = _bars("UP", closes)
    _, metrics = compute_momentum_score(bars)
    assert metrics["macd_histogram"] > 0, f"Accelerating uptrend should produce positive MACD histogram, got {metrics['macd_histogram']}"


def test_macd_histogram_negative_after_sharp_reversal() -> None:
    # 60 bars of steady growth build a positive MACD; 20 bars of sharp decline pull
    # the fast EMA down faster than the signal line catches up → histogram goes negative.
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


def test_sma200_demotes_composite_when_price_below() -> None:
    # Build bars where price ends below the 200-day average (sharp decline at end).
    closes = [100.0 + i * 0.5 for i in range(200)] + [50.0] * 10
    bars = _bars("X", closes)
    _, metrics = compute_momentum_score(bars)
    assert metrics.get("sma200") is not None
    assert bars[-1].close < metrics["sma200"], "Precondition: price must be below SMA200"
    signal_below = build_signal(
        symbol="X",
        bars=bars,
        news_items=[],
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
    )
    # Rebuild bars without the decline to get a baseline composite
    bars_above = _bars("X", [100.0 + i * 0.5 for i in range(210)])
    signal_above = build_signal(
        symbol="X",
        bars=bars_above,
        news_items=[],
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
    )
    assert signal_below.composite_score <= signal_above.composite_score
    assert "200-day" in signal_below.rationale


def test_rsi_overbought_exit_triggers_on_profitable_position() -> None:
    linear_bars = _bars("LINEAR", [100 + i for i in range(80)])
    _, metrics = compute_momentum_score(linear_bars)
    assert metrics["rsi"] > 82, "Precondition: RSI must be > 82 for overbought exit"
    position = PositionSnapshot(
        symbol="LINEAR",
        qty=10,
        avg_entry_price=100.0,
        market_value=1_800.0,
        unrealized_plpc=0.08,
    )
    signal = build_signal(
        symbol="LINEAR",
        bars=linear_bars,
        news_items=[],
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
        position=position,
    )
    assert signal.decision == "sell"
    assert "overbought" in signal.rationale.lower()


def test_rsi_overbought_exit_does_not_trigger_on_unprofitable_position() -> None:
    linear_bars = _bars("LINEAR", [100 + i for i in range(80)])
    position = PositionSnapshot(
        symbol="LINEAR",
        qty=10,
        avg_entry_price=175.0,
        market_value=1_800.0,
        unrealized_plpc=0.02,
    )
    signal = build_signal(
        symbol="LINEAR",
        bars=linear_bars,
        news_items=[],
        bundle=None,
        threshold=0.01,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
        position=position,
    )
    # unrealized_plpc < 0.05, so RSI overbought exit should NOT fire
    assert signal.decision != "sell" or "overbought" not in signal.rationale.lower()


# --- factor alignment bonus and divergence penalty tests ---

def test_aligned_factors_boost_composite() -> None:
    bars = _realistic_uptrend_bars(n=80)
    # Strongly positive earnings aligns with the uptrend momentum
    bundle = EarningsBundle(symbol="TEST", reported_date="2026-01-01", surprise_pct=18.0)
    # Positive news to ensure sentiment_score > 0.20
    news = [
        NewsItem(
            symbol="TEST",
            headline="Company beat expectations and raised full-year guidance significantly.",
            summary="",
            content="",
            source="reuters",
            url="",
            published_at="2099-01-01T00:00:00+00:00",
            sentiment=0.0,
        ),
        NewsItem(
            symbol="TEST",
            headline="Company above consensus estimates and raised guidance on strong demand.",
            summary="",
            content="",
            source="bloomberg",
            url="",
            published_at="2099-01-01T00:00:00+00:00",
            sentiment=0.0,
        ),
    ]
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        news_items=news,
        bundle=bundle,
        threshold=0.20,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=0,
        today=date(2026, 1, 2),  # day after report → recency_decay ≈ 1.0 → earnings_score > 0.20
    )
    # Alignment bonus fires when all three > 0.20; rationale must mention it
    assert "all three factors independently confirm" in signal.rationale


def test_divergent_momentum_sentiment_reduces_composite() -> None:
    bars = _realistic_uptrend_bars(n=80)
    # Strongly negative news creates a momentum–sentiment divergence
    negative_news = [
        NewsItem(
            symbol="TEST",
            headline="Company withdrew guidance and missed consensus estimates badly.",
            summary="",
            content="",
            source="reuters",
            url="",
            published_at="2099-01-01T00:00:00+00:00",
            sentiment=0.0,
        ),
        NewsItem(
            symbol="TEST",
            headline="Management suspended guidance citing severe macro headwinds.",
            summary="",
            content="",
            source="bloomberg",
            url="",
            published_at="2099-01-01T00:00:00+00:00",
            sentiment=0.0,
        ),
    ]
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        news_items=negative_news,
        bundle=None,
        threshold=0.10,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=0,
    )
    # Divergence penalty fires when momentum > 0.20 and sentiment < -0.20
    assert "diverging" in signal.rationale
