from app.models import EarningsBundle, NewsItem, PriceBar
from app.scoring import build_signal, normalize_factor_weights


def _bars(symbol: str, closes: list[float]) -> list[PriceBar]:
    output = []
    for index, close in enumerate(closes):
        output.append(
            PriceBar(
                symbol=symbol,
                timestamp=f"2026-01-{(index % 28) + 1:02d}T00:00:00Z",
                open=close - 1,
                high=close + 1,
                low=close - 2,
                close=close,
                volume=1_000_000,
            )
        )
    return output


def test_buy_signal_when_trend_news_and_earnings_align() -> None:
    bars = _bars("TEST", [100 + i for i in range(70)])
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
    )
    assert signal.decision == "buy"
    assert signal.composite_score > 0.32


def test_watch_signal_when_earnings_are_too_close() -> None:
    bars = _bars("TEST", [100 + i for i in range(70)])
    signal = build_signal(
        symbol="TEST",
        bars=bars,
        news_items=[],
        bundle=EarningsBundle(
            symbol="TEST",
            reported_date="2026-04-18",
            fiscal_date_ending="2026-03-31",
            surprise_pct=10.0,
            transcript_sentiment=0.3,
            upcoming_report_date="2026-04-25",
        ),
        threshold=0.32,
        stop_loss_pct=0.08,
        upcoming_earnings_buffer_days=2,
    )
    assert signal.decision == "watch"


def test_factor_weights_normalize_when_inputs_do_not_sum_to_one() -> None:
    momentum_weight, sentiment_weight, earnings_weight = normalize_factor_weights(4.0, 2.0, 4.0)
    assert round(momentum_weight + sentiment_weight + earnings_weight, 6) == 1.0
    assert momentum_weight == 0.4
    assert sentiment_weight == 0.2
    assert earnings_weight == 0.4
