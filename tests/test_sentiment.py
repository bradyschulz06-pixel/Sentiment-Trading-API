from app.sentiment import aggregate_news_sentiment, score_text
from app.models import NewsItem


def _item(headline: str, published_at: str = "2026-04-25T12:00:00+00:00") -> NewsItem:
    return NewsItem(
        symbol="TEST",
        headline=headline,
        summary="",
        content="",
        source="test",
        url="https://example.com",
        published_at=published_at,
        sentiment=0.0,
    )


def test_positive_phrase_scores_positive() -> None:
    score = score_text("Company beat expectations and raised guidance.")
    assert score > 0.0


def test_negated_positive_phrase_scores_negative() -> None:
    score = score_text("Company did not beat expectations this quarter.")
    assert score < 0.0, f"Expected negative score for negated phrase, got {score}"


def test_negated_positive_token_flips_sign() -> None:
    positive = score_text("Company showed strong growth.")
    negated = score_text("Company did not show strong growth.")
    assert positive > 0.0
    assert negated < positive, "Negation should lower the score"


def test_pure_negative_phrase_scores_negative() -> None:
    score = score_text("Company cut guidance and missed expectations.")
    assert score < 0.0


def test_score_clamped_to_minus_one_and_one() -> None:
    extreme_positive = " ".join(["beat expectations raised guidance record revenue"] * 20)
    extreme_negative = " ".join(["missed expectations cut guidance fraud lawsuit"] * 20)
    assert score_text(extreme_positive) <= 1.0
    assert score_text(extreme_negative) >= -1.0


def test_empty_text_scores_zero() -> None:
    assert score_text("") == 0.0


def test_exponential_decay_downweights_old_news() -> None:
    fresh = _item("Company beat expectations.", published_at="2026-04-25T00:00:00+00:00")
    stale = _item("Company beat expectations.", published_at="2026-04-10T00:00:00+00:00")
    fresh_score = aggregate_news_sentiment([fresh])
    stale_score = aggregate_news_sentiment([stale])
    # Fresh news should receive higher weight than 15-day-old news.
    assert fresh_score > stale_score


def test_aggregate_returns_zero_for_empty_list() -> None:
    assert aggregate_news_sentiment([]) == 0.0
