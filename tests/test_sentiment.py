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
    # A fresh positive item combined with a very stale negative item should net positive
    # because recency weighting heavily favours the recent one.
    fresh_positive = _item("Company beat expectations and raised guidance.", published_at="2099-01-01T00:00:00+00:00")
    stale_negative = _item("Company cut guidance and missed expectations.", published_at="2020-01-01T00:00:00+00:00")
    combined = aggregate_news_sentiment([fresh_positive, stale_negative])
    assert combined > 0.0, "Fresh positive news should outweigh very stale negative news"


def test_aggregate_returns_zero_for_empty_list() -> None:
    assert aggregate_news_sentiment([]) == 0.0


# --- new vocabulary coverage tests ---

def test_profit_warning_scores_negative() -> None:
    score = score_text("Company issues profit warning as demand weakens and margins contract.")
    assert score < 0.0


def test_earnings_beat_phrase_scores_positive() -> None:
    score = score_text("Earnings beat drove the stock higher after strong results.")
    assert score > 0.0


def test_raises_guidance_phrase_scores_positive() -> None:
    score = score_text("Management raises guidance citing accelerating growth.")
    assert score > 0.0


def test_reduced_guidance_scores_negative() -> None:
    score = score_text("Company reduced guidance due to supply chain disruption.")
    assert score < 0.0


def test_layoffs_scores_negative() -> None:
    score = score_text("Company announces layoffs amid restructuring efforts.")
    assert score < 0.0


def test_market_share_gains_scores_positive() -> None:
    score = score_text("The company reported market share gains driven by robust demand.")
    assert score > 0.0


# --- analyst-action phrase tests ---

def test_price_target_raised_scores_positive() -> None:
    score = score_text("Goldman Sachs price target raised to $250, reiterates buy rating.")
    assert score > 0.0


def test_double_upgrade_scores_positive() -> None:
    score = score_text("Analyst issues double upgrade, initiates buy on strong earnings outlook.")
    assert score > 0.0


def test_price_target_cut_scores_negative() -> None:
    score = score_text("Morgan Stanley price target cut to $80 citing deteriorating margins.")
    assert score < 0.0


def test_double_downgrade_scores_negative() -> None:
    score = score_text("Analyst double downgrade, price target reduced to street low.")
    assert score < 0.0


# --- M&A and guidance vocabulary tests ---

def test_acquired_by_scores_positive() -> None:
    score = score_text("Company will be acquired by strategic buyer at significant premium.")
    assert score > 0.2


def test_merger_agreement_scores_positive() -> None:
    score = score_text("Entered merger agreement valued at $12 billion with industry leader.")
    assert score > 0.2


def test_withdrew_guidance_scores_strongly_negative() -> None:
    score = score_text("Management withdrew guidance citing deteriorating macro conditions.")
    assert score < -0.3


def test_suspended_guidance_scores_negative() -> None:
    score = score_text("Board suspended guidance due to market uncertainty.")
    assert score < 0.0


def test_above_consensus_scores_positive() -> None:
    score = score_text("Results came in above consensus estimates for the quarter.")
    assert score > 0.0


def test_below_estimates_scores_negative() -> None:
    score = score_text("Revenue below estimates for the third consecutive quarter.")
    assert score < 0.0


# --- news source quality weighting tests ---

def _sourced_item(headline: str, source: str) -> NewsItem:
    return NewsItem(
        symbol="TEST",
        headline=headline,
        summary="",
        content="",
        source=source,
        url="https://example.com",
        published_at="2099-01-01T00:00:00+00:00",
        sentiment=0.0,
    )


def test_reuters_outweighs_seeking_alpha_same_headline() -> None:
    # A single-item aggregate always equals the item score regardless of multiplier.
    # Mix a positive high-quality source with a negative neutral-source to see the quality effect.
    positive = "Company beat expectations and raised guidance significantly."
    negative = "Company missed expectations and cut guidance."
    reuters_mix = aggregate_news_sentiment([
        _sourced_item(positive, "reuters"),
        _sourced_item(negative, "test"),
    ])
    sa_mix = aggregate_news_sentiment([
        _sourced_item(positive, "seeking alpha"),
        _sourced_item(negative, "test"),
    ])
    assert reuters_mix > sa_mix


def test_unknown_source_uses_neutral_multiplier() -> None:
    headline = "Company beat expectations and raised guidance."
    unknown_score = aggregate_news_sentiment([_sourced_item(headline, "some_unknown_blog")])
    neutral_score = aggregate_news_sentiment([_sourced_item(headline, "test")])
    assert abs(unknown_score - neutral_score) < 1e-6


def test_source_matching_is_case_insensitive() -> None:
    headline = "Company beat expectations and raised guidance."
    lower_score = aggregate_news_sentiment([_sourced_item(headline, "reuters")])
    upper_score = aggregate_news_sentiment([_sourced_item(headline, "Reuters")])
    assert abs(lower_score - upper_score) < 1e-6
