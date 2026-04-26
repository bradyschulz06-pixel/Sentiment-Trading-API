from __future__ import annotations

from datetime import datetime, timezone
import math
import re

from app.models import NewsItem


TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+")

NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without", "lack", "lacks",
    "barely", "hardly", "scarcely", "fail", "fails", "failed", "miss",
    "missed", "unable", "unlikely", "disappoint", "disappoints", "disappointed",
}

PHRASE_WEIGHTS = {
    "beat expectations": 1.7,
    "beats expectations": 1.7,
    "raised guidance": 1.6,
    "raise guidance": 1.6,
    "raises guidance": 1.6,
    "strong demand": 1.2,
    "margin expansion": 1.2,
    "record revenue": 1.4,
    "record earnings": 1.3,
    "record profit": 1.2,
    "earnings beat": 1.5,
    "revenue beat": 1.4,
    "raised outlook": 1.4,
    "raised forecast": 1.5,
    "guidance raised": 1.5,
    "market share gains": 1.1,
    "accelerating growth": 1.2,
    "upgraded": 1.1,
    "outperform": 1.0,
    "blowout quarter": 1.4,
    "strong results": 1.1,
    "beat on revenue": 1.4,
    "beat on earnings": 1.5,
    "buyback program": 0.7,
    "share repurchase": 0.6,
    "dividend increase": 0.7,
    "missed expectations": -1.8,
    "miss expectations": -1.8,
    "cut guidance": -1.8,
    "cuts guidance": -1.8,
    "reduced guidance": -1.7,
    "guidance cut": -1.6,
    "lowered outlook": -1.5,
    "lowered guidance": -1.6,
    "profit warning": -1.8,
    "earnings miss": -1.6,
    "revenue miss": -1.5,
    "margin pressure": -1.2,
    "margin contraction": -1.2,
    "weak demand": -1.2,
    "downgraded": -1.1,
    "market share loss": -1.0,
    "investigation": -1.0,
    "accounting investigation": -1.5,
    "class action": -1.1,
    "regulatory scrutiny": -0.9,
    "supply chain disruption": -0.8,
    "executive departure": -0.7,
    "ceo departure": -0.9,
    "missed on revenue": -1.5,
    "missed on earnings": -1.6,
    "lawsuit": -1.0,
    "price target raised": 1.3,
    "price target increased": 1.2,
    "price target hiked": 1.1,
    "reiterated buy": 0.9,
    "reiterated outperform": 0.8,
    "initiates outperform": 1.1,
    "initiates buy": 1.0,
    "double upgrade": 1.4,
    "price target lowered": -1.3,
    "price target cut": -1.2,
    "price target reduced": -1.1,
    "reiterated sell": -0.9,
    "initiates underperform": -1.2,
    "initiates sell": -1.0,
    "double downgrade": -1.4,
    # M&A catalysts
    "acquired by": 1.5,
    "merger agreement": 1.4,
    "takeover bid": 1.2,
    "acquisition": 0.9,
    "buyout": 1.0,
    "going private": 0.8,
    # Guidance withdrawal
    "withdrew guidance": -1.6,
    "withdraw guidance": -1.6,
    "suspended guidance": -1.5,
    "pulled guidance": -1.5,
    "guidance withdrawn": -1.6,
    # Above/below consensus language
    "raised full-year guidance": 1.7,
    "raised full year guidance": 1.7,
    "above consensus": 1.3,
    "ahead of estimates": 1.3,
    "beats consensus": 1.5,
    "below consensus": -1.3,
    "below estimates": -1.3,
    "missed consensus": -1.5,
}

TOKEN_WEIGHTS = {
    "beat": 0.8,
    "beats": 0.8,
    "strong": 0.5,
    "growth": 0.45,
    "bullish": 0.65,
    "accelerating": 0.55,
    "resilient": 0.45,
    "record": 0.45,
    "upgrade": 0.6,
    "expansion": 0.35,
    "profit": 0.3,
    "momentum": 0.35,
    "outperformed": 0.55,
    "outperforming": 0.55,
    "robust": 0.5,
    "surged": 0.55,
    "surging": 0.5,
    "rebound": 0.45,
    "recovery": 0.4,
    "innovative": 0.3,
    "efficient": 0.3,
    "attractive": 0.35,
    "miss": -0.85,
    "missed": -0.85,
    "weak": -0.55,
    "slowing": -0.55,
    "cut": -0.55,
    "cuts": -0.55,
    "disappointing": -0.65,
    "disappoints": -0.65,
    "headwinds": -0.5,
    "challenges": -0.4,
    "uncertainty": -0.35,
    "restructuring": -0.35,
    "layoffs": -0.5,
    "writedown": -0.7,
    "impairment": -0.65,
    "lawsuit": -0.75,
    "fraud": -0.95,
    "delay": -0.55,
    "decline": -0.55,
    "pressure": -0.4,
    "downgrade": -0.6,
}


SOURCE_QUALITY_MULTIPLIERS: dict[str, float] = {
    # Tier 1: wire services and major financial media (1.4×)
    "reuters": 1.4,
    "bloomberg": 1.4,
    "associated press": 1.4,
    "ap": 1.4,
    "dow jones": 1.4,
    "the wall street journal": 1.4,
    "wall street journal": 1.4,
    "financial times": 1.4,
    # Tier 2: established financial outlets (1.2×)
    "barron's": 1.2,
    "marketwatch": 1.2,
    "cnbc": 1.2,
    "benzinga": 1.1,
    "the street": 1.1,
    # Tier 3: opinion/lower-signal sources (sub-1×)
    "seeking alpha": 0.75,
    "stocktwits": 0.70,
}


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _is_negated(tokens: list[str], token_idx: int, window: int = 4) -> bool:
    """Return True if any negation word appears within `window` tokens before position `token_idx`."""
    start = max(0, token_idx - window)
    return any(t in NEGATION_WORDS for t in tokens[start:token_idx])


def score_text(text: str) -> float:
    lowered = text.lower()
    score = 0.0

    tokens = TOKEN_RE.findall(lowered)

    # Phrase matching with negation check
    for phrase, weight in PHRASE_WEIGHTS.items():
        idx = lowered.find(phrase)
        while idx != -1:
            context_tokens = TOKEN_RE.findall(lowered[max(0, idx - 50) : idx])
            if any(t in NEGATION_WORDS for t in context_tokens[-5:]):
                score -= weight * 0.8
            else:
                score += weight
            idx = lowered.find(phrase, idx + 1)

    # Token matching with negation check
    for i, token in enumerate(tokens):
        token_weight = TOKEN_WEIGHTS.get(token, 0.0)
        if token_weight == 0.0:
            continue
        if _is_negated(tokens, i):
            score -= token_weight * 0.8
        else:
            score += token_weight

    normalizer = max(4.0, math.sqrt(len(tokens) + 1))
    return clamp(score / normalizer)


def _hours_old(published_at: str) -> float:
    try:
        stamp = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return max(0.0, (datetime.now(timezone.utc) - stamp).total_seconds() / 3600)


def aggregate_news_sentiment(items: list[NewsItem]) -> float:
    if not items:
        return 0.0
    weighted_total = 0.0
    weight_sum = 0.0
    for item in items:
        base_score = item.sentiment if item.sentiment else score_text(f"{item.headline}. {item.summary}. {item.content}")
        age_hours = _hours_old(item.published_at)
        # Exponential decay: ~37% weight at 5 days, ~14% at 10 days, floor at 1%
        recency_weight = max(0.01, math.exp(-age_hours / 120.0))
        source_quality = SOURCE_QUALITY_MULTIPLIERS.get(item.source.lower().strip(), 1.0)
        effective_weight = recency_weight * source_quality
        weighted_total += base_score * effective_weight
        weight_sum += effective_weight
    if weight_sum == 0:
        return 0.0
    aggregate = clamp(weighted_total / weight_sum)
    if len(items) == 1:
        aggregate *= 0.80
    return aggregate
