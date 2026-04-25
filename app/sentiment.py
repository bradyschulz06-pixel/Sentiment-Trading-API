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
    "strong demand": 1.2,
    "margin expansion": 1.2,
    "record revenue": 1.4,
    "upgraded": 1.1,
    "outperform": 1.0,
    "missed expectations": -1.8,
    "miss expectations": -1.8,
    "cut guidance": -1.8,
    "cuts guidance": -1.8,
    "margin pressure": -1.2,
    "weak demand": -1.2,
    "downgraded": -1.1,
    "investigation": -1.0,
    "lawsuit": -1.0,
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
    "miss": -0.85,
    "missed": -0.85,
    "weak": -0.55,
    "slowing": -0.55,
    "cut": -0.55,
    "cuts": -0.55,
    "lawsuit": -0.75,
    "fraud": -0.95,
    "delay": -0.55,
    "decline": -0.55,
    "pressure": -0.4,
    "downgrade": -0.6,
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
        # Exponential decay: ~37% weight at 5 days, ~14% at 10 days, floor at 5%
        recency_weight = max(0.05, math.exp(-age_hours / 120.0))
        weighted_total += base_score * recency_weight
        weight_sum += recency_weight
    if weight_sum == 0:
        return 0.0
    return clamp(weighted_total / weight_sum)
