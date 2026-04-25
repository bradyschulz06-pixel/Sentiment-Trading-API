from __future__ import annotations

from datetime import date, datetime, timezone
import math

from app.models import EarningsBundle, NewsItem, PositionSnapshot, PriceBar, SignalScore
from app.sentiment import aggregate_news_sentiment, clamp


def _rolling_annualized_vol(closes: list[float], window: int) -> float:
    """Annualized daily volatility from the last `window` daily returns."""
    if len(closes) < window + 1:
        return 0.15
    daily_rets = [closes[i] / closes[i - 1] - 1 for i in range(len(closes) - window, len(closes))]
    mean = sum(daily_rets) / len(daily_rets)
    variance = sum((r - mean) ** 2 for r in daily_rets) / max(1, len(daily_rets) - 1)
    return max(0.005, math.sqrt(variance * 252))


def _compute_atr(bars: list[PriceBar], period: int = 14) -> float:
    """Average True Range over `period` bars."""
    if len(bars) < period + 1:
        return bars[-1].close * 0.02
    true_ranges = [
        max(
            bars[i].high - bars[i].low,
            abs(bars[i].high - bars[i - 1].close),
            abs(bars[i].low - bars[i - 1].close),
        )
        for i in range(1, len(bars))
    ]
    return sum(true_ranges[-period:]) / period


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def _average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _pct_change(new_value: float, old_value: float) -> float:
    if old_value == 0:
        return 0.0
    return (new_value - old_value) / old_value


def _lookback_close(closes: list[float], days: int) -> float:
    index = max(0, len(closes) - days - 1)
    return closes[index]


def compute_position_vol_scalar(
    bars: list[PriceBar],
    target_annual_vol: float = 0.20,
    min_scalar: float = 0.50,
    max_scalar: float = 1.50,
) -> float:
    """
    Return a multiplier to scale position size inversely by realized volatility.

    A stock whose 21-day annualized vol equals `target_annual_vol` gets a scalar of 1.0
    (full allocation). Higher-vol stocks are scaled down; lower-vol stocks up, bounded by
    [min_scalar, max_scalar] so no single position explodes or shrinks to nothing.
    """
    closes = [bar.close for bar in bars]
    stock_vol = _rolling_annualized_vol(closes, min(21, len(closes) - 1))
    scalar = target_annual_vol / stock_vol
    return max(min_scalar, min(max_scalar, scalar))


def normalize_factor_weights(
    momentum_weight: float,
    sentiment_weight: float,
    earnings_weight: float,
) -> tuple[float, float, float]:
    raw_weights = [
        max(0.0, momentum_weight),
        max(0.0, sentiment_weight),
        max(0.0, earnings_weight),
    ]
    total = sum(raw_weights)
    if total <= 0:
        return 0.45, 0.20, 0.35
    return tuple(weight / total for weight in raw_weights)


def compute_momentum_score(bars: list[PriceBar]) -> tuple[float, dict[str, float]]:
    closes = [bar.close for bar in bars]
    current = closes[-1]
    sma20 = _average(closes[-20:])
    sma50 = _average(closes[-50:]) if len(closes) >= 50 else _average(closes)
    ret21 = _pct_change(current, _lookback_close(closes, 21))
    ret63 = _pct_change(current, _lookback_close(closes, 63))
    trend = 0.0
    if current > sma20:
        trend += 0.4
    if sma20 > sma50:
        trend += 0.6
    vol_21 = _rolling_annualized_vol(closes, 21)
    vol_63 = _rolling_annualized_vol(closes, min(63, len(closes) - 1))
    # Normalize return by expected period volatility: ret / (annualized_vol * sqrt(period/252))
    normalized_21 = clamp(ret21 / (vol_21 * math.sqrt(21 / 252)))
    normalized_63 = clamp(ret63 / (vol_63 * math.sqrt(63 / 252)))
    score = clamp((normalized_21 * 0.45) + (normalized_63 * 0.35) + (trend * 0.20))
    metrics = {
        "current": current,
        "sma20": sma20,
        "sma50": sma50,
        "ret21": ret21,
        "ret63": ret63,
    }
    return score, metrics


def compute_earnings_score(bundle: EarningsBundle | None, today: date, buffer_days: int) -> tuple[float, bool]:
    if bundle is None or bundle.surprise_pct is None:
        return 0.0, False
    reported_date = _parse_iso_date(bundle.reported_date)
    days_since_report = (today - reported_date).days if reported_date else 45
    recency_decay = math.exp(-(max(days_since_report, 0) / 75.0))
    surprise_component = clamp((bundle.surprise_pct or 0.0) / 15.0)
    score = clamp(((surprise_component * 0.70) + (bundle.transcript_sentiment * 0.30)) * recency_decay)
    blocked_for_earnings = False
    upcoming_date = _parse_iso_date(bundle.upcoming_report_date)
    if upcoming_date:
        days_until = (upcoming_date - today).days
        if 0 <= days_until <= buffer_days:
            blocked_for_earnings = True
            score = min(score, 0.1)
    return score, blocked_for_earnings


def build_signal(
    symbol: str,
    bars: list[PriceBar],
    news_items: list[NewsItem],
    bundle: EarningsBundle | None,
    threshold: float,
    stop_loss_pct: float,
    upcoming_earnings_buffer_days: int,
    momentum_weight: float = 0.45,
    sentiment_weight: float = 0.20,
    earnings_weight: float = 0.35,
    position: PositionSnapshot | None = None,
    today: date | None = None,
) -> SignalScore:
    momentum_score, metrics = compute_momentum_score(bars)
    sentiment_score = aggregate_news_sentiment(news_items)
    momentum_weight, sentiment_weight, earnings_weight = normalize_factor_weights(
        momentum_weight,
        sentiment_weight,
        earnings_weight,
    )
    earnings_score, blocked_for_earnings = compute_earnings_score(
        bundle,
        today=today or datetime.now(timezone.utc).date(),
        buffer_days=upcoming_earnings_buffer_days,
    )
    composite = clamp(
        (momentum_score * momentum_weight)
        + (sentiment_score * sentiment_weight)
        + (earnings_score * earnings_weight)
    )
    current_price = metrics["current"]
    stop_price = current_price * (1.0 - stop_loss_pct)
    atr = _compute_atr(bars)
    target_price = current_price + (atr * 3.0)
    if position is not None and position.avg_entry_price > 0:
        stop_price = position.avg_entry_price * (1.0 - stop_loss_pct)
    reasons: list[str] = []
    trend_ok = current_price > metrics["sma20"] > metrics["sma50"]
    if trend_ok:
        reasons.append("trend is above the 20/50-day moving averages")
    else:
        reasons.append("trend is not clean enough yet")
    if sentiment_score > 0.15:
        reasons.append("news tone is positive")
    elif sentiment_score < -0.15:
        reasons.append("news tone is negative")
    if earnings_score > 0.15:
        reasons.append("recent earnings surprise is supportive")
    elif earnings_score < -0.15:
        reasons.append("earnings read-through is weak")
    if blocked_for_earnings:
        reasons.append("an earnings report is too close to open a new swing trade safely")
    decision = "watch"
    if position is not None:
        if current_price <= stop_price or composite < 0.05 or momentum_score < -0.20:
            decision = "sell"
            reasons.append("existing position has lost its edge")
        else:
            decision = "hold"
            reasons.append("existing position still fits the model")
    elif composite >= threshold and momentum_score > 0 and trend_ok and not blocked_for_earnings:
        decision = "buy"
    headline = news_items[0].headline if news_items else ""
    return SignalScore(
        symbol=symbol,
        price=round(current_price, 2),
        momentum_score=round(momentum_score, 4),
        sentiment_score=round(sentiment_score, 4),
        earnings_score=round(earnings_score, 4),
        composite_score=round(composite, 4),
        decision=decision,
        rationale=". ".join(reasons),
        stop_price=round(stop_price, 2),
        target_price=round(target_price, 2),
        next_earnings_date=bundle.upcoming_report_date if bundle else None,
        headline=headline,
    )
