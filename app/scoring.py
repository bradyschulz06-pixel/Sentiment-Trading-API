from __future__ import annotations

from datetime import date, datetime, timezone
import math

from app.models import EarningsBundle, PositionSnapshot, PriceBar, SignalScore
from app.sentiment import clamp


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


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """RSI using Wilder smoothing. Returns 50 (neutral) when there is not enough data."""
    if len(closes) < period + 1:
        return 50.0
    changes = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(0.0, c) for c in changes[:period]]
    losses = [max(0.0, -c) for c in changes[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    for change in changes[period:]:
        avg_gain = (avg_gain * (period - 1) + max(0.0, change)) / period
        avg_loss = (avg_loss * (period - 1) + max(0.0, -change)) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))


def _volume_ratio(bars: list[PriceBar], period: int = 20) -> float:
    """Latest bar volume relative to the prior `period`-bar average. Returns 1.0 when insufficient data."""
    if len(bars) < period + 1:
        return 1.0
    avg_vol = sum(bar.volume for bar in bars[-(period + 1):-1]) / period
    return bars[-1].volume / avg_vol if avg_vol > 0 else 1.0


def _ema(values: list[float], period: int) -> list[float]:
    """Exponential moving average seeded by the first-period SMA. Returns [] when insufficient data."""
    if len(values) < period:
        return []
    k = 2.0 / (period + 1)
    ema_vals = [sum(values[:period]) / period]
    for v in values[period:]:
        ema_vals.append(ema_vals[-1] * (1 - k) + v * k)
    return ema_vals


def _compute_macd(
    closes: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[float, float, float]:
    """MACD line, signal line, and histogram for the most recent bar.
    Returns (0.0, 0.0, 0.0) when there is insufficient data."""
    if len(closes) < slow + signal:
        return 0.0, 0.0, 0.0
    ema_fast = _ema(closes, fast)
    ema_slow = _ema(closes, slow)
    # ema_fast covers closes[fast-1:], ema_slow covers closes[slow-1:].
    # Align by dropping the leading portion of ema_fast so both series start at closes[slow-1:].
    offset = slow - fast
    macd_line = [f - s for f, s in zip(ema_fast[offset:], ema_slow)]
    if len(macd_line) < signal:
        return 0.0, 0.0, 0.0
    signal_line = _ema(macd_line, signal)
    if not signal_line:
        return 0.0, 0.0, 0.0
    macd_current = macd_line[-1]
    signal_current = signal_line[-1]
    return macd_current, signal_current, macd_current - signal_current


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


def compute_conviction_sizing(
    composite_score: float,
    base_position_pct: float = 0.08,
    min_scalar: float = 0.75,
    max_scalar: float = 1.25
) -> float:
    """
    Scale position size based on signal conviction (composite score).

    Higher conviction scores get larger allocations, bounded by min/max scalars.
    """
    # Normalize score to 0-1 range for sizing
    normalized_score = max(0.0, min(1.0, composite_score))

    # Linear scaling from base position
    conviction_scalar = min_scalar + (normalized_score * (max_scalar - min_scalar))

    return base_position_pct * conviction_scalar


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


def compute_momentum_score(bars: list[PriceBar]) -> tuple[float, dict]:
    closes = [bar.close for bar in bars]
    current = closes[-1]
    sma20 = _average(closes[-20:])
    sma50 = _average(closes[-50:]) if len(closes) >= 50 else _average(closes)
    sma200 = _average(closes[-200:]) if len(closes) >= 200 else None
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
    raw_score = clamp((normalized_21 * 0.45) + (normalized_63 * 0.35) + (trend * 0.20))
    # MACD bonus: histogram normalised by 2× ATR so the ±0.10 cap requires a historically large
    # histogram reading — avoids over-weighting MACD on every small divergence.
    _, _, macd_histogram = _compute_macd(closes)
    atr = _compute_atr(bars)
    macd_bonus = clamp(macd_histogram / max(atr * 2.0, 1e-6), -0.10, 0.10)
    score = clamp(raw_score + macd_bonus)
    rsi = _compute_rsi(closes)
    metrics = {
        "current": current,
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "ret21": ret21,
        "ret63": ret63,
        "rsi": rsi,
        "macd_histogram": macd_histogram,
    }
    return score, metrics


def compute_earnings_score(bundle: EarningsBundle | None, today: date, buffer_days: int) -> tuple[float, bool]:
    if bundle is None or bundle.surprise_pct is None:
        return 0.0, False
    reported_date = _parse_iso_date(bundle.reported_date)
    days_since_report = (today - reported_date).days if reported_date else 45
    recency_decay = math.exp(-(max(days_since_report, 0) / 30.0))
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


def compute_profit_levels(
    entry_price: float,
    target_price: float,
    num_levels: int = 3
) -> list[float]:
    """
    Calculate price levels for multi-level profit taking.

    Distributes profit targets evenly between entry and final target,
    allowing traders to lock in gains at multiple levels.
    """
    if num_levels < 1:
        return []

    profit_levels = []
    price_range = target_price - entry_price

    for i in range(1, num_levels + 1):
        # Distribute levels evenly (e.g., 33%, 67%, 100% of range)
        level_price = entry_price + (price_range * (i / num_levels))
        profit_levels.append(round(level_price, 2))

    return profit_levels


def compute_trailing_stop(
    position: PositionSnapshot,
    current_price: float,
    bars: list[PriceBar],
    trailing_pct: float = 0.06,
    activation_pct: float = 0.03
) -> float:
    """
    Compute a trailing stop price for an existing position.

    The trailing stop activates once the position is up by activation_pct,
    then trails the highest price seen by trailing_pct.
    """
    if position.avg_entry_price <= 0:
        return current_price * (1.0 - trailing_pct)

    unrealized_pnl = (current_price - position.avg_entry_price) / position.avg_entry_price

    # Only activate trailing stop if we're in profit
    if unrealized_pnl < activation_pct:
        return position.avg_entry_price * (1.0 - 0.08)  # Use original stop

    # Calculate the highest price since entry (using recent bars)
    closes = [bar.close for bar in bars]
    entry_index = max(0, len(closes) - 30)  # Look back up to 30 bars
    recent_highs = [max(closes[entry_index:])]
    highest_since_entry = max(recent_highs)

    # Trail from the high
    trailing_stop = highest_since_entry * (1.0 - trailing_pct)

    # Never trail below the original stop
    original_stop = position.avg_entry_price * (1.0 - 0.08)
    return max(trailing_stop, original_stop)


def build_signal(
    symbol: str,
    bars: list[PriceBar],
    bundle: EarningsBundle | None,
    threshold: float,
    stop_loss_pct: float,
    upcoming_earnings_buffer_days: int,
    position: PositionSnapshot | None = None,
    today: date | None = None,
) -> SignalScore:
    momentum_score, metrics = compute_momentum_score(bars)
    earnings_score, blocked_for_earnings = compute_earnings_score(
        bundle,
        today=today or datetime.now(timezone.utc).date(),
        buffer_days=upcoming_earnings_buffer_days,
    )
    # Enhanced composite: combine momentum and earnings for stronger signals
    # Momentum provides timing, earnings provides fundamental catalyst
    composite = (momentum_score * 0.40) + (earnings_score * 0.60)
    current_price = metrics["current"]
    rsi = metrics.get("rsi", 50.0)
    stop_price = current_price * (1.0 - stop_loss_pct)
    # Enhanced target: use risk-reward ratio based on volatility
    atr = _compute_atr(bars)
    volatility_adjusted_target = stop_loss_pct * 2.0  # 2:1 risk-reward minimum
    # Add momentum bonus for strong trends
    if momentum_score > 0.3:
        volatility_adjusted_target *= 1.3  # Extend targets for strong momentum
    target_price = current_price * (1.0 + volatility_adjusted_target)
    if position is not None and position.avg_entry_price > 0:
        # Use trailing stop for existing positions
        trailing_stop = compute_trailing_stop(position, current_price, bars)
        stop_price = max(stop_price, trailing_stop)
    reasons: list[str] = []
    trend_ok = current_price > metrics["sma20"] > metrics["sma50"]
    if trend_ok:
        reasons.append("trend is above the 20/50-day moving averages")
    else:
        reasons.append("trend is not clean enough yet")
    if earnings_score > 0.15:
        reasons.append("recent earnings surprise is supportive")
    elif earnings_score < -0.15:
        reasons.append("earnings read-through is weak")
    if blocked_for_earnings:
        reasons.append("an earnings report is too close to open a new swing trade safely")
    if rsi > 75:
        reasons.append(f"RSI is elevated ({rsi:.0f}) — waiting for momentum to cool before entry")
    elif rsi < 35:
        reasons.append(f"RSI shows short-term weakness ({rsi:.0f})")
    # Enhanced decision logic with multiple confirmation factors
    decision = "watch"
    if position is not None:
        # Exit logic for existing positions
        unrealized_pnl = (current_price - position.avg_entry_price) / position.avg_entry_price
        # Momentum-based exit: if momentum turns negative while in profit, take gains
        if momentum_score < -0.2 and unrealized_pnl > 0.02:
            decision = "sell"
            reasons.append("momentum has turned negative - taking profits")
        elif current_price <= stop_price or composite < 0.10:
            decision = "sell"
            reasons.append("existing position has lost its edge")
        else:
            decision = "hold"
            reasons.append("existing position still fits the model")
    elif composite >= threshold and not blocked_for_earnings:
        # Enhanced entry requirements
        if rsi > 75.0:
            decision = "watch"
            reasons.append("RSI too elevated for entry")
        elif momentum_score < 0.1:
            decision = "watch"
            reasons.append("momentum not strong enough for entry")
        else:
            decision = "buy"
            reasons.append("strong earnings and momentum alignment")
    # Calculate conviction-based position sizing
    conviction_sizing = compute_conviction_sizing(composite)

    # Calculate multi-level profit targets
    profit_levels = compute_profit_levels(current_price, target_price, num_levels=3)

    return SignalScore(
        symbol=symbol,
        price=round(current_price, 2),
        momentum_score=round(momentum_score, 4),
        sentiment_score=0.0,
        earnings_score=round(earnings_score, 4),
        composite_score=round(composite, 4),
        decision=decision,
        rationale=". ".join(reasons),
        stop_price=round(stop_price, 2),
        target_price=round(target_price, 2),
        next_earnings_date=bundle.upcoming_report_date if bundle else None,
        headline="",
        conviction_sizing=round(conviction_sizing, 4),
        profit_levels=profit_levels,
    )
