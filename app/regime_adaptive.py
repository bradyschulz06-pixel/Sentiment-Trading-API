"""Regime-adaptive parameters for dynamic strategy adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class MarketRegime(Enum):
    """Market regime classification."""
    BULL_VOLATILE = "bull_volatile"      # Strong uptrend with high volatility
    BULL_STABLE = "bull_stable"          # Strong uptrend with low volatility
    BULL_NEUTRAL = "bull_neutral"        # Moderate uptrend
    NEUTRAL = "neutral"                  # Sideways/choppy market
    BEAR_VOLATILE = "bear_volatile"      # Strong downtrend with high volatility
    BEAR_STABLE = "bear_stable"          # Strong downtrend with low volatility
    CRISIS = "crisis"                    # Market crash conditions


@dataclass(slots=True)
class RegimeParameters:
    """Strategy parameters optimized for specific market regimes."""
    regime: MarketRegime
    signal_threshold: float
    max_positions: int
    position_size_multiplier: float
    stop_loss_multiplier: float
    target_multiplier: float
    momentum_weight: float
    earnings_weight: float
    trailing_stop_enabled: bool
    trailing_stop_pct: float
    description: str


# Default regime-specific parameters
REGIME_PARAMETERS: Dict[MarketRegime, RegimeParameters] = {
    MarketRegime.BULL_VOLATILE: RegimeParameters(
        regime=MarketRegime.BULL_VOLATILE,
        signal_threshold=0.25,           # Lower threshold in volatile bull
        max_positions=5,                 # More positions to spread risk
        position_size_multiplier=0.8,   # Smaller positions due to volatility
        stop_loss_multiplier=1.2,       # Wider stops to avoid whipsaws
        target_multiplier=1.5,          # Higher targets for big moves
        momentum_weight=0.50,            # Emphasize momentum in volatile bull
        earnings_weight=0.50,
        trailing_stop_enabled=True,
        trailing_stop_pct=0.08,         # Wider trailing stops
        description="Strong uptrend with high volatility - wider stops, momentum focus"
    ),

    MarketRegime.BULL_STABLE: RegimeParameters(
        regime=MarketRegime.BULL_STABLE,
        signal_threshold=0.30,           # Standard threshold
        max_positions=4,                 # Standard position count
        position_size_multiplier=1.1,   # Slightly larger positions
        stop_loss_multiplier=0.9,       # Tighter stops in stable market
        target_multiplier=1.3,          # Standard targets
        momentum_weight=0.40,            # Balanced approach
        earnings_weight=0.60,
        trailing_stop_enabled=True,
        trailing_stop_pct=0.05,         # Standard trailing stops
        description="Strong stable uptrend - balanced approach with quality focus"
    ),

    MarketRegime.BULL_NEUTRAL: RegimeParameters(
        regime=MarketRegime.BULL_NEUTRAL,
        signal_threshold=0.35,           # Higher threshold for selectivity
        max_positions=3,                 # Fewer, higher-quality positions
        position_size_multiplier=1.0,   # Standard sizing
        stop_loss_multiplier=1.0,       # Standard stops
        target_multiplier=1.2,          # Conservative targets
        momentum_weight=0.35,            # More conservative
        earnings_weight=0.65,
        trailing_stop_enabled=True,
        trailing_stop_pct=0.06,
        description="Moderate uptrend - selective, quality-focused approach"
    ),

    MarketRegime.NEUTRAL: RegimeParameters(
        regime=MarketRegime.NEUTRAL,
        signal_threshold=0.40,           # High threshold for selectivity
        max_positions=2,                 # Very selective
        position_size_multiplier=0.9,   # Smaller positions
        stop_loss_multiplier=1.1,       # Slightly wider stops
        target_multiplier=1.1,          # Conservative targets
        momentum_weight=0.30,            # Earnings focus in neutral market
        earnings_weight=0.70,
        trailing_stop_enabled=True,
        trailing_stop_pct=0.07,
        description="Sideways market - very selective, earnings-focused"
    ),

    MarketRegime.BEAR_VOLATILE: RegimeParameters(
        regime=MarketRegime.BEAR_VOLATILE,
        signal_threshold=0.50,           # Very high threshold
        max_positions=1,                 # Maximum 1 position if any
        position_size_multiplier=0.5,   # Very small positions
        stop_loss_multiplier=1.5,       # Very wide stops
        target_multiplier=0.8,          # Quick profit taking
        momentum_weight=0.20,            # Minimal momentum weight
        earnings_weight=0.80,
        trailing_stop_enabled=False,     # No trailing stops in bear
        trailing_stop_pct=0.10,
        description="Volatile bear market - extremely selective, minimal exposure"
    ),

    MarketRegime.BEAR_STABLE: RegimeParameters(
        regime=MarketRegime.BEAR_STABLE,
        signal_threshold=0.45,           # High threshold
        max_positions=2,                 # Limited positions
        position_size_multiplier=0.7,   # Smaller positions
        stop_loss_multiplier=1.3,       # Wide stops
        target_multiplier=0.9,          # Conservative targets
        momentum_weight=0.25,            # Earnings focus
        earnings_weight=0.75,
        trailing_stop_enabled=False,
        trailing_stop_pct=0.08,
        description="Stable bear market - selective, defensive positioning"
    ),

    MarketRegime.CRISIS: RegimeParameters(
        regime=MarketRegime.CRISIS,
        signal_threshold=0.60,           # Extremely high threshold
        max_positions=0,                 # No new positions
        position_size_multiplier=0.0,   # No new sizing
        stop_loss_multiplier=2.0,       # Very wide stops for existing
        target_multiplier=0.5,          # Take any profit quickly
        momentum_weight=0.00,            # No momentum weight
        earnings_weight=1.00,            # Pure earnings focus
        trailing_stop_enabled=False,
        trailing_stop_pct=0.15,
        description="Crisis conditions - defensive mode, no new entries"
    ),
}


def detect_market_regime(
    spy_trend: float,           # SPY trend (positive = up, negative = down)
    spy_volatility: float,      # SPY volatility (annualized)
    vix_level: float,           # VIX level
    breadth: float,             # Market breadth (advance/decline ratio)
    recent_drawdown: float = 0.0  # Recent market drawdown
) -> MarketRegime:
    """
    Detect current market regime from multiple indicators.

    Args:
        spy_trend: SPY price trend (e.g., 20-day return)
        spy_volatility: Annualized volatility of SPY
        vix_level: Current VIX level
        breadth: Market breadth (advancing/declining issues ratio)
        recent_drawdown: Recent market drawdown from peak

    Returns:
        Detected MarketRegime
    """
    # Crisis detection
    if recent_drawdown < -0.20 or vix_level > 40:
        return MarketRegime.CRISIS

    # Bear market detection
    if spy_trend < -0.10:  # More than 10% decline
        if vix_level > 30 or spy_volatility > 0.35:
            return MarketRegime.BEAR_VOLATILE
        else:
            return MarketRegime.BEAR_STABLE

    # Bull market detection
    if spy_trend > 0.10:  # More than 10% gain
        if vix_level > 25 or spy_volatility > 0.30:
            return MarketRegime.BULL_VOLATILE
        elif spy_trend > 0.20:
            return MarketRegime.BULL_STABLE
        else:
            return MarketRegime.BULL_NEUTRAL

    # Neutral market
    return MarketRegime.NEUTRAL


def get_regime_parameters(
    current_regime: MarketRegime,
    base_parameters: Optional[Dict] = None
) -> RegimeParameters:
    """
    Get regime-specific parameters, with optional base parameter overrides.

    Args:
        current_regime: Detected market regime
        base_parameters: Optional base parameters to blend with regime parameters

    Returns:
        RegimeParameters optimized for current regime
    """
    regime_params = REGIME_PARAMETERS.get(current_regime)

    if regime_params is None:
        # Default to neutral if regime not found
        regime_params = REGIME_PARAMETERS[MarketRegime.NEUTRAL]

    # If base parameters provided, blend them with regime parameters
    if base_parameters:
        return blend_parameters(regime_params, base_parameters)

    return regime_params


def blend_parameters(
    regime_params: RegimeParameters,
    base_params: Dict,
    blend_factor: float = 0.7
) -> RegimeParameters:
    """
    Blend regime-specific parameters with base parameters.

    Args:
        regime_params: Regime-specific parameters
        base_params: Base parameters from configuration
        blend_factor: How much to weight regime params (0-1)

    Returns:
        Blended RegimeParameters
    """
    def blend_value(regime_value: float, base_value: float) -> float:
        return regime_value * blend_factor + base_value * (1 - blend_factor)

    return RegimeParameters(
        regime=regime_params.regime,
        signal_threshold=blend_value(
            regime_params.signal_threshold,
            base_params.get("signal_threshold", 0.30)
        ),
        max_positions=int(blend_value(
            float(regime_params.max_positions),
            float(base_params.get("max_positions", 4))
        )),
        position_size_multiplier=blend_value(
            regime_params.position_size_multiplier,
            base_params.get("position_size_multiplier", 1.0)
        ),
        stop_loss_multiplier=blend_value(
            regime_params.stop_loss_multiplier,
            base_params.get("stop_loss_multiplier", 1.0)
        ),
        target_multiplier=blend_value(
            regime_params.target_multiplier,
            base_params.get("target_multiplier", 1.0)
        ),
        momentum_weight=blend_value(
            regime_params.momentum_weight,
            base_params.get("momentum_weight", 0.40)
        ),
        earnings_weight=blend_value(
            regime_params.earnings_weight,
            base_params.get("earnings_weight", 0.60)
        ),
        trailing_stop_enabled=regime_params.trailing_stop_enabled,
        trailing_stop_pct=blend_value(
            regime_params.trailing_stop_pct,
            base_params.get("trailing_stop_pct", 0.06)
        ),
        description=f"Blended: {regime_params.description}"
    )


def apply_regime_adjustments(
    signal_threshold: float,
    max_positions: int,
    stop_loss_pct: float,
    target_multiplier: float,
    regime_params: RegimeParameters
) -> tuple[float, int, float, float]:
    """
    Apply regime-specific adjustments to base parameters.

    Args:
        signal_threshold: Base signal threshold
        max_positions: Base max positions
        stop_loss_pct: Base stop loss percentage
        target_multiplier: Base target multiplier
        regime_params: Regime-specific parameters

    Returns:
        Tuple of (adjusted_threshold, adjusted_max_positions, adjusted_stop_loss, adjusted_target)
    """
    adjusted_threshold = signal_threshold * regime_params.signal_threshold / 0.30  # Normalize to base
    adjusted_max_positions = min(max_positions, regime_params.max_positions)
    adjusted_stop_loss = stop_loss_pct * regime_params.stop_loss_multiplier
    adjusted_target = target_multiplier * regime_params.target_multiplier

    return (
        adjusted_threshold,
        adjusted_max_positions,
        adjusted_stop_loss,
        adjusted_target
    )


def get_regime_transition_guidance(
    old_regime: MarketRegime,
    new_regime: MarketRegime
) -> List[str]:
    """
    Get guidance for transitioning between market regimes.

    Args:
        old_regime: Previous market regime
        new_regime: New market regime

    Returns:
        List of guidance strings for the transition
    """
    guidance = []

    # Crisis transitions
    if new_regime == MarketRegime.CRISIS:
        guidance.append("CRISIS MODE: Immediately reduce exposure, tighten risk management")
        guidance.append("Consider closing all but strongest positions")
        guidance.append("Focus on capital preservation over returns")

    # Bear market transitions
    elif new_regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.BEAR_STABLE]:
        if old_regime not in [MarketRegime.BEAR_VOLATILE, MarketRegime.BEAR_STABLE, MarketRegime.CRISIS]:
            guidance.append("BEAR MARKET: Shift to defensive posture")
            guidance.append("Reduce position sizes and increase selectivity")
            guidance.append("Focus on earnings quality over momentum")

    # Bull market transitions
    elif new_regime in [MarketRegime.BULL_VOLATILE, MarketRegime.BULL_STABLE, MarketRegime.BULL_NEUTRAL]:
        if old_regime in [MarketRegime.BEAR_VOLATILE, MarketRegime.BEAR_STABLE, MarketRegime.CRISIS]:
            guidance.append("BULL MARKET: Can increase exposure gradually")
            guidance.append("Look for quality momentum names")
            guidance.append("Consider wider stops for volatile conditions")

    # Neutral market
    elif new_regime == MarketRegime.NEUTRAL:
        guidance.append("NEUTRAL MARKET: Be selective and patient")
        guidance.append("Focus on best risk/reward opportunities")
        guidance.append("Maintain defensive positioning")

    return guidance