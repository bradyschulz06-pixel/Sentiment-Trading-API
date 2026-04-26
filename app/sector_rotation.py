"""Sector analysis and rotation strategies for enhanced profit capability."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List


# Sector classification for major stocks
SECTOR_MAPPING = {
    # Technology
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "GOOG": "Technology",
    "META": "Technology",
    "NVDA": "Technology",
    "AMD": "Technology",
    "INTC": "Technology",
    "CSCO": "Technology",
    "ORCL": "Technology",
    "IBM": "Technology",

    # Financials
    "JPM": "Financials",
    "BAC": "Financials",
    "WFC": "Financials",
    "C": "Financials",
    "GS": "Financials",
    "MS": "Financials",
    "BLK": "Financials",
    "SCHW": "Financials",
    "AXP": "Financials",

    # Healthcare
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "PFE": "Healthcare",
    "ABBV": "Healthcare",
    "MRK": "Healthcare",
    "T": "Healthcare",
    "LLY": "Healthcare",
    "ABT": "Healthcare",
    "DHR": "Healthcare",

    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "TJX": "Consumer Discretionary",

    # Consumer Staples
    "PG": "Consumer Staples",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "COST": "Consumer Staples",
    "WMT": "Consumer Staples",
    "PM": "Consumer Staples",
    "MO": "Consumer Staples",
    "CL": "Consumer Staples",

    # Energy
    "XOM": "Energy",
    "CVX": "Energy",
    "COP": "Energy",
    "SLB": "Energy",
    "EOG": "Energy",
    "PXD": "Energy",
    "MPC": "Energy",
    "PSX": "Energy",

    # Industrials
    "CAT": "Industrials",
    "GE": "Industrials",
    "HON": "Industrials",
    "UPS": "Industrials",
    "RTX": "Industrials",
    "BA": "Industrials",
    "DE": "Industrials",
    "LMT": "Industrials",

    # Utilities
    "NEE": "Utilities",
    "DUK": "Utilities",
    "SO": "Utilities",
    "D": "Utilities",
    "EXC": "Utilities",
    "AEP": "Utilities",
    "SRE": "Utilities",

    # Real Estate
    "AMT": "Real Estate",
    "PLD": "Real Estate",
    "CCI": "Real Estate",
    "EQIX": "Real Estate",
    "PSA": "Real Estate",

    # Materials
    "LIN": "Materials",
    "APD": "Materials",
    "SHW": "Materials",
    "DOW": "Materials",
    "DD": "Materials",
    "FCX": "Materials",
    "NEM": "Materials",

    # Communication Services
    "VZ": "Communication Services",
    "T": "Communication Services",
    "DIS": "Communication Services",
    "CMCSA": "Communication Services",
    "NFLX": "Communication Services",
    "GOOGL": "Communication Services",  # Also in Tech
    "GOOG": "Communication Services",   # Also in Tech
    "META": "Communication Services",  # Also in Tech
}


@dataclass(slots=True)
class SectorPerformance:
    """Performance metrics for a sector."""
    sector_name: str
    symbols: List[str]
    avg_momentum_score: float
    avg_earnings_score: float
    avg_composite_score: float
    total_signals: int
    buy_signals: int
    strength_rank: int  # 1 = strongest, higher = weaker
    rotation_signal: str  # "overweight", "neutral", "underweight"


def get_sector_for_symbol(symbol: str) -> str:
    """Get the sector classification for a given symbol."""
    return SECTOR_MAPPING.get(symbol.upper(), "Other")


def analyze_sector_performance(
    signals: List[dict],
    lookback_days: int = 30
) -> Dict[str, SectorPerformance]:
    """
    Analyze performance across sectors from recent signals.

    Args:
        signals: List of signal dictionaries with symbol, momentum_score, earnings_score, composite_score, decision
        lookback_days: Number of days to consider for analysis

    Returns:
        Dictionary mapping sector names to SectorPerformance objects
    """
    sector_data = defaultdict(lambda: {
        "symbols": [],
        "momentum_scores": [],
        "earnings_scores": [],
        "composite_scores": [],
        "buy_signals": 0,
        "total_signals": 0
    })

    # Aggregate data by sector
    for signal in signals:
        symbol = signal.get("symbol", "")
        sector = get_sector_for_symbol(symbol)

        sector_data[sector]["symbols"].append(symbol)
        sector_data[sector]["momentum_scores"].append(signal.get("momentum_score", 0.0))
        sector_data[sector]["earnings_scores"].append(signal.get("earnings_score", 0.0))
        sector_data[sector]["composite_scores"].append(signal.get("composite_score", 0.0))
        sector_data[sector]["total_signals"] += 1

        if signal.get("decision") == "buy":
            sector_data[sector]["buy_signals"] += 1

    # Calculate sector performance metrics
    sector_performances = {}
    for sector, data in sector_data.items():
        if not data["total_signals"]:
            continue

        avg_momentum = sum(data["momentum_scores"]) / len(data["momentum_scores"]) if data["momentum_scores"] else 0.0
        avg_earnings = sum(data["earnings_scores"]) / len(data["earnings_scores"]) if data["earnings_scores"] else 0.0
        avg_composite = sum(data["composite_scores"]) / len(data["composite_scores"]) if data["composite_scores"] else 0.0

        sector_performances[sector] = SectorPerformance(
            sector_name=sector,
            symbols=data["symbols"],
            avg_momentum_score=round(avg_momentum, 4),
            avg_earnings_score=round(avg_earnings, 4),
            avg_composite_score=round(avg_composite, 4),
            total_signals=data["total_signals"],
            buy_signals=data["buy_signals"],
            strength_rank=0,  # Will be calculated below
            rotation_signal="neutral"  # Will be calculated below
        )

    # Rank sectors by composite score
    sorted_sectors = sorted(
        sector_performances.values(),
        key=lambda x: x.avg_composite_score,
        reverse=True
    )

    for rank, sector_perf in enumerate(sorted_sectors, start=1):
        sector_perf.strength_rank = rank

        # Determine rotation signal
        if rank <= 2:  # Top 2 sectors
            sector_perf.rotation_signal = "overweight"
        elif rank >= len(sorted_sectors) - 1:  # Bottom sector
            sector_perf.rotation_signal = "underweight"
        else:
            sector_perf.rotation_signal = "neutral"

    return sector_performances


def get_sector_rotation_weights(
    sector_performances: Dict[str, SectorPerformance],
    base_weight: float = 0.10,
    overweight_boost: float = 0.05,
    underweight_penalty: float = 0.03
) -> Dict[str, float]:
    """
    Calculate position sizing weights based on sector rotation signals.

    Args:
        sector_performances: Dictionary of sector performance analysis
        base_weight: Base weight for neutral sectors
        overweight_boost: Additional weight for overweight sectors
        underweight_penalty: Weight reduction for underweight sectors

    Returns:
        Dictionary mapping sector names to recommended position weights
    """
    weights = {}

    for sector, perf in sector_performances.items():
        if perf.rotation_signal == "overweight":
            weights[sector] = base_weight + overweight_boost
        elif perf.rotation_signal == "underweight":
            weights[sector] = max(0.01, base_weight - underweight_penalty)
        else:  # neutral
            weights[sector] = base_weight

    return weights


def filter_signals_by_sector_rotation(
    signals: List[dict],
    sector_performances: Dict[str, SectorPerformance],
    max_underweight_signals: int = 1
) -> List[dict]:
    """
    Filter signals based on sector rotation preferences.

    Prioritizes signals from overweight and neutral sectors,
    limits signals from underweight sectors.

    Args:
        signals: List of signal dictionaries
        sector_performances: Dictionary of sector performance analysis
        max_underweight_signals: Maximum number of signals to keep from underweight sectors

    Returns:
        Filtered list of signals prioritized by sector strength
    """
    overweight_signals = []
    neutral_signals = []
    underweight_signals = []

    for signal in signals:
        symbol = signal.get("symbol", "")
        sector = get_sector_for_symbol(symbol)
        sector_perf = sector_performances.get(sector)

        if not sector_perf:
            neutral_signals.append(signal)
        elif sector_perf.rotation_signal == "overweight":
            overweight_signals.append(signal)
        elif sector_perf.rotation_signal == "neutral":
            neutral_signals.append(signal)
        else:  # underweight
            underweight_signals.append(signal)

    # Prioritize: overweight > neutral > limited underweight
    filtered_signals = (
        overweight_signals +
        neutral_signals +
        underweight_signals[:max_underweight_signals]
    )

    return filtered_signals


def get_sector_diversification_score(
    current_positions: List[dict],
    target_sectors: List[str] = None
) -> float:
    """
    Calculate how well-diversified current positions are across sectors.

    Args:
        current_positions: List of position dictionaries with symbol field
        target_sectors: Optional list of sectors to target (default: all major sectors)

    Returns:
        Diversification score from 0.0 (poorly diversified) to 1.0 (well diversified)
    """
    if not current_positions:
        return 1.0  # No positions means perfectly diversified (vacuously true)

    if target_sectors is None:
        target_sectors = [
            "Technology", "Financials", "Healthcare", "Consumer Discretionary",
            "Consumer Staples", "Energy", "Industrials", "Utilities"
        ]

    # Count sectors represented in current positions
    represented_sectors = set()
    for position in current_positions:
        symbol = position.get("symbol", "")
        sector = get_sector_for_symbol(symbol)
        represented_sectors.add(sector)

    # Calculate diversification ratio
    diversification_ratio = len(represented_sectors) / len(target_sectors)

    return min(1.0, diversification_ratio)