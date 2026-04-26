from __future__ import annotations

from dataclasses import dataclass, field

from app.config import Settings


SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "GOOGL": "Technology", "META": "Technology", "AMZN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary", "JPM": "Financials", "BAC": "Financials",
    "GS": "Financials", "MA": "Financials", "V": "Financials",
    "UNH": "Health Care", "LLY": "Health Care", "ABBV": "Health Care",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "CAT": "Industrials", "HON": "Industrials", "RTX": "Industrials",
    "LIN": "Materials", "APD": "Materials",
    "NEE": "Utilities", "DUK": "Utilities",
    "PLD": "Real Estate", "AMT": "Real Estate",
    "COST": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
}
_DEFAULT_SECTOR = "Other"


@dataclass
class RiskState:
    """Mutable intra-run risk counters. Create one per run_once() call."""
    stops_today: int = 0
    trades_by_symbol: dict[str, int] = field(default_factory=dict)

    def record_stop(self) -> None:
        self.stops_today += 1

    def record_trade(self, symbol: str) -> None:
        self.trades_by_symbol[symbol] = self.trades_by_symbol.get(symbol, 0) + 1


@dataclass
class RiskVerdict:
    approved: bool
    reason: str = ""


class RiskGate:
    """Independent veto layer between signal scoring and trade execution."""

    def __init__(self, settings: Settings) -> None:
        self._s = settings

    def evaluate_buy(
        self,
        symbol: str,
        *,
        daily_pnl_pct: float,
        state: RiskState,
        current_sector_counts: dict[str, int],
    ) -> RiskVerdict:
        s = self._s

        if daily_pnl_pct < -s.max_daily_loss_pct:
            return RiskVerdict(False, f"Daily loss limit hit ({daily_pnl_pct:.1%})")

        if state.stops_today >= 3:
            return RiskVerdict(False, "Circuit breaker: ≥3 hard stops today")

        if state.trades_by_symbol.get(symbol, 0) >= s.max_trades_per_symbol_per_day:
            return RiskVerdict(False, f"Trade frequency limit for {symbol}")

        sector = SECTOR_MAP.get(symbol, _DEFAULT_SECTOR)
        if current_sector_counts.get(sector, 0) >= s.max_positions_per_sector:
            return RiskVerdict(False, f"Sector cap reached for {sector}")

        return RiskVerdict(True)
