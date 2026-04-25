from __future__ import annotations


DEFAULT_UNIVERSE_PRESET = "balanced_quality"

UNIVERSE_PRESETS = {
    "balanced_quality": {
        "label": "Balanced Quality",
        "description": "A tighter 16-name large-cap basket with strong liquidity and sector balance.",
        "symbols": [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "AVGO",
            "JPM",
            "V",
            "LLY",
            "COST",
            "WMT",
            "XOM",
            "CVX",
            "CAT",
            "GE",
        ],
    },
    "mega_cap_focus": {
        "label": "Mega Cap Focus",
        "description": "Concentrates on the biggest liquid names where news and earnings coverage are usually strongest.",
        "symbols": [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "AVGO",
            "TSLA",
            "NFLX",
            "AMD",
            "JPM",
            "V",
        ],
    },
    "sector_leaders": {
        "label": "Sector Leaders",
        "description": "Keeps breadth but trims weaker redundancy by using one to two leaders per major sector.",
        "symbols": [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOGL",
            "META",
            "JPM",
            "GS",
            "V",
            "LLY",
            "UNH",
            "COST",
            "WMT",
            "XOM",
            "CVX",
            "CAT",
            "GE",
            "ORCL",
        ],
    },
}


def get_universe_presets() -> dict[str, dict]:
    return UNIVERSE_PRESETS


def normalize_universe_preset(preset: str) -> str:
    return preset if preset in UNIVERSE_PRESETS else DEFAULT_UNIVERSE_PRESET


def get_universe(raw_value: str, preset: str = DEFAULT_UNIVERSE_PRESET) -> list[str]:
    if not raw_value.strip():
        normalized = normalize_universe_preset(preset)
        return UNIVERSE_PRESETS[normalized]["symbols"].copy()
    parsed = [item.strip().upper() for item in raw_value.split(",") if item.strip()]
    return sorted(set(parsed))


def get_union_for_presets(presets: list[str]) -> list[str]:
    symbols: set[str] = set()
    for preset in presets:
        normalized = normalize_universe_preset(preset)
        symbols.update(UNIVERSE_PRESETS[normalized]["symbols"])
    return sorted(symbols)
