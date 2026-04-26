from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import secrets


ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


_load_env_file(ROOT_DIR / ".env")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def _env_float_clamp(name: str, default: float, lo: float, hi: float) -> float:
    import warnings as _warnings
    val = _env_float(name, default)
    if not (lo <= val <= hi):
        _warnings.warn(
            f"Config: {name}={val} is outside [{lo}, {hi}]; clamping to valid range.",
            stacklevel=2,
        )
        return max(lo, min(hi, val))
    return val


def _normalize_alpaca_base_url(value: str, default: str) -> str:
    base_url = os.getenv(value, default).rstrip("/")
    if base_url.endswith("/v2"):
        return base_url[:-3]
    return base_url


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str
    environment: str
    host: str
    port: int
    debug: bool
    admin_username: str
    admin_password: str
    admin_password_hash: str
    session_secret: str
    alpaca_api_key: str
    alpaca_secret_key: str
    alpaca_trading_base_url: str
    alpaca_data_base_url: str
    alpha_vantage_api_key: str
    universe_preset: str
    universe_symbols: str
    lookback_days: int
    news_window_days: int
    earnings_lookup_limit: int
    upcoming_earnings_buffer_days: int
    backtest_min_bars: int
    max_positions: int
    max_position_pct: float
    stop_loss_pct: float
    signal_threshold: float
    factor_momentum_weight: float
    factor_sentiment_weight: float
    factor_earnings_weight: float
    market_regime_filter_enabled: bool
    market_regime_cautious_threshold_boost: float
    market_regime_cautious_max_positions_multiplier: float
    auto_trade_enabled: bool
    engine_run_interval_minutes: int
    backtest_benchmark_symbol: str
    backtest_commission_per_order: float
    backtest_slippage_bps: float
    backtest_max_hold_days: int
    backtest_min_hold_days: int
    backtest_trailing_stop_pct: float
    backtest_trailing_arm_pct: float
    backtest_take_profit_pct: float
    backtest_breakeven_arm_pct: float
    backtest_breakeven_floor_pct: float
    backtest_reentry_cooldown_days: int
    backtest_rolling_drawdown_window: int
    backtest_rolling_drawdown_limit: float
    conviction_sizing_enabled: bool
    conviction_sizing_min_scalar: float
    conviction_sizing_max_scalar: float
    alpha_vantage_requests_per_minute: int
    max_daily_loss_pct: float
    max_trades_per_symbol_per_day: int
    max_positions_per_sector: int
    db_path: Path

    @property
    def trading_configured(self) -> bool:
        return bool(self.alpaca_api_key and self.alpaca_secret_key)

    @property
    def earnings_configured(self) -> bool:
        return bool(self.alpha_vantage_api_key)

    @property
    def has_password_warning(self) -> bool:
        return not self.admin_password_hash and self.admin_password == "change-me-now"


def get_settings() -> Settings:
    db_path = Path(os.getenv("DB_PATH", ROOT_DIR / "trading.db"))
    return Settings(
        app_name=os.getenv("APP_NAME", "Northstar Swing Trader"),
        environment=os.getenv("ENVIRONMENT", "development"),
        host=os.getenv("HOST", "127.0.0.1"),
        port=_env_int("PORT", 8000),
        debug=_env_bool("DEBUG", True),
        admin_username=os.getenv("ADMIN_USERNAME", "admin"),
        admin_password=os.getenv("ADMIN_PASSWORD", "change-me-now"),
        admin_password_hash=os.getenv("ADMIN_PASSWORD_HASH", ""),
        session_secret=os.getenv("SESSION_SECRET", secrets.token_urlsafe(32)),
        alpaca_api_key=os.getenv("ALPACA_API_KEY", ""),
        alpaca_secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        alpaca_trading_base_url=_normalize_alpaca_base_url("ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets"),
        alpaca_data_base_url=os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets").rstrip("/"),
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY", ""),
        universe_preset=os.getenv("UNIVERSE_PRESET", "sector_leaders").strip(),
        universe_symbols=os.getenv("UNIVERSE_SYMBOLS", ""),
        lookback_days=_env_int("LOOKBACK_DAYS", 180),
        news_window_days=_env_int("NEWS_WINDOW_DAYS", 7),
        earnings_lookup_limit=_env_int("EARNINGS_LOOKUP_LIMIT", 8),
        upcoming_earnings_buffer_days=_env_int("UPCOMING_EARNINGS_BUFFER_DAYS", 2),
        backtest_min_bars=_env_int("BACKTEST_MIN_BARS", 0),
        max_positions=_env_int("MAX_POSITIONS", 4),
        max_position_pct=_env_float_clamp("MAX_POSITION_PCT", 0.08, 0.01, 0.50),
        stop_loss_pct=_env_float_clamp("STOP_LOSS_PCT", 0.07, 0.01, 0.30),
        signal_threshold=_env_float_clamp("SIGNAL_THRESHOLD", 0.30, 0.05, 0.95),
        factor_momentum_weight=_env_float_clamp("FACTOR_MOMENTUM_WEIGHT", 0.00, 0.0, 1.0),
        factor_sentiment_weight=_env_float_clamp("FACTOR_SENTIMENT_WEIGHT", 0.00, 0.0, 1.0),
        factor_earnings_weight=_env_float_clamp("FACTOR_EARNINGS_WEIGHT", 1.00, 0.0, 1.0),
        market_regime_filter_enabled=_env_bool("MARKET_REGIME_FILTER_ENABLED", True),
        market_regime_cautious_threshold_boost=_env_float_clamp("MARKET_REGIME_CAUTIOUS_THRESHOLD_BOOST", 0.04, 0.0, 0.50),
        market_regime_cautious_max_positions_multiplier=_env_float_clamp("MARKET_REGIME_CAUTIOUS_MAX_POSITIONS_MULTIPLIER", 0.50, 0.0, 1.0),
        auto_trade_enabled=_env_bool("AUTO_TRADE_ENABLED", False),
        engine_run_interval_minutes=_env_int("ENGINE_RUN_INTERVAL_MINUTES", 0),
        backtest_benchmark_symbol=os.getenv("BACKTEST_BENCHMARK_SYMBOL", "SPY").strip().upper() or "SPY",
        backtest_commission_per_order=_env_float("BACKTEST_COMMISSION_PER_ORDER", 1.00),
        backtest_slippage_bps=_env_float("BACKTEST_SLIPPAGE_BPS", 20.0),
        backtest_max_hold_days=_env_int("BACKTEST_MAX_HOLD_DAYS", 20),
        backtest_min_hold_days=_env_int("BACKTEST_MIN_HOLD_DAYS", 3),
        backtest_trailing_stop_pct=_env_float("BACKTEST_TRAILING_STOP_PCT", 0.00),
        backtest_trailing_arm_pct=_env_float("BACKTEST_TRAILING_ARM_PCT", 0.00),
        backtest_take_profit_pct=_env_float("BACKTEST_TAKE_PROFIT_PCT", 0.00),
        backtest_breakeven_arm_pct=_env_float_clamp("BACKTEST_BREAKEVEN_ARM_PCT", 0.00, 0.0, 0.20),
        backtest_breakeven_floor_pct=_env_float_clamp("BACKTEST_BREAKEVEN_FLOOR_PCT", 0.005, 0.0, 0.05),
        backtest_reentry_cooldown_days=_env_int("BACKTEST_REENTRY_COOLDOWN_DAYS", 3),
        backtest_rolling_drawdown_window=_env_int("BACKTEST_ROLLING_DRAWDOWN_WINDOW", 10),
        backtest_rolling_drawdown_limit=_env_float_clamp("BACKTEST_ROLLING_DRAWDOWN_LIMIT", 0.05, 0.01, 0.20),
        conviction_sizing_enabled=_env_bool("CONVICTION_SIZING_ENABLED", True),
        conviction_sizing_min_scalar=_env_float_clamp("CONVICTION_SIZING_MIN_SCALAR", 0.75, 0.10, 1.00),
        conviction_sizing_max_scalar=_env_float_clamp("CONVICTION_SIZING_MAX_SCALAR", 1.25, 1.00, 2.00),
        alpha_vantage_requests_per_minute=_env_int("ALPHA_VANTAGE_REQUESTS_PER_MINUTE", 5),
        max_daily_loss_pct=_env_float_clamp("MAX_DAILY_LOSS_PCT", 0.02, 0.001, 0.20),
        max_trades_per_symbol_per_day=_env_int("MAX_TRADES_PER_SYMBOL_PER_DAY", 1),
        max_positions_per_sector=_env_int("MAX_POSITIONS_PER_SECTOR", 2),
        db_path=db_path,
    )
