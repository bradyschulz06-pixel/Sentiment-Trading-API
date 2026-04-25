from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class PriceBar:
    symbol: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class NewsItem:
    symbol: str
    headline: str
    summary: str
    content: str
    source: str
    url: str
    published_at: str
    sentiment: float = 0.0


@dataclass(slots=True)
class EarningsBundle:
    symbol: str
    fiscal_date_ending: str | None = None
    reported_date: str | None = None
    reported_eps: float | None = None
    estimated_eps: float | None = None
    surprise: float | None = None
    surprise_pct: float | None = None
    report_time: str | None = None
    transcript_sentiment: float = 0.0
    transcript_excerpt: str = ""
    upcoming_report_date: str | None = None


@dataclass(slots=True)
class PositionSnapshot:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_plpc: float
    side: str = "long"


@dataclass(slots=True)
class SignalScore:
    symbol: str
    price: float
    momentum_score: float
    sentiment_score: float
    earnings_score: float
    composite_score: float
    decision: str
    rationale: str
    stop_price: float
    target_price: float
    next_earnings_date: str | None = None
    headline: str = ""


@dataclass(slots=True)
class TradeIntent:
    symbol: str
    side: str
    qty: int
    notional: float
    reason: str
    status: str = "planned"
    broker_order_id: str = ""


@dataclass(slots=True)
class EngineRunResult:
    status: str
    summary: str
    started_at: str
    completed_at: str
    trigger: str
    signals: list[SignalScore] = field(default_factory=list)
    news_items: list[NewsItem] = field(default_factory=list)
    positions: list[PositionSnapshot] = field(default_factory=list)
    trades: list[TradeIntent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""


@dataclass(slots=True)
class BacktestTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    qty: int
    gross_pnl: float
    pnl: float
    commissions_paid: float
    return_pct: float
    hold_days: int
    exit_reason: str


@dataclass(slots=True)
class BacktestPoint:
    date: str
    equity: float
    benchmark_equity: float
    cash: float
    positions: int


@dataclass(slots=True)
class BacktestResult:
    status: str
    summary: str
    start_date: str
    end_date: str
    period_days: int
    starting_capital: float
    ending_equity: float
    total_return_pct: float
    benchmark_symbol: str
    benchmark_return_pct: float
    outperformance_pct: float
    max_drawdown_pct: float
    universe_preset: str
    universe_label: str
    signal_threshold: float
    factor_momentum_weight: float
    factor_sentiment_weight: float
    factor_earnings_weight: float
    total_trades: int
    win_rate_pct: float
    average_trade_return_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    chart_points: str
    benchmark_chart_points: str
    benchmark_ending_equity: float
    commission_per_order: float
    slippage_bps: float
    max_hold_days: int
    min_hold_days: int
    trailing_stop_pct: float
    trailing_arm_pct: float
    take_profit_pct: float
    notes: list[str] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)
    daily_points: list[BacktestPoint] = field(default_factory=list)


@dataclass(slots=True)
class WalkForwardWindowResult:
    period_days: int
    total_return_pct: float
    benchmark_return_pct: float
    outperformance_pct: float
    max_drawdown_pct: float
    total_trades: int


@dataclass(slots=True)
class WalkForwardCandidateResult:
    label: str
    universe_preset: str
    universe_label: str
    factor_profile_name: str
    signal_threshold: float
    factor_momentum_weight: float
    factor_sentiment_weight: float
    factor_earnings_weight: float
    stability_score: float
    average_return_pct: float
    average_outperformance_pct: float
    worst_outperformance_pct: float
    positive_window_ratio: float
    benchmark_win_ratio: float
    average_drawdown_pct: float
    outperformance_stddev_pct: float
    total_trades: int
    windows: list[WalkForwardWindowResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class WalkForwardResult:
    status: str
    summary: str
    benchmark_symbol: str
    starting_capital: float
    windows_tested: list[int] = field(default_factory=list)
    thresholds_tested: list[float] = field(default_factory=list)
    candidates: list[WalkForwardCandidateResult] = field(default_factory=list)
    best_candidate: WalkForwardCandidateResult | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class JournalClosedTrade:
    symbol: str
    qty: int
    entry_at: str
    exit_at: str
    entry_price: float
    exit_price: float
    pnl: float
    return_pct: float
    hold_days: int | None
    dominant_factor: str
    entry_composite_score: float
    entry_momentum_score: float
    entry_sentiment_score: float
    entry_earnings_score: float
    entry_rationale: str
    entry_headline: str


@dataclass(slots=True)
class JournalOpenPosition:
    symbol: str
    qty: float
    avg_entry_price: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_plpc: float
    hold_days: int | None
    dominant_factor: str
    entry_composite_score: float
    entry_momentum_score: float
    entry_sentiment_score: float
    entry_earnings_score: float
    entry_rationale: str
    entry_headline: str


@dataclass(slots=True)
class JournalFactorSummary:
    factor_name: str
    total_trades: int
    win_rate_pct: float
    average_return_pct: float
    total_pnl: float


@dataclass(slots=True)
class JournalOrderActivity:
    symbol: str
    side: str
    qty: int
    filled_price: float
    status: str
    submitted_at: str
    filled_at: str


@dataclass(slots=True)
class PaperJournalResult:
    status: str
    summary: str
    total_closed_trades: int
    win_rate_pct: float
    realized_pnl: float
    average_return_pct: float
    open_positions_count: int
    unrealized_pnl: float
    realized_plus_unrealized_pnl: float
    best_trade_pnl: float
    worst_trade_pnl: float
    closed_trades: list[JournalClosedTrade] = field(default_factory=list)
    open_positions: list[JournalOpenPosition] = field(default_factory=list)
    factor_summaries: list[JournalFactorSummary] = field(default_factory=list)
    recent_orders: list[JournalOrderActivity] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
