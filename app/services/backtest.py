from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

from app.config import Settings
from app.db import get_connection, initialize_database
from app.models import BacktestPoint, BacktestResult, BacktestTrade, EarningsBundle, PositionSnapshot, PriceBar
from app.scoring import _compute_rsi, build_signal, compute_position_vol_scalar, normalize_factor_weights
from app.services.alpaca import AlpacaService
from app.services.alpha_vantage import AlphaVantageService
from app.services.market_regime import apply_market_regime_to_signal, evaluate_market_regime
from app.universe import get_universe, get_universe_presets, normalize_universe_preset


@dataclass(slots=True)
class _OpenPosition:
    symbol: str
    qty: int
    entry_price: float
    entry_date: str
    peak_price: float
    commissions_paid: float


def _date_key(timestamp: str) -> str:
    return timestamp[:10]


def _find_latest_earnings(rows: list[EarningsBundle], as_of_date: str) -> EarningsBundle | None:
    latest: EarningsBundle | None = None
    for row in rows:
        if not row.reported_date or row.reported_date[:10] > as_of_date:
            break
        latest = row
    return latest


def _apply_slippage(price: float, side: str, slippage_bps: float) -> float:
    slip_multiplier = slippage_bps / 10_000.0
    if side == "buy":
        return price * (1.0 + slip_multiplier)
    return price * (1.0 - slip_multiplier)


def _series_to_points(values: list[float], low: float, high: float) -> str:
    if not values:
        return ""
    spread = max(high - low, 1.0)
    if len(values) == 1:
        return "0,60"
    coords = []
    for index, value in enumerate(values):
        x = (index / (len(values) - 1)) * 100
        y = 60 - (((value - low) / spread) * 60)
        coords.append(f"{x:.2f},{y:.2f}")
    return " ".join(coords)


def _comparison_chart_points(points: list[BacktestPoint]) -> tuple[str, str]:
    if not points:
        return "", ""
    strategy = [point.equity for point in points]
    benchmark = [point.benchmark_equity for point in points]
    low = min(strategy + benchmark)
    high = max(strategy + benchmark)
    return _series_to_points(strategy, low, high), _series_to_points(benchmark, low, high)


def _max_drawdown_pct(points: list[BacktestPoint]) -> float:
    peak = 0.0
    max_drawdown = 0.0
    for point in points:
        peak = max(peak, point.equity)
        if peak <= 0:
            continue
        drawdown = (point.equity - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)
    return max_drawdown


def _sharpe_ratio(daily_points: list[BacktestPoint], risk_free_annual: float = 0.05) -> float:
    """Annualized Sharpe ratio from daily equity curve. Returns 0.0 if fewer than 2 points."""
    if len(daily_points) < 2:
        return 0.0
    equities = [p.equity for p in daily_points]
    daily_rets = [(equities[i] - equities[i - 1]) / equities[i - 1] for i in range(1, len(equities)) if equities[i - 1] > 0]
    if not daily_rets:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = [r - rf_daily for r in daily_rets]
    mean_excess = sum(excess) / len(excess)
    variance = sum((r - mean_excess) ** 2 for r in excess) / max(1, len(excess) - 1)
    std = math.sqrt(variance)
    if std == 0:
        return 0.0
    return round((mean_excess / std) * math.sqrt(252), 4)


def _sortino_ratio(daily_points: list[BacktestPoint], risk_free_annual: float = 0.05) -> float:
    """Annualized Sortino ratio (penalises only downside deviation). Returns 0.0 if fewer than 2 points."""
    if len(daily_points) < 2:
        return 0.0
    equities = [p.equity for p in daily_points]
    daily_rets = [(equities[i] - equities[i - 1]) / equities[i - 1] for i in range(1, len(equities)) if equities[i - 1] > 0]
    if not daily_rets:
        return 0.0
    rf_daily = (1 + risk_free_annual) ** (1 / 252) - 1
    excess = [r - rf_daily for r in daily_rets]
    mean_excess = sum(excess) / len(excess)
    downside = [r for r in excess if r < 0]
    if not downside:
        return 0.0
    downside_variance = sum(r ** 2 for r in downside) / len(downside)
    downside_std = math.sqrt(downside_variance)
    if downside_std == 0:
        return 0.0
    return round((mean_excess / downside_std) * math.sqrt(252), 4)


def _holding_days(date_index: dict[str, int], start_date: str, end_date: str) -> int:
    return max(1, date_index[end_date] - date_index[start_date] + 1)


def _determine_exit_reason(
    *,
    current_price: float,
    position: _OpenPosition,
    hold_days: int,
    signal,
    settings: Settings,
    signal_threshold: float,
    max_hold_days: int,
    min_hold_days: int,
    trailing_stop_pct: float,
    trailing_arm_pct: float,
    take_profit_pct: float,
    rsi: float = 50.0,
    breakeven_arm_pct: float = 0.03,
    breakeven_floor_pct: float = 0.005,
) -> str | None:
    hard_stop_price = position.entry_price * (1.0 - settings.stop_loss_pct)
    trailing_stop_price = position.peak_price * (1.0 - trailing_stop_pct)
    take_profit_price = position.entry_price * (1.0 + take_profit_pct)
    trailing_armed = position.peak_price >= position.entry_price * (1.0 + trailing_arm_pct)
    unrealized_return = (current_price - position.entry_price) / max(position.entry_price, 1e-6)
    # Promote hard stop to breakeven floor once peak gain has ever exceeded the arm threshold.
    # Using peak_price (not current_price) makes the floor permanent once armed.
    breakeven_floor_price = position.entry_price * (1.0 + breakeven_floor_pct)
    peak_return = (position.peak_price - position.entry_price) / max(position.entry_price, 1e-6)
    if peak_return >= breakeven_arm_pct and hard_stop_price < breakeven_floor_price:
        hard_stop_price = breakeven_floor_price

    if current_price <= hard_stop_price:
        return "Hard stop hit."
    if hold_days >= min_hold_days and trailing_armed and current_price <= trailing_stop_price:
        return "Trailing stop hit."
    if hold_days >= min_hold_days and current_price >= take_profit_price:
        return "Profit target reached."
    if hold_days >= max_hold_days:
        return "Time stop reached."
    # Sell into strength: exit a profitable position when RSI reaches extreme overbought territory
    # to avoid giving back gains on a sharp reversal.
    if hold_days >= min_hold_days and rsi > 82 and unrealized_return >= 0.05:
        return "RSI overbought exit on profitable position."
    if hold_days >= min_hold_days and (signal.decision == "sell" or signal.composite_score < max(0.05, signal_threshold * 0.55)):
        return "Signal quality faded."
    return None


def simulate_backtest(
    settings: Settings,
    price_map: dict[str, list[PriceBar]],
    earnings_map: dict[str, list[EarningsBundle]],
    benchmark_symbol: str,
    period_days: int,
    starting_capital: float,
    *,
    universe_preset: str | None = None,
    signal_threshold: float | None = None,
    factor_momentum_weight: float | None = None,
    factor_sentiment_weight: float | None = None,
    factor_earnings_weight: float | None = None,
    commission_per_order: float | None = None,
    slippage_bps: float | None = None,
    max_hold_days: int | None = None,
    min_hold_days: int | None = None,
    trailing_stop_pct: float | None = None,
    trailing_arm_pct: float | None = None,
    take_profit_pct: float | None = None,
    breakeven_arm_pct: float | None = None,
    breakeven_floor_pct: float | None = None,
) -> BacktestResult:
    universe_preset = normalize_universe_preset(universe_preset or settings.universe_preset)
    universe_meta = get_universe_presets()[universe_preset]
    signal_threshold = settings.signal_threshold if signal_threshold is None else min(max(signal_threshold, 0.05), 0.95)
    factor_momentum_weight, factor_sentiment_weight, factor_earnings_weight = normalize_factor_weights(
        settings.factor_momentum_weight if factor_momentum_weight is None else factor_momentum_weight,
        settings.factor_sentiment_weight if factor_sentiment_weight is None else factor_sentiment_weight,
        settings.factor_earnings_weight if factor_earnings_weight is None else factor_earnings_weight,
    )
    commission_per_order = settings.backtest_commission_per_order if commission_per_order is None else max(0.0, commission_per_order)
    slippage_bps = settings.backtest_slippage_bps if slippage_bps is None else max(0.0, slippage_bps)
    max_hold_days = settings.backtest_max_hold_days if max_hold_days is None else max(2, max_hold_days)
    min_hold_days = settings.backtest_min_hold_days if min_hold_days is None else max(1, min_hold_days)
    trailing_stop_pct = settings.backtest_trailing_stop_pct if trailing_stop_pct is None else max(0.01, trailing_stop_pct)
    trailing_arm_pct = settings.backtest_trailing_arm_pct if trailing_arm_pct is None else max(0.0, trailing_arm_pct)
    take_profit_pct = settings.backtest_take_profit_pct if take_profit_pct is None else max(0.01, take_profit_pct)
    breakeven_arm_pct = settings.backtest_breakeven_arm_pct if breakeven_arm_pct is None else max(0.0, breakeven_arm_pct)
    breakeven_floor_pct = settings.backtest_breakeven_floor_pct if breakeven_floor_pct is None else max(0.0, breakeven_floor_pct)
    min_hold_days = min(min_hold_days, max_hold_days)

    benchmark_bars = price_map.get(benchmark_symbol, [])
    if len(benchmark_bars) < max(2, period_days):
        raise RuntimeError("Not enough benchmark price history to run the backtest.")

    ordered_dates = [_date_key(bar.timestamp) for bar in benchmark_bars][-period_days:]
    date_index = {stamp: index for index, stamp in enumerate(ordered_dates)}
    bar_lookup = {
        symbol: {_date_key(bar.timestamp): bar for bar in bars}
        for symbol, bars in price_map.items()
    }
    benchmark_lookup = bar_lookup[benchmark_symbol]

    open_positions: dict[str, _OpenPosition] = {}
    closed_trades: list[BacktestTrade] = []
    daily_points: list[BacktestPoint] = []
    regime_counts = {"supportive": 0, "cautious": 0, "risk_off": 0, "inactive": 0}
    cash = starting_capital
    benchmark_start = benchmark_bars[-period_days].close
    benchmark_shares = starting_capital / benchmark_start if benchmark_start else 0.0
    # Minimum bars a symbol must have before it is eligible for entry.
    # Config override (BACKTEST_MIN_BARS) wins; otherwise floor at 65 or period/3.
    min_bars = settings.backtest_min_bars if settings.backtest_min_bars > 0 else max(65, period_days // 3)

    for current_date in ordered_dates:
        candidate_signals = []
        current_prices: dict[str, float] = {}
        symbol_bars: dict[str, list[PriceBar]] = {}
        earnings_by_symbol: dict[str, EarningsBundle | None] = {}
        current_benchmark_price = benchmark_lookup[current_date].close
        benchmark_symbol_bars = [bar for bar in benchmark_bars if _date_key(bar.timestamp) <= current_date]

        for symbol, bars in price_map.items():
            if symbol == benchmark_symbol:
                continue
            filtered_bars = [bar for bar in bars if _date_key(bar.timestamp) <= current_date]
            if len(filtered_bars) < min_bars:
                continue

            current_bar = filtered_bars[-1]
            current_price = current_bar.close
            symbol_bars[symbol] = filtered_bars
            current_prices[symbol] = current_price
            earnings_by_symbol[symbol] = _find_latest_earnings(earnings_map.get(symbol, []), current_date)

        regime = evaluate_market_regime(settings, benchmark_symbol, benchmark_symbol_bars, symbol_bars)
        regime_counts[regime.label] = regime_counts.get(regime.label, 0) + 1
        effective_threshold = regime.effective_signal_threshold(signal_threshold)

        for symbol, filtered_bars in symbol_bars.items():
            current_price = current_prices[symbol]
            earnings_bundle = earnings_by_symbol.get(symbol)
            open_position = open_positions.get(symbol)
            position_snapshot = None
            if open_position is not None:
                position_snapshot = PositionSnapshot(
                    symbol=symbol,
                    qty=open_position.qty,
                    avg_entry_price=open_position.entry_price,
                    market_value=open_position.qty * current_price,
                    unrealized_plpc=((current_price - open_position.entry_price) / open_position.entry_price),
                )

            signal = build_signal(
                symbol=symbol,
                bars=filtered_bars,
                news_items=[],
                bundle=earnings_bundle,
                threshold=effective_threshold,
                stop_loss_pct=settings.stop_loss_pct,
                upcoming_earnings_buffer_days=0,
                momentum_weight=factor_momentum_weight,
                sentiment_weight=factor_sentiment_weight,
                earnings_weight=factor_earnings_weight,
                position=position_snapshot,
                today=date.fromisoformat(current_date),
            )
            signal = apply_market_regime_to_signal(signal, regime, signal_threshold)

            if open_position is not None:
                open_position.peak_price = max(open_position.peak_price, current_price)
                hold_days = _holding_days(date_index, open_position.entry_date, current_date)
                closes = [bar.close for bar in filtered_bars]
                exit_reason = _determine_exit_reason(
                    current_price=current_price,
                    position=open_position,
                    hold_days=hold_days,
                    signal=signal,
                    settings=settings,
                    signal_threshold=signal_threshold,
                    max_hold_days=max_hold_days,
                    min_hold_days=min_hold_days,
                    trailing_stop_pct=trailing_stop_pct,
                    trailing_arm_pct=trailing_arm_pct,
                    take_profit_pct=take_profit_pct,
                    rsi=_compute_rsi(closes),
                    breakeven_arm_pct=breakeven_arm_pct,
                    breakeven_floor_pct=breakeven_floor_pct,
                )
                if exit_reason:
                    exit_price = _apply_slippage(current_price, "sell", slippage_bps)
                    gross_proceeds = exit_price * open_position.qty
                    net_proceeds = gross_proceeds - commission_per_order
                    gross_pnl = (exit_price - open_position.entry_price) * open_position.qty
                    net_pnl = net_proceeds - (open_position.entry_price * open_position.qty) - open_position.commissions_paid
                    cash += net_proceeds
                    closed_trades.append(
                        BacktestTrade(
                            symbol=symbol,
                            entry_date=open_position.entry_date,
                            exit_date=current_date,
                            entry_price=round(open_position.entry_price, 2),
                            exit_price=round(exit_price, 2),
                            qty=open_position.qty,
                            gross_pnl=round(gross_pnl, 2),
                            pnl=round(net_pnl, 2),
                            commissions_paid=round(open_position.commissions_paid + commission_per_order, 2),
                            return_pct=round(net_pnl / ((open_position.entry_price * open_position.qty) + open_position.commissions_paid), 4),
                            hold_days=hold_days,
                            exit_reason=exit_reason,
                        )
                    )
                    del open_positions[symbol]
                continue

            if signal.decision == "buy":
                candidate_signals.append(signal)

        candidate_signals.sort(key=lambda item: item.composite_score, reverse=True)
        current_equity = cash + sum(position.qty * current_prices.get(symbol, position.entry_price) for symbol, position in open_positions.items())
        effective_max_positions = regime.effective_max_positions(settings.max_positions)
        open_slots = max(0, effective_max_positions - len(open_positions))
        base_budget = current_equity * settings.max_position_pct

        for signal in candidate_signals:
            if open_slots <= 0:
                break
            if signal.symbol in open_positions:
                continue
            raw_price = current_prices.get(signal.symbol, signal.price)
            entry_price = _apply_slippage(raw_price, "buy", slippage_bps)
            # Scale position budget by realized vol so high-vol stocks get smaller allocations.
            vol_scalar = compute_position_vol_scalar(symbol_bars.get(signal.symbol, []))
            if settings.conviction_sizing_enabled:
                threshold_range = max(1.0 - signal_threshold, 0.01)
                conviction = max(0.0, min(1.0, (signal.composite_score - signal_threshold) / threshold_range))
                conviction_scalar = (
                    settings.conviction_sizing_min_scalar
                    + (settings.conviction_sizing_max_scalar - settings.conviction_sizing_min_scalar) * conviction
                )
            else:
                conviction_scalar = 1.0
            budget_per_trade = min(
                base_budget * vol_scalar * conviction_scalar,
                base_budget * settings.conviction_sizing_max_scalar,
            )
            tradable_cash = max(0.0, min(budget_per_trade, cash - commission_per_order))
            qty = int(tradable_cash // entry_price)
            if qty < 1:
                continue
            total_cost = (entry_price * qty) + commission_per_order
            if total_cost > cash:
                continue
            cash -= total_cost
            open_positions[signal.symbol] = _OpenPosition(
                symbol=signal.symbol,
                qty=qty,
                entry_price=entry_price,
                entry_date=current_date,
                peak_price=raw_price,
                commissions_paid=commission_per_order,
            )
            open_slots -= 1

        equity = cash + sum(position.qty * current_prices.get(symbol, position.entry_price) for symbol, position in open_positions.items())
        daily_points.append(
            BacktestPoint(
                date=current_date,
                equity=round(equity, 2),
                benchmark_equity=round(benchmark_shares * current_benchmark_price, 2),
                cash=round(cash, 2),
                positions=len(open_positions),
            )
        )

    final_date = ordered_dates[-1]
    for symbol, position in list(open_positions.items()):
        final_bar = bar_lookup[symbol].get(final_date)
        if final_bar is None:
            continue
        exit_price = _apply_slippage(final_bar.close, "sell", slippage_bps)
        gross_proceeds = exit_price * position.qty
        net_proceeds = gross_proceeds - commission_per_order
        gross_pnl = (exit_price - position.entry_price) * position.qty
        net_pnl = net_proceeds - (position.entry_price * position.qty) - position.commissions_paid
        hold_days = _holding_days(date_index, position.entry_date, final_date)
        closed_trades.append(
            BacktestTrade(
                symbol=symbol,
                entry_date=position.entry_date,
                exit_date=final_date,
                entry_price=round(position.entry_price, 2),
                exit_price=round(exit_price, 2),
                qty=position.qty,
                gross_pnl=round(gross_pnl, 2),
                pnl=round(net_pnl, 2),
                commissions_paid=round(position.commissions_paid + commission_per_order, 2),
                return_pct=round(net_pnl / ((position.entry_price * position.qty) + position.commissions_paid), 4),
                hold_days=hold_days,
                exit_reason="Marked to market on final backtest day.",
            )
        )
        cash += net_proceeds
        del open_positions[symbol]

    if daily_points:
        daily_points[-1].equity = round(cash, 2)
        daily_points[-1].cash = round(cash, 2)
        daily_points[-1].positions = 0

    ending_equity = daily_points[-1].equity if daily_points else starting_capital
    benchmark_ending_equity = daily_points[-1].benchmark_equity if daily_points else starting_capital
    total_return_pct = ((ending_equity - starting_capital) / starting_capital) if starting_capital else 0.0
    benchmark_return_pct = ((benchmark_ending_equity - starting_capital) / starting_capital) if starting_capital else 0.0
    winning_trades = [trade for trade in closed_trades if trade.pnl > 0]
    average_trade_return_pct = sum(trade.return_pct for trade in closed_trades) / len(closed_trades) if closed_trades else 0.0
    sharpe = _sharpe_ratio(daily_points)
    sortino = _sortino_ratio(daily_points)
    strategy_chart_points, benchmark_chart_points = _comparison_chart_points(daily_points)

    return BacktestResult(
        status="ok",
        summary=f"Simulated {len(closed_trades)} completed trades across {period_days} trading days with slippage and commissions.",
        start_date=ordered_dates[0],
        end_date=ordered_dates[-1],
        period_days=period_days,
        starting_capital=round(starting_capital, 2),
        ending_equity=round(ending_equity, 2),
        total_return_pct=round(total_return_pct, 4),
        benchmark_symbol=benchmark_symbol,
        benchmark_return_pct=round(benchmark_return_pct, 4),
        outperformance_pct=round(total_return_pct - benchmark_return_pct, 4),
        max_drawdown_pct=round(_max_drawdown_pct(daily_points), 4),
        universe_preset=universe_preset,
        universe_label=universe_meta["label"],
        signal_threshold=round(signal_threshold, 4),
        factor_momentum_weight=round(factor_momentum_weight, 4),
        factor_sentiment_weight=round(factor_sentiment_weight, 4),
        factor_earnings_weight=round(factor_earnings_weight, 4),
        total_trades=len(closed_trades),
        win_rate_pct=round((len(winning_trades) / len(closed_trades)), 4) if closed_trades else 0.0,
        average_trade_return_pct=round(average_trade_return_pct, 4),
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        chart_points=strategy_chart_points,
        benchmark_chart_points=benchmark_chart_points,
        benchmark_ending_equity=round(benchmark_ending_equity, 2),
        commission_per_order=round(commission_per_order, 2),
        slippage_bps=round(slippage_bps, 2),
        max_hold_days=max_hold_days,
        min_hold_days=min_hold_days,
        trailing_stop_pct=round(trailing_stop_pct, 4),
        trailing_arm_pct=round(trailing_arm_pct, 4),
        take_profit_pct=round(take_profit_pct, 4),
        notes=[
            f"Universe preset: {universe_meta['label']} ({len(price_map) - 1} tradable symbols in this replay set).",
            f"Composite weights were momentum {factor_momentum_weight * 100:.0f}%, sentiment {factor_sentiment_weight * 100:.0f}%, and earnings {factor_earnings_weight * 100:.0f}% with a {signal_threshold:.2f} buy threshold.",
            (
                "Market regime filter counted "
                f"{regime_counts.get('supportive', 0)} supportive days, "
                f"{regime_counts.get('cautious', 0)} cautious days, and "
                f"{regime_counts.get('risk_off', 0)} risk-off days."
            ),
            f"Each order assumes ${commission_per_order:.2f} commission and {slippage_bps:.2f} bps of slippage.",
            f"Exit model uses {max_hold_days} max hold days, {trailing_stop_pct * 100:.1f}% trailing stop armed after {trailing_arm_pct * 100:.1f}% profit, and {take_profit_pct * 100:.1f}% profit take.",
            "Historical news sentiment is excluded from replay to avoid pretending we have cheap point-in-time news coverage.",
            "Historical earnings surprise is replayed only after each reported date becomes known.",
        ],
        trades=closed_trades[::-1],
        daily_points=daily_points,
    )


class BacktestService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.alpaca = AlpacaService(settings)
        self.alpha = AlphaVantageService(settings)
        initialize_database(settings)

    def load_market_data(
        self,
        symbols: list[str],
        *,
        fetch_days: int,
    ) -> tuple[dict[str, list[PriceBar]], dict[str, list[EarningsBundle]]]:
        price_map: dict[str, list[PriceBar]] = {}
        earnings_map: dict[str, list[EarningsBundle]] = {}

        for symbol in symbols:
            price_map[symbol] = self.alpaca.get_daily_bars(symbol, fetch_days)

        if self.settings.earnings_configured:
            with get_connection(self.settings) as conn:
                for symbol in symbols:
                    earnings_map[symbol] = self.alpha.get_quarterly_earnings_history(symbol, conn)

        return price_map, earnings_map

    def run(
        self,
        period_days: int = 120,
        starting_capital: float = 100_000.0,
        *,
        universe_preset: str | None = None,
        signal_threshold: float | None = None,
        factor_momentum_weight: float | None = None,
        factor_sentiment_weight: float | None = None,
        factor_earnings_weight: float | None = None,
        commission_per_order: float | None = None,
        slippage_bps: float | None = None,
        max_hold_days: int | None = None,
        min_hold_days: int | None = None,
        trailing_stop_pct: float | None = None,
        trailing_arm_pct: float | None = None,
        take_profit_pct: float | None = None,
        benchmark_symbol: str | None = None,
    ) -> BacktestResult:
        if not self.settings.trading_configured:
            raise RuntimeError("Add your Alpaca keys before running a backtest.")

        universe_preset = normalize_universe_preset(universe_preset or self.settings.universe_preset)
        universe = get_universe(self.settings.universe_symbols, universe_preset)
        benchmark_symbol = (benchmark_symbol or self.settings.backtest_benchmark_symbol).strip().upper()
        fetch_days = max(period_days + 120, 220)
        symbols = sorted(set(universe + [benchmark_symbol]))
        price_map, earnings_map = self.load_market_data(symbols, fetch_days=fetch_days)

        return simulate_backtest(
            settings=self.settings,
            price_map=price_map,
            earnings_map=earnings_map,
            benchmark_symbol=benchmark_symbol,
            period_days=period_days,
            starting_capital=starting_capital,
            universe_preset=universe_preset,
            signal_threshold=signal_threshold,
            factor_momentum_weight=factor_momentum_weight,
            factor_sentiment_weight=factor_sentiment_weight,
            factor_earnings_weight=factor_earnings_weight,
            commission_per_order=commission_per_order,
            slippage_bps=slippage_bps,
            max_hold_days=max_hold_days,
            min_hold_days=min_hold_days,
            trailing_stop_pct=trailing_stop_pct,
            trailing_arm_pct=trailing_arm_pct,
            take_profit_pct=take_profit_pct,
        )
