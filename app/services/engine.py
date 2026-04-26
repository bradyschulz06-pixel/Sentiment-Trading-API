from __future__ import annotations

import httpx

from app.config import Settings
from app.db import get_connection, initialize_database, save_run, utc_now_iso
from app.models import EngineRunResult, PositionSnapshot, SignalScore, TradeIntent
from app.scoring import build_signal, compute_momentum_score, compute_position_vol_scalar
from app.services.alpaca import AlpacaService
from app.services.alpha_vantage import AlphaVantageService
from app.services.market_regime import apply_market_regime_to_signal, evaluate_market_regime
from app.services.risk import RiskGate, RiskState, SECTOR_MAP, _DEFAULT_SECTOR
from app.universe import get_universe


class TradingEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.alpaca = AlpacaService(settings)
        self.alpha = AlphaVantageService(settings)
        initialize_database(settings)

    def _plan_trades(
        self,
        signals: list[SignalScore],
        positions: list[PositionSnapshot],
        equity: float,
        buying_power: float,
        regime,
        price_map: dict[str, list] | None = None,
        signal_threshold: float = 0.30,
        gate: RiskGate | None = None,
        risk_state: RiskState | None = None,
        daily_pnl_pct: float = 0.0,
    ) -> list[TradeIntent]:
        intents: list[TradeIntent] = []
        held_symbols = {position.symbol for position in positions}
        remaining_positions = len(positions)
        for signal in signals:
            if signal.symbol in held_symbols and signal.decision == "sell":
                position = next(item for item in positions if item.symbol == signal.symbol)
                intents.append(
                    TradeIntent(
                        symbol=signal.symbol,
                        side="sell",
                        qty=max(1, int(abs(position.qty))),
                        notional=position.market_value,
                        reason=signal.rationale,
                    )
                )
                remaining_positions -= 1
        effective_max_positions = regime.effective_max_positions(self.settings.max_positions)
        open_slots = max(0, effective_max_positions - remaining_positions)
        base_budget = equity * self.settings.max_position_pct
        # Pre-compute current sector counts from held positions for the risk gate.
        sector_counts: dict[str, int] = {}
        for pos in positions:
            sec = SECTOR_MAP.get(pos.symbol, _DEFAULT_SECTOR)
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        for signal in signals:
            if open_slots <= 0:
                break
            if signal.decision != "buy" or signal.symbol in held_symbols:
                continue
            # Independent risk gate veto.
            if gate is not None and risk_state is not None:
                verdict = gate.evaluate_buy(
                    signal.symbol,
                    daily_pnl_pct=daily_pnl_pct,
                    state=risk_state,
                    current_sector_counts=sector_counts,
                )
                if not verdict.approved:
                    continue
            # Scale position budget by realized vol and signal conviction.
            if price_map and signal.symbol in price_map:
                vol_scalar = compute_position_vol_scalar(price_map[signal.symbol])
            else:
                vol_scalar = 1.0
            if self.settings.conviction_sizing_enabled:
                threshold_range = max(1.0 - signal_threshold, 0.01)
                conviction = max(0.0, min(1.0, (signal.composite_score - signal_threshold) / threshold_range))
                conviction_scalar = (
                    self.settings.conviction_sizing_min_scalar
                    + (self.settings.conviction_sizing_max_scalar - self.settings.conviction_sizing_min_scalar)
                    * conviction
                )
            else:
                conviction_scalar = 1.0
            budget = min(
                base_budget * vol_scalar * conviction_scalar,
                base_budget * self.settings.conviction_sizing_max_scalar,
            )
            qty = int(min(budget, buying_power) // signal.price)
            if qty < 1:
                continue
            notional = qty * signal.price
            intents.append(
                TradeIntent(
                    symbol=signal.symbol,
                    side="buy",
                    qty=qty,
                    notional=notional,
                    reason=signal.rationale,
                )
            )
            if risk_state is not None:
                risk_state.record_trade(signal.symbol)
                sec = SECTOR_MAP.get(signal.symbol, _DEFAULT_SECTOR)
                sector_counts[sec] = sector_counts.get(sec, 0) + 1
            buying_power -= notional
            open_slots -= 1
        return intents

    def _execute_trade_intents(self, intents: list[TradeIntent]) -> list[TradeIntent]:
        executed: list[TradeIntent] = []
        for intent in intents:
            try:
                if intent.side == "buy":
                    executed.append(self.alpaca.submit_market_order(intent.symbol, intent.qty, "buy", intent.reason))
                else:
                    response = self.alpaca.close_position(intent.symbol)
                    executed.append(
                        TradeIntent(
                            symbol=intent.symbol,
                            side="sell",
                            qty=intent.qty,
                            notional=float(response.get("filled_avg_price") or 0) * intent.qty,
                            reason=intent.reason,
                            status=response.get("status", "submitted"),
                            broker_order_id=response.get("order_id", ""),
                        )
                    )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {401, 403}:
                    reason = f"{intent.reason}. Order rejected — Alpaca auth failed (check API keys)."
                elif exc.response.status_code == 422:
                    reason = f"{intent.reason}. Order rejected by Alpaca (unprocessable: {exc.response.text[:120]})."
                else:
                    reason = f"{intent.reason}. Alpaca HTTP {exc.response.status_code} — may be transient."
                executed.append(
                    TradeIntent(
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        notional=intent.notional,
                        reason=reason,
                        status="failed",
                    )
                )
            except Exception as exc:  # noqa: BLE001
                executed.append(
                    TradeIntent(
                        symbol=intent.symbol,
                        side=intent.side,
                        qty=intent.qty,
                        notional=intent.notional,
                        reason=f"{intent.reason}. Execution failed: {exc}",
                        status="failed",
                    )
                )
        return executed

    def run_once(self, trigger: str = "manual", execute_trades: bool = False) -> EngineRunResult:
        started_at = utc_now_iso()
        warnings: list[str] = []
        if not self.settings.trading_configured:
            result = EngineRunResult(
                status="error",
                summary="Add your Alpaca keys before the engine can fetch market data.",
                started_at=started_at,
                completed_at=utc_now_iso(),
                trigger=trigger,
                error="Missing Alpaca credentials.",
            )
            with get_connection(self.settings) as conn:
                save_run(conn, result)
            return result
        positions: list[PositionSnapshot] = []
        signals: list[SignalScore] = []
        with get_connection(self.settings) as conn:
            try:
                account = self.alpaca.get_account()
                positions = self.alpaca.get_positions()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in {401, 403}:
                    summary = "Alpaca rejected the request — check your API key and secret."
                else:
                    summary = f"Alpaca returned HTTP {exc.response.status_code}. The service may be temporarily unavailable."
                result = EngineRunResult(
                    status="error",
                    summary=summary,
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    trigger=trigger,
                    error=str(exc),
                )
                save_run(conn, result)
                return result
            except Exception as exc:  # noqa: BLE001
                result = EngineRunResult(
                    status="error",
                    summary="Could not connect to Alpaca. Check your API keys and paper endpoint.",
                    started_at=started_at,
                    completed_at=utc_now_iso(),
                    trigger=trigger,
                    error=str(exc),
                )
                save_run(conn, result)
                return result
            position_map = {position.symbol: position for position in positions}
            pre_rank: list[tuple[float, str, list]] = []
            price_map: dict[str, list] = {}
            universe = get_universe(self.settings.universe_symbols, self.settings.universe_preset)
            benchmark_symbol = self.settings.backtest_benchmark_symbol
            try:
                benchmark_bars = self.alpaca.get_daily_bars(benchmark_symbol, self.settings.lookback_days)
            except Exception as exc:  # noqa: BLE001
                benchmark_bars = []
                warnings.append(f"{benchmark_symbol}: benchmark fetch failed ({exc}).")
            # Benchmark 21-day return used later for relative-strength ranking.
            spy_ret21 = 0.0
            if benchmark_bars:
                _, spy_metrics = compute_momentum_score(benchmark_bars)
                spy_ret21 = spy_metrics.get("ret21", 0.0)
            ret21_map: dict[str, float] = {}
            for symbol in universe:
                try:
                    bars = self.alpaca.get_daily_bars(symbol, self.settings.lookback_days)
                    if len(bars) < 55:
                        warnings.append(f"{symbol}: not enough price history to score cleanly.")
                        continue
                    price_map[symbol] = bars
                    momentum_score, momo_metrics = compute_momentum_score(bars)
                    ret21_map[symbol] = momo_metrics.get("ret21", 0.0)
                    pre_rank.append((momentum_score, symbol, bars))
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{symbol}: data fetch failed ({exc}).")
            regime = evaluate_market_regime(self.settings, benchmark_symbol, benchmark_bars, price_map)
            warnings.insert(0, regime.to_warning())
            # Fetch earnings for ALL pre-ranked symbols since earnings is the sole signal.
            alpha_symbols: set[str] = {item[1] for item in pre_rank}
            alpha_symbols.update(position.symbol for position in positions)
            earnings_map = {}
            for symbol in alpha_symbols:
                if not self.settings.earnings_configured:
                    warnings.append("Alpha Vantage key is missing, so earnings surprise is neutral.")
                    break
                try:
                    earnings_map[symbol] = self.alpha.get_earnings_bundle(symbol, conn, include_calendar=True)
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"{symbol}: earnings lookup failed ({exc}).")
            effective_threshold = regime.effective_signal_threshold(self.settings.signal_threshold)
            for _, symbol, bars in pre_rank:
                signal = build_signal(
                    symbol=symbol,
                    bars=bars,
                    bundle=earnings_map.get(symbol),
                    threshold=effective_threshold,
                    stop_loss_pct=self.settings.stop_loss_pct,
                    upcoming_earnings_buffer_days=self.settings.upcoming_earnings_buffer_days,
                    position=position_map.get(symbol),
                )
                signal = apply_market_regime_to_signal(signal, regime, self.settings.signal_threshold)
                signals.append(signal)
            # Rank by composite score adjusted for relative strength vs the benchmark.
            # A stock outperforming SPY by ≥5% on a 21-day basis gets up to a 20% boost;
            # a laggard by ≥5% gets up to a 20% reduction.  composite_score itself is unchanged.
            def _rel_adjusted_rank(sig: SignalScore) -> float:
                rel = ret21_map.get(sig.symbol, 0.0) - spy_ret21
                rel_factor = max(-1.0, min(1.0, rel / 0.05))
                return sig.composite_score * (1.0 + 0.20 * rel_factor)
            signals.sort(key=_rel_adjusted_rank, reverse=True)
            equity = float(account.get("equity") or 0.0)
            buying_power = float(account.get("buying_power") or equity)
            last_equity = float(account.get("last_equity") or equity)
            daily_pnl_pct = (equity - last_equity) / max(last_equity, 1.0)
            gate = RiskGate(self.settings)
            risk_state = RiskState()
            planned_trades = self._plan_trades(
                signals, positions, equity, buying_power, regime, price_map,
                signal_threshold=effective_threshold,
                gate=gate,
                risk_state=risk_state,
                daily_pnl_pct=daily_pnl_pct,
            )
            executed_trades = planned_trades
            if execute_trades and self.settings.auto_trade_enabled:
                executed_trades = self._execute_trade_intents(planned_trades)
            summary = (
                f"Scored {len(signals)} symbols in a {regime.label.replace('_', ' ')} market regime "
                f"and generated {len(planned_trades)} trade ideas."
            )
            result = EngineRunResult(
                status="ok",
                summary=summary,
                started_at=started_at,
                completed_at=utc_now_iso(),
                trigger=trigger,
                signals=signals,
                positions=positions,
                trades=executed_trades,
                warnings=warnings,
            )
            save_run(conn, result)
            return result
