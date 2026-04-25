from __future__ import annotations

import asyncio
import contextlib
import csv
import io
import json
from pathlib import Path

from fastapi import FastAPI, Form, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware

from app.auth import verify_password
from app.config import get_settings
from app.db import (
    fetch_latest_run,
    fetch_news_for_run,
    fetch_positions_for_run,
    fetch_recent_runs,
    fetch_signal_by_symbol,
    fetch_signals_for_run,
    fetch_trades_for_run,
    get_connection,
    initialize_database,
)
from app.services.alpaca import AlpacaService
from app.services.backtest import BacktestService
from app.services.engine import TradingEngine
from app.services.journal import PaperJournalService
from app.services.market_regime import parse_regime_warning
from app.services.walkforward import (
    WalkForwardService,
    default_walkforward_thresholds,
    default_walkforward_windows,
    get_factor_profiles,
)
from app.universe import get_universe_presets, normalize_universe_preset


settings = get_settings()
initialize_database(settings)
APP_DIR = Path(__file__).resolve().parent

app = FastAPI(title=settings.app_name, debug=settings.debug)
app.add_middleware(SessionMiddleware, secret_key=settings.session_secret)
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
templates = Jinja2Templates(directory=APP_DIR / "templates")


def percent_value(value) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def score_value(value) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def currency_value(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "n/a"


templates.env.filters["pct"] = percent_value
templates.env.filters["score"] = score_value
templates.env.filters["currency"] = currency_value


def _is_authenticated(request: Request) -> bool:
    return bool(request.session.get("authenticated"))


def _redirect_login() -> RedirectResponse:
    return RedirectResponse("/login", status_code=status.HTTP_303_SEE_OTHER)


def _parse_int_csv(raw_value: str, default: list[int], *, minimum: int, maximum: int) -> list[int]:
    output: list[int] = []
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            continue
        output.append(max(minimum, min(value, maximum)))
    return sorted(set(output)) or default


def _parse_float_csv(raw_value: str, default: list[float], *, minimum: float, maximum: float) -> list[float]:
    output: list[float] = []
    for chunk in raw_value.split(","):
        item = chunk.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            continue
        output.append(round(max(minimum, min(value, maximum)), 2))
    return sorted(set(output)) or default


@app.on_event("startup")
async def startup_event() -> None:
    if settings.engine_run_interval_minutes > 0:
        app.state.scheduler_task = asyncio.create_task(engine_scheduler())


@app.on_event("shutdown")
async def shutdown_event() -> None:
    task = getattr(app.state, "scheduler_task", None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


async def engine_scheduler() -> None:
    engine = TradingEngine(settings)
    while True:
        await asyncio.sleep(settings.engine_run_interval_minutes * 60)
        await run_in_threadpool(engine.run_once, "scheduler", settings.auto_trade_enabled)


def _split_regime_warning(raw_warnings: list[str]) -> tuple[dict | None, list[str]]:
    regime = None
    remaining: list[str] = []
    for warning in raw_warnings:
        parsed = parse_regime_warning(warning)
        if parsed is not None:
            regime = parsed
            continue
        remaining.append(warning)
    return regime, remaining


def _build_dashboard_context(request: Request, message: str = "") -> dict:
    with get_connection(settings) as conn:
        latest_run = fetch_latest_run(conn)
        signals = fetch_signals_for_run(conn, latest_run["id"]) if latest_run else []
        news_items = fetch_news_for_run(conn, latest_run["id"]) if latest_run else []
        stored_positions = fetch_positions_for_run(conn, latest_run["id"]) if latest_run else []
        trades = fetch_trades_for_run(conn, latest_run["id"]) if latest_run else []
        runs = fetch_recent_runs(conn, limit=8)
    latest_warnings = json.loads(latest_run["warnings_json"]) if latest_run else []
    latest_regime, latest_warnings = _split_regime_warning(latest_warnings)
    account = None
    live_positions = stored_positions
    orders = trades
    broker_error = ""
    if settings.trading_configured:
        alpaca = AlpacaService(settings)
        try:
            account = alpaca.get_account()
            live_positions = alpaca.get_positions()
            orders = alpaca.get_orders(limit=10)
        except Exception as exc:  # noqa: BLE001
            broker_error = str(exc)
    return {
        "request": request,
        "settings": settings,
        "message": message,
        "latest_run": latest_run,
        "latest_regime": latest_regime,
        "latest_warnings": latest_warnings,
        "signals": signals,
        "news_items": news_items,
        "positions": live_positions,
        "orders": orders,
        "runs": runs,
        "account": account,
        "broker_error": broker_error,
    }


def _build_backtest_context(
    request: Request,
    *,
    result=None,
    error: str = "",
    period_days: int = 120,
    starting_capital: float = 100_000.0,
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
) -> dict:
    last_result = result if result is not None else getattr(app.state, "last_backtest", None)
    return {
        "request": request,
        "settings": settings,
        "universe_presets": get_universe_presets(),
        "result": last_result,
        "error": error,
        "period_days": period_days,
        "starting_capital": starting_capital,
        "universe_preset": normalize_universe_preset(
            last_result.universe_preset if universe_preset is None and last_result is not None else (
                settings.universe_preset if universe_preset is None else universe_preset
            )
        ),
        "signal_threshold": (
            last_result.signal_threshold if signal_threshold is None and last_result is not None else (
                settings.signal_threshold if signal_threshold is None else signal_threshold
            )
        ),
        "factor_momentum_weight": (
            last_result.factor_momentum_weight if factor_momentum_weight is None and last_result is not None else (
                settings.factor_momentum_weight if factor_momentum_weight is None else factor_momentum_weight
            )
        ),
        "factor_sentiment_weight": (
            last_result.factor_sentiment_weight if factor_sentiment_weight is None and last_result is not None else (
                settings.factor_sentiment_weight if factor_sentiment_weight is None else factor_sentiment_weight
            )
        ),
        "factor_earnings_weight": (
            last_result.factor_earnings_weight if factor_earnings_weight is None and last_result is not None else (
                settings.factor_earnings_weight if factor_earnings_weight is None else factor_earnings_weight
            )
        ),
        "commission_per_order": (
            last_result.commission_per_order if commission_per_order is None and last_result is not None else (
                settings.backtest_commission_per_order if commission_per_order is None else commission_per_order
            )
        ),
        "slippage_bps": (
            last_result.slippage_bps if slippage_bps is None and last_result is not None else (
                settings.backtest_slippage_bps if slippage_bps is None else slippage_bps
            )
        ),
        "max_hold_days": (
            last_result.max_hold_days if max_hold_days is None and last_result is not None else (
                settings.backtest_max_hold_days if max_hold_days is None else max_hold_days
            )
        ),
        "min_hold_days": (
            last_result.min_hold_days if min_hold_days is None and last_result is not None else (
                settings.backtest_min_hold_days if min_hold_days is None else min_hold_days
            )
        ),
        "trailing_stop_pct": (
            last_result.trailing_stop_pct if trailing_stop_pct is None and last_result is not None else (
                settings.backtest_trailing_stop_pct if trailing_stop_pct is None else trailing_stop_pct
            )
        ),
        "trailing_arm_pct": (
            last_result.trailing_arm_pct if trailing_arm_pct is None and last_result is not None else (
                settings.backtest_trailing_arm_pct if trailing_arm_pct is None else trailing_arm_pct
            )
        ),
        "take_profit_pct": (
            last_result.take_profit_pct if take_profit_pct is None and last_result is not None else (
                settings.backtest_take_profit_pct if take_profit_pct is None else take_profit_pct
            )
        ),
    }


def _build_walkforward_context(
    request: Request,
    *,
    result=None,
    error: str = "",
    windows_csv: str | None = None,
    thresholds_csv: str | None = None,
    starting_capital: float = 100_000.0,
) -> dict:
    last_result = result if result is not None else getattr(app.state, "last_walkforward", None)
    windows_default = last_result.windows_tested if last_result is not None else default_walkforward_windows()
    thresholds_default = last_result.thresholds_tested if last_result is not None else default_walkforward_thresholds(settings)
    return {
        "request": request,
        "settings": settings,
        "result": last_result,
        "error": error,
        "windows_csv": windows_csv or ", ".join(str(item) for item in windows_default),
        "thresholds_csv": thresholds_csv or ", ".join(f"{item:.2f}" for item in thresholds_default),
        "starting_capital": (
            result.starting_capital if result is not None else (
                starting_capital if error else (last_result.starting_capital if last_result is not None else starting_capital)
            )
        ),
        "factor_profiles": get_factor_profiles(),
        "universe_presets": get_universe_presets(),
    }


def _build_journal_context(
    request: Request,
    *,
    result=None,
    error: str = "",
) -> dict:
    return {
        "request": request,
        "settings": settings,
        "result": result,
        "error": error,
    }


def _backtest_result_or_none():
    return getattr(app.state, "last_backtest", None)


def _backtest_trades_csv(result) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "symbol",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "qty",
            "gross_pnl",
            "net_pnl",
            "commissions_paid",
            "return_pct",
            "hold_days",
            "exit_reason",
        ]
    )
    for trade in result.trades:
        writer.writerow(
            [
                trade.symbol,
                trade.entry_date,
                trade.exit_date,
                trade.entry_price,
                trade.exit_price,
                trade.qty,
                trade.gross_pnl,
                trade.pnl,
                trade.commissions_paid,
                trade.return_pct,
                trade.hold_days,
                trade.exit_reason,
            ]
        )
    return buffer.getvalue()


def _backtest_equity_csv(result) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["date", "strategy_equity", "benchmark_equity", "cash", "positions"])
    for point in result.daily_points:
        writer.writerow([point.date, point.equity, point.benchmark_equity, point.cash, point.positions])
    return buffer.getvalue()


@app.get("/")
async def dashboard(request: Request, message: str = ""):
    if not _is_authenticated(request):
        return _redirect_login()
    return templates.TemplateResponse(request, "dashboard.html", _build_dashboard_context(request, message))


@app.get("/login")
async def login_page(request: Request, error: str = ""):
    if _is_authenticated(request):
        return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse(request, "login.html", {"request": request, "settings": settings, "error": error})


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    username_ok = username == settings.admin_username
    password_source = settings.admin_password_hash or settings.admin_password
    password_ok = verify_password(password, password_source)
    if not (username_ok and password_ok):
        return templates.TemplateResponse(
            request,
            "login.html",
            {"request": request, "settings": settings, "error": "Invalid username or password."},
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    request.session["authenticated"] = True
    request.session["username"] = username
    return RedirectResponse("/", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/logout")
async def logout(request: Request):
    request.session.clear()
    return _redirect_login()


@app.post("/run")
async def run_engine(request: Request):
    if not _is_authenticated(request):
        return _redirect_login()
    engine = TradingEngine(settings)
    await run_in_threadpool(engine.run_once, "manual", False)
    return RedirectResponse("/?message=Fresh+scan+completed", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/backtest")
async def backtest_page(request: Request):
    if not _is_authenticated(request):
        return _redirect_login()
    return templates.TemplateResponse(request, "backtest.html", _build_backtest_context(request))


@app.get("/journal")
async def journal_page(request: Request):
    if not _is_authenticated(request):
        return _redirect_login()
    service = PaperJournalService(settings)
    try:
        result = await run_in_threadpool(service.run)
        return templates.TemplateResponse(request, "journal.html", _build_journal_context(request, result=result))
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            request,
            "journal.html",
            _build_journal_context(request, error=str(exc)),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.post("/backtest")
async def run_backtest(
    request: Request,
    period_days: int = Form(120),
    starting_capital: float = Form(100000.0),
    universe_preset: str = Form(settings.universe_preset),
    signal_threshold: float = Form(settings.signal_threshold),
    factor_momentum_weight: float = Form(settings.factor_momentum_weight),
    factor_sentiment_weight: float = Form(settings.factor_sentiment_weight),
    factor_earnings_weight: float = Form(settings.factor_earnings_weight),
    commission_per_order: float = Form(settings.backtest_commission_per_order),
    slippage_bps: float = Form(settings.backtest_slippage_bps),
    max_hold_days: int = Form(settings.backtest_max_hold_days),
    min_hold_days: int = Form(settings.backtest_min_hold_days),
    trailing_stop_pct: float = Form(settings.backtest_trailing_stop_pct),
    trailing_arm_pct: float = Form(settings.backtest_trailing_arm_pct),
    take_profit_pct: float = Form(settings.backtest_take_profit_pct),
):
    if not _is_authenticated(request):
        return _redirect_login()
    period_days = max(60, min(period_days, 252))
    starting_capital = max(10_000.0, min(starting_capital, 1_000_000.0))
    universe_preset = normalize_universe_preset(universe_preset)
    signal_threshold = max(0.10, min(signal_threshold, 0.90))
    factor_momentum_weight = max(0.0, min(factor_momentum_weight, 1.0))
    factor_sentiment_weight = max(0.0, min(factor_sentiment_weight, 1.0))
    factor_earnings_weight = max(0.0, min(factor_earnings_weight, 1.0))
    commission_per_order = max(0.0, min(commission_per_order, 25.0))
    slippage_bps = max(0.0, min(slippage_bps, 100.0))
    max_hold_days = max(3, min(max_hold_days, 90))
    min_hold_days = max(1, min(min_hold_days, max_hold_days))
    trailing_stop_pct = max(0.01, min(trailing_stop_pct, 0.30))
    trailing_arm_pct = max(0.0, min(trailing_arm_pct, 0.20))
    take_profit_pct = max(0.01, min(take_profit_pct, 0.50))
    service = BacktestService(settings)
    try:
        result = await run_in_threadpool(
            service.run,
            period_days,
            starting_capital,
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
        app.state.last_backtest = result
        return templates.TemplateResponse(
            request,
            "backtest.html",
            _build_backtest_context(
                request,
                result=result,
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
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            request,
            "backtest.html",
            _build_backtest_context(
                request,
                error=str(exc),
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
            ),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.get("/walk-forward")
async def walkforward_page(request: Request):
    if not _is_authenticated(request):
        return _redirect_login()
    return templates.TemplateResponse(request, "walkforward.html", _build_walkforward_context(request))


@app.post("/walk-forward")
async def run_walkforward(
    request: Request,
    windows_csv: str = Form(", ".join(str(item) for item in default_walkforward_windows())),
    thresholds_csv: str = Form(", ".join(f"{item:.2f}" for item in default_walkforward_thresholds(settings))),
    starting_capital: float = Form(100000.0),
):
    if not _is_authenticated(request):
        return _redirect_login()
    windows = _parse_int_csv(windows_csv, default_walkforward_windows(), minimum=60, maximum=252)
    thresholds = _parse_float_csv(
        thresholds_csv,
        default_walkforward_thresholds(settings),
        minimum=0.10,
        maximum=0.90,
    )
    starting_capital = max(10_000.0, min(starting_capital, 1_000_000.0))
    service = WalkForwardService(settings)
    try:
        result = await run_in_threadpool(
            service.run,
            windows=windows,
            thresholds=thresholds,
            starting_capital=starting_capital,
        )
        app.state.last_walkforward = result
        return templates.TemplateResponse(
            request,
            "walkforward.html",
            _build_walkforward_context(
                request,
                result=result,
                windows_csv=", ".join(str(item) for item in windows),
                thresholds_csv=", ".join(f"{item:.2f}" for item in thresholds),
                starting_capital=starting_capital,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return templates.TemplateResponse(
            request,
            "walkforward.html",
            _build_walkforward_context(
                request,
                error=str(exc),
                windows_csv=", ".join(str(item) for item in windows),
                thresholds_csv=", ".join(f"{item:.2f}" for item in thresholds),
                starting_capital=starting_capital,
            ),
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@app.get("/backtest/export/trades.csv")
async def export_backtest_trades(request: Request):
    if not _is_authenticated(request):
        return _redirect_login()
    result = _backtest_result_or_none()
    if result is None:
        return RedirectResponse("/backtest", status_code=status.HTTP_303_SEE_OTHER)
    return Response(
        content=_backtest_trades_csv(result),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=backtest-trades.csv"},
    )


@app.get("/backtest/export/equity.csv")
async def export_backtest_equity(request: Request):
    if not _is_authenticated(request):
        return _redirect_login()
    result = _backtest_result_or_none()
    if result is None:
        return RedirectResponse("/backtest", status_code=status.HTTP_303_SEE_OTHER)
    return Response(
        content=_backtest_equity_csv(result),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=backtest-equity.csv"},
    )


@app.post("/signals/{symbol}/buy")
async def buy_symbol(request: Request, symbol: str):
    if not _is_authenticated(request):
        return _redirect_login()
    with get_connection(settings) as conn:
        signal = fetch_signal_by_symbol(conn, symbol.upper())
    if signal is None:
        return RedirectResponse("/?message=No+recent+signal+found+for+that+symbol", status_code=status.HTTP_303_SEE_OTHER)
    if signal["decision"] != "buy":
        return RedirectResponse("/?message=That+symbol+is+not+a+current+buy+candidate", status_code=status.HTTP_303_SEE_OTHER)
    alpaca = AlpacaService(settings)
    try:
        account = alpaca.get_account()
        equity = float(account.get("equity") or 0.0)
        qty = int((equity * settings.max_position_pct) // float(signal["price"]))
        qty = max(1, qty)
        alpaca.submit_market_order(symbol.upper(), qty, "buy", "Manual dashboard buy")
        message = f"Submitted paper buy for {symbol.upper()}."
    except Exception as exc:  # noqa: BLE001
        message = f"Buy failed for {symbol.upper()}: {exc}"
    return RedirectResponse(f"/?message={message.replace(' ', '+')}", status_code=status.HTTP_303_SEE_OTHER)


@app.post("/positions/{symbol}/sell")
async def sell_symbol(request: Request, symbol: str):
    if not _is_authenticated(request):
        return _redirect_login()
    alpaca = AlpacaService(settings)
    try:
        alpaca.close_position(symbol.upper())
        message = f"Submitted close order for {symbol.upper()}."
    except Exception as exc:  # noqa: BLE001
        message = f"Sell failed for {symbol.upper()}: {exc}"
    return RedirectResponse(f"/?message={message.replace(' ', '+')}", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "app": settings.app_name,
        "trading_configured": settings.trading_configured,
        "earnings_configured": settings.earnings_configured,
        "auto_trade_enabled": settings.auto_trade_enabled,
    }
