# AI Stock Trader MVP

`E:\ai-stock-trader` is a fast MVP for a password-protected swing-trading dashboard that combines:

- Momentum from daily price action
- Sentiment from recent news and earnings-call text
- Earnings surprise from reported vs estimated EPS
- Alpaca paper trading for order execution

The app is intentionally conservative:

- It starts in manual approval mode
- It uses paper trading by default
- It caps the portfolio at 5 positions
- It blocks new entries right before earnings
- It is designed to be understandable before it is clever

## Stack

- Python 3.14
- FastAPI + Jinja templates for the web app
- SQLite for local persistence and API caching
- Alpaca REST APIs for market data and paper trades
- Alpha Vantage for earnings surprise and call transcripts

## Why the universe is limited

The default universe is a curated basket of liquid U.S. large-cap names so the MVP stays:

- Easier to understand
- Easier to monitor
- More realistic for swing trading
- More respectful of free/cheap API limits

The engine also limits Alpha Vantage lookups to the strongest early candidates plus existing positions, which helps keep the earnings pipeline inside free-tier constraints.

## Quick Start

1. Create a virtual environment and install dependencies:

```powershell
cd E:\ai-stock-trader
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
```

2. Create your local environment file:

```powershell
Copy-Item .env.example .env
```

3. Edit `.env` and set:

- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `ALPHA_VANTAGE_API_KEY`
- `ADMIN_PASSWORD` or `ADMIN_PASSWORD_HASH`
- `SESSION_SECRET`

4. Start the app:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

5. Open [http://127.0.0.1:8000](http://127.0.0.1:8000)

The dashboard links directly to:

- `/journal` for paper-trade analytics and factor attribution
- `/backtest` for single-parameter replay
- `/walk-forward` for multi-window stability testing

## Password Hashing

For deployment, prefer a password hash over plaintext:

```powershell
python -c "from app.auth import create_password_hash; print(create_password_hash('your-strong-password'))"
```

Put the output in `ADMIN_PASSWORD_HASH` and leave `ADMIN_PASSWORD` blank.

## Key Controls

- `AUTO_TRADE_ENABLED=false` keeps the system in manual approval mode
- `ENGINE_RUN_INTERVAL_MINUTES=0` disables background scheduling
- `UNIVERSE_PRESET=sector_leaders` uses the current walk-forward winner from the tighter research baskets
- `UNIVERSE_SYMBOLS=` lets you override the default watchlist with a comma-separated list
- `SIGNAL_THRESHOLD=0.30` controls how selective the model is
- `FACTOR_MOMENTUM_WEIGHT=0.40`, `FACTOR_SENTIMENT_WEIGHT=0.25`, and `FACTOR_EARNINGS_WEIGHT=0.35` control the composite score mix
- `MARKET_REGIME_FILTER_ENABLED=true` blocks new longs in clearly weak tape and throttles them in mixed tape
- `MARKET_REGIME_CAUTIOUS_THRESHOLD_BOOST=0.04` raises the live entry bar when the market is choppy
- `MARKET_REGIME_CAUTIOUS_MAX_POSITIONS_MULTIPLIER=0.50` cuts open-slot capacity in half during cautious conditions
- `UPCOMING_EARNINGS_BUFFER_DAYS=2` blocks entries too close to earnings
- `BACKTEST_COMMISSION_PER_ORDER=1.00` adds a per-order commission assumption to the replay
- `BACKTEST_SLIPPAGE_BPS=8` adds execution slippage to entries and exits
- `BACKTEST_MAX_HOLD_DAYS=18` forces a time stop in the stricter replay model
- `BACKTEST_TRAILING_STOP_PCT=0.06` is the current stable default from the multi-window replay
- `BACKTEST_TRAILING_ARM_PCT=0.00` stays off by default because a `3%` arm helped the longest window but hurt the shorter test windows; the control remains available for more research
- `BACKTEST_TAKE_PROFIT_PCT=0.14` tightens exits before automation

## Strategy Notes

The current composite score is:

- `40%` momentum
- `25%` recent news sentiment
- `35%` earnings surprise plus transcript tone

The app now supports tighter universe presets:

- `balanced_quality`: a 16-name quality-heavy research basket
- `mega_cap_focus`: the most liquid mega-caps
- `sector_leaders`: the current walk-forward winner and default sector-balanced list

That weighting is a starting point for the MVP, not the finished research model. Once we have paper-trading logs, we can tune:

- Weights
- Holding period rules
- Exit logic
- Universe size
- News window length
- Earnings decay

## Tests

```powershell
.venv\Scripts\python -m pytest
```

## Backtesting

The `/backtest` page now includes:

- Commission and slippage assumptions
- Factor-weight and universe-preset controls
- A market-regime gate using `SPY` trend plus universe breadth
- A benchmark comparison chart against `SPY` by default
- CSV export for closed trades and the equity curve
- Stricter exits using hard stop, trailing stop, time stop, and profit-take rules

The `/walk-forward` page ranks parameter sets across multiple windows so you can see which combinations are stable instead of just lucky once.

The `/journal` page reconstructs Alpaca paper fills into closed trades and open positions, then matches each entry to the nearest stored signal snapshot so you can review factor attribution and post-entry behavior.

This is still a replay, not an institutional-grade simulator. It does not model intraday fills, partial fills, or perfect point-in-time news reconstruction.

## Next upgrades

- Backtesting against cached daily data
- Better transcript summarization
- Sector exposure limits
- Volatility-adjusted position sizing
- Email or text alerts
- Proper user management for multi-user deployment
