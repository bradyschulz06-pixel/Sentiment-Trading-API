"""Demo script for backtesting validation and parameter optimization."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List
import random

from app.models import PriceBar, EarningsBundle
from app.backtest_validation import (
    BacktestValidator,
    ValidationConfig,
    ValidationPhase,
    create_default_validation_config
)
from app.parameter_optimization import OptimizationMethod


def generate_sample_price_data(
    symbols: List[str],
    start_date: str = "2025-01-01",
    end_date: str = "2026-04-26",
    base_price: float = 100.0
) -> Dict[str, List[PriceBar]]:
    """
    Generate sample price data for testing.

    Args:
        symbols: List of stock symbols
        start_date: Start date for data generation
        end_date: End date for data generation
        base_price: Base price for all stocks

    Returns:
        Dictionary mapping symbols to price bar lists
    """
    price_map = {}

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days

    for symbol in symbols:
        bars = []
        current_price = base_price

        # Add some randomness to base price per symbol
        symbol_adjustment = random.uniform(0.8, 1.2)
        current_price *= symbol_adjustment

        for day in range(days):
            date_str = (start + timedelta(days=day)).strftime("%Y-%m-%d")

            # Generate realistic price movement
            daily_return = random.gauss(0.0005, 0.015)  # Small positive drift with volatility
            current_price *= (1 + daily_return)

            # Generate OHLC
            high = current_price * random.uniform(1.0, 1.02)
            low = current_price * random.uniform(0.98, 1.0)
            open_price = current_price * random.uniform(0.99, 1.01)
            close = current_price

            # Generate volume
            volume = random.uniform(1_000_000, 10_000_000)

            bar = PriceBar(
                symbol=symbol,
                timestamp=f"{date_str}T00:00:00Z",
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=int(volume)
            )

            bars.append(bar)

        price_map[symbol] = bars

    return price_map


def generate_sample_earnings_data(
    symbols: List[str],
    start_date: str = "2025-01-01",
    end_date: str = "2026-04-26"
) -> Dict[str, List[EarningsBundle]]:
    """
    Generate sample earnings data for testing.

    Args:
        symbols: List of stock symbols
        start_date: Start date for data generation
        end_date: End date for data generation

    Returns:
        Dictionary mapping symbols to earnings bundle lists
    """
    earnings_map = {}

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = (end - start).days

    for symbol in symbols:
        bundles = []

        # Generate 2-3 earnings reports per symbol
        num_earnings = random.randint(2, 3)

        for i in range(num_earnings):
            # Random date within range
            days_offset = random.randint(30, days - 30)
            report_date = (start + timedelta(days=days_offset)).strftime("%Y-%m-%d")

            # Generate earnings data
            estimated_eps = random.uniform(1.0, 5.0)
            surprise_pct = random.uniform(-0.15, 0.20)  # -15% to +20% surprise
            reported_eps = estimated_eps * (1 + surprise_pct)

            # Generate upcoming earnings date
            upcoming_offset = random.randint(30, 90)
            upcoming_date = (start + timedelta(days=days_offset + upcoming_offset)).strftime("%Y-%m-%d")

            bundle = EarningsBundle(
                symbol=symbol,
                fiscal_date_ending=f"{2025 + i}-03-31",
                reported_date=report_date,
                reported_eps=round(reported_eps, 2),
                estimated_eps=round(estimated_eps, 2),
                surprise=round(reported_eps - estimated_eps, 2),
                surprise_pct=round(surprise_pct, 4),
                report_time=random.choice(["AMC", "After market close"]),
                transcript_sentiment=random.uniform(-0.5, 0.8),
                transcript_excerpt="Sample transcript text for testing purposes.",
                upcoming_report_date=upcoming_date
            )

            bundles.append(bundle)

        # Sort by reported date
        bundles.sort(key=lambda x: x.reported_date or "")

        earnings_map[symbol] = bundles

    return earnings_map


def run_validation_demo():
    """Run a demonstration of the validation framework."""
    print("="*60)
    print("BACKTESTING VALIDATION DEMO")
    print("="*60)

    # Generate sample data
    print("\nGenerating sample data...")
    symbols = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "JNJ", "AMZN", "NVDA"]

    # Use shorter test period for demo
    test_periods = [
        ("2025-10-26", "2026-04-26"),  # 6 months
    ]

    price_map = generate_sample_price_data(symbols, "2025-01-01", "2026-04-26")
    earnings_map = generate_sample_earnings_data(symbols, "2025-01-01", "2026-04-26")

    print(f"Generated price data for {len(symbols)} symbols")
    print(f"Generated earnings data for {len(symbols)} symbols")

    # Create validation config
    config = ValidationConfig(
        test_periods=test_periods,
        starting_capital=100_000.0,
        benchmark_symbol="AAPL",  # Use one of our symbols as benchmark
        commission_per_order=1.00,
        slippage_bps=8.0,
        enable_enhanced_signals=True,
        enable_sector_rotation=True,
        enable_regime_adaptive=True,
        enable_multi_level_profit=True,
        enable_optimization=True,
        optimization_method=OptimizationMethod.RANDOM_SEARCH,
        optimization_iterations=50  # Reduced for demo
    )

    # Run validation
    validator = BacktestValidator(config)

    try:
        # Run full validation
        validation_results = validator.run_full_validation(price_map, earnings_map)

        # Run parameter optimization
        print("\n" + "="*60)
        print("RUNNING PARAMETER OPTIMIZATION")
        print("="*60)

        optimization_results = validator.run_parameter_optimization(
            price_map, earnings_map, test_periods[0]
        )

        # Save validation report
        report_path = validator.save_validation_report()

        print("\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"\nValidation report saved to: {report_path}")
        print(f"Optimization status: {optimization_results.get('status', 'unknown')}")

        if optimization_results.get("status") == "success":
            print(f"Best parameters found: {optimization_results.get('best_parameters')}")
            print(f"Best fitness score: {optimization_results.get('fitness_score', 0):.3f}")

    except Exception as e:
        print(f"\nDemo encountered an error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_validation_demo()