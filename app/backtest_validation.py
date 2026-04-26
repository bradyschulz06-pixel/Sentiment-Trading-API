"""Comprehensive backtesting validation framework for profit enhancements."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import statistics

from app.config import Settings
from app.models import BacktestResult, BacktestTrade, PriceBar, EarningsBundle
from app.services.backtest import simulate_backtest
from app.parameter_optimization import (
    ParameterRange,
    ParameterOptimizer,
    OptimizationMethod,
    create_default_parameter_ranges,
    optimize_trading_parameters
)


class ValidationPhase(Enum):
    """Phases of backtesting validation."""
    BASELINE = "baseline"
    ENHANCED_SIGNALS = "enhanced_signals"
    SECTOR_ROTATION = "sector_rotation"
    REGIME_ADAPTIVE = "regime_adaptive"
    MULTI_LEVEL_PROFIT = "multi_level_profit"
    FULL_ENHANCED = "full_enhanced"


@dataclass(slots=True)
class ValidationConfig:
    """Configuration for backtesting validation."""
    test_periods: List[Tuple[str, str]]  # List of (start_date, end_date) tuples
    starting_capital: float = 100_000.0
    benchmark_symbol: str = "SPY"
    commission_per_order: float = 1.00
    slippage_bps: float = 8.0

    # Feature flags for testing individual enhancements
    enable_enhanced_signals: bool = True
    enable_sector_rotation: bool = True
    enable_regime_adaptive: bool = True
    enable_multi_level_profit: bool = True

    # Optimization settings
    enable_optimization: bool = True
    optimization_method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH
    optimization_iterations: int = 200


@dataclass(slots=True)
class ValidationResult:
    """Results from backtesting validation."""
    phase: ValidationPhase
    config: ValidationConfig
    backtest_results: List[BacktestResult] = field(default_factory=list)
    aggregate_metrics: Dict[str, float] = field(default_factory=dict)
    improvements: Dict[str, float] = field(default_factory=dict)
    passed: bool = True
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ComparisonResult:
    """Comparison between different validation phases."""
    baseline: ValidationResult
    enhanced: ValidationResult
    performance_delta: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class BacktestValidator:
    """Comprehensive backtesting validation system."""

    def __init__(self, config: ValidationConfig):
        """
        Initialize the backtest validator.

        Args:
            config: Validation configuration
        """
        self.config = config
        self.validation_results: Dict[ValidationPhase, ValidationResult] = {}

    def run_full_validation(
        self,
        price_map: Dict[str, List[PriceBar]],
        earnings_map: Dict[str, List[EarningsBundle]]
    ) -> Dict[ValidationPhase, ValidationResult]:
        """
        Run comprehensive validation across all phases.

        Args:
            price_map: Historical price data
            earnings_map: Historical earnings data

        Returns:
            Dictionary mapping phases to validation results
        """
        print("Starting comprehensive backtesting validation...")

        # Run each validation phase
        phases = [
            ValidationPhase.BASELINE,
            ValidationPhase.ENHANCED_SIGNALS,
            ValidationPhase.SECTOR_ROTATION,
            ValidationPhase.REGIME_ADAPTIVE,
            ValidationPhase.MULTI_LEVEL_PROFIT,
            ValidationPhase.FULL_ENHANCED
        ]

        for phase in phases:
            print(f"\n{'='*60}")
            print(f"Running validation phase: {phase.value}")
            print(f"{'='*60}")

            result = self._run_validation_phase(phase, price_map, earnings_map)
            self.validation_results[phase] = result

            print(f"Phase {phase.value} completed: {'PASSED' if result.passed else 'FAILED'}")
            if result.notes:
                print("Notes:")
                for note in result.notes:
                    print(f"  - {note}")
            if result.warnings:
                print("Warnings:")
                for warning in result.warnings:
                    print(f"  - {warning}")

        # Generate comparison report
        print(f"\n{'='*60}")
        print("GENERATING COMPARISON REPORT")
        print(f"{'='*60}")

        comparison = self._generate_comparison_report()
        self._print_comparison_report(comparison)

        return self.validation_results

    def _run_validation_phase(
        self,
        phase: ValidationPhase,
        price_map: Dict[str, List[PriceBar]],
        earnings_map: Dict[str, List[EarningsBundle]]
    ) -> ValidationResult:
        """Run a single validation phase."""
        result = ValidationResult(phase=phase, config=self.config)

        # Configure settings based on phase
        settings = self._get_phase_settings(phase)

        # Run backtests for each test period
        for start_date, end_date in self.config.test_periods:
            try:
                backtest_result = simulate_backtest(
                    settings=settings,
                    price_map=price_map,
                    earnings_map=earnings_map,
                    benchmark_symbol=self.config.benchmark_symbol,
                    period_days=self._calculate_period_days(start_date, end_date),
                    starting_capital=self.config.starting_capital,
                    start_date=start_date,
                    end_date=end_date,
                    commission_per_order=self.config.commission_per_order,
                    slippage_bps=self.config.slippage_bps,
                )

                result.backtest_results.append(backtest_result)

            except Exception as e:
                result.warnings.append(f"Backtest failed for {start_date} to {end_date}: {str(e)}")
                result.passed = False

        # Calculate aggregate metrics
        result.aggregate_metrics = self._calculate_aggregate_metrics(result.backtest_results)

        # Validate results
        result.passed, result.notes, result.warnings = self._validate_results(
            result.aggregate_metrics, phase
        )

        return result

    def _get_phase_settings(self, phase: ValidationPhase) -> Settings:
        """Get settings configured for a specific validation phase."""
        # Base settings with default values
        base_settings = {
            "app_name": "Validation",
            "environment": "test",
            "host": "127.0.0.1",
            "port": 8000,
            "debug": False,
            "admin_username": "admin",
            "admin_password": "test",
            "admin_password_hash": "",
            "session_secret": "test_secret",
            "alpaca_api_key": "test_key",
            "alpaca_secret_key": "test_secret",
            "alpaca_trading_base_url": "https://paper-api.alpaca.markets",
            "alpaca_data_base_url": "https://data.alpaca.markets",
            "alpha_vantage_api_key": "test_key",
            "universe_preset": "sector_leaders",
            "universe_symbols": "",
            "lookback_days": 180,
            "news_window_days": 7,
            "earnings_lookup_limit": 8,
            "upcoming_earnings_buffer_days": 2,
            "backtest_min_bars": 0,
            "max_position_pct": 0.08,
            "market_regime_filter_enabled": True,
            "market_regime_cautious_threshold_boost": 0.04,
            "market_regime_cautious_max_positions_multiplier": 0.50,
            "auto_trade_enabled": False,
            "engine_run_interval_minutes": 0,
            "backtest_benchmark_symbol": self.config.benchmark_symbol,
            "backtest_commission_per_order": self.config.commission_per_order,
            "backtest_slippage_bps": self.config.slippage_bps,
            "backtest_min_hold_days": 3,
            "backtest_trailing_arm_pct": 0.00,
            "backtest_take_profit_pct": 0.00,
            "backtest_breakeven_arm_pct": 0.00,
            "backtest_breakeven_floor_pct": 0.005,
            "backtest_reentry_cooldown_days": 3,
            "backtest_rolling_drawdown_window": 10,
            "backtest_rolling_drawdown_limit": 0.05,
            "conviction_sizing_enabled": True,
            "conviction_sizing_min_scalar": 0.75,
            "conviction_sizing_max_scalar": 1.25,
            "alpha_vantage_requests_per_minute": 5,
            "max_daily_loss_pct": 0.02,
            "max_trades_per_symbol_per_day": 1,
            "max_positions_per_sector": 2,
            "db_path": Path("test_validation.db")
        }

        # Apply phase-specific settings
        if phase == ValidationPhase.BASELINE:
            # Original simple settings
            base_settings.update({
                "max_positions": 4,
                "stop_loss_pct": 0.07,
                "signal_threshold": 0.30,
                "factor_momentum_weight": 0.00,
                "factor_sentiment_weight": 0.00,
                "factor_earnings_weight": 1.00,
                "backtest_trailing_stop_pct": 0.00,
                "backtest_max_hold_days": 20,
            })

        elif phase == ValidationPhase.ENHANCED_SIGNALS:
            # Enhanced signal generation
            base_settings.update({
                "max_positions": 4,
                "stop_loss_pct": 0.07,
                "signal_threshold": 0.30,
                "factor_momentum_weight": 0.40,
                "factor_sentiment_weight": 0.00,
                "factor_earnings_weight": 0.60,
                "backtest_trailing_stop_pct": 0.00,
                "backtest_max_hold_days": 20,
            })

        elif phase == ValidationPhase.SECTOR_ROTATION:
            # Enhanced signals + sector rotation
            base_settings.update({
                "max_positions": 4,
                "stop_loss_pct": 0.07,
                "signal_threshold": 0.30,
                "factor_momentum_weight": 0.40,
                "factor_sentiment_weight": 0.00,
                "factor_earnings_weight": 0.60,
                "backtest_trailing_stop_pct": 0.00,
                "backtest_max_hold_days": 20,
            })

        elif phase == ValidationPhase.REGIME_ADAPTIVE:
            # Enhanced signals + regime adaptive
            base_settings.update({
                "max_positions": 4,
                "stop_loss_pct": 0.07,
                "signal_threshold": 0.30,
                "factor_momentum_weight": 0.40,
                "factor_sentiment_weight": 0.00,
                "factor_earnings_weight": 0.60,
                "backtest_trailing_stop_pct": 0.00,
                "backtest_max_hold_days": 20,
            })

        elif phase == ValidationPhase.MULTI_LEVEL_PROFIT:
            # Enhanced signals + multi-level profit taking
            base_settings.update({
                "max_positions": 4,
                "stop_loss_pct": 0.07,
                "signal_threshold": 0.30,
                "factor_momentum_weight": 0.40,
                "factor_sentiment_weight": 0.00,
                "factor_earnings_weight": 0.60,
                "backtest_trailing_stop_pct": 0.06,
                "backtest_trailing_arm_pct": 0.03,
                "backtest_max_hold_days": 20,
            })

        elif phase == ValidationPhase.FULL_ENHANCED:
            # All enhancements enabled
            base_settings.update({
                "max_positions": 4,
                "stop_loss_pct": 0.07,
                "signal_threshold": 0.30,
                "factor_momentum_weight": 0.40,
                "factor_sentiment_weight": 0.00,
                "factor_earnings_weight": 0.60,
                "backtest_trailing_stop_pct": 0.06,
                "backtest_trailing_arm_pct": 0.03,
                "backtest_max_hold_days": 20,
            })

        return Settings(**base_settings)

    def _calculate_period_days(self, start_date: str, end_date: str) -> int:
        """Calculate number of days between dates."""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        return (end - start).days

    def _calculate_aggregate_metrics(self, results: List[BacktestResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across multiple backtest results."""
        if not results:
            return {}

        metrics = {
            "total_return_pct": statistics.mean([r.total_return_pct for r in results]),
            "benchmark_return_pct": statistics.mean([r.benchmark_return_pct for r in results]),
            "outperformance_pct": statistics.mean([r.outperformance_pct for r in results]),
            "max_drawdown_pct": statistics.mean([r.max_drawdown_pct for r in results]),
            "sharpe_ratio": statistics.mean([r.sharpe_ratio for r in results]),
            "total_trades": sum([r.total_trades for r in results]),
            "win_rate_pct": statistics.mean([r.win_rate_pct for r in results]),
            "average_trade_return_pct": statistics.mean([r.average_trade_return_pct for r in results]),
        }

        # Calculate standard deviations for robustness
        if len(results) > 1:
            metrics["total_return_std"] = statistics.stdev([r.total_return_pct for r in results])
            metrics["sharpe_std"] = statistics.stdev([r.sharpe_ratio for r in results])
            metrics["drawdown_std"] = statistics.stdev([r.max_drawdown_pct for r in results])

        return metrics

    def _validate_results(
        self,
        metrics: Dict[str, float],
        phase: ValidationPhase
    ) -> Tuple[bool, List[str], List[str]]:
        """Validate backtest results against expected criteria."""
        notes = []
        warnings = []
        passed = True

        # Basic validation criteria
        if metrics.get("total_trades", 0) == 0:
            warnings.append("No trades generated - strategy may be too restrictive")
            passed = False

        if metrics.get("win_rate_pct", 0) < 0.30:
            warnings.append(f"Low win rate ({metrics.get('win_rate_pct', 0):.1%}) - consider parameter adjustments")

        if metrics.get("max_drawdown_pct", 100) > 0.30:
            warnings.append(f"High drawdown ({metrics.get('max_drawdown_pct', 100):.1%}) - risk management may need improvement")

        if metrics.get("sharpe_ratio", 0) < 0.5:
            warnings.append(f"Low Sharpe ratio ({metrics.get('sharpe_ratio', 0):.2f}) - risk-adjusted returns may be suboptimal")

        # Phase-specific validation
        if phase == ValidationPhase.ENHANCED_SIGNALS:
            if metrics.get("total_return_pct", 0) > 0:
                notes.append("Enhanced signals generating positive returns")

        elif phase == ValidationPhase.FULL_ENHANCED:
            if metrics.get("total_return_pct", 0) > 10:
                notes.append("Full enhanced system showing strong performance")
            if metrics.get("sharpe_ratio", 0) > 1.0:
                notes.append("Excellent risk-adjusted returns with full enhancements")

        # Check for robustness across periods
        if "total_return_std" in metrics:
            if metrics["total_return_std"] > 15:
                warnings.append(f"High return variability ({metrics['total_return_std']:.1f}%) - strategy may not be robust")

        return passed, notes, warnings

    def _generate_comparison_report(self) -> ComparisonResult:
        """Generate comprehensive comparison between baseline and enhanced."""
        baseline = self.validation_results.get(ValidationPhase.BASELINE)
        enhanced = self.validation_results.get(ValidationPhase.FULL_ENHANCED)

        if not baseline or not enhanced:
            return ComparisonResult(
                baseline=baseline or ValidationResult(phase=ValidationPhase.BASELINE, config=self.config),
                enhanced=enhanced or ValidationResult(phase=ValidationPhase.FULL_ENHANCED, config=self.config)
            )

        comparison = ComparisonResult(baseline=baseline, enhanced=enhanced)

        # Calculate performance deltas
        baseline_metrics = baseline.aggregate_metrics
        enhanced_metrics = enhanced.aggregate_metrics

        comparison.performance_delta = {
            "total_return_pct": enhanced_metrics.get("total_return_pct", 0) - baseline_metrics.get("total_return_pct", 0),
            "sharpe_ratio": enhanced_metrics.get("sharpe_ratio", 0) - baseline_metrics.get("sharpe_ratio", 0),
            "max_drawdown_pct": enhanced_metrics.get("max_drawdown_pct", 100) - baseline_metrics.get("max_drawdown_pct", 100),
            "win_rate_pct": enhanced_metrics.get("win_rate_pct", 0) - baseline_metrics.get("win_rate_pct", 0),
            "total_trades": enhanced_metrics.get("total_trades", 0) - baseline_metrics.get("total_trades", 0),
        }

        # Generate recommendations
        comparison.recommendations = self._generate_recommendations(comparison)

        return comparison

    def _generate_recommendations(self, comparison: ComparisonResult) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []

        delta = comparison.performance_delta

        # Return improvement analysis
        if delta["total_return_pct"] > 5:
            recommendations.append(f"Strong return improvement: +{delta['total_return_pct']:.1f}%")
        elif delta["total_return_pct"] > 0:
            recommendations.append(f"Modest return improvement: +{delta['total_return_pct']:.1f}%")
        else:
            recommendations.append(f"Return degradation: {delta['total_return_pct']:.1f}% - consider parameter adjustments")

        # Sharpe ratio analysis
        if delta["sharpe_ratio"] > 0.3:
            recommendations.append(f"Significant risk-adjusted return improvement: +{delta['sharpe_ratio']:.2f} Sharpe")
        elif delta["sharpe_ratio"] > 0:
            recommendations.append(f"Modest Sharpe improvement: +{delta['sharpe_ratio']:.2f}")
        else:
            recommendations.append(f"Sharpe ratio decreased: {delta['sharpe_ratio']:.2f}")

        # Drawdown analysis
        if delta["max_drawdown_pct"] < -0.02:
            recommendations.append(f"Drawdown reduced by {abs(delta['max_drawdown_pct'])*100:.1f}% - improved risk management")
        elif delta["max_drawdown_pct"] > 0.02:
            recommendations.append(f"Drawdown increased by {delta['max_drawdown_pct']*100:.1f}% - monitor risk")

        # Win rate analysis
        if delta["win_rate_pct"] > 0.05:
            recommendations.append(f"Win rate improved by {delta['win_rate_pct']*100:.1f}%")
        elif delta["win_rate_pct"] < -0.05:
            recommendations.append(f"Win rate decreased by {abs(delta['win_rate_pct'])*100:.1f}%")

        # Trade frequency analysis
        if delta["total_trades"] > 10:
            recommendations.append(f"Trade frequency increased by {delta['total_trades']} trades")
        elif delta["total_trades"] < -10:
            recommendations.append(f"Trade frequency decreased by {abs(delta['total_trades'])} trades")

        # Overall assessment
        if (delta["total_return_pct"] > 0 and delta["sharpe_ratio"] > 0 and
            delta["max_drawdown_pct"] < 0):
            recommendations.append("OVERALL: Enhanced system shows comprehensive improvement")
        elif delta["total_return_pct"] > 0:
            recommendations.append("OVERALL: Enhanced system improves returns but monitor risk metrics")
        else:
            recommendations.append("OVERALL: Consider parameter optimization - current enhancements may need tuning")

        return recommendations

    def _print_comparison_report(self, comparison: ComparisonResult):
        """Print detailed comparison report."""
        print("\n" + "="*60)
        print("BASELINE vs ENHANCED COMPARISON")
        print("="*60)

        baseline_metrics = comparison.baseline.aggregate_metrics
        enhanced_metrics = comparison.enhanced.aggregate_metrics
        delta = comparison.performance_delta

        print("\nPerformance Metrics:")
        print(f"{'Metric':<25} {'Baseline':>12} {'Enhanced':>12} {'Delta':>12}")
        print("-" * 61)

        metrics_to_show = [
            ("Total Return %", "total_return_pct"),
            ("Sharpe Ratio", "sharpe_ratio"),
            ("Max Drawdown %", "max_drawdown_pct"),
            ("Win Rate %", "win_rate_pct"),
            ("Total Trades", "total_trades"),
        ]

        for display_name, metric_key in metrics_to_show:
            baseline_val = baseline_metrics.get(metric_key, 0)
            enhanced_val = enhanced_metrics.get(metric_key, 0)
            delta_val = delta.get(metric_key, 0)

            if metric_key in ["total_return_pct", "win_rate_pct", "max_drawdown_pct"]:
                baseline_str = f"{baseline_val*100:.1f}%"
                enhanced_str = f"{enhanced_val*100:.1f}%"
                delta_str = f"{delta_val*100:+.1f}%"
            else:
                baseline_str = f"{baseline_val:.2f}"
                enhanced_str = f"{enhanced_val:.2f}"
                delta_str = f"{delta_val:+.2f}"

            print(f"{display_name:<25} {baseline_str:>12} {enhanced_str:>12} {delta_str:>12}")

        print("\nRecommendations:")
        for rec in comparison.recommendations:
            print(f"  - {rec}")

        print("\n" + "="*60)

    def run_parameter_optimization(
        self,
        price_map: Dict[str, List[PriceBar]],
        earnings_map: Dict[str, List[EarningsBundle]],
        optimization_period: Tuple[str, str] = None
    ) -> Dict[str, any]:
        """
        Run parameter optimization to find optimal settings.

        Args:
            price_map: Historical price data
            earnings_map: Historical earnings data
            optimization_period: Optional (start_date, end_date) for optimization

        Returns:
            Dictionary with optimization results
        """
        if not self.config.enable_optimization:
            return {"status": "skipped", "reason": "Optimization disabled in config"}

        print("\n" + "="*60)
        print("RUNNING PARAMETER OPTIMIZATION")
        print("="*60)

        # Use first test period if none specified
        if optimization_period is None and self.config.test_periods:
            optimization_period = self.config.test_periods[0]

        if not optimization_period:
            return {"status": "error", "reason": "No optimization period specified"}

        start_date, end_date = optimization_period
        period_days = self._calculate_period_days(start_date, end_date)

        print(f"Optimization period: {start_date} to {end_date} ({period_days} days)")
        print(f"Optimization method: {self.config.optimization_method.value}")
        print(f"Max iterations: {self.config.optimization_iterations}")

        # Define evaluation function for optimization
        def evaluate_parameters(params: Dict[str, float]) -> Dict[str, float]:
            """Evaluate a set of parameters via backtesting."""
            try:
                # Create settings with optimized parameters
                settings = Settings(
                    app_name="Optimization",
                    environment="test",
                    host="127.0.0.1",
                    port=8000,
                    debug=False,
                    admin_username="admin",
                    admin_password="test",
                    admin_password_hash="",
                    session_secret="test_secret",
                    alpaca_api_key="test_key",
                    alpaca_secret_key="test_secret",
                    alpaca_trading_base_url="https://paper-api.alpaca.markets",
                    alpaca_data_base_url="https://data.alpaca.markets",
                    alpha_vantage_api_key="test_key",
                    universe_preset="sector_leaders",
                    universe_symbols="",
                    lookback_days=180,
                    news_window_days=7,
                    earnings_lookup_limit=8,
                    upcoming_earnings_buffer_days=2,
                    backtest_min_bars=0,
                    max_positions=int(params.get("max_positions", 4)),
                    max_position_pct=0.08,
                    stop_loss_pct=params.get("stop_loss_pct", 0.07),
                    signal_threshold=params.get("signal_threshold", 0.30),
                    factor_momentum_weight=params.get("momentum_weight", 0.40),
                    factor_sentiment_weight=0.00,
                    factor_earnings_weight=params.get("earnings_weight", 0.60),
                    market_regime_filter_enabled=True,
                    market_regime_cautious_threshold_boost=0.04,
                    market_regime_cautious_max_positions_multiplier=0.50,
                    auto_trade_enabled=False,
                    engine_run_interval_minutes=0,
                    backtest_benchmark_symbol=self.config.benchmark_symbol,
                    backtest_commission_per_order=self.config.commission_per_order,
                    backtest_slippage_bps=self.config.slippage_bps,
                    backtest_max_hold_days=int(params.get("max_hold_days", 20)),
                    backtest_min_hold_days=3,
                    backtest_trailing_stop_pct=params.get("trailing_stop_pct", 0.06),
                    backtest_trailing_arm_pct=0.03,
                    backtest_take_profit_pct=0.00,
                    backtest_breakeven_arm_pct=0.00,
                    backtest_breakeven_floor_pct=0.005,
                    backtest_reentry_cooldown_days=3,
                    backtest_rolling_drawdown_window=10,
                    backtest_rolling_drawdown_limit=0.05,
                    conviction_sizing_enabled=True,
                    conviction_sizing_min_scalar=0.75,
                    conviction_sizing_max_scalar=1.25,
                    alpha_vantage_requests_per_minute=5,
                    max_daily_loss_pct=0.02,
                    max_trades_per_symbol_per_day=1,
                    max_positions_per_sector=2,
                    db_path=Path("test_optimization.db")
                )

                # Run backtest
                result = simulate_backtest(
                    settings=settings,
                    price_map=price_map,
                    earnings_map=earnings_map,
                    benchmark_symbol=self.config.benchmark_symbol,
                    period_days=period_days,
                    starting_capital=self.config.starting_capital,
                    start_date=start_date,
                    end_date=end_date,
                )

                # Return performance metrics
                return {
                    "total_return_pct": result.total_return_pct,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown_pct": result.max_drawdown_pct,
                    "win_rate_pct": result.win_rate_pct,
                    "total_trades": result.total_trades,
                    "average_trade_return_pct": result.average_trade_return_pct,
                }

            except Exception as e:
                print(f"Error evaluating parameters {params}: {e}")
                # Return poor metrics for failed evaluations
                return {
                    "total_return_pct": -100.0,
                    "sharpe_ratio": -10.0,
                    "max_drawdown_pct": 1.0,
                    "win_rate_pct": 0.0,
                    "total_trades": 0,
                    "average_trade_return_pct": -100.0,
                }

        # Run optimization
        try:
            optimization_result = optimize_trading_parameters(
                evaluation_function=evaluate_parameters,
                optimization_method=self.config.optimization_method,
                max_iterations=self.config.optimization_iterations
            )

            print(f"\nOptimization completed!")
            print(f"Best parameters: {optimization_result.best_parameter_set.parameters}")
            print(f"Best fitness score: {optimization_result.best_parameter_set.get_fitness_score():.3f}")
            print(f"Total iterations: {optimization_result.total_iterations}")
            print(f"Computation time: {optimization_result.computation_time_seconds:.1f}s")

            print("\nOptimization recommendations:")
            for rec in optimization_result.recommendations:
                print(f"  - {rec}")

            return {
                "status": "success",
                "best_parameters": optimization_result.best_parameter_set.parameters,
                "best_metrics": optimization_result.best_parameter_set.performance_metrics,
                "fitness_score": optimization_result.best_parameter_set.get_fitness_score(),
                "total_iterations": optimization_result.total_iterations,
                "computation_time": optimization_result.computation_time_seconds,
                "recommendations": optimization_result.recommendations
            }

        except Exception as e:
            print(f"Optimization failed: {e}")
            return {
                "status": "error",
                "reason": str(e)
            }

    def save_validation_report(self, output_path: Path = None) -> Path:
        """
        Save comprehensive validation report to file.

        Args:
            output_path: Optional path for output file

        Returns:
            Path to saved report
        """
        if output_path is None:
            output_path = Path("validation_report.json")

        report_data = {
            "config": {
                "test_periods": self.config.test_periods,
                "starting_capital": self.config.starting_capital,
                "benchmark_symbol": self.config.benchmark_symbol,
                "enable_optimization": self.config.enable_optimization,
            },
            "validation_results": {
                phase.value: {
                    "passed": result.passed,
                    "aggregate_metrics": result.aggregate_metrics,
                    "notes": result.notes,
                    "warnings": result.warnings,
                }
                for phase, result in self.validation_results.items()
            }
        }

        # Add comparison if available
        if ValidationPhase.BASELINE in self.validation_results and ValidationPhase.FULL_ENHANCED in self.validation_results:
            comparison = self._generate_comparison_report()
            report_data["comparison"] = {
                "performance_delta": comparison.performance_delta,
                "recommendations": comparison.recommendations
            }

        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)

        print(f"\nValidation report saved to: {output_path}")
        return output_path


def create_default_validation_config() -> ValidationConfig:
    """
    Create default validation configuration.

    Returns:
        ValidationConfig with sensible defaults
    """
    # Define test periods covering different market conditions
    test_periods = [
        # Recent 6 months
        ("2025-10-26", "2026-04-26"),
        # Bull market period (example)
        ("2023-01-01", "2023-12-31"),
        # Bear market period (example)
        ("2022-01-01", "2022-12-31"),
    ]

    return ValidationConfig(
        test_periods=test_periods,
        starting_capital=100_000.0,
        benchmark_symbol="SPY",
        commission_per_order=1.00,
        slippage_bps=8.0,
        enable_enhanced_signals=True,
        enable_sector_rotation=True,
        enable_regime_adaptive=True,
        enable_multi_level_profit=True,
        enable_optimization=True,
        optimization_method=OptimizationMethod.RANDOM_SEARCH,
        optimization_iterations=200
    )