"""Advanced parameter optimization system for maximum profit capability."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import itertools
import math
from typing import Dict, List, Optional, Tuple, Callable
import random


class OptimizationMethod(Enum):
    """Methods for parameter optimization."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"  # Simplified Bayesian optimization
    GENETIC = "genetic"   # Genetic algorithm


@dataclass(slots=True)
class ParameterRange:
    """Definition of a parameter to optimize."""
    name: str
    min_value: float
    max_value: float
    step: float
    is_integer: bool = False
    description: str = ""

    def get_values(self) -> List[float]:
        """Get all possible values for this parameter."""
        if self.is_integer:
            return [float(v) for v in range(int(self.min_value), int(self.max_value) + 1, int(self.step))]
        else:
            values = []
            current = self.min_value
            while current <= self.max_value + 1e-9:  # Small epsilon for float comparison
                values.append(round(current, 6))
                current += self.step
            return values


@dataclass(slots=True)
class ParameterSet:
    """A specific set of parameter values."""
    parameters: Dict[str, float]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    rank: int = 0
    validation_score: float = 0.0

    def get_fitness_score(self, weights: Dict[str, float] = None) -> float:
        """
        Calculate overall fitness score from performance metrics.

        Args:
            weights: Optional weights for different metrics (default: balanced)

        Returns:
            Combined fitness score
        """
        if weights is None:
            weights = {
                "total_return_pct": 0.30,
                "sharpe_ratio": 0.25,
                "max_drawdown_pct": -0.20,  # Negative because lower is better
                "win_rate_pct": 0.15,
                "profit_factor": 0.10
            }

        score = 0.0
        for metric, weight in weights.items():
            if metric in self.performance_metrics:
                # Normalize metrics to roughly 0-1 range
                value = self.performance_metrics[metric]

                if metric == "max_drawdown_pct":
                    # Lower drawdown is better, normalize to 0-1
                    normalized = max(0.0, min(1.0, (0.50 - value) / 0.50))
                elif metric == "sharpe_ratio":
                    # Sharpe > 2 is excellent, normalize
                    normalized = max(0.0, min(1.0, value / 2.0))
                elif metric == "total_return_pct":
                    # 50% annual return is excellent
                    normalized = max(0.0, min(1.0, value / 50.0))
                elif metric == "win_rate_pct":
                    # Win rate is already 0-1
                    normalized = value
                elif metric == "profit_factor":
                    # Profit factor > 2 is excellent
                    normalized = max(0.0, min(1.0, value / 2.0))
                else:
                    normalized = 0.5  # Neutral for unknown metrics

                score += weight * normalized

        return score


@dataclass(slots=True)
class OptimizationResult:
    """Results from parameter optimization."""
    best_parameter_set: ParameterSet
    all_parameter_sets: List[ParameterSet]
    optimization_method: OptimizationMethod
    total_iterations: int
    computation_time_seconds: float
    convergence_info: Dict[str, any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class ParameterOptimizer:
    """Advanced parameter optimization system."""

    def __init__(
        self,
        parameter_ranges: List[ParameterRange],
        evaluation_function: Callable[[Dict[str, float]], Dict[str, float]],
        optimization_method: OptimizationMethod = OptimizationMethod.GRID_SEARCH,
        max_iterations: int = 1000,
        early_stopping_patience: int = 50,
        random_seed: int = 42
    ):
        """
        Initialize the parameter optimizer.

        Args:
            parameter_ranges: List of parameters to optimize
            evaluation_function: Function that takes parameters and returns performance metrics
            optimization_method: Method to use for optimization
            max_iterations: Maximum number of iterations
            early_stopping_patience: Stop if no improvement for this many iterations
            random_seed: Random seed for reproducibility
        """
        self.parameter_ranges = parameter_ranges
        self.evaluation_function = evaluation_function
        self.optimization_method = optimization_method
        self.max_iterations = max_iterations
        self.early_stopping_patience = early_stopping_patience
        self.random_seed = random_seed
        random.seed(random_seed)

    def optimize(self) -> OptimizationResult:
        """
        Run the optimization process.

        Returns:
            OptimizationResult with best parameters and all tested sets
        """
        start_time = datetime.now()

        if self.optimization_method == OptimizationMethod.GRID_SEARCH:
            result = self._grid_search()
        elif self.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            result = self._random_search()
        elif self.optimization_method == OptimizationMethod.BAYESIAN:
            result = self._bayesian_optimization()
        elif self.optimization_method == OptimizationMethod.GENETIC:
            result = self._genetic_algorithm()
        else:
            raise ValueError(f"Unknown optimization method: {self.optimization_method}")

        computation_time = (datetime.now() - start_time).total_seconds()
        result.computation_time_seconds = computation_time
        result.total_iterations = len(result.all_parameter_sets)

        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)

        return result

    def _grid_search(self) -> OptimizationResult:
        """Exhaustive grid search over all parameter combinations."""
        parameter_values = [pr.get_values() for pr in self.parameter_ranges]
        parameter_names = [pr.name for pr in self.parameter_ranges]

        all_combinations = list(itertools.product(*parameter_values))

        # Limit combinations if too many
        if len(all_combinations) > self.max_iterations:
            # Sample randomly from the grid
            indices = random.sample(range(len(all_combinations)), self.max_iterations)
            all_combinations = [all_combinations[i] for i in indices]

        parameter_sets = []
        best_score = -float('inf')
        best_set = None

        for combination in all_combinations:
            params = dict(zip(parameter_names, combination))
            metrics = self.evaluation_function(params)

            param_set = ParameterSet(parameters=params, performance_metrics=metrics)
            fitness = param_set.get_fitness_score()

            if fitness > best_score:
                best_score = fitness
                best_set = param_set

            parameter_sets.append(param_set)

        # Rank all parameter sets
        parameter_sets.sort(key=lambda ps: ps.get_fitness_score(), reverse=True)
        for i, ps in enumerate(parameter_sets):
            ps.rank = i + 1

        return OptimizationResult(
            best_parameter_set=best_set or parameter_sets[0],
            all_parameter_sets=parameter_sets,
            optimization_method=self.optimization_method,
            total_iterations=len(parameter_sets),
            computation_time_seconds=0.0,
            convergence_info={
                "method": "grid_search",
                "total_combinations": len(all_combinations),
                "sampled": len(parameter_sets)
            }
        )

    def _random_search(self) -> OptimizationResult:
        """Random search over parameter space."""
        parameter_names = [pr.name for pr in self.parameter_ranges]

        parameter_sets = []
        best_score = -float('inf')
        best_set = None
        no_improvement_count = 0

        for iteration in range(self.max_iterations):
            # Generate random parameters
            params = {}
            for pr in self.parameter_ranges:
                if pr.is_integer:
                    value = random.randint(int(pr.min_value), int(pr.max_value))
                else:
                    value = random.uniform(pr.min_value, pr.max_value)
                    # Round to step precision
                    value = round(value / pr.step) * pr.step
                    value = max(pr.min_value, min(pr.max_value, value))
                params[pr.name] = value

            metrics = self.evaluation_function(params)
            param_set = ParameterSet(parameters=params, performance_metrics=metrics)
            fitness = param_set.get_fitness_score()

            if fitness > best_score:
                best_score = fitness
                best_set = param_set
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            parameter_sets.append(param_set)

            # Early stopping
            if no_improvement_count >= self.early_stopping_patience:
                break

        # Rank all parameter sets
        parameter_sets.sort(key=lambda ps: ps.get_fitness_score(), reverse=True)
        for i, ps in enumerate(parameter_sets):
            ps.rank = i + 1

        return OptimizationResult(
            best_parameter_set=best_set or parameter_sets[0],
            all_parameter_sets=parameter_sets,
            optimization_method=self.optimization_method,
            total_iterations=len(parameter_sets),
            computation_time_seconds=0.0,
            convergence_info={
                "method": "random_search",
                "early_stopped": no_improvement_count >= self.early_stopping_patience
            }
        )

    def _bayesian_optimization(self) -> OptimizationResult:
        """Simplified Bayesian optimization using exploration-exploitation."""
        # Start with random samples
        initial_samples = min(20, self.max_iterations // 4)
        parameter_names = [pr.name for pr in self.parameter_ranges]

        parameter_sets = []
        evaluated_params = []

        # Initial random exploration
        for _ in range(initial_samples):
            params = {}
            for pr in self.parameter_ranges:
                if pr.is_integer:
                    value = random.randint(int(pr.min_value), int(pr.max_value))
                else:
                    value = random.uniform(pr.min_value, pr.max_value)
                    value = round(value / pr.step) * pr.step
                    value = max(pr.min_value, min(pr.max_value, value))
                params[pr.name] = value

            metrics = self.evaluation_function(params)
            param_set = ParameterSet(parameters=params, performance_metrics=metrics)
            parameter_sets.append(param_set)
            evaluated_params.append(params)

        # Exploitation phase - sample around best performers
        remaining_iterations = self.max_iterations - initial_samples

        for _ in range(remaining_iterations):
            # Select a random good performer to explore around
            if parameter_sets:
                # Weight selection towards better performers
                top_performers = sorted(parameter_sets, key=lambda ps: ps.get_fitness_score(), reverse=True)[:5]
                base_params = random.choice(top_performers).parameters
            else:
                base_params = {pr.name: (pr.min_value + pr.max_value) / 2 for pr in self.parameter_ranges}

            # Add noise to explore around the base parameters
            params = {}
            for pr in self.parameter_ranges:
                base_value = base_params.get(pr.name, (pr.min_value + pr.max_value) / 2)

                # Add Gaussian noise scaled by parameter range
                noise_scale = (pr.max_value - pr.min_value) * 0.1
                noise = random.gauss(0, noise_scale)

                if pr.is_integer:
                    value = int(round(base_value + noise))
                    value = max(int(pr.min_value), min(int(pr.max_value), value))
                else:
                    value = base_value + noise
                    value = round(value / pr.step) * pr.step
                    value = max(pr.min_value, min(pr.max_value, value))

                params[pr.name] = value

            metrics = self.evaluation_function(params)
            param_set = ParameterSet(parameters=params, performance_metrics=metrics)
            parameter_sets.append(param_set)

        # Rank all parameter sets
        parameter_sets.sort(key=lambda ps: ps.get_fitness_score(), reverse=True)
        for i, ps in enumerate(parameter_sets):
            ps.rank = i + 1

        return OptimizationResult(
            best_parameter_set=parameter_sets[0] if parameter_sets else None,
            all_parameter_sets=parameter_sets,
            optimization_method=self.optimization_method,
            total_iterations=len(parameter_sets),
            computation_time_seconds=0.0,
            convergence_info={
                "method": "bayesian",
                "initial_samples": initial_samples,
                "exploitation_samples": len(parameter_sets) - initial_samples
            }
        )

    def _genetic_algorithm(self) -> OptimizationResult:
        """Genetic algorithm for parameter optimization."""
        population_size = min(50, self.max_iterations // 10)
        generations = self.max_iterations // population_size
        mutation_rate = 0.1
        crossover_rate = 0.7

        parameter_names = [pr.name for pr in self.parameter_ranges]

        # Initialize population
        population = []
        for _ in range(population_size):
            params = {}
            for pr in self.parameter_ranges:
                if pr.is_integer:
                    value = random.randint(int(pr.min_value), int(pr.max_value))
                else:
                    value = random.uniform(pr.min_value, pr.max_value)
                    value = round(value / pr.step) * pr.step
                    value = max(pr.min_value, min(pr.max_value, value))
                params[pr.name] = value

            metrics = self.evaluation_function(params)
            param_set = ParameterSet(parameters=params, performance_metrics=metrics)
            population.append(param_set)

        all_parameter_sets = population.copy()

        for generation in range(generations):
            # Sort by fitness
            population.sort(key=lambda ps: ps.get_fitness_score(), reverse=True)

            # Selection - keep top performers
            survivors = population[:population_size // 2]

            # Crossover
            offspring = []
            while len(offspring) < population_size // 2:
                parent1 = random.choice(survivors)
                parent2 = random.choice(survivors)

                if random.random() < crossover_rate:
                    # Single-point crossover
                    crossover_point = random.randint(1, len(parameter_names) - 1)
                    child_params = {}

                    for i, param_name in enumerate(parameter_names):
                        if i < crossover_point:
                            child_params[param_name] = parent1.parameters[param_name]
                        else:
                            child_params[param_name] = parent2.parameters[param_name]

                    # Ensure parameters are within valid ranges
                    for pr in self.parameter_ranges:
                        if pr.name in child_params:
                            child_params[pr.name] = max(pr.min_value, min(pr.max_value, child_params[pr.name]))
                            if pr.is_integer:
                                child_params[pr.name] = int(round(child_params[pr.name]))

                    metrics = self.evaluation_function(child_params)
                    child = ParameterSet(parameters=child_params, performance_metrics=metrics)
                    offspring.append(child)
                    all_parameter_sets.append(child)

            # Mutation
            for individual in survivors + offspring:
                if random.random() < mutation_rate:
                    # Mutate a random parameter
                    param_to_mutate = random.choice(parameter_names)
                    pr = next(p for p in self.parameter_ranges if p.name == param_to_mutate)

                    if pr.is_integer:
                        new_value = random.randint(int(pr.min_value), int(pr.max_value))
                    else:
                        new_value = random.uniform(pr.min_value, pr.max_value)
                        new_value = round(new_value / pr.step) * pr.step
                        new_value = max(pr.min_value, min(pr.max_value, new_value))

                    individual.parameters[param_to_mutate] = new_value

                    # Re-evaluate mutated individual
                    metrics = self.evaluation_function(individual.parameters)
                    individual.performance_metrics = metrics

            # Create new population
            population = survivors + offspring

        # Rank all parameter sets
        all_parameter_sets.sort(key=lambda ps: ps.get_fitness_score(), reverse=True)
        for i, ps in enumerate(all_parameter_sets):
            ps.rank = i + 1

        return OptimizationResult(
            best_parameter_set=all_parameter_sets[0] if all_parameter_sets else None,
            all_parameter_sets=all_parameter_sets,
            optimization_method=self.optimization_method,
            total_iterations=len(all_parameter_sets),
            computation_time_seconds=0.0,
            convergence_info={
                "method": "genetic",
                "population_size": population_size,
                "generations": generations
            }
        )

    def _generate_recommendations(self, result: OptimizationResult) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []

        if not result.all_parameter_sets:
            return ["No optimization results available"]

        best = result.best_parameter_set

        # Analyze best parameters
        recommendations.append(f"Best parameters found: {best.parameters}")

        # Compare with default parameters
        default_params = {pr.name: (pr.min_value + pr.max_value) / 2 for pr in self.parameter_ranges}
        significant_changes = []

        for param_name, best_value in best.parameters.items():
            default_value = default_params.get(param_name, 0)
            if default_value > 0:
                change_pct = abs((best_value - default_value) / default_value)
                if change_pct > 0.2:  # More than 20% change
                    significant_changes.append(f"{param_name}: {default_value:.3f} -> {best_value:.3f}")

        if significant_changes:
            recommendations.append("Significant parameter changes from defaults:")
            recommendations.extend(f"  - {change}" for change in significant_changes)

        # Performance insights
        metrics = best.performance_metrics
        if metrics.get("total_return_pct", 0) > 20:
            recommendations.append("Strong return potential detected (>20% annualized)")

        if metrics.get("sharpe_ratio", 0) > 1.5:
            recommendations.append("Excellent risk-adjusted returns (Sharpe > 1.5)")

        if metrics.get("max_drawdown_pct", 100) < 15:
            recommendations.append("Conservative drawdown profile (<15%)")

        if metrics.get("win_rate_pct", 0) > 0.55:
            recommendations.append("Above-average win rate (>55%)")

        # Method-specific recommendations
        if result.optimization_method == OptimizationMethod.GRID_SEARCH:
            recommendations.append("Grid search provides exhaustive coverage but may be computationally expensive")

        elif result.optimization_method == OptimizationMethod.RANDOM_SEARCH:
            recommendations.append("Random search is efficient for high-dimensional spaces")

        elif result.optimization_method == OptimizationMethod.BAYESIAN:
            recommendations.append("Bayesian optimization balances exploration and exploitation efficiently")

        elif result.optimization_method == OptimizationMethod.GENETIC:
            recommendations.append("Genetic algorithm is effective for complex, non-linear parameter relationships")

        return recommendations


def create_default_parameter_ranges() -> List[ParameterRange]:
    """
    Create default parameter ranges for optimization.

    Returns:
        List of ParameterRange objects for common trading parameters
    """
    return [
        ParameterRange(
            name="signal_threshold",
            min_value=0.15,
            max_value=0.50,
            step=0.05,
            description="Minimum composite score for buy signals"
        ),
        ParameterRange(
            name="max_positions",
            min_value=2,
            max_value=8,
            step=1,
            is_integer=True,
            description="Maximum number of concurrent positions"
        ),
        ParameterRange(
            name="stop_loss_pct",
            min_value=0.05,
            max_value=0.15,
            step=0.01,
            description="Stop loss percentage"
        ),
        ParameterRange(
            name="max_hold_days",
            min_value=10,
            max_value=30,
            step=5,
            is_integer=True,
            description="Maximum holding period in days"
        ),
        ParameterRange(
            name="trailing_stop_pct",
            min_value=0.00,
            max_value=0.10,
            step=0.02,
            description="Trailing stop percentage (0 = disabled)"
        ),
        ParameterRange(
            name="momentum_weight",
            min_value=0.20,
            max_value=0.60,
            step=0.10,
            description="Weight of momentum in composite score"
        ),
        ParameterRange(
            name="earnings_weight",
            min_value=0.40,
            max_value=0.80,
            step=0.10,
            description="Weight of earnings in composite score"
        ),
    ]


def optimize_trading_parameters(
    evaluation_function: Callable[[Dict[str, float]], Dict[str, float]],
    parameter_ranges: List[ParameterRange] = None,
    optimization_method: OptimizationMethod = OptimizationMethod.RANDOM_SEARCH,
    max_iterations: int = 500
) -> OptimizationResult:
    """
    Convenience function to optimize trading parameters.

    Args:
        evaluation_function: Function that evaluates parameter performance
        parameter_ranges: Optional custom parameter ranges
        optimization_method: Method to use for optimization
        max_iterations: Maximum number of iterations

    Returns:
        OptimizationResult with optimal parameters
    """
    if parameter_ranges is None:
        parameter_ranges = create_default_parameter_ranges()

    optimizer = ParameterOptimizer(
        parameter_ranges=parameter_ranges,
        evaluation_function=evaluation_function,
        optimization_method=optimization_method,
        max_iterations=max_iterations
    )

    return optimizer.optimize()