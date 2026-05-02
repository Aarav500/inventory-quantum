"""
Distributionally Robust Optimization (DRO).

Optimizes for worst-case distributions within an ambiguity set.
Reference: Ben-Tal et al. (2013) - Robust Optimization
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class DROResult:
    """Result from DRO optimization."""
    optimal_decision: np.ndarray
    worst_case_cost: float
    robust_cost: float
    nominal_cost: float
    ambiguity_radius: float


class WassersteinDRO:
    """
    Distributionally Robust Optimization with Wasserstein ambiguity set.
    
    Optimizes: min_x max_{P: W(P, P_0) <= epsilon} E_P[c(x, xi)]
    
    The Wasserstein ball captures all distributions within epsilon
    distance of the empirical distribution.
    """
    
    def __init__(self, epsilon: float = 0.1):
        """
        Args:
            epsilon: Radius of Wasserstein ball (ambiguity level)
        """
        self.epsilon = epsilon
    
    def _wasserstein_worst_case(
        self,
        x: np.ndarray,
        scenarios: np.ndarray,
        cost_fn: callable,
    ) -> float:
        """
        Compute worst-case expected cost over Wasserstein ball.
        
        Uses CVaR reformulation for tractability.
        """
        costs = np.array([cost_fn(x, s) for s in scenarios])
        n = len(scenarios)
        
        # Worst-case is upper bound: CVaR-like formulation
        # For small epsilon, worst-case ≈ mean + epsilon * Lipschitz_constant
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        
        # Approximate Lipschitz constant from cost variability
        lipschitz_approx = std_cost / (np.std(scenarios) + 1e-8)
        
        worst_case = mean_cost + self.epsilon * lipschitz_approx
        
        return worst_case
    
    def optimize(
        self,
        scenarios: np.ndarray,
        cost_fn: callable,
        x0: np.ndarray,
        bounds: List[Tuple[float, float]]
    ) -> DROResult:
        """
        Solve DRO problem.
        
        Args:
            scenarios: Empirical scenarios (n_scenarios, dim)
            cost_fn: Cost function (decision, scenario) -> cost
            x0: Initial decision
            bounds: Variable bounds
        
        Returns:
            DROResult with robust optimal decision
        """
        # Objective: minimize worst-case cost
        def objective(x):
            return self._wasserstein_worst_case(x, scenarios, cost_fn)
        
        result = minimize(
            objective,
            x0,
            bounds=bounds,
            method='L-BFGS-B'
        )
        
        optimal_x = result.x
        worst_case_cost = result.fun
        
        # Nominal cost (without robustness)
        nominal_cost = np.mean([cost_fn(optimal_x, s) for s in scenarios])
        
        # Cost of robust solution under nominal distribution
        robust_cost = np.mean([cost_fn(optimal_x, s) for s in scenarios])
        
        return DROResult(
            optimal_decision=optimal_x,
            worst_case_cost=worst_case_cost,
            robust_cost=robust_cost,
            nominal_cost=nominal_cost,
            ambiguity_radius=self.epsilon
        )


class MomentDRO:
    """
    Moment-based Distributionally Robust Optimization.
    
    Optimizes over all distributions matching given moments.
    """
    
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        """
        Args:
            mean: First moment (mean vector)
            cov: Second moment (covariance matrix)
        """
        self.mean = mean
        self.cov = cov
    
    def worst_case_expectation(self, c: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute worst-case E[c'xi] over moment ambiguity set.
        
        Closed form: max E[c'xi] = c'mu + ||Sigma^{1/2} c||
        """
        expected = c @ self.mean
        
        # Variance term
        try:
            L = np.linalg.cholesky(self.cov)
            std_term = np.linalg.norm(L.T @ c)
        except:
            std_term = np.sqrt(c @ self.cov @ c)
        
        worst_case = expected + std_term
        
        return worst_case, self.mean + self.cov @ c / (std_term + 1e-8)


class RobustInventoryOptimizer:
    """
    Robust inventory optimization under demand uncertainty.
    """
    
    def __init__(
        self,
        demand_mean: float,
        demand_std: float,
        holding_cost: float = 0.1,
        stockout_cost: float = 10.0,
        ordering_cost: float = 50.0,
        robustness_level: float = 0.1
    ):
        self.demand_mean = demand_mean
        self.demand_std = demand_std
        self.h = holding_cost
        self.p = stockout_cost
        self.K = ordering_cost
        self.epsilon = robustness_level
    
    def _cost_function(self, x: np.ndarray, demand: float) -> float:
        """Inventory cost for given decision and demand realization."""
        reorder_point, order_qty = x
        
        # Expected inventory on hand
        avg_inventory = reorder_point + order_qty / 2 - demand
        holding = self.h * max(0, avg_inventory)
        
        # Expected stockout
        stockout = self.p * max(0, demand - reorder_point)
        
        # Ordering cost (assuming we order)
        ordering = self.K / (order_qty + 1)
        
        return holding + stockout + ordering
    
    def optimize_nominal(self) -> Dict:
        """Standard (non-robust) optimization."""
        # Classic newsvendor-like solution
        critical_ratio = self.p / (self.p + self.h)
        from scipy.stats import norm
        z = norm.ppf(critical_ratio)
        
        reorder_point = self.demand_mean + z * self.demand_std
        
        # EOQ
        D = self.demand_mean * 365  # Annual demand
        order_qty = np.sqrt(2 * D * self.K / self.h)
        
        return {
            'reorder_point': reorder_point,
            'order_quantity': order_qty,
            'method': 'nominal'
        }
    
    def optimize_robust(self, n_scenarios: int = 1000) -> Dict:
        """Distributionally robust optimization."""
        # Generate scenarios
        scenarios = np.random.normal(self.demand_mean, self.demand_std, n_scenarios)
        scenarios = np.maximum(0, scenarios)  # Non-negative demand
        
        # DRO
        dro = WassersteinDRO(epsilon=self.epsilon)
        
        x0 = np.array([self.demand_mean, 100])
        bounds = [(0, self.demand_mean * 3), (10, 500)]
        
        result = dro.optimize(
            scenarios.reshape(-1, 1),
            lambda x, s: self._cost_function(x, s[0]),
            x0,
            bounds
        )
        
        return {
            'reorder_point': result.optimal_decision[0],
            'order_quantity': result.optimal_decision[1],
            'worst_case_cost': result.worst_case_cost,
            'nominal_cost': result.nominal_cost,
            'robustness_premium': result.worst_case_cost - result.nominal_cost,
            'method': 'robust_wasserstein'
        }
    
    def compare_approaches(self) -> Dict:
        """Compare nominal vs robust solutions."""
        nominal = self.optimize_nominal()
        robust = self.optimize_robust()
        
        # Evaluate under worst-case
        worst_demand = self.demand_mean + 2 * self.demand_std
        
        nominal_worst = self._cost_function(
            np.array([nominal['reorder_point'], nominal['order_quantity']]),
            worst_demand
        )
        robust_worst = self._cost_function(
            np.array([robust['reorder_point'], robust['order_quantity']]),
            worst_demand
        )
        
        return {
            'nominal_solution': nominal,
            'robust_solution': robust,
            'worst_case_comparison': {
                'nominal_cost_at_worst': nominal_worst,
                'robust_cost_at_worst': robust_worst,
                'improvement': nominal_worst - robust_worst
            },
            'insight': (
                f"Robust solution increases reorder point by "
                f"{robust['reorder_point'] - nominal['reorder_point']:.1f} units "
                f"to hedge against demand uncertainty."
            )
        }
