"""
Monte Carlo Simulation for Risk Analysis.

Simulates thousands of demand scenarios to:
- Estimate stockout probability
- Calculate Value at Risk (VaR)
- Optimize under uncertainty
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class RiskMetrics:
    """Risk analysis results."""
    expected_cost: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # Conditional VaR (Expected Shortfall)
    stockout_probability: float
    service_level: float
    scenario_distribution: np.ndarray


class MonteCarloSimulator:
    """
    Monte Carlo simulation for inventory risk analysis.
    """
    
    def __init__(
        self,
        n_simulations: int = 10000,
        random_seed: int = 42
    ):
        self.n_simulations = n_simulations
        np.random.seed(random_seed)
    
    def simulate_demand(
        self,
        mean: float,
        std: float,
        horizon: int,
        distribution: str = 'normal'
    ) -> np.ndarray:
        """
        Simulate demand scenarios.
        
        Returns: (n_simulations, horizon) array
        """
        if distribution == 'normal':
            demand = np.random.normal(mean, std, (self.n_simulations, horizon))
        elif distribution == 'poisson':
            demand = np.random.poisson(mean, (self.n_simulations, horizon))
        elif distribution == 'negative_binomial':
            # Parameterize by mean and std
            p = mean / (std ** 2)
            r = mean * p / (1 - p)
            demand = np.random.negative_binomial(max(1, int(r)), min(0.99, p), 
                                                  (self.n_simulations, horizon))
        else:
            demand = np.maximum(0, np.random.normal(mean, std, (self.n_simulations, horizon)))
        
        return np.maximum(0, demand)
    
    def simulate_inventory(
        self,
        initial_inventory: float,
        reorder_point: float,
        order_quantity: float,
        lead_time: int,
        demand_scenarios: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate inventory levels under demand scenarios.
        
        Returns: (inventory_levels, stockouts) arrays
        """
        n_sim, horizon = demand_scenarios.shape
        
        inventory = np.zeros((n_sim, horizon))
        stockouts = np.zeros((n_sim, horizon))
        orders_in_transit = np.zeros((n_sim, lead_time + 1))
        
        current_inventory = np.full(n_sim, initial_inventory)
        
        for t in range(horizon):
            # Receive orders
            current_inventory += orders_in_transit[:, 0]
            orders_in_transit = np.roll(orders_in_transit, -1, axis=1)
            orders_in_transit[:, -1] = 0
            
            # Fulfill demand
            demand = demand_scenarios[:, t]
            fulfilled = np.minimum(current_inventory, demand)
            stockouts[:, t] = demand - fulfilled
            current_inventory -= fulfilled
            
            # Place orders if below reorder point
            needs_order = current_inventory <= reorder_point
            orders_in_transit[needs_order, lead_time] += order_quantity
            
            inventory[:, t] = current_inventory
        
        return inventory, stockouts
    
    def calculate_costs(
        self,
        inventory_levels: np.ndarray,
        stockouts: np.ndarray,
        holding_cost: float = 0.1,
        stockout_cost: float = 10.0,
        ordering_cost: float = 50.0,
    ) -> np.ndarray:
        """Calculate total cost for each scenario."""
        holding = holding_cost * np.sum(inventory_levels, axis=1)
        stockout = stockout_cost * np.sum(stockouts, axis=1)
        
        # Count orders (when inventory drops to ROP)
        n_orders = np.sum(np.diff(inventory_levels, axis=1) > 0, axis=1)
        ordering = ordering_cost * n_orders
        
        return holding + stockout + ordering
    
    def analyze_risk(
        self,
        demand_mean: float,
        demand_std: float,
        initial_inventory: float,
        reorder_point: float,
        order_quantity: float,
        lead_time: int = 7,
        horizon: int = 30,
        holding_cost: float = 0.1,
        stockout_cost: float = 10.0,
    ) -> RiskMetrics:
        """
        Full risk analysis via Monte Carlo simulation.
        """
        # Simulate demand
        demand = self.simulate_demand(demand_mean, demand_std, horizon)
        
        # Simulate inventory
        inventory, stockouts = self.simulate_inventory(
            initial_inventory, reorder_point, order_quantity, lead_time, demand
        )
        
        # Calculate costs
        costs = self.calculate_costs(inventory, stockouts, holding_cost, stockout_cost)
        
        # Risk metrics
        expected_cost = np.mean(costs)
        var_95 = np.percentile(costs, 95)
        cvar_95 = np.mean(costs[costs >= var_95])
        
        stockout_days = np.sum(stockouts > 0, axis=1)
        stockout_prob = np.mean(stockout_days > 0)
        service_level = 1 - np.mean(stockouts.sum(axis=1) / demand.sum(axis=1))
        
        return RiskMetrics(
            expected_cost=float(expected_cost),
            var_95=float(var_95),
            cvar_95=float(cvar_95),
            stockout_probability=float(stockout_prob),
            service_level=float(service_level),
            scenario_distribution=costs
        )
    
    def optimize_under_uncertainty(
        self,
        demand_mean: float,
        demand_std: float,
        initial_inventory: float,
        lead_time: int = 7,
    ) -> Dict:
        """
        Find optimal (ROP, Q) under uncertainty.
        """
        best_cost = np.inf
        best_params = None
        results = []
        
        for rop in range(int(demand_mean * 0.5), int(demand_mean * 2), 10):
            for q in range(20, 200, 20):
                metrics = self.analyze_risk(
                    demand_mean, demand_std,
                    initial_inventory, rop, q, lead_time
                )
                
                # Optimize CVaR (risk-averse)
                objective = 0.5 * metrics.expected_cost + 0.5 * metrics.cvar_95
                
                results.append({
                    'reorder_point': rop,
                    'order_quantity': q,
                    'expected_cost': metrics.expected_cost,
                    'cvar_95': metrics.cvar_95,
                    'service_level': metrics.service_level,
                    'objective': objective
                })
                
                if objective < best_cost:
                    best_cost = objective
                    best_params = (rop, q, metrics)
        
        return {
            'optimal_reorder_point': best_params[0],
            'optimal_order_quantity': best_params[1],
            'optimal_metrics': best_params[2],
            'all_results': results
        }
