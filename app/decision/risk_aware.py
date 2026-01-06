"""Risk-aware optimization with uncertainty quantification."""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from scipy import stats, optimize

from app.models.inventory import ForecastPoint, DecisionResult


class RiskAwareOptimizer:
    """
    Uncertainty-aware inventory optimization.
    
    Uses CVaR (Conditional Value at Risk) to account for
    tail risks in demand uncertainty.
    """
    
    def __init__(
        self,
        holding_cost: float = 0.1,
        ordering_cost: float = 50.0,
        stockout_cost: float = 10.0,
        service_level: float = 0.95,
        lead_time: int = 7,
        risk_aversion: float = 0.8,  # CVaR quantile
        n_scenarios: int = 1000,
    ):
        """
        Initialize optimizer.
        
        Args:
            holding_cost: Cost per unit per day
            ordering_cost: Fixed cost per order
            stockout_cost: Cost per unit stockout
            service_level: Target service level
            lead_time: Lead time in days
            risk_aversion: CVaR quantile (higher = more risk averse)
            n_scenarios: Number of demand scenarios
        """
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.service_level = service_level
        self.lead_time = lead_time
        self.risk_aversion = risk_aversion
        self.n_scenarios = n_scenarios
    
    def _generate_demand_scenarios(
        self,
        forecast: List[ForecastPoint]
    ) -> np.ndarray:
        """
        Generate demand scenarios from forecast with uncertainty.
        
        Returns:
            Array of shape (n_scenarios, horizon) with demand samples
        """
        horizon = len(forecast)
        scenarios = np.zeros((self.n_scenarios, horizon))
        
        for t, point in enumerate(forecast):
            mean = point.predicted
            
            # Infer std from prediction interval if available
            if point.upper_bound and point.lower_bound:
                std = (point.upper_bound - point.lower_bound) / 4  # ~95% CI
            else:
                std = mean * 0.3  # Default 30% CV
            
            # Sample from truncated normal (no negative demand)
            scenarios[:, t] = np.maximum(0, np.random.normal(mean, std, self.n_scenarios))
        
        return scenarios
    
    def _calculate_scenario_costs(
        self,
        order_quantity: float,
        demand_scenarios: np.ndarray,
        initial_inventory: float = 0,
    ) -> np.ndarray:
        """
        Calculate costs for each demand scenario.
        
        Returns:
            Array of costs for each scenario
        """
        n_scenarios = demand_scenarios.shape[0]
        horizon = demand_scenarios.shape[1]
        costs = np.zeros(n_scenarios)
        
        for i in range(n_scenarios):
            inventory = initial_inventory + order_quantity
            total_cost = self.ordering_cost if order_quantity > 0 else 0
            
            for t in range(min(horizon, self.lead_time)):
                demand = demand_scenarios[i, t]
                
                # Holding cost
                total_cost += max(0, inventory) * self.holding_cost
                
                # Stockout cost
                if inventory < demand:
                    total_cost += (demand - inventory) * self.stockout_cost
                
                # Update inventory
                inventory = inventory - demand
            
            costs[i] = total_cost
        
        return costs
    
    def _cvar(self, costs: np.ndarray, alpha: float) -> float:
        """Calculate Conditional Value at Risk."""
        var_idx = int(len(costs) * alpha)
        sorted_costs = np.sort(costs)
        return np.mean(sorted_costs[var_idx:])
    
    def _objective(
        self,
        order_quantity: float,
        demand_scenarios: np.ndarray,
        initial_inventory: float,
        lambda_cvar: float = 0.5,
    ) -> float:
        """
        Objective: weighted combination of expected cost and CVaR.
        
        Args:
            order_quantity: Order quantity to evaluate
            demand_scenarios: Demand scenarios
            initial_inventory: Current inventory
            lambda_cvar: Weight on CVaR (0 = risk neutral, 1 = risk averse)
        
        Returns:
            Weighted objective value
        """
        costs = self._calculate_scenario_costs(
            order_quantity, demand_scenarios, initial_inventory
        )
        
        expected_cost = np.mean(costs)
        cvar_cost = self._cvar(costs, self.risk_aversion)
        
        return (1 - lambda_cvar) * expected_cost + lambda_cvar * cvar_cost
    
    def _calculate_service_level(
        self,
        order_quantity: float,
        demand_scenarios: np.ndarray,
        initial_inventory: float,
    ) -> float:
        """Calculate expected service level (fill rate)."""
        n_scenarios = demand_scenarios.shape[0]
        horizon = min(demand_scenarios.shape[1], self.lead_time)
        
        total_demand = 0
        total_fulfilled = 0
        
        for i in range(n_scenarios):
            inventory = initial_inventory + order_quantity
            
            for t in range(horizon):
                demand = demand_scenarios[i, t]
                fulfilled = min(demand, max(0, inventory))
                
                total_demand += demand
                total_fulfilled += fulfilled
                
                inventory = inventory - demand
        
        return total_fulfilled / (total_demand + 1e-6)
    
    def optimize(
        self,
        data: pd.DataFrame,
        forecast: List[ForecastPoint] = None
    ) -> DecisionResult:
        """
        Find risk-aware optimal order quantity.
        
        Args:
            data: Historical data
            forecast: Demand forecast with uncertainty
        
        Returns:
            DecisionResult with risk-aware optimization
        """
        sku = data['sku'].iloc[0]
        
        # Generate forecast if not provided
        if forecast is None:
            from app.forecasting.lightgbm_model import LightGBMForecaster
            forecaster = LightGBMForecaster()
            forecast = forecaster.forecast(data, horizon=self.lead_time * 2)
        
        # Generate demand scenarios
        demand_scenarios = self._generate_demand_scenarios(forecast)
        
        # Get initial inventory
        initial_inventory = 0
        if 'quantity_on_hand' in data.columns:
            initial_inventory = data['quantity_on_hand'].iloc[-1] or 0
        
        # Optimize with service level constraint
        mean_demand = np.mean([f.predicted for f in forecast[:self.lead_time]])
        max_order = mean_demand * self.lead_time * 3
        
        # Grid search for optimal quantity
        best_quantity = 0
        best_objective = float('inf')
        
        for q in np.linspace(0, max_order, 100):
            # Check service level constraint
            service = self._calculate_service_level(
                q, demand_scenarios, initial_inventory
            )
            
            if service >= self.service_level:
                obj = self._objective(
                    q, demand_scenarios, initial_inventory, lambda_cvar=0.5
                )
                if obj < best_objective:
                    best_objective = obj
                    best_quantity = q
        
        # If no feasible solution, find minimum to meet service level
        if best_quantity == 0:
            for q in np.linspace(0, max_order, 200):
                service = self._calculate_service_level(
                    q, demand_scenarios, initial_inventory
                )
                if service >= self.service_level:
                    best_quantity = q
                    best_objective = self._objective(
                        q, demand_scenarios, initial_inventory, lambda_cvar=0.5
                    )
                    break
        
        # Calculate final metrics
        final_service = self._calculate_service_level(
            best_quantity, demand_scenarios, initial_inventory
        )
        
        # Reorder point
        lead_time_std = np.std(demand_scenarios[:, :self.lead_time].sum(axis=1))
        reorder_point = mean_demand * self.lead_time + 1.645 * lead_time_std
        
        return DecisionResult(
            sku=sku,
            policy='risk_aware',
            reorder_point=round(reorder_point, 2),
            reorder_quantity=round(best_quantity, 2),
            expected_cost=round(best_objective, 2),
            expected_service_level=round(final_service, 4),
            uncertainty_quantile=self.risk_aversion,
        )
