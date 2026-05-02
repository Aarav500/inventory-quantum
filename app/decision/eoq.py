"""EOQ and service-level constrained policy."""

import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats
from scipy.optimize import minimize_scalar

from app.models.inventory import ForecastPoint, DecisionResult


class EOQPolicy:
    """
    Economic Order Quantity (EOQ) with service level constraint.
    
    Optimizes order quantity to minimize total cost while meeting
    a target service level (fill rate) constraint.
    """
    
    def __init__(
        self,
        holding_cost: float = 0.1,
        ordering_cost: float = 50.0,
        stockout_cost: float = 10.0,
        service_level: float = 0.95,
        lead_time: int = 7,
    ):
        """
        Initialize policy.
        
        Args:
            holding_cost: Cost per unit per day
            ordering_cost: Fixed cost per order
            stockout_cost: Cost per unit of stockout
            service_level: Target fill rate
            lead_time: Lead time in days
        """
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.service_level = service_level
        self.lead_time = lead_time
    
    def _calculate_fill_rate(
        self,
        Q: float,
        reorder_point: float,
        mean_demand: float,
        std_demand: float
    ) -> float:
        """Calculate expected fill rate for given Q and ROP."""
        lead_time_std = std_demand * np.sqrt(self.lead_time)
        
        # Expected shortage per cycle
        z = (reorder_point - mean_demand * self.lead_time) / (lead_time_std + 1e-6)
        expected_shortage = lead_time_std * (stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))
        
        # Fill rate = 1 - (expected shortage / Q)
        fill_rate = 1 - expected_shortage / (Q + 1e-6)
        return max(0, min(1, fill_rate))
    
    def _total_cost(
        self,
        Q: float,
        mean_demand: float,
        std_demand: float
    ) -> float:
        """Calculate total annual cost for given order quantity."""
        if Q <= 0:
            return float('inf')
        
        annual_demand = mean_demand * 365
        
        # Holding cost (average cycle stock)
        holding_cost = 0.5 * Q * self.holding_cost * 365
        
        # Ordering cost
        ordering_cost = (annual_demand / Q) * self.ordering_cost
        
        # Calculate reorder point for target service level
        z = stats.norm.ppf(self.service_level)
        lead_time_std = std_demand * np.sqrt(self.lead_time)
        reorder_point = mean_demand * self.lead_time + z * lead_time_std
        
        # Expected stockout cost
        expected_shortage = lead_time_std * (stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))
        stockout_cost = (annual_demand / Q) * expected_shortage * self.stockout_cost
        
        return holding_cost + ordering_cost + stockout_cost
    
    def optimize(
        self,
        data: pd.DataFrame,
        forecast: List[ForecastPoint] = None
    ) -> DecisionResult:
        """
        Find optimal order quantity and reorder point.
        
        Args:
            data: Historical data
            forecast: Optional forecast
        
        Returns:
            DecisionResult with optimized Q and ROP
        """
        sku = data['sku'].iloc[0]
        
        # Calculate demand statistics
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        daily_demand = df.groupby('date')['quantity_sold'].sum()
        
        mean_demand = daily_demand.mean()
        std_demand = daily_demand.std()
        
        if forecast:
            forecast_values = [f.predicted for f in forecast]
            mean_demand = np.mean(forecast_values)
            if forecast[0].upper_bound and forecast[0].lower_bound:
                std_demand = np.mean([
                    (f.upper_bound - f.lower_bound) / 4
                    for f in forecast
                ])
        
        # Initial EOQ
        annual_demand = mean_demand * 365
        eoq_initial = np.sqrt(2 * annual_demand * self.ordering_cost / 
                              (self.holding_cost * 365 + 1e-6))
        
        # Optimize Q to minimize total cost
        result = minimize_scalar(
            lambda Q: self._total_cost(Q, mean_demand, std_demand),
            bounds=(1, eoq_initial * 3),
            method='bounded'
        )
        
        optimal_Q = max(1, result.x)
        
        # Calculate reorder point for service level
        z = stats.norm.ppf(self.service_level)
        lead_time_std = std_demand * np.sqrt(self.lead_time)
        reorder_point = mean_demand * self.lead_time + z * lead_time_std
        
        # Calculate expected cost and actual service level
        total_cost = self._total_cost(optimal_Q, mean_demand, std_demand)
        actual_service = self._calculate_fill_rate(
            optimal_Q, reorder_point, mean_demand, std_demand
        )
        
        return DecisionResult(
            sku=sku,
            policy='eoq',
            reorder_point=round(reorder_point, 2),
            reorder_quantity=round(optimal_Q, 2),
            expected_cost=round(total_cost, 2),
            expected_service_level=round(actual_service, 4),
            uncertainty_quantile=None,
        )
