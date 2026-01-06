"""Reorder point policy baseline."""

import numpy as np
import pandas as pd
from typing import List, Optional
from scipy import stats

from app.models.inventory import ForecastPoint, DecisionResult


class ReorderPointPolicy:
    """
    Traditional reorder point (ROP) policy.
    
    ROP = Expected demand during lead time + Safety stock
    Safety stock = z * σ_L where σ_L is demand std during lead time
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
            service_level: Target service level (fill rate)
            lead_time: Lead time in days
        """
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.service_level = service_level
        self.lead_time = lead_time
    
    def optimize(
        self,
        data: pd.DataFrame,
        forecast: List[ForecastPoint] = None
    ) -> DecisionResult:
        """
        Calculate optimal reorder point and quantity.
        
        Args:
            data: Historical data
            forecast: Optional forecast (uses history if not provided)
        
        Returns:
            DecisionResult with ROP and order quantity
        """
        # Get SKU
        sku = data['sku'].iloc[0]
        
        # Calculate demand statistics from history
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        daily_demand = df.groupby('date')['quantity_sold'].sum()
        
        mean_demand = daily_demand.mean()
        std_demand = daily_demand.std()
        
        # Use forecast if provided
        if forecast:
            forecast_values = [f.predicted for f in forecast]
            mean_demand = np.mean(forecast_values)
            
            # Use forecast uncertainty if available
            if forecast[0].upper_bound and forecast[0].lower_bound:
                # Approximate std from prediction interval
                std_demand = np.mean([
                    (f.upper_bound - f.lower_bound) / 4  # ~4 std in 95% CI
                    for f in forecast
                ])
        
        # Calculate z-score for service level
        z = stats.norm.ppf(self.service_level)
        
        # Demand during lead time
        lead_time_demand = mean_demand * self.lead_time
        lead_time_std = std_demand * np.sqrt(self.lead_time)
        
        # Safety stock
        safety_stock = z * lead_time_std
        
        # Reorder point
        reorder_point = lead_time_demand + safety_stock
        
        # Economic Order Quantity (EOQ) for order size
        annual_demand = mean_demand * 365
        eoq = np.sqrt(2 * annual_demand * self.ordering_cost / (self.holding_cost * 365))
        
        # Expected costs (simplified)
        holding_cost = 0.5 * eoq * self.holding_cost * 365  # Avg inventory * cost
        ordering_cost = (annual_demand / eoq) * self.ordering_cost
        
        # Stockout probability
        stockout_prob = 1 - self.service_level
        expected_stockout = stockout_prob * mean_demand * self.lead_time
        stockout_cost = expected_stockout * self.stockout_cost * (365 / self.lead_time)
        
        total_cost = holding_cost + ordering_cost + stockout_cost
        
        return DecisionResult(
            sku=sku,
            policy='reorder_point',
            reorder_point=round(reorder_point, 2),
            reorder_quantity=round(eoq, 2),
            expected_cost=round(total_cost, 2),
            expected_service_level=self.service_level,
            uncertainty_quantile=None,
        )
