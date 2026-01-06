"""QUBO formulation for inventory optimization."""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from app.models.inventory import ForecastPoint, DecisionResult
from app.decision.simulated_annealing import SimulatedAnnealingSolver


class QUBOOptimizer:
    """
    Quadratic Unconstrained Binary Optimization (QUBO) formulation
    for inventory replenishment decisions.
    
    Encodes order quantities as binary vectors and constructs Q matrix
    that represents the total cost objective with constraints.
    """
    
    def __init__(
        self,
        holding_cost: float = 0.1,
        ordering_cost: float = 50.0,
        stockout_cost: float = 10.0,
        service_level: float = 0.95,
        lead_time: int = 7,
        n_bits: int = 8,  # 2^8 = 256 quantity levels
        penalty_weight: float = 100.0,
    ):
        """
        Initialize QUBO optimizer.
        
        Args:
            holding_cost: Cost per unit per day
            ordering_cost: Fixed cost per order
            stockout_cost: Cost per unit stockout
            service_level: Target service level
            lead_time: Lead time in days
            n_bits: Number of bits for binary encoding
            penalty_weight: Penalty weight for constraint violations
        """
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.service_level = service_level
        self.lead_time = lead_time
        self.n_bits = n_bits
        self.penalty_weight = penalty_weight
        self._solver = SimulatedAnnealingSolver()
    
    def _binary_to_quantity(self, binary: np.ndarray) -> int:
        """Convert binary array to integer quantity."""
        return int(sum(b * (2 ** i) for i, b in enumerate(binary)))
    
    def _quantity_to_binary(self, quantity: int) -> np.ndarray:
        """Convert integer quantity to binary array."""
        binary = np.zeros(self.n_bits, dtype=int)
        for i in range(self.n_bits):
            binary[i] = (quantity >> i) & 1
        return binary
    
    def _build_qubo_matrix(
        self,
        mean_demand: float,
        std_demand: float,
        current_inventory: float = 0,
    ) -> np.ndarray:
        """
        Construct the QUBO matrix Q where cost = x^T Q x.
        
        The objective encodes:
        1. Holding cost: proportional to expected inventory
        2. Ordering cost: fixed cost for any positive order
        3. Stockout cost: expected shortage penalty
        4. Service level constraint: penalty for low service level
        
        Args:
            mean_demand: Mean daily demand
            std_demand: Standard deviation of demand
            current_inventory: Current inventory level
        
        Returns:
            QUBO matrix Q of shape (n_bits, n_bits)
        """
        n = self.n_bits
        Q = np.zeros((n, n))
        
        # Expected demand during lead time
        lead_time_demand = mean_demand * self.lead_time
        lead_time_std = std_demand * np.sqrt(self.lead_time)
        
        # Binary encoding: quantity = sum(x_i * 2^i)
        # Holding cost contribution
        for i in range(n):
            for j in range(n):
                # E[inventory] ~ (quantity + current_inventory - demand) / 2
                # Contribution from x_i * x_j term
                coeff = (2 ** i) * (2 ** j) * self.holding_cost * self.lead_time / 2
                if i == j:
                    Q[i, i] += coeff
                else:
                    Q[i, j] += coeff / 2
                    Q[j, i] += coeff / 2
        
        # Stockout cost contribution (linear in shortage)
        for i in range(n):
            # Expected shortage decreases with order quantity
            # Approximate: shortage ~ max(0, demand - inventory - quantity)
            shortage_coeff = -self.stockout_cost * (2 ** i)
            Q[i, i] += shortage_coeff
        
        # Service level constraint penalty
        # Penalize if quantity is too low to meet service level
        required_safety = lead_time_std * 1.645  # ~95% service level
        min_order = max(0, lead_time_demand + required_safety - current_inventory)
        
        # Add penalty for orders below minimum
        for i in range(n):
            # Penalty decreases as quantity increases toward minimum
            penalty_coeff = -self.penalty_weight * (2 ** i) / (min_order + 1)
            Q[i, i] += penalty_coeff
        
        # Ordering cost (fixed cost for any order > 0)
        # This is tricky in QUBO - approximate by adding small cost per bit
        for i in range(n):
            Q[i, i] += self.ordering_cost / n
        
        return Q
    
    def _evaluate_solution(
        self,
        quantity: int,
        mean_demand: float,
        std_demand: float,
        current_inventory: float = 0,
    ) -> Tuple[float, float]:
        """
        Evaluate total cost and service level for a given order quantity.
        
        Returns:
            Tuple of (total_cost, service_level)
        """
        lead_time_demand = mean_demand * self.lead_time
        lead_time_std = std_demand * np.sqrt(self.lead_time)
        
        # Expected inventory
        expected_inventory = current_inventory + quantity - lead_time_demand
        avg_inventory = max(0, expected_inventory) / 2
        
        # Holding cost
        holding = avg_inventory * self.holding_cost * self.lead_time
        
        # Ordering cost (if any order placed)
        ordering = self.ordering_cost if quantity > 0 else 0
        
        # Stockout calculation
        if quantity + current_inventory > 0:
            z = (quantity + current_inventory - lead_time_demand) / (lead_time_std + 1e-6)
            service = min(1, max(0, 0.5 + 0.5 * np.tanh(z / 2)))
        else:
            service = 0
        
        # Expected shortage
        shortage = max(0, lead_time_demand - quantity - current_inventory) * (1 - service)
        stockout = shortage * self.stockout_cost
        
        total_cost = holding + ordering + stockout
        
        return total_cost, service
    
    def optimize(
        self,
        data: pd.DataFrame,
        forecast: List[ForecastPoint] = None
    ) -> DecisionResult:
        """
        Find optimal order quantity using QUBO + simulated annealing.
        
        Args:
            data: Historical data
            forecast: Optional demand forecast
        
        Returns:
            DecisionResult with QUBO-optimized decision
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
        
        # Get current inventory if available
        current_inventory = 0
        if 'quantity_on_hand' in df.columns:
            current_inventory = df['quantity_on_hand'].iloc[-1] or 0
        
        # Build QUBO matrix
        Q = self._build_qubo_matrix(mean_demand, std_demand, current_inventory)
        
        # Solve using simulated annealing
        best_binary = self._solver.solve(Q)
        optimal_quantity = self._binary_to_quantity(best_binary)
        
        # Evaluate solution
        total_cost, service_level = self._evaluate_solution(
            optimal_quantity, mean_demand, std_demand, current_inventory
        )
        
        # Calculate reorder point
        lead_time_std = std_demand * np.sqrt(self.lead_time)
        reorder_point = mean_demand * self.lead_time + 1.645 * lead_time_std
        
        return DecisionResult(
            sku=sku,
            policy='qubo',
            reorder_point=round(reorder_point, 2),
            reorder_quantity=round(optimal_quantity, 2),
            expected_cost=round(total_cost, 2),
            expected_service_level=round(service_level, 4),
            uncertainty_quantile=0.95,
        )
    
    def compare_solvers(
        self,
        data: pd.DataFrame,
        forecast: List[ForecastPoint] = None
    ) -> Dict[str, float]:
        """
        Compare different solvers on the same QUBO problem.
        
        Returns:
            Dictionary with solver names and their solution costs
        """
        sku = data['sku'].iloc[0]
        
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        daily_demand = df.groupby('date')['quantity_sold'].sum()
        
        mean_demand = daily_demand.mean()
        std_demand = daily_demand.std()
        
        if forecast:
            forecast_values = [f.predicted for f in forecast]
            mean_demand = np.mean(forecast_values)
        
        current_inventory = 0
        if 'quantity_on_hand' in df.columns:
            current_inventory = df['quantity_on_hand'].iloc[-1] or 0
        
        Q = self._build_qubo_matrix(mean_demand, std_demand, current_inventory)
        
        results = {}
        
        # Simulated annealing
        sa_solver = SimulatedAnnealingSolver()
        sa_binary = sa_solver.solve(Q)
        sa_quantity = self._binary_to_quantity(sa_binary)
        sa_cost, _ = self._evaluate_solution(
            sa_quantity, mean_demand, std_demand, current_inventory
        )
        results['simulated_annealing'] = round(sa_cost, 2)
        
        # Random search baseline
        best_random_cost = float('inf')
        for _ in range(1000):
            random_binary = np.random.randint(0, 2, self.n_bits)
            random_qty = self._binary_to_quantity(random_binary)
            cost, _ = self._evaluate_solution(
                random_qty, mean_demand, std_demand, current_inventory
            )
            best_random_cost = min(best_random_cost, cost)
        results['random_search'] = round(best_random_cost, 2)
        
        # Greedy (round to nearest power of 2)
        greedy_qty = int(mean_demand * self.lead_time * 1.5)
        greedy_cost, _ = self._evaluate_solution(
            greedy_qty, mean_demand, std_demand, current_inventory
        )
        results['greedy'] = round(greedy_cost, 2)
        
        return results
