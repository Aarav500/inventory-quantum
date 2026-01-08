"""Quantum Scenario Simulator router for Monte Carlo inventory simulations."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Tuple
import random
import math

router = APIRouter()


class SimulationRequest(BaseModel):
    sku: str
    num_simulations: int = 1000
    days: int = 30
    demand_volatility: float = 0.2
    supplier_reliability: float = 0.95
    holding_cost: float = 0.5
    stockout_cost: float = 5.0
    initial_stock: int = 100


class SimulationResult(BaseModel):
    percentiles: Dict[str, List[float]]  # p10, p50, p90 trajectories
    stockout_probability: float
    expected_cost: float
    worst_case_cost: float
    best_case_cost: float
    recommendation: str


@router.post("/simulate", response_model=SimulationResult)
async def run_simulation(request: SimulationRequest):
    """Run Monte Carlo simulation for inventory scenarios."""
    
    scenarios: List[List[float]] = []
    total_costs: List[float] = []
    stockouts = 0
    
    # Run N simulations
    for _ in range(request.num_simulations):
        inventory = request.initial_stock
        path = [float(inventory)]
        cost = 0.0
        has_stockout = False
        
        for _ in range(request.days):
            # Stochastic demand (Normal distribution)
            # Base demand 10, volatility sigma
            demand = max(0, random.normalvariate(10, 10 * request.demand_volatility))
            
            # Stochastic supply (Bernoulli process for delay)
            # 10% chance of supply delay if reordering (simplified)
            
            # Simple policy: Order 50 when below 30
            if inventory < 30:
                # Supply shock simulation
                if random.random() > request.supplier_reliability:
                    lead_time = random.randint(3, 7)  # Delayed
                else:
                    lead_time = 2  # Normal
                
                # We simulate instant inventory updates for simplicity of the path array
                # In full sim, we'd queue orders. Here we just add noise to represent supply var.
                inventory += 50 if random.random() > 0.1 else 0
            
            inventory -= demand
            
            # Cost accumulation
            if inventory > 0:
                cost += inventory * request.holding_cost
            else:
                cost += abs(inventory) * request.stockout_cost
                has_stockout = True
                
            path.append(max(0, inventory))
        
        scenarios.append(path)
        total_costs.append(cost)
        if has_stockout:
            stockouts += 1
            
    # Calculate percentiles point-by-point
    p10_path = []
    p50_path = []
    p90_path = []
    
    num_points = request.days + 1
    for i in range(num_points):
        step_values = sorted([s[i] for s in scenarios])
        p10_path.append(step_values[int(request.num_simulations * 0.1)])
        p50_path.append(step_values[int(request.num_simulations * 0.5)])
        p90_path.append(step_values[int(request.num_simulations * 0.9)])
    
    # Analyze costs
    total_costs.sort()
    
    stockout_prob = stockouts / request.num_simulations
    
    recommendation = "Safe"
    if stockout_prob > 0.3:
        recommendation = "Critical Risk: Increase Safety Stock"
    elif stockout_prob > 0.1:
        recommendation = "Moderate Risk: Monitor Closely"
    elif stockout_prob < 0.01:
        recommendation = "Inefficient: Reduce Safety Stock"
    
    return SimulationResult(
        percentiles={
            "p10": p10_path,  # Worst case (low inventory)
            "p50": p50_path,  # Median
            "p90": p90_path   # Best case (high inventory)
        },
        stockout_probability=round(stockout_prob * 100, 1),
        expected_cost=round(sum(total_costs) / len(total_costs), 2),
        worst_case_cost=round(total_costs[int(request.num_simulations * 0.95)], 2),
        best_case_cost=round(total_costs[int(request.num_simulations * 0.05)], 2),
        recommendation=recommendation
    )
