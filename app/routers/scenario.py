"""Quantum Scenario Simulator router for Monte Carlo inventory simulations."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Tuple
import random
import math

from app.routers.upload import get_data

router = APIRouter()


class SimulationRequest(BaseModel):
    sku: str
    num_simulations: int = 1000
    days: int = 30
    demand_volatility: float = None  # inferred from data if None
    supplier_reliability: float = 0.95
    holding_cost: float = 0.5
    stockout_cost: float = 5.0
    initial_stock: int = None # inferred from data if None
    avg_daily_demand: float = None # inferred from data if None


class SimulationResult(BaseModel):
    percentiles: Dict[str, List[float]]  # p10, p50, p90 trajectories
    stockout_probability: float
    expected_cost: float
    worst_case_cost: float
    best_case_cost: float
    recommendation: str
    params_used: dict


@router.post("/simulate", response_model=SimulationResult)
async def run_simulation(request: SimulationRequest):
    """Run Monte Carlo simulation for inventory scenarios.
    
    If params are missing, attempts to infer them from uploaded CSV data
    for the specified SKU.
    """
    df = get_data()
    
    # Defaults
    avg_demand = 10.0
    volatility = 0.2
    initial_stock = 100
    
    # Infer/Override from data
    if df is not None:
        sku_data = df[df['sku'] == request.sku]
        if len(sku_data) > 0:
            avg_demand = sku_data['quantity_sold'].mean()
            if len(sku_data) > 1:
                std_dev = sku_data['quantity_sold'].std()
                volatility = std_dev / avg_demand if avg_demand > 0 else 0.2
            
            if 'quantity_on_hand' in sku_data.columns and not sku_data['quantity_on_hand'].isna().all():
                initial_stock = int(sku_data['quantity_on_hand'].iloc[-1])
            else:
                initial_stock = int(avg_demand * 10)
    
    # Use request overrides if provided
    avg_demand = request.avg_daily_demand if request.avg_daily_demand is not None else avg_demand
    volatility = request.demand_volatility if request.demand_volatility is not None else volatility
    initial_stock = request.initial_stock if request.initial_stock is not None else initial_stock
    
    scenarios: List[List[float]] = []
    total_costs: List[float] = []
    stockouts = 0
    
    # Run N simulations
    for _ in range(request.num_simulations):
        inventory = float(initial_stock)
        path = [inventory]
        cost = 0.0
        has_stockout = False
        
        for _ in range(request.days):
            # Stochastic demand (Normal distribution)
            # Clip at 0
            demand = max(0, random.normalvariate(avg_demand, avg_demand * volatility))
            
            # Simple policy: Order 5 days of stock when below 3 days
            reorder_point = avg_demand * 3
            order_qty = avg_demand * 5
            
            if inventory < reorder_point:
                # Supply shock simulation
                if random.random() > request.supplier_reliability:
                    lead_time = random.randint(3, 7)  # Delayed
                else:
                    lead_time = 2  # Normal
                
                # Assume simplified delivery logic (instant for path chart, but maybe skip replenishment for lead_time days?)
                # For this viz, we'll just add it effectively "next day" with noise to keep it simple
                if random.random() > 0.1: # 90% chance it arrives
                     inventory += order_qty
            
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
            "p10": p10_path,
            "p50": p50_path,
            "p90": p90_path
        },
        stockout_probability=round(stockout_prob * 100, 1),
        expected_cost=round(sum(total_costs) / len(total_costs), 2),
        worst_case_cost=round(total_costs[int(request.num_simulations * 0.95)], 2),
        best_case_cost=round(total_costs[int(request.num_simulations * 0.05)], 2),
        recommendation=recommendation,
        params_used={
            "avg_demand": round(avg_demand, 2),
            "volatility": round(volatility, 2),
            "initial_stock": initial_stock
        }
    )
