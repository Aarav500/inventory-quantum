"""Decision and optimization router."""

from typing import List
from fastapi import APIRouter, HTTPException, Query

from app.models.inventory import (
    DecisionRequest, DecisionResult, QUBOAblation
)
from app.models.responses import APIResponse
from app.routers.upload import get_data
from app.routers.forecast import MODELS
from app.decision.reorder_point import ReorderPointPolicy
from app.decision.eoq import EOQPolicy
from app.decision.qubo import QUBOOptimizer
from app.decision.simulated_annealing import SimulatedAnnealingSolver
from app.decision.risk_aware import RiskAwareOptimizer
from app.rl.cql import CQLPolicy

router = APIRouter()

# Policy registry
POLICIES = {
    'reorder_point': ReorderPointPolicy,
    'eoq': EOQPolicy,
    'qubo': QUBOOptimizer,
    'risk_aware': RiskAwareOptimizer,
    'rl': CQLPolicy,
}


@router.post("/optimize", response_model=APIResponse[DecisionResult])
async def optimize_decision(request: DecisionRequest):
    """Get optimal reorder decision for a SKU."""
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    sku_data = df[df['sku'] == request.sku]
    if len(sku_data) == 0:
        raise HTTPException(status_code=404, detail=f"SKU '{request.sku}' not found")
    
    # Get forecast if not provided
    forecast = request.forecast
    if forecast is None:
        forecaster = MODELS['lightgbm']()
        forecast = forecaster.forecast(sku_data, horizon=30)
    
    # Get policy
    if request.policy not in POLICIES:
        raise HTTPException(status_code=400, detail=f"Unknown policy: {request.policy}")
    
    policy_class = POLICIES[request.policy]
    policy = policy_class(
        holding_cost=request.holding_cost or 0.1,
        ordering_cost=request.ordering_cost or 50.0,
        stockout_cost=request.stockout_cost or 10.0,
        service_level=request.service_level,
    )
    
    result = policy.optimize(sku_data, forecast)
    
    return APIResponse(
        success=True,
        data=result,
        message=f"Optimized using {request.policy} policy"
    )


@router.get("/qubo/ablation", response_model=APIResponse[QUBOAblation])
async def qubo_ablation(
    sku: str = Query(..., description="SKU to analyze"),
):
    """Run ablation study comparing QUBO with baseline policies."""
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    sku_data = df[df['sku'] == sku]
    if len(sku_data) == 0:
        raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found")
    
    # Get forecast
    forecaster = MODELS['lightgbm']()
    forecast = forecaster.forecast(sku_data, horizon=30)
    
    # Run all policies
    results = {}
    for policy_name in ['reorder_point', 'eoq', 'qubo']:
        policy = POLICIES[policy_name](
            holding_cost=0.1,
            ordering_cost=50.0,
            stockout_cost=10.0,
            service_level=0.95,
        )
        results[policy_name] = policy.optimize(sku_data, forecast)
    
    # Calculate improvement
    baseline_cost = results['eoq'].expected_cost
    qubo_cost = results['qubo'].expected_cost
    improvement = (baseline_cost - qubo_cost) / baseline_cost * 100 if baseline_cost > 0 else 0
    
    # Solver comparison (SA vs random)
    qubo_optimizer = QUBOOptimizer()
    solver_comparison = qubo_optimizer.compare_solvers(sku_data, forecast)
    
    return APIResponse(
        success=True,
        data=QUBOAblation(
            sku=sku,
            policies=results,
            qubo_improvement_pct=improvement,
            solver_comparison=solver_comparison,
        ),
        message="Ablation study complete"
    )


@router.post("/batch", response_model=APIResponse[List[DecisionResult]])
async def batch_optimize(
    policy: str = Query(default="qubo"),
    service_level: float = Query(default=0.95, ge=0.5, le=0.999),
):
    """Optimize decisions for all SKUs."""
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    skus = df['sku'].unique()
    results = []
    
    for sku in skus:
        sku_data = df[df['sku'] == sku]
        
        # Get forecast
        forecaster = MODELS['lightgbm']()
        forecast = forecaster.forecast(sku_data, horizon=30)
        
        # Optimize
        policy_instance = POLICIES[policy](
            holding_cost=0.1,
            ordering_cost=50.0,
            stockout_cost=10.0,
            service_level=service_level,
        )
        result = policy_instance.optimize(sku_data, forecast)
        results.append(result)
    
    return APIResponse(
        success=True,
        data=results,
        message=f"Optimized {len(results)} SKUs"
    )
