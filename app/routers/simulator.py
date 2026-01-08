"""Simulator router for what-if scenario analysis."""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
import random
import uuid

from app.routers.upload import get_data

router = APIRouter()


class ScenarioType(str, Enum):
    DEMAND_SHOCK = "demand_shock"
    SUPPLY_DISRUPTION = "supply_disruption"
    PRICE_CHANGE = "price_change"
    PROMOTION = "promotion"
    SEASONAL_EVENT = "seasonal_event"
    NEW_COMPETITOR = "new_competitor"
    LEAD_TIME_CHANGE = "lead_time_change"


class ScenarioRequest(BaseModel):
    """Request to run a what-if scenario."""
    scenario_type: ScenarioType
    name: Optional[str] = None
    parameters: Dict = Field(default_factory=dict)
    sku_filter: Optional[List[str]] = None
    duration_days: int = Field(default=30, ge=1, le=365)


class ScenarioImpact(BaseModel):
    """Impact of scenario on a metric."""
    metric: str
    baseline_value: float
    projected_value: float
    change_percent: float
    confidence: float


class ScenarioResult(BaseModel):
    """Result of a what-if scenario simulation."""
    id: str
    scenario_type: ScenarioType
    name: str
    description: str
    parameters: Dict
    duration_days: int
    impacts: List[ScenarioImpact]
    recommendations: List[str]
    risk_level: str  # low, medium, high
    created_at: str
    data_source: str = "demo"


class ScenarioPreset(BaseModel):
    """Preset scenario configuration."""
    id: str
    name: str
    description: str
    scenario_type: ScenarioType
    default_parameters: Dict
    typical_impact: str


class CompareRequest(BaseModel):
    """Request to compare multiple scenarios."""
    scenario_ids: List[str]


# Preset scenarios
scenario_presets = [
    ScenarioPreset(
        id="preset-001",
        name="Demand Surge (+50%)",
        description="Simulate a sudden 50% increase in demand across all products",
        scenario_type=ScenarioType.DEMAND_SHOCK,
        default_parameters={"change_percent": 50, "ramp_days": 3},
        typical_impact="Stockout risk increases, safety stock depletes faster"
    ),
    ScenarioPreset(
        id="preset-002",
        name="Demand Drop (-30%)",
        description="Simulate a 30% decrease in demand, typical during economic downturn",
        scenario_type=ScenarioType.DEMAND_SHOCK,
        default_parameters={"change_percent": -30, "ramp_days": 7},
        typical_impact="Excess inventory accumulates, holding costs increase"
    ),
    ScenarioPreset(
        id="preset-003",
        name="Supplier Delay (2 weeks)",
        description="Simulate a 2-week delay from primary supplier",
        scenario_type=ScenarioType.SUPPLY_DISRUPTION,
        default_parameters={"delay_days": 14, "affected_suppliers": ["primary"]},
        typical_impact="Stockout risk for affected SKUs, need backup suppliers"
    ),
    ScenarioPreset(
        id="preset-004",
        name="Flash Sale Promotion",
        description="Simulate a 3-day flash sale with 40% discount",
        scenario_type=ScenarioType.PROMOTION,
        default_parameters={"discount_percent": 40, "duration_days": 3, "demand_multiplier": 3.5},
        typical_impact="Temporary demand spike, potential for stockouts"
    ),
    # ... other presets ...
]

# Store simulation results
simulation_results: List[ScenarioResult] = []


def simulate_scenario(request: ScenarioRequest) -> ScenarioResult:
    """Run a scenario simulation and generate results using uploaded data as baseline."""
    
    # Get baseline data from CSV
    df = get_data()
    data_source = "demo"
    base_demand = 10000.0
    current_stock = 50000.0
    
    if df is not None:
        data_source = "uploaded"
        # Calculate real baseline from data
        if 'quantity_sold' in df.columns:
            # Estimate monthly demand
            daily_avg = df.groupby('date')['quantity_sold'].sum().mean()
            base_demand = daily_avg * 30
            
        if 'quantity_on_hand' in df.columns:
            current_stock = df.groupby('sku')['quantity_on_hand'].last().sum()
    
    # Generate impacts based on scenario type
    impacts = []
    recommendations = []
    risk = "medium"
    
    if request.scenario_type == ScenarioType.DEMAND_SHOCK:
        change = request.parameters.get("change_percent", 20)
        new_demand = base_demand * (1 + change / 100)
        
        impacts = [
            ScenarioImpact(metric="Monthly Demand", baseline_value=base_demand, 
                          projected_value=new_demand, change_percent=change, confidence=0.85),
            ScenarioImpact(metric="Stockout Risk Score", baseline_value=15, 
                          projected_value=min(95, 15 + abs(change) * 1.2), 
                          change_percent=abs(change) * 5, confidence=0.75),
            ScenarioImpact(metric="Safety Stock Need", baseline_value=current_stock * 0.1, 
                          projected_value=(current_stock * 0.1) * (1 + abs(change) / 100), 
                          change_percent=abs(change), confidence=0.8),
        ]
        
        if change > 0:
            recommendations = [
                f"Increase annual safety stock budget by {int(change)}%",
                "Secure additional warehouse space",
                "Review reorder points"
            ]
        else:
            recommendations = [
                "Reduce purchase orders immediately",
                "Plan clearance sales"
            ]
        risk = "high" if abs(change) > 40 else "medium"
        
    elif request.scenario_type == ScenarioType.SUPPLY_DISRUPTION:
        delay = request.parameters.get("delay_days", 7)
        impacts = [
            ScenarioImpact(metric="Avg Lead Time", baseline_value=7, 
                          projected_value=7 + delay, change_percent=delay/7*100, confidence=0.9),
            ScenarioImpact(metric="Fulfillment Rate", baseline_value=98, 
                          projected_value=max(60, 98 - delay * 2), change_percent=-delay*2, confidence=0.7),
        ]
        recommendations = ["Activate backup suppliers", "Increase safety stock"]
        risk = "high" if delay > 10 else "medium"
        
    # ... handle other types generically for brevity ...
    else:
        impacts = [
            ScenarioImpact(metric="Demand", baseline_value=base_demand, 
                          projected_value=base_demand * 1.1, change_percent=10, confidence=0.7),
        ]
        recommendations = ["Monitor situation"]
        
    
    preset = next((p for p in scenario_presets if p.scenario_type == request.scenario_type), None)
    description = preset.description if preset else f"Custom {request.scenario_type.value} scenario"
    
    return ScenarioResult(
        id=f"sim-{uuid.uuid4().hex[:8]}",
        scenario_type=request.scenario_type,
        name=request.name or f"{request.scenario_type.value.replace('_', ' ').title()} Scenario",
        description=description,
        parameters=request.parameters,
        duration_days=request.duration_days,
        impacts=impacts,
        recommendations=recommendations,
        risk_level=risk,
        created_at=datetime.now().isoformat(),
        data_source=data_source
    )


@router.post("/scenario", response_model=ScenarioResult)
async def run_scenario(request: ScenarioRequest):
    """Run a what-if scenario simulation."""
    result = simulate_scenario(request)
    simulation_results.append(result)
    return result


@router.get("/presets", response_model=List[ScenarioPreset])
async def get_scenario_presets():
    """Get preset scenario configurations."""
    return scenario_presets


@router.post("/preset/{preset_id}", response_model=ScenarioResult)
async def run_preset_scenario(
    preset_id: str,
    duration_days: int = Query(30, ge=1, le=365)
):
    """Run a preset scenario."""
    preset = next((p for p in scenario_presets if p.id == preset_id), None)
    if not preset:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Preset not found")
    
    request = ScenarioRequest(
        scenario_type=preset.scenario_type,
        name=preset.name,
        parameters=preset.default_parameters,
        duration_days=duration_days
    )
    
    result = simulate_scenario(request)
    simulation_results.append(result)
    return result


@router.get("/history", response_model=List[ScenarioResult])
async def get_simulation_history(
    limit: int = Query(20, ge=1, le=100)
):
    """Get past simulation results."""
    return simulation_results[:limit]


@router.post("/compare")
async def compare_scenarios(request: CompareRequest):
    """Compare multiple scenario results."""
    results = []
    for sid in request.scenario_ids:
        result = next((r for r in simulation_results if r.id == sid), None)
        if result:
            results.append(result)
    
    if len(results) < 2:
        return {"message": "Need at least 2 scenarios to compare", "found": len(results)}
    
    return {
        "scenarios": [
            {
                "id": r.id,
                "name": r.name,
                "type": r.scenario_type.value,
                "risk_level": r.risk_level,
                "impacts_summary": {i.metric: i.change_percent for i in r.impacts}
            }
            for r in results
        ],
        "recommendation": "Choose scenario with lowest risk that meets business objectives"
    }


@router.get("/summary")
async def get_simulator_summary():
    """Get simulation usage summary."""
    return {
        "total_simulations": len(simulation_results),
        "available_presets": len(scenario_presets),
        "recent_simulations": [
            {"id": r.id, "name": r.name, "risk": r.risk_level}
            for r in simulation_results[-5:]
        ]
    }
