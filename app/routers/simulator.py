"""Simulator router for what-if scenario analysis."""

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum
import random
import uuid

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
    ScenarioPreset(
        id="preset-005",
        name="Holiday Season",
        description="Simulate holiday shopping season demand patterns",
        scenario_type=ScenarioType.SEASONAL_EVENT,
        default_parameters={"demand_multiplier": 2.5, "peak_week": 3},
        typical_impact="Extended high demand period, logistics pressure"
    ),
    ScenarioPreset(
        id="preset-006",
        name="Price Increase (+15%)",
        description="Simulate impact of 15% price increase on demand",
        scenario_type=ScenarioType.PRICE_CHANGE,
        default_parameters={"price_change_percent": 15, "elasticity": -1.2},
        typical_impact="Demand reduction, revenue impact depends on elasticity"
    ),
    ScenarioPreset(
        id="preset-007",
        name="New Competitor Entry",
        description="Simulate a new competitor entering the market",
        scenario_type=ScenarioType.NEW_COMPETITOR,
        default_parameters={"market_share_loss": 15, "price_pressure": 10},
        typical_impact="Gradual demand reduction, pricing pressure"
    ),
    ScenarioPreset(
        id="preset-008",
        name="Global Shipping Crisis",
        description="Simulate extended lead times due to shipping disruption",
        scenario_type=ScenarioType.LEAD_TIME_CHANGE,
        default_parameters={"lead_time_multiplier": 2.5, "cost_increase": 30},
        typical_impact="Need larger safety stocks, higher costs"
    ),
]

# Store simulation results
simulation_results: List[ScenarioResult] = []


def simulate_scenario(request: ScenarioRequest) -> ScenarioResult:
    """Run a scenario simulation and generate results."""
    
    # Generate impacts based on scenario type
    impacts = []
    recommendations = []
    
    if request.scenario_type == ScenarioType.DEMAND_SHOCK:
        change = request.parameters.get("change_percent", 20)
        base_demand = 10000
        new_demand = base_demand * (1 + change / 100)
        
        impacts = [
            ScenarioImpact(metric="Daily Demand", baseline_value=base_demand/30, 
                          projected_value=new_demand/30, change_percent=change, confidence=0.85),
            ScenarioImpact(metric="Stockout Risk", baseline_value=5, 
                          projected_value=min(95, 5 + abs(change) * 1.5), 
                          change_percent=abs(change) * 1.5, confidence=0.75),
            ScenarioImpact(metric="Safety Stock Need", baseline_value=500, 
                          projected_value=500 * (1 + abs(change) / 100), 
                          change_percent=abs(change), confidence=0.8),
        ]
        
        if change > 0:
            recommendations = [
                f"Increase safety stock by {int(change)}%",
                "Consider expedited shipping from suppliers",
                "Review reorder points for high-velocity items",
                "Prepare alternative supplier options"
            ]
        else:
            recommendations = [
                "Reduce incoming orders to prevent overstock",
                "Consider promotional activities to clear inventory",
                "Negotiate delayed delivery with suppliers",
                "Review storage costs"
            ]
        risk = "high" if abs(change) > 40 else "medium" if abs(change) > 20 else "low"
        
    elif request.scenario_type == ScenarioType.SUPPLY_DISRUPTION:
        delay = request.parameters.get("delay_days", 7)
        impacts = [
            ScenarioImpact(metric="Lead Time (days)", baseline_value=7, 
                          projected_value=7 + delay, change_percent=delay/7*100, confidence=0.9),
            ScenarioImpact(metric="Order Fulfillment Rate", baseline_value=98, 
                          projected_value=max(60, 98 - delay * 3), change_percent=-delay*3, confidence=0.7),
            ScenarioImpact(metric="Stockout Probability", baseline_value=2, 
                          projected_value=min(80, 2 + delay * 5), change_percent=delay*5, confidence=0.75),
        ]
        recommendations = [
            f"Increase safety stock to cover {delay} additional days",
            "Activate backup suppliers immediately",
            "Prioritize high-margin items for available stock",
            "Communicate delays to customers proactively"
        ]
        risk = "high" if delay > 10 else "medium" if delay > 5 else "low"
        
    elif request.scenario_type == ScenarioType.PROMOTION:
        multiplier = request.parameters.get("demand_multiplier", 2.0)
        discount = request.parameters.get("discount_percent", 20)
        
        impacts = [
            ScenarioImpact(metric="Daily Sales Volume", baseline_value=100, 
                          projected_value=100 * multiplier, change_percent=(multiplier-1)*100, confidence=0.7),
            ScenarioImpact(metric="Revenue per Unit", baseline_value=50, 
                          projected_value=50 * (1 - discount/100), change_percent=-discount, confidence=0.95),
            ScenarioImpact(metric="Gross Margin", baseline_value=40, 
                          projected_value=max(5, 40 - discount * 0.8), change_percent=-discount*0.8, confidence=0.85),
        ]
        recommendations = [
            f"Pre-stock {int(multiplier)}x normal inventory before promotion",
            "Ensure warehouse capacity for demand surge",
            "Staff up fulfillment team",
            "Set up inventory alerts for critical thresholds"
        ]
        risk = "medium" if multiplier > 2 else "low"
        
    else:
        # Generic impacts for other scenario types
        impacts = [
            ScenarioImpact(metric="Demand", baseline_value=1000, 
                          projected_value=1000 * random.uniform(0.7, 1.3), 
                          change_percent=random.uniform(-30, 30), confidence=0.7),
            ScenarioImpact(metric="Inventory Cost", baseline_value=50000, 
                          projected_value=50000 * random.uniform(0.9, 1.2), 
                          change_percent=random.uniform(-10, 20), confidence=0.75),
        ]
        recommendations = [
            "Monitor key metrics closely",
            "Prepare contingency plans",
            "Review supplier contracts"
        ]
        risk = "medium"
    
    # Generate description
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
        created_at=datetime.now().isoformat()
    )


@router.post("/scenario", response_model=ScenarioResult)
async def run_scenario(request: ScenarioRequest):
    """
    Run a what-if scenario simulation.
    
    Simulates the impact of various business scenarios on inventory metrics.
    """
    result = simulate_scenario(request)
    simulation_results.append(result)
    return result


@router.get("/presets", response_model=List[ScenarioPreset])
async def get_scenario_presets():
    """
    Get preset scenario configurations.
    
    Returns commonly used scenarios ready to run.
    """
    return scenario_presets


@router.post("/preset/{preset_id}", response_model=ScenarioResult)
async def run_preset_scenario(
    preset_id: str,
    duration_days: int = Query(30, ge=1, le=365)
):
    """
    Run a preset scenario.
    """
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
    """
    Get past simulation results.
    """
    return simulation_results[:limit]


@router.post("/compare")
async def compare_scenarios(request: CompareRequest):
    """
    Compare multiple scenario results.
    
    Provides side-by-side comparison of scenario impacts.
    """
    results = []
    for sid in request.scenario_ids:
        result = next((r for r in simulation_results if r.id == sid), None)
        if result:
            results.append(result)
    
    if len(results) < 2:
        return {"message": "Need at least 2 scenarios to compare", "found": len(results)}
    
    comparison = {
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
    
    return comparison


@router.get("/summary")
async def get_simulator_summary():
    """Get simulation usage summary."""
    return {
        "total_simulations": len(simulation_results),
        "available_presets": len(scenario_presets),
        "scenario_types": [st.value for st in ScenarioType],
        "recent_simulations": [
            {"id": r.id, "name": r.name, "risk": r.risk_level}
            for r in simulation_results[-5:]
        ]
    }
