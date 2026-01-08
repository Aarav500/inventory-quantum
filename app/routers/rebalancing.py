"""Multi-Location Rebalancing router for network optimization."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import random
import math

router = APIRouter()


class Location(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    type: str  # "warehouse", "store", "distribution_center"
    inventory_level: int
    optimal_level: int
    status: str  # "overstock", "understock", "balanced"


class Transfer(BaseModel):
    source_id: str
    source_name: str
    target_id: str
    target_name: str
    sku: str
    quantity: int
    distance_km: float
    cost_estimate: float
    priority: str  # "high", "medium", "low"


class NetworkResponse(BaseModel):
    locations: List[Location]
    total_inventory: int
    imbalance_score: float  # 0-100 scale


@router.get("/network")
async def get_network():
    """Get network status and location data."""
    locations = [
        Location(
            id="DC-NY", name="New York DC", lat=40.7128, lng=-74.0060,
            type="distribution_center", inventory_level=8500, optimal_level=5000,
            status="overstock"
        ),
        Location(
            id="WH-CHI", name="Chicago Warehouse", lat=41.8781, lng=-87.6298,
            type="warehouse", inventory_level=3200, optimal_level=4000,
            status="understock"
        ),
        Location(
            id="WH-LA", name="Los Angeles Warehouse", lat=34.0522, lng=-118.2437,
            type="warehouse", inventory_level=6000, optimal_level=5500,
            status="balanced"
        ),
        Location(
            id="ST-MIA", name="Miami Store", lat=25.7617, lng=-80.1918,
            type="store", inventory_level=200, optimal_level=800,
            status="understock"
        ),
        Location(
            id="ST-SEA", name="Seattle Store", lat=47.6062, lng=-122.3321,
            type="store", inventory_level=1200, optimal_level=600,
            status="overstock"
        ),
        Location(
            id="ST-AUS", name="Austin Store", lat=30.2672, lng=-97.7431,
            type="store", inventory_level=450, optimal_level=500,
            status="balanced"
        )
    ]
    
    total_inv = sum(l.inventory_level for l in locations)
    # Simple imbalance score calculation
    imbalance = sum(abs(l.inventory_level - l.optimal_level) for l in locations)
    imbalance_score = min(100, (imbalance / total_inv) * 100)
    
    return NetworkResponse(
        locations=locations,
        total_inventory=total_inv,
        imbalance_score=round(imbalance_score, 1)
    )


def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula for distance."""
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * \
        math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c


@router.post("/optimize")
async def calculate_transfers():
    """Calculate optimal stock transfers to balance network."""
    # Simulated optimization result
    transfers = [
        Transfer(
            source_id="DC-NY", source_name="New York DC",
            target_id="ST-MIA", target_name="Miami Store",
            sku="MXT-99", quantity=500,
            distance_km=1756.0, cost_estimate=450.00, priority="high"
        ),
        Transfer(
            source_id="ST-SEA", source_name="Seattle Store",
            target_id="WH-LA", target_name="Los Angeles Warehouse",
            sku="GEN-22", quantity=400,
            distance_km=1543.0, cost_estimate=320.00, priority="medium"
        ),
        Transfer(
            source_id="DC-NY", source_name="New York DC",
            target_id="WH-CHI", target_name="Chicago Warehouse",
            sku="PRM-01", quantity=800,
            distance_km=1145.0, cost_estimate=600.00, priority="high"
        )
    ]
    
    return {
        "transfers": transfers,
        "total_cost": sum(t.cost_estimate for t in transfers),
        "total_units_moved": sum(t.quantity for t in transfers),
        "optimization_method": "Minimum Cost Network Flow"
    }
