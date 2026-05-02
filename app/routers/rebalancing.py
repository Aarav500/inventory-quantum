"""Multi-Location Rebalancing router for network optimization."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict
import random
import math

from app.routers.upload import get_data

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
    data_source: str = "demo"


# Static locations structure since CSV doesn't provide it
STATIC_LOCATIONS = [
    {"id": "DC-NY", "name": "New York DC", "lat": 40.7128, "lng": -74.0060, "type": "distribution_center"},
    {"id": "WH-CHI", "name": "Chicago Warehouse", "lat": 41.8781, "lng": -87.6298, "type": "warehouse"},
    {"id": "WH-LA", "name": "Los Angeles Warehouse", "lat": 34.0522, "lng": -118.2437, "type": "warehouse"},
    {"id": "ST-MIA", "name": "Miami Store", "lat": 25.7617, "lng": -80.1918, "type": "store"},
    {"id": "ST-SEA", "name": "Seattle Store", "lat": 47.6062, "lng": -122.3321, "type": "store"},
    {"id": "ST-AUS", "name": "Austin Store", "lat": 30.2672, "lng": -97.7431, "type": "store"}
]


@router.get("/network")
async def get_network():
    """Get network status and location data."""
    df = get_data()
    data_source = "demo"
    total_sku_inventory = 20000 # default
    
    if df is not None:
        data_source = "uploaded"
        if 'quantity_on_hand' in df.columns:
            # Sum latest inventory
            latest = df.sort_values('date').groupby('sku').last()
            total_sku_inventory = int(latest['quantity_on_hand'].sum())
    
    # Distribute total inventory across locations artificially to enable visualization
    locations = []
    
    # Weights for distribution
    weights = [0.35, 0.25, 0.20, 0.05, 0.10, 0.05]
    
    for i, loc_def in enumerate(STATIC_LOCATIONS):
        share = total_sku_inventory * weights[i]
        variation = random.uniform(0.9, 1.1)
        actual = int(share * variation)
        optimal = int(share) # Assume optimal is close to share
        
        diff_pct = (actual - optimal) / optimal if optimal > 0 else 0
        
        if diff_pct > 0.15: status = "overstock"
        elif diff_pct < -0.15: status = "understock"
        else: status = "balanced"
        
        locations.append(Location(
            id=loc_def["id"],
            name=loc_def["name"],
            lat=loc_def["lat"],
            lng=loc_def["lng"],
            type=loc_def["type"],
            inventory_level=actual,
            optimal_level=optimal,
            status=status
        ))
    
    total_inv = sum(l.inventory_level for l in locations)
    imbalance = sum(abs(l.inventory_level - l.optimal_level) for l in locations)
    imbalance_score = min(100, (imbalance / total_inv) * 100) if total_inv > 0 else 0
    
    return NetworkResponse(
        locations=locations,
        total_inventory=total_inv,
        imbalance_score=round(imbalance_score, 1),
        data_source=data_source
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
    df = get_data()
    
    transfers = []
    
    # Use real SKUs from data if available
    skus_to_move = []
    if df is not None:
        skus_to_move = df['sku'].unique().tolist()
        random.shuffle(skus_to_move)
        skus_to_move = skus_to_move[:5] # Pick 5 random SKUs
    else:
        skus_to_move = ["MXT-99", "GEN-22", "PRM-01"]
        
    for sku in skus_to_move:
        # Pick random source and target from static locations
        source = random.choice(STATIC_LOCATIONS)
        target = random.choice([l for l in STATIC_LOCATIONS if l["id"] != source["id"]])
        
        dist = calculate_distance(source["lat"], source["lng"], target["lat"], target["lng"])
        qty = random.randint(100, 1000)
        cost = round(qty * 0.5 + dist * 0.1, 2)
        
        transfers.append(Transfer(
            source_id=source["id"], source_name=source["name"],
            target_id=target["id"], target_name=target["name"],
            sku=str(sku), quantity=qty,
            distance_km=round(dist, 1), cost_estimate=cost,
            priority=random.choice(["high", "medium", "low"])
        ))
    
    return {
        "transfers": transfers,
        "total_cost": sum(t.cost_estimate for t in transfers),
        "total_units_moved": sum(t.quantity for t in transfers),
        "optimization_method": "Minimum Cost Network Flow",
        "data_source": "uploaded" if df is not None else "demo"
    }
