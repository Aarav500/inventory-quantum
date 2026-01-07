"""Locations router for multi-warehouse inventory management."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
import uuid
import random

router = APIRouter()


class Location(BaseModel):
    """Warehouse or store location."""
    id: Optional[str] = None
    name: str
    type: str = Field(..., description="warehouse, store, distribution_center")
    address: str
    city: str
    state: str
    country: str = "USA"
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    capacity: int = Field(..., description="Maximum storage capacity in units")
    current_utilization: Optional[float] = None
    is_active: bool = True
    created_at: Optional[str] = None


class LocationInventory(BaseModel):
    """Inventory at a specific location."""
    sku: str
    name: str
    quantity: int
    min_level: int
    max_level: int
    reorder_point: int
    last_updated: str


class TransferRequest(BaseModel):
    """Stock transfer between locations."""
    from_location_id: str
    to_location_id: str
    sku: str
    quantity: int
    reason: Optional[str] = None


class TransferResult(BaseModel):
    """Result of a stock transfer."""
    transfer_id: str
    status: str
    from_location: str
    to_location: str
    sku: str
    quantity: int
    estimated_arrival: str
    created_at: str


class OptimizationResult(BaseModel):
    """Stock distribution optimization result."""
    location_id: str
    location_name: str
    sku: str
    current_stock: int
    optimal_stock: int
    action: str  # "increase", "decrease", "maintain"
    quantity_change: int
    reason: str


# Demo locations
demo_locations = [
    Location(
        id="loc-001",
        name="East Coast Distribution Center",
        type="distribution_center",
        address="123 Industrial Blvd",
        city="Newark",
        state="NJ",
        country="USA",
        latitude=40.7357,
        longitude=-74.1724,
        capacity=50000,
        current_utilization=72.5,
        is_active=True,
        created_at=datetime.now().isoformat()
    ),
    Location(
        id="loc-002",
        name="West Coast Warehouse",
        type="warehouse",
        address="456 Pacific Way",
        city="Los Angeles",
        state="CA",
        country="USA",
        latitude=34.0522,
        longitude=-118.2437,
        capacity=35000,
        current_utilization=68.3,
        is_active=True,
        created_at=datetime.now().isoformat()
    ),
    Location(
        id="loc-003",
        name="Central Hub",
        type="distribution_center",
        address="789 Commerce St",
        city="Dallas",
        state="TX",
        country="USA",
        latitude=32.7767,
        longitude=-96.7970,
        capacity=45000,
        current_utilization=81.2,
        is_active=True,
        created_at=datetime.now().isoformat()
    ),
    Location(
        id="loc-004",
        name="Downtown Store",
        type="store",
        address="100 Main Street",
        city="Chicago",
        state="IL",
        country="USA",
        latitude=41.8781,
        longitude=-87.6298,
        capacity=5000,
        current_utilization=55.0,
        is_active=True,
        created_at=datetime.now().isoformat()
    ),
]

locations = demo_locations.copy()


@router.get("", response_model=List[Location])
async def get_locations(
    type: Optional[str] = Query(None, description="Filter by location type"),
    is_active: Optional[bool] = Query(None, description="Filter by active status")
):
    """
    List all locations.
    
    Returns warehouses, stores, and distribution centers.
    """
    result = locations.copy()
    
    if type:
        result = [loc for loc in result if loc.type == type]
    
    if is_active is not None:
        result = [loc for loc in result if loc.is_active == is_active]
    
    return result


@router.get("/{location_id}", response_model=Location)
async def get_location(location_id: str):
    """Get a specific location by ID."""
    for loc in locations:
        if loc.id == location_id:
            return loc
    raise HTTPException(status_code=404, detail="Location not found")


@router.post("", response_model=Location)
async def create_location(location: Location):
    """
    Add a new warehouse or store location.
    """
    location.id = f"loc-{uuid.uuid4().hex[:8]}"
    location.created_at = datetime.now().isoformat()
    location.current_utilization = 0.0
    locations.append(location)
    return location


@router.put("/{location_id}", response_model=Location)
async def update_location(location_id: str, updated: Location):
    """Update an existing location."""
    for i, loc in enumerate(locations):
        if loc.id == location_id:
            updated.id = location_id
            updated.created_at = loc.created_at
            locations[i] = updated
            return updated
    raise HTTPException(status_code=404, detail="Location not found")


@router.delete("/{location_id}")
async def delete_location(location_id: str):
    """Delete a location."""
    for i, loc in enumerate(locations):
        if loc.id == location_id:
            locations.pop(i)
            return {"message": "Location deleted", "id": location_id}
    raise HTTPException(status_code=404, detail="Location not found")


@router.get("/{location_id}/inventory", response_model=List[LocationInventory])
async def get_location_inventory(
    location_id: str,
    limit: int = Query(50, ge=1, le=500)
):
    """
    Get inventory at a specific location.
    """
    # Verify location exists
    if not any(loc.id == location_id for loc in locations):
        raise HTTPException(status_code=404, detail="Location not found")
    
    # Generate demo inventory
    inventory = []
    for i in range(min(limit, 30)):
        inventory.append(LocationInventory(
            sku=f"SKU-{str(i+1).zfill(4)}",
            name=f"Product {i+1}",
            quantity=random.randint(0, 500),
            min_level=random.randint(10, 50),
            max_level=random.randint(200, 500),
            reorder_point=random.randint(20, 100),
            last_updated=datetime.now().isoformat()
        ))
    
    return inventory


@router.post("/transfer", response_model=TransferResult)
async def create_transfer(request: TransferRequest):
    """
    Initiate stock transfer between locations.
    """
    # Verify locations exist
    from_loc = None
    to_loc = None
    for loc in locations:
        if loc.id == request.from_location_id:
            from_loc = loc
        if loc.id == request.to_location_id:
            to_loc = loc
    
    if not from_loc:
        raise HTTPException(status_code=404, detail="Source location not found")
    if not to_loc:
        raise HTTPException(status_code=404, detail="Destination location not found")
    
    # Create transfer
    transfer = TransferResult(
        transfer_id=f"xfer-{uuid.uuid4().hex[:8]}",
        status="in_transit",
        from_location=from_loc.name,
        to_location=to_loc.name,
        sku=request.sku,
        quantity=request.quantity,
        estimated_arrival=(datetime.now()).isoformat(),
        created_at=datetime.now().isoformat()
    )
    
    return transfer


@router.get("/optimize", response_model=List[OptimizationResult])
async def optimize_stock_distribution(
    sku: Optional[str] = Query(None, description="Optimize for specific SKU")
):
    """
    Calculate optimal stock distribution across locations.
    
    Uses demand patterns and location proximity to recommend stock rebalancing.
    """
    results = []
    demo_skus = [sku] if sku else ["SKU-0001", "SKU-0005", "SKU-0012"]
    
    for loc in locations:
        for s in demo_skus:
            current = random.randint(50, 300)
            optimal = random.randint(80, 250)
            diff = optimal - current
            
            if diff > 20:
                action = "increase"
                reason = "High demand forecast, low current stock"
            elif diff < -20:
                action = "decrease"
                reason = "Low demand forecast, excess inventory"
            else:
                action = "maintain"
                reason = "Stock levels optimal"
            
            results.append(OptimizationResult(
                location_id=loc.id,
                location_name=loc.name,
                sku=s,
                current_stock=current,
                optimal_stock=optimal,
                action=action,
                quantity_change=abs(diff),
                reason=reason
            ))
    
    return results


@router.get("/map-data")
async def get_map_data():
    """
    Get location data formatted for map visualization.
    """
    return {
        "locations": [
            {
                "id": loc.id,
                "name": loc.name,
                "type": loc.type,
                "lat": loc.latitude,
                "lng": loc.longitude,
                "city": loc.city,
                "state": loc.state,
                "utilization": loc.current_utilization,
                "capacity": loc.capacity
            }
            for loc in locations if loc.latitude and loc.longitude
        ],
        "summary": {
            "total_locations": len(locations),
            "warehouses": len([l for l in locations if l.type == "warehouse"]),
            "distribution_centers": len([l for l in locations if l.type == "distribution_center"]),
            "stores": len([l for l in locations if l.type == "store"]),
            "total_capacity": sum(l.capacity for l in locations),
            "avg_utilization": round(sum(l.current_utilization or 0 for l in locations) / len(locations), 1)
        }
    }
