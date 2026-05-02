"""Reorder router for automatic purchase order recommendations."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import math
import numpy as np
import pandas as pd

from app.routers.upload import get_data

router = APIRouter()


class UrgencyLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReorderSuggestion(BaseModel):
    """A reorder recommendation."""
    id: str
    sku: str
    product_name: str
    current_stock: int
    reorder_point: int
    safety_stock: int
    suggested_quantity: int
    unit_cost: float
    total_cost: float
    urgency: UrgencyLevel
    days_until_stockout: int
    supplier: str
    lead_time_days: int
    created_at: str
    status: str = "pending"  # pending, approved, ordered, received


class EOQCalculation(BaseModel):
    """Economic Order Quantity calculation result."""
    sku: str
    annual_demand: int
    ordering_cost: float
    holding_cost_per_unit: float
    eoq: int
    orders_per_year: float
    order_cycle_days: float
    total_annual_cost: float


class ReorderHistory(BaseModel):
    """Historical reorder record."""
    id: str
    sku: str
    quantity: int
    unit_cost: float
    total_cost: float
    ordered_at: str
    received_at: Optional[str]
    status: str
    supplier: str


# Demo suppliers
demo_suppliers = [
    "Global Supply Co.",
    "FastTrack Logistics",
    "Premier Components",
    "Reliable Distributors",
    "Direct Source Inc."
]

# Store suggestions/history in memory (in a real app this would be in DB)
suggestions_cache: List[ReorderSuggestion] = []
history_cache: List[ReorderHistory] = []


def calculate_eoq(annual_demand: int, ordering_cost: float, holding_cost: float) -> int:
    """Calculate Economic Order Quantity."""
    if holding_cost <= 0 or annual_demand <= 0:
        return 100
    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return max(1, round(eoq))


def generate_suggestions_from_data(df) -> List[ReorderSuggestion]:
    """Generate reorder suggestions from uploaded CSV data."""
    suggestions = []
    
    # Get unique SKUs
    skus = df['sku'].unique()
    
    for sku in skus:
        sku_data = df[df['sku'] == sku].sort_values('date')
        
        # Calculate demand stats
        avg_demand = sku_data['quantity_sold'].mean()
        std_demand = sku_data['quantity_sold'].std() if len(sku_data) > 1 else avg_demand * 0.2
        
        # Current stock (latest)
        if 'quantity_on_hand' in sku_data.columns and not sku_data['quantity_on_hand'].isna().all():
            current_stock = int(sku_data['quantity_on_hand'].iloc[-1])
        else:
            # Fallback if no stock info
            current_stock = int(avg_demand * 10)
            
        # Lead time
        lead_time = 7
        if 'lead_time_days' in sku_data.columns:
             val = sku_data['lead_time_days'].iloc[0]
             if not pd.isna(val):
                 lead_time = int(val)
        
        # Safety Stock formula: Z * std_dev * sqrt(lead_time)
        # Assuming 95% service level (Z=1.65)
        safety_stock = int(1.65 * std_demand * math.sqrt(lead_time))
        
        # Reorder Point = (Avg Demand * Lead Time) + Safety Stock
        reorder_point = int((avg_demand * lead_time) + safety_stock)
        
        # If below reorder point, suggest order
        if current_stock <= reorder_point:
            # Days until stockout
            days_out = int(current_stock / avg_demand) if avg_demand > 0 else 30
            
            # Urgency
            if days_out <= lead_time / 2:
                urgency = UrgencyLevel.CRITICAL
            elif days_out <= lead_time:
                urgency = UrgencyLevel.HIGH
            elif days_out <= lead_time * 2:
                urgency = UrgencyLevel.MEDIUM
            else:
                urgency = UrgencyLevel.LOW
                
            # Suggest Quantity (EOQ based)
            # Default costs if not in data
            ordering_cost = 50.0
            unit_cost = 10.0
            if 'price' in sku_data.columns:
                 val = sku_data['price'].iloc[0]
                 if not pd.isna(val):
                     unit_cost = float(val)
            
            holding_cost = unit_cost * 0.2 # 20% holding cost rate
            annual_demand = int(avg_demand * 365)
            
            suggested_qty = calculate_eoq(annual_demand, ordering_cost, holding_cost)
            
            # Ensure we order enough to cover lead time deficit
            deficit = reorder_point - current_stock
            suggested_qty = max(suggested_qty, deficit + int(avg_demand * 7))
            
            suggestions.append(ReorderSuggestion(
                id=f"sug-{uuid.uuid4().hex[:8]}",
                sku=str(sku),
                product_name=str(sku),
                current_stock=current_stock,
                reorder_point=reorder_point,
                safety_stock=safety_stock,
                suggested_quantity=suggested_qty,
                unit_cost=round(unit_cost, 2),
                total_cost=round(unit_cost * suggested_qty, 2),
                urgency=urgency,
                days_until_stockout=days_out,
                supplier=demo_suppliers[hash(str(sku)) % len(demo_suppliers)],
                lead_time_days=lead_time,
                created_at=datetime.now().isoformat(),
                status="pending"
            ))
            
    # Sort by urgency
    urgency_order = {UrgencyLevel.CRITICAL: 0, UrgencyLevel.HIGH: 1, 
                     UrgencyLevel.MEDIUM: 2, UrgencyLevel.LOW: 3}
    suggestions.sort(key=lambda x: urgency_order[x.urgency])
    return suggestions


def generate_demo_suggestions(count: int = 20) -> List[ReorderSuggestion]:
    """Generate demo reorder suggestions."""
    suggestions = []
    import random
    
    for i in range(count):
        current = random.randint(5, 100)
        reorder_point = random.randint(20, 80)
        safety = random.randint(10, 30)
        
        if current <= reorder_point:
            days_out = max(1, int((current - safety) / max(1, random.randint(5, 20))))
            
            if days_out <= 3:
                urgency = UrgencyLevel.CRITICAL
            elif days_out <= 7:
                urgency = UrgencyLevel.HIGH
            elif days_out <= 14:
                urgency = UrgencyLevel.MEDIUM
            else:
                urgency = UrgencyLevel.LOW
            
            unit_cost = round(random.uniform(5, 200), 2)
            suggested_qty = random.randint(50, 300)
            
            suggestions.append(ReorderSuggestion(
                id=f"sug-{uuid.uuid4().hex[:8]}",
                sku=f"DEMO-SKU-{str(i+1).zfill(4)}",
                product_name=f"Demo Product {i+1}",
                current_stock=current,
                reorder_point=reorder_point,
                safety_stock=safety,
                suggested_quantity=suggested_qty,
                unit_cost=unit_cost,
                total_cost=round(unit_cost * suggested_qty, 2),
                urgency=urgency,
                days_until_stockout=days_out,
                supplier=random.choice(demo_suppliers),
                lead_time_days=random.randint(3, 14),
                created_at=datetime.now().isoformat(),
                status="pending"
            ))
            
    suggestions.sort(key=lambda x: x.days_until_stockout)
    return suggestions


@router.get("/suggestions", response_model=List[ReorderSuggestion])
async def get_reorder_suggestions(
    urgency: Optional[UrgencyLevel] = Query(None, description="Filter by urgency level"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get reorder recommendations using uploaded data.
    """
    global suggestions_cache
    
    # Try to populate from data if cache is empty
    if not suggestions_cache:
        df = get_data()
        if df is not None:
            suggestions_cache = generate_suggestions_from_data(df)
        else:
            suggestions_cache = generate_demo_suggestions(30)
    
    result = suggestions_cache.copy()
    
    if urgency:
        result = [s for s in result if s.urgency == urgency]
    
    if status:
        result = [s for s in result if s.status == status]
    
    return result[:limit]


class CalculateRequest(BaseModel):
    """Request for reorder calculation."""
    sku: str
    annual_demand: Optional[int] = None
    ordering_cost: float = Field(default=50.0, description="Cost per order")
    holding_cost_rate: float = Field(default=0.25, description="Annual holding cost as % of item cost")
    unit_cost: float = Field(default=10.0, description="Cost per unit")


@router.post("/calculate", response_model=EOQCalculation)
async def calculate_optimal_quantity(request: CalculateRequest):
    """
    Calculate optimal reorder quantity using EOQ model.
    """
    annual_demand = request.annual_demand
    
    # Try to get annual demand from real data if not provided
    if annual_demand is None:
        df = get_data()
        if df is not None:
            sku_data = df[df['sku'] == request.sku]
            if len(sku_data) > 0:
                avg_daily = sku_data['quantity_sold'].mean()
                annual_demand = int(avg_daily * 365)
    
    if annual_demand is None:
        import random
        annual_demand = random.randint(1000, 10000)
        
    holding_cost = request.unit_cost * request.holding_cost_rate
    
    eoq = calculate_eoq(annual_demand, request.ordering_cost, holding_cost)
    orders_per_year = annual_demand / eoq if eoq > 0 else 12
    order_cycle = 365 / orders_per_year if orders_per_year > 0 else 30
    
    total_ordering = request.ordering_cost * orders_per_year
    total_holding = (eoq / 2) * holding_cost
    total_cost = total_ordering + total_holding
    
    return EOQCalculation(
        sku=request.sku,
        annual_demand=annual_demand,
        ordering_cost=request.ordering_cost,
        holding_cost_per_unit=round(holding_cost, 2),
        eoq=eoq,
        orders_per_year=round(orders_per_year, 1),
        order_cycle_days=round(order_cycle, 1),
        total_annual_cost=round(total_cost, 2)
    )


@router.post("/{suggestion_id}/approve")
async def approve_suggestion(suggestion_id: str):
    """
    Approve a reorder suggestion.
    """
    for sug in suggestions_cache:
        if sug.id == suggestion_id:
            sug.status = "approved"
            return {
                "message": "Suggestion approved",
                "suggestion": sug
            }
    
    raise HTTPException(status_code=404, detail="Suggestion not found")


@router.post("/approve-all")
async def approve_all_critical():
    """
    Approve all critical and high urgency suggestions.
    """
    approved = []
    for sug in suggestions_cache:
        if sug.urgency in [UrgencyLevel.CRITICAL, UrgencyLevel.HIGH] and sug.status == "pending":
            sug.status = "approved"
            approved.append(sug.id)
    
    return {
        "message": f"Approved {len(approved)} suggestions",
        "approved_ids": approved
    }


@router.post("/{suggestion_id}/reject")
async def reject_suggestion(suggestion_id: str):
    """Reject a reorder suggestion."""
    for i, sug in enumerate(suggestions_cache):
        if sug.id == suggestion_id:
            suggestions_cache.pop(i)
            return {"message": "Suggestion rejected and removed", "id": suggestion_id}
    
    raise HTTPException(status_code=404, detail="Suggestion not found")


@router.get("/history", response_model=List[ReorderHistory])
async def get_reorder_history(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=200)
):
    """
    Get historical reorder records.
    """
    # This remains demo data since we don't have a database of order history
    import random
    history = []
    
    for i in range(min(limit, 20)):
        ordered_at = datetime.now() - timedelta(days=random.randint(1, days))
        received = random.random() > 0.2
        
        history.append(ReorderHistory(
            id=f"ord-{uuid.uuid4().hex[:8]}",
            sku=f"DEMO-SKU-{str(random.randint(1, 100)).zfill(4)}",
            quantity=random.randint(50, 500),
            unit_cost=round(random.uniform(5, 200), 2),
            total_cost=round(random.uniform(500, 10000), 2),
            ordered_at=ordered_at.isoformat(),
            received_at=(ordered_at + timedelta(days=random.randint(3, 14))).isoformat() if received else None,
            status="received" if received else random.choice(["ordered", "in_transit"]),
            supplier=random.choice(demo_suppliers)
        ))
    
    history.sort(key=lambda x: x.ordered_at, reverse=True)
    return history


@router.get("/stats")
async def get_reorder_stats():
    """Get reorder statistics summary."""
    # Ensure suggestions are generated
    if not suggestions_cache:
        df = get_data()
        if df is not None:
            suggestions_cache.extend(generate_suggestions_from_data(df))
        else:
            suggestions_cache.extend(generate_demo_suggestions(30))
            
    suggestions = suggestions_cache
    
    return {
        "pending_suggestions": len([s for s in suggestions if s.status == "pending"]),
        "approved_awaiting_order": len([s for s in suggestions if s.status == "approved"]),
        "critical_items": len([s for s in suggestions if s.urgency == UrgencyLevel.CRITICAL]),
        "high_urgency_items": len([s for s in suggestions if s.urgency == UrgencyLevel.HIGH]),
        "total_pending_value": round(sum(s.total_cost for s in suggestions if s.status == "pending"), 2),
        "avg_days_to_stockout": round(sum(s.days_until_stockout for s in suggestions) / max(1, len(suggestions)), 1),
        "top_suppliers": list(set(s.supplier for s in suggestions[:10]))
    }
