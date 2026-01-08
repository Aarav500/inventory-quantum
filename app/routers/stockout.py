"""Stockout Predictor router for forecasting inventory shortages."""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import random

router = APIRouter()


class StockoutRisk(BaseModel):
    sku: str
    name: str
    current_stock: int
    avg_daily_demand: float
    days_until_stockout: int
    risk_level: str  # "critical", "high", "medium", "low"
    risk_score: float  # 0-100
    recommended_order_qty: int
    last_order_date: str


class StockoutPrediction(BaseModel):
    predictions: List[StockoutRisk]
    summary: dict


def calculate_risk_level(days: int) -> tuple:
    """Calculate risk level and score based on days until stockout."""
    if days <= 3:
        return "critical", 95 + random.uniform(0, 5)
    elif days <= 7:
        return "high", 75 + random.uniform(0, 20)
    elif days <= 14:
        return "medium", 40 + random.uniform(0, 35)
    else:
        return "low", random.uniform(5, 40)


def generate_demo_predictions(limit: int = 20) -> List[StockoutRisk]:
    """Generate demo stockout predictions."""
    products = [
        ("SKU-001", "Premium Widget A"),
        ("SKU-002", "Standard Widget B"),
        ("SKU-003", "Economy Widget C"),
        ("SKU-004", "Industrial Gadget X"),
        ("SKU-005", "Consumer Gadget Y"),
        ("SKU-006", "Professional Tool Pro"),
        ("SKU-007", "Basic Tool Lite"),
        ("SKU-008", "Advanced Module Alpha"),
        ("SKU-009", "Core Module Beta"),
        ("SKU-010", "Essential Part Gamma"),
        ("SKU-011", "Premium Accessory Delta"),
        ("SKU-012", "Standard Accessory Epsilon"),
        ("SKU-013", "Heavy Duty Component"),
        ("SKU-014", "Lightweight Component"),
        ("SKU-015", "Specialty Item Zeta"),
    ]
    
    predictions = []
    for sku, name in products[:limit]:
        current_stock = random.randint(5, 200)
        avg_daily_demand = random.uniform(5, 50)
        days_until_stockout = max(1, int(current_stock / avg_daily_demand))
        risk_level, risk_score = calculate_risk_level(days_until_stockout)
        
        predictions.append(StockoutRisk(
            sku=sku,
            name=name,
            current_stock=current_stock,
            avg_daily_demand=round(avg_daily_demand, 1),
            days_until_stockout=days_until_stockout,
            risk_level=risk_level,
            risk_score=round(risk_score, 1),
            recommended_order_qty=int(avg_daily_demand * 30),  # 30-day supply
            last_order_date=(datetime.now() - timedelta(days=random.randint(10, 60))).strftime("%Y-%m-%d")
        ))
    
    # Sort by risk score (highest first)
    predictions.sort(key=lambda x: x.risk_score, reverse=True)
    return predictions


@router.get("/predictions", response_model=StockoutPrediction)
async def get_stockout_predictions(
    limit: int = Query(20, ge=1, le=100),
    risk_filter: Optional[str] = Query(None, description="Filter by risk level: critical, high, medium, low")
):
    """Get stockout predictions for all items.
    
    Returns items sorted by risk score (highest first).
    """
    predictions = generate_demo_predictions(limit)
    
    if risk_filter:
        predictions = [p for p in predictions if p.risk_level == risk_filter.lower()]
    
    critical_count = len([p for p in predictions if p.risk_level == "critical"])
    high_count = len([p for p in predictions if p.risk_level == "high"])
    
    return StockoutPrediction(
        predictions=predictions,
        summary={
            "total_items": len(predictions),
            "critical_count": critical_count,
            "high_count": high_count,
            "items_needing_action": critical_count + high_count,
            "average_days_to_stockout": round(sum(p.days_until_stockout for p in predictions) / len(predictions), 1) if predictions else 0
        }
    )


@router.get("/risk-score/{sku}")
async def get_risk_score(sku: str):
    """Get detailed risk score for a specific SKU."""
    current_stock = random.randint(10, 150)
    avg_daily_demand = random.uniform(5, 40)
    days_until_stockout = max(1, int(current_stock / avg_daily_demand))
    risk_level, risk_score = calculate_risk_level(days_until_stockout)
    
    return {
        "sku": sku,
        "current_stock": current_stock,
        "avg_daily_demand": round(avg_daily_demand, 1),
        "days_until_stockout": days_until_stockout,
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "factors": {
            "demand_volatility": round(random.uniform(0.1, 0.5), 2),
            "lead_time_days": random.randint(5, 21),
            "supplier_reliability": round(random.uniform(0.85, 0.99), 2),
            "seasonal_factor": round(random.uniform(0.8, 1.3), 2)
        },
        "recommendations": [
            f"Order {int(avg_daily_demand * 30)} units to maintain 30-day supply",
            "Consider increasing safety stock due to demand volatility",
            "Monitor supplier delivery times closely"
        ] if risk_level in ["critical", "high"] else [
            "Current stock levels are adequate",
            "Continue monitoring demand patterns"
        ]
    }


@router.get("/timeline")
async def get_stockout_timeline():
    """Get timeline view of when items will stock out."""
    predictions = generate_demo_predictions(15)
    
    # Group by timeframe
    timeline = {
        "next_3_days": [p.dict() for p in predictions if p.days_until_stockout <= 3],
        "next_7_days": [p.dict() for p in predictions if 3 < p.days_until_stockout <= 7],
        "next_14_days": [p.dict() for p in predictions if 7 < p.days_until_stockout <= 14],
        "next_30_days": [p.dict() for p in predictions if 14 < p.days_until_stockout <= 30],
    }
    
    return {
        "timeline": timeline,
        "action_required": len(timeline["next_3_days"]) + len(timeline["next_7_days"]),
        "total_monitored": len(predictions)
    }
