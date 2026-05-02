"""Stockout Predictor router for forecasting inventory shortages."""

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import numpy as np

from app.routers.upload import get_data

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


def calculate_risk_level(days: int, demand_volatility: float = 0.2) -> tuple:
    """Calculate risk level and score based on days until stockout."""
    # Add volatility factor to risk score
    volatility_penalty = min(20, demand_volatility * 40)
    
    if days <= 3:
        return "critical", min(100, 95 + volatility_penalty / 4)
    elif days <= 7:
        return "high", min(100, 75 + volatility_penalty / 2)
    elif days <= 14:
        return "medium", min(100, 40 + volatility_penalty)
    else:
        return "low", max(5, 30 - days + volatility_penalty)


def calculate_predictions_from_data(df, limit: int = 20) -> List[StockoutRisk]:
    """Calculate stockout predictions from uploaded CSV data."""
    predictions = []
    
    # Get unique SKUs
    skus = df['sku'].unique()
    
    for sku in skus[:limit]:
        sku_data = df[df['sku'] == sku].copy()
        
        # Calculate average daily demand
        if 'quantity_sold' in sku_data.columns:
            avg_daily_demand = sku_data['quantity_sold'].mean()
            demand_std = sku_data['quantity_sold'].std() if len(sku_data) > 1 else 0
            demand_volatility = demand_std / avg_daily_demand if avg_daily_demand > 0 else 0
        else:
            avg_daily_demand = 10.0
            demand_volatility = 0.2
        
        # Get current stock (use latest quantity_on_hand or estimate)
        if 'quantity_on_hand' in sku_data.columns and not sku_data['quantity_on_hand'].isna().all():
            current_stock = int(sku_data['quantity_on_hand'].iloc[-1])
        else:
            # Estimate as 14 days of average demand
            current_stock = int(avg_daily_demand * 14)
        
        # Calculate days until stockout
        if avg_daily_demand > 0:
            days_until_stockout = max(1, int(current_stock / avg_daily_demand))
        else:
            days_until_stockout = 999
        
        risk_level, risk_score = calculate_risk_level(days_until_stockout, demand_volatility)
        
        # Get lead time for order quantity calculation
        lead_time = 7
        if 'lead_time_days' in sku_data.columns and not sku_data['lead_time_days'].isna().all():
            lead_time = int(sku_data['lead_time_days'].iloc[0])
        
        # Recommended order: enough for lead time + safety stock (2 weeks)
        recommended_qty = int(avg_daily_demand * (lead_time + 14))
        
        # Last order date (use last date in data as proxy)
        last_date = sku_data['date'].max()
        if hasattr(last_date, 'strftime'):
            last_order_date = last_date.strftime("%Y-%m-%d")
        else:
            last_order_date = str(last_date)[:10]
        
        predictions.append(StockoutRisk(
            sku=str(sku),
            name=str(sku),  # Use SKU as name since CSV doesn't have product names
            current_stock=max(0, current_stock),
            avg_daily_demand=round(max(0.1, avg_daily_demand), 1),
            days_until_stockout=min(999, days_until_stockout),
            risk_level=risk_level,
            risk_score=round(risk_score, 1),
            recommended_order_qty=max(1, recommended_qty),
            last_order_date=last_order_date
        ))
    
    # Sort by risk score (highest first)
    predictions.sort(key=lambda x: x.risk_score, reverse=True)
    return predictions


def generate_demo_predictions(limit: int = 20) -> List[StockoutRisk]:
    """Generate demo stockout predictions when no data is uploaded."""
    import random
    
    products = [
        ("DEMO-SKU-001", "Demo Widget A"),
        ("DEMO-SKU-002", "Demo Widget B"),
        ("DEMO-SKU-003", "Demo Gadget C"),
        ("DEMO-SKU-004", "Demo Tool D"),
        ("DEMO-SKU-005", "Demo Part E"),
    ]
    
    predictions = []
    for sku, name in products[:limit]:
        current_stock = random.randint(5, 200)
        avg_daily_demand = random.uniform(5, 50)
        days_until_stockout = max(1, int(current_stock / avg_daily_demand))
        risk_level, risk_score = calculate_risk_level(days_until_stockout)
        
        predictions.append(StockoutRisk(
            sku=sku,
            name=name + " (Demo Data)",
            current_stock=current_stock,
            avg_daily_demand=round(avg_daily_demand, 1),
            days_until_stockout=days_until_stockout,
            risk_level=risk_level,
            risk_score=round(risk_score, 1),
            recommended_order_qty=int(avg_daily_demand * 30),
            last_order_date=(datetime.now() - timedelta(days=random.randint(10, 60))).strftime("%Y-%m-%d")
        ))
    
    predictions.sort(key=lambda x: x.risk_score, reverse=True)
    return predictions


@router.get("/predictions", response_model=StockoutPrediction)
async def get_stockout_predictions(
    limit: int = Query(20, ge=1, le=100),
    risk_filter: Optional[str] = Query(None, description="Filter by risk level: critical, high, medium, low")
):
    """Get stockout predictions for all items.
    
    Uses uploaded CSV data if available, otherwise returns demo data.
    Returns items sorted by risk score (highest first).
    """
    df = get_data()
    
    if df is not None:
        predictions = calculate_predictions_from_data(df, limit)
        data_source = "uploaded"
    else:
        predictions = generate_demo_predictions(limit)
        data_source = "demo"
    
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
            "average_days_to_stockout": round(sum(p.days_until_stockout for p in predictions) / len(predictions), 1) if predictions else 0,
            "data_source": data_source
        }
    )


@router.get("/risk-score/{sku}")
async def get_risk_score(sku: str):
    """Get detailed risk score for a specific SKU."""
    df = get_data()
    
    if df is not None:
        sku_data = df[df['sku'] == sku]
        if len(sku_data) == 0:
            raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found in uploaded data")
        
        # Calculate from real data
        avg_daily_demand = sku_data['quantity_sold'].mean() if 'quantity_sold' in sku_data.columns else 10
        demand_std = sku_data['quantity_sold'].std() if len(sku_data) > 1 else avg_daily_demand * 0.2
        demand_volatility = demand_std / avg_daily_demand if avg_daily_demand > 0 else 0.2
        
        if 'quantity_on_hand' in sku_data.columns and not sku_data['quantity_on_hand'].isna().all():
            current_stock = int(sku_data['quantity_on_hand'].iloc[-1])
        else:
            current_stock = int(avg_daily_demand * 14)
        
        lead_time = int(sku_data['lead_time_days'].iloc[0]) if 'lead_time_days' in sku_data.columns else 7
        
        days_until_stockout = max(1, int(current_stock / avg_daily_demand)) if avg_daily_demand > 0 else 999
        risk_level, risk_score = calculate_risk_level(days_until_stockout, demand_volatility)
        
        data_source = "uploaded"
    else:
        # Demo mode
        import random
        current_stock = random.randint(10, 150)
        avg_daily_demand = random.uniform(5, 40)
        demand_volatility = random.uniform(0.1, 0.5)
        days_until_stockout = max(1, int(current_stock / avg_daily_demand))
        risk_level, risk_score = calculate_risk_level(days_until_stockout, demand_volatility)
        lead_time = random.randint(5, 21)
        data_source = "demo"
    
    return {
        "sku": sku,
        "current_stock": current_stock,
        "avg_daily_demand": round(avg_daily_demand, 1),
        "days_until_stockout": days_until_stockout,
        "risk_level": risk_level,
        "risk_score": round(risk_score, 1),
        "data_source": data_source,
        "factors": {
            "demand_volatility": round(demand_volatility, 2),
            "lead_time_days": lead_time,
            "supplier_reliability": 0.95,  # Default since not in CSV
            "seasonal_factor": 1.0  # Default since not calculated
        },
        "recommendations": [
            f"Order {int(avg_daily_demand * (lead_time + 14))} units to maintain buffer",
            "Consider increasing safety stock due to demand volatility" if demand_volatility > 0.3 else "Demand is stable",
            "Monitor supplier delivery times closely"
        ] if risk_level in ["critical", "high"] else [
            "Current stock levels are adequate",
            "Continue monitoring demand patterns"
        ]
    }


@router.get("/timeline")
async def get_stockout_timeline():
    """Get timeline view of when items will stock out."""
    df = get_data()
    
    if df is not None:
        predictions = calculate_predictions_from_data(df, 50)
        data_source = "uploaded"
    else:
        predictions = generate_demo_predictions(15)
        data_source = "demo"
    
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
        "total_monitored": len(predictions),
        "data_source": data_source
    }
