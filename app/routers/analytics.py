"""Analytics router for historical accuracy and trend analysis."""

from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime, timedelta
import random

router = APIRouter()


class AccuracyPoint(BaseModel):
    """Single accuracy measurement point."""
    date: str
    mape: float  # Mean Absolute Percentage Error
    rmse: float  # Root Mean Square Error
    forecast_count: int


class DemandTrend(BaseModel):
    """Demand trend for a SKU."""
    sku: str
    trend: str  # "increasing", "decreasing", "stable"
    average_demand: float
    growth_rate: float  # percentage
    data_points: List[dict]


class SeasonalPattern(BaseModel):
    """Seasonal pattern detection."""
    sku: str
    has_seasonality: bool
    peak_months: List[int]
    low_months: List[int]
    seasonality_strength: float  # 0-1


class ComparisonResult(BaseModel):
    """Forecast vs actual comparison."""
    sku: str
    period: str
    forecast: float
    actual: float
    error: float
    error_percent: float


# In-memory storage for demo
accuracy_history = []


def generate_demo_accuracy_history(days: int = 30) -> List[AccuracyPoint]:
    """Generate demo accuracy history data."""
    history = []
    base_mape = 12.0
    base_rmse = 45.0
    
    for i in range(days):
        date = (datetime.now() - timedelta(days=days - i)).strftime("%Y-%m-%d")
        # Simulate improving accuracy over time
        improvement = i * 0.1
        noise = random.uniform(-2, 2)
        
        history.append(AccuracyPoint(
            date=date,
            mape=max(5, base_mape - improvement + noise),
            rmse=max(20, base_rmse - improvement * 3 + noise * 5),
            forecast_count=random.randint(50, 200)
        ))
    
    return history


@router.get("/accuracy-history", response_model=List[AccuracyPoint])
async def get_accuracy_history(
    days: int = Query(30, ge=7, le=365, description="Number of days of history")
):
    """
    Get historical forecast accuracy metrics.
    
    Returns MAPE and RMSE trends over time to track model performance.
    """
    return generate_demo_accuracy_history(days)


@router.get("/demand-trends", response_model=List[DemandTrend])
async def get_demand_trends(
    sku: Optional[str] = Query(None, description="Filter by specific SKU"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Analyze demand trends for SKUs.
    
    Identifies increasing, decreasing, or stable demand patterns.
    """
    demo_skus = ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005",
                 "SKU-006", "SKU-007", "SKU-008", "SKU-009", "SKU-010"]
    
    if sku:
        demo_skus = [sku]
    
    trends = []
    for s in demo_skus[:limit]:
        trend_type = random.choice(["increasing", "decreasing", "stable"])
        growth = random.uniform(-15, 25) if trend_type != "stable" else random.uniform(-3, 3)
        
        # Generate historical data points
        data_points = []
        base_demand = random.uniform(100, 500)
        for i in range(12):
            month = (datetime.now() - timedelta(days=30 * (12 - i))).strftime("%Y-%m")
            demand = base_demand * (1 + growth / 100 * i / 12) + random.uniform(-20, 20)
            data_points.append({"month": month, "demand": round(demand, 2)})
        
        trends.append(DemandTrend(
            sku=s,
            trend=trend_type,
            average_demand=round(base_demand, 2),
            growth_rate=round(growth, 2),
            data_points=data_points
        ))
    
    return trends


@router.get("/seasonality", response_model=List[SeasonalPattern])
async def get_seasonality(
    sku: Optional[str] = Query(None, description="Filter by specific SKU"),
    limit: int = Query(10, ge=1, le=100)
):
    """
    Detect seasonal patterns in demand.
    
    Identifies peak and low demand months for inventory planning.
    """
    demo_skus = ["SKU-001", "SKU-002", "SKU-003", "SKU-004", "SKU-005"]
    
    if sku:
        demo_skus = [sku]
    
    patterns = []
    for s in demo_skus[:limit]:
        has_season = random.random() > 0.3  # 70% have seasonality
        
        if has_season:
            # Generate realistic seasonal patterns
            peak_months = random.sample(range(1, 13), random.randint(2, 4))
            low_months = random.sample([m for m in range(1, 13) if m not in peak_months], 
                                       random.randint(2, 3))
            strength = random.uniform(0.4, 0.9)
        else:
            peak_months = []
            low_months = []
            strength = random.uniform(0, 0.2)
        
        patterns.append(SeasonalPattern(
            sku=s,
            has_seasonality=has_season,
            peak_months=sorted(peak_months),
            low_months=sorted(low_months),
            seasonality_strength=round(strength, 2)
        ))
    
    return patterns


class CompareRequest(BaseModel):
    """Request for forecast comparison."""
    sku: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@router.post("/compare", response_model=List[ComparisonResult])
async def compare_forecast_actual(request: CompareRequest):
    """
    Compare forecasted values against actual values.
    
    Useful for model evaluation and identifying problem areas.
    """
    demo_skus = [request.sku] if request.sku else ["SKU-001", "SKU-002", "SKU-003"]
    
    results = []
    for sku in demo_skus:
        for i in range(7):  # Last 7 periods
            period = (datetime.now() - timedelta(days=i * 7)).strftime("%Y-W%W")
            forecast = random.uniform(100, 500)
            actual = forecast * random.uniform(0.85, 1.15)  # +/- 15% variance
            error = actual - forecast
            error_pct = (error / forecast) * 100
            
            results.append(ComparisonResult(
                sku=sku,
                period=period,
                forecast=round(forecast, 2),
                actual=round(actual, 2),
                error=round(error, 2),
                error_percent=round(error_pct, 2)
            ))
    
    return results


@router.get("/summary")
async def get_analytics_summary():
    """
    Get a summary of analytics metrics.
    
    Provides quick overview of forecast performance.
    """
    return {
        "current_mape": round(random.uniform(8, 15), 2),
        "current_rmse": round(random.uniform(30, 60), 2),
        "forecast_accuracy": round(random.uniform(82, 95), 1),
        "total_forecasts_30d": random.randint(500, 2000),
        "skus_with_seasonality": random.randint(15, 40),
        "trending_up": random.randint(10, 30),
        "trending_down": random.randint(5, 20),
        "stable": random.randint(20, 50)
    }
