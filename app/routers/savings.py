"""Cost Savings Calculator router for ROI analysis."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import random

router = APIRouter()


class SavingsInput(BaseModel):
    current_holding_cost: Optional[float] = 50000
    current_stockout_cost: Optional[float] = 25000
    current_ordering_cost: Optional[float] = 15000
    num_skus: Optional[int] = 150
    monthly_revenue: Optional[float] = 500000


class SavingsBreakdown(BaseModel):
    category: str
    before: float
    after: float
    savings: float
    savings_percent: float


class SavingsResponse(BaseModel):
    total_monthly_savings: float
    total_annual_savings: float
    roi_percent: float
    payback_months: float
    breakdown: List[SavingsBreakdown]
    recommendations: List[str]


@router.post("/calculate", response_model=SavingsResponse)
async def calculate_savings(input_data: SavingsInput):
    """Calculate potential cost savings from optimization.
    
    Analyzes current costs and estimates savings from:
    - Holding cost reduction through better demand forecasting
    - Stockout cost reduction through safety stock optimization
    - Ordering cost reduction through optimal order quantities
    """
    # Calculate savings percentages (based on typical optimization results)
    holding_reduction = random.uniform(0.15, 0.25)  # 15-25% reduction
    stockout_reduction = random.uniform(0.40, 0.60)  # 40-60% reduction
    ordering_reduction = random.uniform(0.10, 0.20)  # 10-20% reduction
    
    holding_savings = input_data.current_holding_cost * holding_reduction
    stockout_savings = input_data.current_stockout_cost * stockout_reduction
    ordering_savings = input_data.current_ordering_cost * ordering_reduction
    
    total_monthly = holding_savings + stockout_savings + ordering_savings
    total_annual = total_monthly * 12
    
    # Assume implementation cost of ~2 months of savings
    implementation_cost = total_monthly * 2
    roi = (total_annual / implementation_cost) * 100 if implementation_cost > 0 else 0
    payback = implementation_cost / total_monthly if total_monthly > 0 else 0
    
    breakdown = [
        SavingsBreakdown(
            category="Holding Costs",
            before=input_data.current_holding_cost,
            after=input_data.current_holding_cost - holding_savings,
            savings=round(holding_savings, 2),
            savings_percent=round(holding_reduction * 100, 1)
        ),
        SavingsBreakdown(
            category="Stockout Costs",
            before=input_data.current_stockout_cost,
            after=input_data.current_stockout_cost - stockout_savings,
            savings=round(stockout_savings, 2),
            savings_percent=round(stockout_reduction * 100, 1)
        ),
        SavingsBreakdown(
            category="Ordering Costs",
            before=input_data.current_ordering_cost,
            after=input_data.current_ordering_cost - ordering_savings,
            savings=round(ordering_savings, 2),
            savings_percent=round(ordering_reduction * 100, 1)
        )
    ]
    
    recommendations = [
        f"Implement safety stock optimization to reduce stockout costs by ${stockout_savings:,.0f}/month",
        f"Use demand forecasting to reduce holding costs by ${holding_savings:,.0f}/month",
        f"Apply EOQ models to reduce ordering frequency and save ${ordering_savings:,.0f}/month",
        "Consider ABC analysis to prioritize high-value items",
        "Review lead times with suppliers to improve planning accuracy"
    ]
    
    return SavingsResponse(
        total_monthly_savings=round(total_monthly, 2),
        total_annual_savings=round(total_annual, 2),
        roi_percent=round(roi, 1),
        payback_months=round(payback, 1),
        breakdown=breakdown,
        recommendations=recommendations
    )


@router.get("/summary")
async def get_savings_summary():
    """Get quick summary of potential savings."""
    monthly_savings = random.uniform(25000, 55000)
    
    return {
        "potential_monthly_savings": round(monthly_savings, 2),
        "potential_annual_savings": round(monthly_savings * 12, 2),
        "key_opportunities": [
            {
                "area": "Demand Forecasting",
                "impact": "High",
                "potential_savings": round(monthly_savings * 0.4, 2),
                "description": "ML-based forecasting reduces overstock and stockouts"
            },
            {
                "area": "Safety Stock Optimization",
                "impact": "High",
                "potential_savings": round(monthly_savings * 0.35, 2),
                "description": "Dynamic safety stock based on demand variability"
            },
            {
                "area": "Order Quantity Optimization",
                "impact": "Medium",
                "potential_savings": round(monthly_savings * 0.25, 2),
                "description": "EOQ with quantum-inspired optimization"
            }
        ],
        "current_efficiency_score": random.randint(55, 75),
        "potential_efficiency_score": random.randint(85, 95)
    }


@router.get("/comparison")
async def get_before_after_comparison():
    """Get before/after comparison of key metrics."""
    return {
        "metrics": [
            {
                "name": "Average Days of Stock",
                "before": random.randint(35, 50),
                "after": random.randint(18, 28),
                "unit": "days",
                "improvement": "better"
            },
            {
                "name": "Stockout Rate",
                "before": round(random.uniform(5, 12), 1),
                "after": round(random.uniform(1, 3), 1),
                "unit": "%",
                "improvement": "lower"
            },
            {
                "name": "Inventory Turnover",
                "before": round(random.uniform(3, 5), 1),
                "after": round(random.uniform(6, 9), 1),
                "unit": "x/year",
                "improvement": "higher"
            },
            {
                "name": "Fill Rate",
                "before": round(random.uniform(88, 93), 1),
                "after": round(random.uniform(96, 99), 1),
                "unit": "%",
                "improvement": "higher"
            },
            {
                "name": "Carrying Cost %",
                "before": round(random.uniform(22, 30), 1),
                "after": round(random.uniform(15, 20), 1),
                "unit": "%",
                "improvement": "lower"
            }
        ],
        "overall_improvement": f"{random.randint(25, 45)}%"
    }
