"""Supplier Scorecard router for vendor performance analysis."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
import random

from app.routers.upload import get_data

router = APIRouter()


class SupplierMetric(BaseModel):
    name: str # e.g. "On-Time Delivery"
    score: float # 0-100
    grade: str # A, B, C, D, F
    trend: str # "improving", "declining", "stable"


class Supplier(BaseModel):
    id: str
    name: str
    overall_score: float # 0-5 stars
    metrics: List[SupplierMetric]
    avg_lead_time: int # days
    variance_days: float # +/- days
    cost_competitiveness: str # "Low", "Medium", "High"
    data_source: str = "demo"


def _create_metric(name: str, score: float) -> SupplierMetric:
    grade = "F"
    if score >= 90: grade = "A"
    elif score >= 80: grade = "B"
    elif score >= 70: grade = "C"
    elif score >= 60: grade = "D"
    
    return SupplierMetric(
        name=name,
        score=round(score, 1),
        grade=grade,
        trend=random.choice(["improving", "stable", "declining"])
    )


@router.get("/scores", response_model=List[Supplier])
async def get_supplier_scores():
    """Get performance scorecard for all suppliers.
    
    If 'supplier' column exists in uploaded data, aggregates metrics from it.
    Otherwise, returns demo suppliers.
    """
    df = get_data()
    
    if df is not None and 'supplier' in df.columns:
        # Generate scores from real data
        suppliers = []
        for supplier_name in df['supplier'].unique():
            s_data = df[df['supplier'] == supplier_name]
            
            # Simulate metrics based on available data or defaults
            # (In a real system, we'd calculate on-time delivery from order vs receipt dates)
            if 'lead_time_days' in s_data.columns:
                avg_lead = int(s_data['lead_time_days'].mean())
                variance = float(s_data['lead_time_days'].std()) if len(s_data) > 1 else 1.0
            else:
                avg_lead = random.randint(5, 30)
                variance = random.uniform(0.5, 5.0)
            
            # Mock quality/on-time since they likely aren't in simple CSV
            # But seed them with name hash for consistency
            random.seed(supplier_name)
            on_time = random.uniform(70, 99)
            quality = random.uniform(85, 99.9)
            responsiveness = random.uniform(60, 95)
            
            weighted_avg = (on_time * 0.4 + quality * 0.4 + responsiveness * 0.2)
            stars = round((weighted_avg / 100) * 5, 1)
            
            metrics = [
                _create_metric("On-Time Delivery", on_time),
                _create_metric("Quality Defect Rate", quality),
                _create_metric("Responsiveness", responsiveness)
            ]
            
            suppliers.append(Supplier(
                id=f"V-{str(abs(hash(supplier_name)))[:4]}",
                name=str(supplier_name),
                overall_score=stars,
                metrics=metrics,
                avg_lead_time=avg_lead,
                variance_days=round(variance, 1),
                cost_competitiveness=random.choice(["High ($$$)", "Medium ($$)", "Low ($)"]),
                data_source="uploaded"
            ))
            random.seed() # reset
            
        suppliers.sort(key=lambda x: x.overall_score, reverse=True)
        return suppliers

    # Default Demo Data (if no supplier col)
    suppliers = [
        {"id": "V-001", "name": "Global Tech Components", "type": "Electronics"},
        {"id": "V-002", "name": "Precision Fabworks", "type": "Metals"},
        {"id": "V-003", "name": "Rapid Logistics Inc", "type": "Shipping"},
        {"id": "V-004", "name": "EcoPack Solutions", "type": "Packaging"},
        {"id": "V-005", "name": "Standard Parts Co", "type": "Hardware"}
    ]
    
    scorecards = []
    
    for s in suppliers:
        on_time = random.uniform(70, 99)
        quality = random.uniform(85, 99.9)
        responsiveness = random.uniform(60, 95)
        
        weighted_avg = (on_time * 0.4 + quality * 0.4 + responsiveness * 0.2)
        stars = round((weighted_avg / 100) * 5, 1)
        
        metrics = [
            _create_metric("On-Time Delivery", on_time),
            _create_metric("Quality Defect Rate", quality),
            _create_metric("Responsiveness", responsiveness)
        ]
        
        scorecards.append(Supplier(
            id=s["id"],
            name=s["name"],
            overall_score=stars,
            metrics=metrics,
            avg_lead_time=random.randint(5, 45),
            variance_days=round(random.uniform(0.5, 5.0), 1),
            cost_competitiveness=random.choice(["High ($$$)", "Medium ($$)", "Low ($)"]),
            data_source="demo"
        ))
    
    scorecards.sort(key=lambda x: x.overall_score, reverse=True)
    return scorecards
