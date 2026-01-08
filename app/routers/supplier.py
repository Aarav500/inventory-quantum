"""Supplier Scorecard router for vendor performance analysis."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
import random

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


@router.get("/scores", response_model=List[Supplier])
async def get_supplier_scores():
    """Get performance scorecard for all suppliers."""
    
    suppliers = [
        {"id": "V-001", "name": "Global Tech Components", "type": "Electronics"},
        {"id": "V-002", "name": "Precision Fabworks", "type": "Metals"},
        {"id": "V-003", "name": "Rapid Logistics Inc", "type": "Shipping"},
        {"id": "V-004", "name": "EcoPack Solutions", "type": "Packaging"},
        {"id": "V-005", "name": "Standard Parts Co", "type": "Hardware"}
    ]
    
    scorecards = []
    
    for s in suppliers:
        # Simulate realistic manufacturing metrics
        on_time = random.uniform(70, 99)
        quality = random.uniform(85, 99.9)
        responsiveness = random.uniform(60, 95)
        
        # Calculate weighted score (5-star scale)
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
            cost_competitiveness=random.choice(["High ($$$)", "Medium ($$)", "Low ($)"])
        ))
    
    # Sort by score descending
    scorecards.sort(key=lambda x: x.overall_score, reverse=True)
    return scorecards


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
