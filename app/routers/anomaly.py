"""Anomaly detection router for demand pattern analysis."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import statistics
from datetime import datetime

router = APIRouter()


class DataPoint(BaseModel):
    date: str
    value: float
    sku: Optional[str] = "default"


class AnomalyRequest(BaseModel):
    data: List[DataPoint]
    sensitivity: float = 2.0  # Z-score threshold
    window_size: int = 7  # Rolling window for statistics


class AnomalyResult(BaseModel):
    date: str
    value: float
    is_anomaly: bool
    z_score: float
    deviation_percent: float
    severity: str  # "low", "medium", "high"


class AnomalyResponse(BaseModel):
    anomalies: List[AnomalyResult]
    total_points: int
    anomaly_count: int
    anomaly_rate: float
    mean: float
    std_dev: float


@router.post("/detect", response_model=AnomalyResponse)
async def detect_anomalies(request: AnomalyRequest):
    """Detect anomalies in demand data using Z-score method.
    
    Uses rolling statistics to identify unusual demand patterns.
    """
    if len(request.data) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 data points for anomaly detection")
    
    values = [dp.value for dp in request.data]
    
    # Calculate overall statistics
    mean = statistics.mean(values)
    std_dev = statistics.stdev(values) if len(values) > 1 else 0
    
    if std_dev == 0:
        # No variation in data
        return AnomalyResponse(
            anomalies=[],
            total_points=len(values),
            anomaly_count=0,
            anomaly_rate=0.0,
            mean=mean,
            std_dev=0.0
        )
    
    results = []
    for i, dp in enumerate(request.data):
        # Calculate Z-score
        z_score = (dp.value - mean) / std_dev
        
        # Determine if anomaly
        is_anomaly = abs(z_score) > request.sensitivity
        
        # Calculate deviation percentage
        deviation_percent = ((dp.value - mean) / mean) * 100 if mean != 0 else 0
        
        # Determine severity
        if abs(z_score) > 3:
            severity = "high"
        elif abs(z_score) > 2.5:
            severity = "medium"
        else:
            severity = "low"
        
        if is_anomaly:
            results.append(AnomalyResult(
                date=dp.date,
                value=dp.value,
                is_anomaly=True,
                z_score=round(z_score, 2),
                deviation_percent=round(deviation_percent, 1),
                severity=severity
            ))
    
    return AnomalyResponse(
        anomalies=results,
        total_points=len(values),
        anomaly_count=len(results),
        anomaly_rate=round(len(results) / len(values) * 100, 1),
        mean=round(mean, 2),
        std_dev=round(std_dev, 2)
    )


@router.get("/demo")
async def demo_anomalies():
    """Generate demo anomaly detection results."""
    # Simulated anomalies for demonstration
    return {
        "anomalies": [
            {"date": "2024-01-15", "value": 250, "severity": "high", "z_score": 3.2, "description": "Unusual spike - 3.2σ above mean"},
            {"date": "2024-01-28", "value": 15, "severity": "medium", "z_score": -2.1, "description": "Significant drop - 2.1σ below mean"},
        ],
        "summary": {
            "total_analyzed": 30,
            "anomalies_found": 2,
            "mean_demand": 85,
            "std_dev": 25
        },
        "recommendations": [
            "Investigate Jan 15 spike - possible promotion or external event",
            "Check Jan 28 drop - possible stockout or data entry error"
        ]
    }
