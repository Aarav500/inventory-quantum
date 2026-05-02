"""Monitoring and drift detection router."""

from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Query

from app.models.inventory import DriftMetrics
from app.models.responses import APIResponse
from app.routers.upload import get_data
from app.monitoring.drift import DriftDetector

router = APIRouter()


@router.get("/drift", response_model=APIResponse[List[DriftMetrics]])
async def check_drift(
    sku: Optional[str] = Query(None, description="SKU to check, or all if None"),
    reference_days: int = Query(default=90, description="Reference period in days"),
    test_days: int = Query(default=30, description="Test period in days"),
):
    """
    Check for distributional drift in demand patterns.
    
    Uses:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Feature-level drift detection
    """
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    if sku:
        df = df[df['sku'] == sku]
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found")
    
    detector = DriftDetector()
    results = detector.detect_drift(
        df,
        reference_days=reference_days,
        test_days=test_days
    )
    
    drifted_count = sum(1 for r in results if r.is_drifted)
    
    return APIResponse(
        success=True,
        data=results,
        message=f"Checked {len(results)} SKUs, {drifted_count} show drift"
    )


@router.get("/health")
async def monitoring_health():
    """Get monitoring system health status."""
    df = get_data()
    has_data = df is not None
    
    return APIResponse(
        success=True,
        data={
            "data_loaded": has_data,
            "total_records": len(df) if has_data else 0,
            "last_check": datetime.now().isoformat(),
        },
        message="Monitoring system healthy"
    )


@router.get("/summary")
async def monitoring_summary():
    """Get overall monitoring summary."""
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    detector = DriftDetector()
    results = detector.detect_drift(df, reference_days=90, test_days=30)
    
    drifted_skus = [r.sku for r in results if r.is_drifted]
    avg_psi = sum(r.psi for r in results) / len(results) if results else 0
    
    return APIResponse(
        success=True,
        data={
            "total_skus": len(results),
            "drifted_skus": drifted_skus,
            "drifted_count": len(drifted_skus),
            "average_psi": avg_psi,
            "timestamp": datetime.now().isoformat(),
        },
        message="Monitoring summary generated"
    )
