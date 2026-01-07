"""Reports router for PDF/Excel report generation."""

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum
import uuid

router = APIRouter()


class ReportFormat(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"


class ReportType(str, Enum):
    INVENTORY_SUMMARY = "inventory_summary"
    FORECAST_REPORT = "forecast_report"
    ABC_ANALYSIS = "abc_analysis"
    REORDER_SUGGESTIONS = "reorder_suggestions"
    PERFORMANCE_METRICS = "performance_metrics"
    LOCATION_COMPARISON = "location_comparison"
    DEMAND_TRENDS = "demand_trends"


class ReportRequest(BaseModel):
    """Request to generate a report."""
    report_type: ReportType
    format: ReportFormat = ReportFormat.PDF
    title: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    sku_filter: Optional[List[str]] = None
    location_filter: Optional[List[str]] = None
    include_charts: bool = True


class Report(BaseModel):
    """Generated report metadata."""
    id: str
    report_type: ReportType
    format: ReportFormat
    title: str
    status: str  # pending, generating, ready, failed
    created_at: str
    completed_at: Optional[str] = None
    file_size: Optional[int] = None
    download_url: Optional[str] = None
    expires_at: Optional[str] = None


class ReportTemplate(BaseModel):
    """Report template information."""
    id: str
    name: str
    description: str
    report_type: ReportType
    default_format: ReportFormat
    available_formats: List[ReportFormat]
    parameters: List[dict]


# In-memory storage for demo
reports: List[Report] = []


# Demo templates
report_templates = [
    ReportTemplate(
        id="tpl-001",
        name="Inventory Summary Report",
        description="Complete overview of current inventory levels, value, and status across all locations",
        report_type=ReportType.INVENTORY_SUMMARY,
        default_format=ReportFormat.PDF,
        available_formats=[ReportFormat.PDF, ReportFormat.EXCEL, ReportFormat.CSV],
        parameters=[
            {"name": "include_low_stock", "type": "boolean", "default": True},
            {"name": "include_valuation", "type": "boolean", "default": True}
        ]
    ),
    ReportTemplate(
        id="tpl-002",
        name="Demand Forecast Report",
        description="AI-generated demand forecasts with confidence intervals and accuracy metrics",
        report_type=ReportType.FORECAST_REPORT,
        default_format=ReportFormat.PDF,
        available_formats=[ReportFormat.PDF, ReportFormat.EXCEL],
        parameters=[
            {"name": "forecast_horizon", "type": "integer", "default": 30},
            {"name": "include_confidence", "type": "boolean", "default": True}
        ]
    ),
    ReportTemplate(
        id="tpl-003",
        name="ABC Analysis Report",
        description="Pareto analysis with item categorization and management recommendations",
        report_type=ReportType.ABC_ANALYSIS,
        default_format=ReportFormat.PDF,
        available_formats=[ReportFormat.PDF, ReportFormat.EXCEL],
        parameters=[
            {"name": "include_recommendations", "type": "boolean", "default": True}
        ]
    ),
    ReportTemplate(
        id="tpl-004",
        name="Reorder Suggestions Report",
        description="Recommended purchase orders based on forecasts and current stock levels",
        report_type=ReportType.REORDER_SUGGESTIONS,
        default_format=ReportFormat.EXCEL,
        available_formats=[ReportFormat.PDF, ReportFormat.EXCEL, ReportFormat.CSV],
        parameters=[
            {"name": "include_costs", "type": "boolean", "default": True},
            {"name": "urgent_only", "type": "boolean", "default": False}
        ]
    ),
    ReportTemplate(
        id="tpl-005",
        name="Performance Metrics Report",
        description="KPI dashboard with forecast accuracy, stockout rates, and inventory turnover",
        report_type=ReportType.PERFORMANCE_METRICS,
        default_format=ReportFormat.PDF,
        available_formats=[ReportFormat.PDF, ReportFormat.EXCEL],
        parameters=[
            {"name": "period", "type": "string", "default": "monthly"}
        ]
    ),
    ReportTemplate(
        id="tpl-006",
        name="Location Comparison Report",
        description="Side-by-side comparison of inventory across all warehouse locations",
        report_type=ReportType.LOCATION_COMPARISON,
        default_format=ReportFormat.PDF,
        available_formats=[ReportFormat.PDF, ReportFormat.EXCEL],
        parameters=[
            {"name": "include_utilization", "type": "boolean", "default": True}
        ]
    ),
    ReportTemplate(
        id="tpl-007",
        name="Demand Trends Report",
        description="Historical demand patterns with seasonality analysis and trend detection",
        report_type=ReportType.DEMAND_TRENDS,
        default_format=ReportFormat.PDF,
        available_formats=[ReportFormat.PDF, ReportFormat.EXCEL],
        parameters=[
            {"name": "lookback_months", "type": "integer", "default": 12}
        ]
    ),
]


@router.get("/templates", response_model=List[ReportTemplate])
async def get_report_templates():
    """
    Get available report templates.
    
    Returns list of report types that can be generated.
    """
    return report_templates


@router.post("/generate", response_model=Report)
async def generate_report(request: ReportRequest):
    """
    Generate a new report.
    
    Creates a report based on the specified type and parameters.
    In production, this would trigger async report generation.
    """
    report_id = f"rpt-{uuid.uuid4().hex[:8]}"
    
    # Find template for title
    template = next((t for t in report_templates if t.report_type == request.report_type), None)
    title = request.title or (template.name if template else f"{request.report_type.value} Report")
    
    report = Report(
        id=report_id,
        report_type=request.report_type,
        format=request.format,
        title=title,
        status="ready",  # In demo, immediately ready
        created_at=datetime.now().isoformat(),
        completed_at=datetime.now().isoformat(),
        file_size=1024 * (50 + hash(report_id) % 200),  # Demo file size
        download_url=f"/reports/{report_id}/download",
        expires_at=datetime.now().isoformat()  # Would be future date
    )
    
    reports.append(report)
    return report


@router.get("", response_model=List[Report])
async def list_reports(
    limit: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """
    List generated reports.
    
    Returns recent reports with their status and download links.
    """
    result = reports.copy()
    
    if status:
        result = [r for r in result if r.status == status]
    
    return result[:limit]


@router.get("/{report_id}", response_model=Report)
async def get_report(report_id: str):
    """Get report details by ID."""
    for report in reports:
        if report.id == report_id:
            return report
    raise HTTPException(status_code=404, detail="Report not found")


@router.get("/{report_id}/download")
async def download_report(report_id: str):
    """
    Download a generated report.
    
    In production, this would return the actual file.
    Demo returns mock download info.
    """
    for report in reports:
        if report.id == report_id:
            if report.status != "ready":
                raise HTTPException(status_code=400, detail="Report not ready for download")
            
            return {
                "message": "Demo mode - report download initiated",
                "report_id": report_id,
                "filename": f"{report.title.replace(' ', '_')}.{report.format.value}",
                "format": report.format.value,
                "size_bytes": report.file_size,
                "note": "In production, this would return the actual file"
            }
    
    raise HTTPException(status_code=404, detail="Report not found")


@router.delete("/{report_id}")
async def delete_report(report_id: str):
    """Delete a report."""
    for i, report in enumerate(reports):
        if report.id == report_id:
            reports.pop(i)
            return {"message": "Report deleted", "id": report_id}
    raise HTTPException(status_code=404, detail="Report not found")


@router.get("/stats/summary")
async def get_report_stats():
    """Get report generation statistics."""
    return {
        "total_reports": len(reports),
        "reports_this_week": len(reports),  # Demo
        "by_type": {
            rt.value: len([r for r in reports if r.report_type == rt])
            for rt in ReportType
        },
        "by_format": {
            rf.value: len([r for r in reports if r.format == rf])
            for rf in ReportFormat
        },
        "available_templates": len(report_templates)
    }
