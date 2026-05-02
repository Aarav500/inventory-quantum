"""API response models."""

from typing import Generic, TypeVar, Optional, List
from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    """Standard API response wrapper."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    message: Optional[str] = None


class UploadResponse(BaseModel):
    """Response for file upload."""
    filename: str
    s3_key: str
    records_processed: int
    skus_found: List[str]
    date_range: tuple[str, str]
    validation_warnings: List[str] = []


class ReportResponse(BaseModel):
    """Response with generated report."""
    report_type: str
    content: str
    s3_key: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    s3_connected: bool = False
    models_loaded: List[str] = []
