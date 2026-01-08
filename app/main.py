"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os

from app.config import get_settings
from app.routers import upload, forecast, decision, monitoring
from app.routers import analytics, alerts, abc, locations, reports, reorder, simulator, integrations
from app.routers import export, anomaly

settings = get_settings()

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Inventory forecasting and quantum-inspired optimization platform",
    root_path=os.getenv("ROOT_PATH", ""),
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(upload.router, prefix="/upload", tags=["Upload"])
app.include_router(forecast.router, prefix="/forecast", tags=["Forecasting"])
app.include_router(decision.router, prefix="/decision", tags=["Decision"])
app.include_router(monitoring.router, prefix="/monitoring", tags=["Monitoring"])

# New feature routers
app.include_router(analytics.router, prefix="/analytics", tags=["Analytics"])
app.include_router(alerts.router, prefix="/alerts", tags=["Alerts"])
app.include_router(abc.router, prefix="/abc", tags=["ABC Analysis"])
app.include_router(locations.router, prefix="/locations", tags=["Locations"])
app.include_router(reports.router, prefix="/reports", tags=["Reports"])
app.include_router(reorder.router, prefix="/reorder", tags=["Reorder"])
app.include_router(simulator.router, prefix="/simulator", tags=["Simulator"])
app.include_router(integrations.router, prefix="/integrations", tags=["Integrations"])

# Export and Anomaly Detection
app.include_router(export.router, prefix="/export", tags=["Export"])
app.include_router(anomaly.router, prefix="/anomaly", tags=["Anomaly Detection"])

# Mount static files
# Mount static files
static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path), html=True), name="static")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": settings.api_version}

@app.get("/debug/diagnose")
async def diagnose_paths():
    """Diagnose file system paths."""
    import os
    current_file = Path(__file__).resolve()
    static_path_absolute = current_file.parent / "static"
    cwd = os.getcwd()
    ls_static = []
    if static_path_absolute.exists():
        ls_static = os.listdir(str(static_path_absolute))
    
    return {
        "cwd": cwd,
        "__file__": str(current_file),
        "static_path_absolute": str(static_path_absolute),
        "static_exists": static_path_absolute.exists(),
        "static_files": ls_static,
        "root_path_env": os.getenv("ROOT_PATH"),
    }


@app.get("/")
async def root():
    """Redirect to dashboard."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/static/index.html")

