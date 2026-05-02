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
from app.routers import chat, stockout, savings
from app.routers import rebalancing, scenario, supplier

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

# Health and diagnostic endpoints must be registered before app.mount() so that
# Starlette's route-match loop finds them before the StaticFiles Mount, which
# returns 405 for HEAD on paths it does not own and would short-circuit the loop.
from fastapi.responses import FileResponse


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check():
    return {"status": "healthy", "version": settings.api_version}


@app.get("/debug/diagnose")
async def diagnose_paths():
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

# New AI Features
app.include_router(chat.router, prefix="/chat", tags=["AI Chat"])
app.include_router(stockout.router, prefix="/stockout", tags=["Stockout Predictor"])
app.include_router(savings.router, prefix="/savings", tags=["Cost Savings"])

# Phase 2: Advanced Features
app.include_router(rebalancing.router, prefix="/rebalancing", tags=["Rebalancing"])
app.include_router(scenario.router, prefix="/scenario", tags=["Scenario Simulator"])
app.include_router(supplier.router, prefix="/supplier", tags=["Supplier Scorecard"])

# Mount static files — must come after /health and /debug routes
static_path = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_path), html=True), name="static")


@app.get("/")
async def root():
    return FileResponse(static_path / "index.html")


@app.get("/{page}.html")
async def serve_html(page: str):
    file_path = static_path / f"{page}.html"
    if file_path.exists():
        return FileResponse(file_path)
    return {"detail": "Not Found", "version": settings.api_version}, 404

