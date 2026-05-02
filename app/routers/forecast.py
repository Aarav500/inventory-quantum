"""Forecasting router."""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from app.models.inventory import (
    ForecastRequest, ForecastResult, ForecastPoint,
    BacktestRequest, BacktestMetrics
)
from app.models.responses import APIResponse
from app.routers.upload import get_data
from app.forecasting.naive import SeasonalNaiveForecaster
from app.forecasting.arima import ARIMAForecaster
from app.forecasting.lightgbm_model import LightGBMForecaster
from app.forecasting.deep_model import DeepForecaster
from app.forecasting.backtest import run_backtest

router = APIRouter()

# Model registry
MODELS = {
    'naive': SeasonalNaiveForecaster,
    'arima': ARIMAForecaster,
    'lightgbm': LightGBMForecaster,
    'deep': DeepForecaster,
}


@router.post("/{sku}", response_model=APIResponse[List[ForecastResult]])
async def generate_forecast(
    sku: str,
    request: ForecastRequest = None
):
    """Generate forecasts for a specific SKU."""
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded. Please upload CSV first.")
    
    # Filter for SKU
    sku_data = df[df['sku'] == sku]
    if len(sku_data) == 0:
        raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found")
    
    if request is None:
        request = ForecastRequest(sku=sku)
    
    results = []
    for model_name in request.models:
        if model_name not in MODELS:
            continue
        
        try:
            forecaster = MODELS[model_name]()
            predictions = forecaster.forecast(
                sku_data,
                horizon=request.horizon,
                include_intervals=request.include_intervals
            )
            
            results.append(ForecastResult(
                sku=sku,
                model=model_name,
                horizon=request.horizon,
                predictions=predictions,
                metrics=forecaster.last_metrics
            ))
        except Exception as e:
            # Log error but continue with other models
            results.append(ForecastResult(
                sku=sku,
                model=model_name,
                horizon=request.horizon,
                predictions=[],
                metrics={"error": str(e)}
            ))
    
    return APIResponse(
        success=True,
        data=results,
        message=f"Generated {len(results)} forecasts for SKU {sku}"
    )


@router.get("/compare")
async def compare_models(
    sku: Optional[str] = Query(None, description="SKU to compare, or all if None"),
    models: List[str] = Query(default=["naive", "arima", "lightgbm"]),
):
    """Compare forecast models on a dataset."""
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    if sku:
        df = df[df['sku'] == sku]
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"SKU '{sku}' not found")
    
    # Run comparison backtest
    results = run_backtest(df, models, n_windows=3, window_size=30)
    
    return APIResponse(
        success=True,
        data=results,
        message=f"Compared {len(models)} models"
    )


@router.post("/backtest", response_model=APIResponse[List[BacktestMetrics]])
async def run_backtest_endpoint(request: BacktestRequest):
    """Run rolling backtest for forecasting models."""
    df = get_data()
    if df is None:
        raise HTTPException(status_code=404, detail="No data uploaded")
    
    if request.sku:
        df = df[df['sku'] == request.sku]
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"SKU '{request.sku}' not found")
    
    results = run_backtest(
        df,
        request.models,
        n_windows=request.n_windows,
        window_size=request.window_size
    )
    
    return APIResponse(
        success=True,
        data=results,
        message=f"Backtest complete with {request.n_windows} windows"
    )
