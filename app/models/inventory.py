"""Pydantic models for inventory data."""

from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field


class InventoryRecord(BaseModel):
    """Single inventory record from CSV."""
    date: date
    sku: str = Field(..., description="Stock Keeping Unit identifier")
    quantity_sold: float = Field(..., ge=0, description="Units sold")
    quantity_on_hand: Optional[float] = Field(None, ge=0, description="Current inventory level")
    price: Optional[float] = Field(None, ge=0, description="Unit price")
    lead_time_days: Optional[int] = Field(None, ge=0, description="Lead time in days")
    holding_cost: Optional[float] = Field(None, ge=0, description="Holding cost per unit per day")
    ordering_cost: Optional[float] = Field(None, ge=0, description="Fixed ordering cost")
    stockout_cost: Optional[float] = Field(None, ge=0, description="Stockout cost per unit")


class InventoryDataset(BaseModel):
    """Complete inventory dataset."""
    records: List[InventoryRecord]
    skus: List[str]
    date_range: tuple[date, date]
    total_records: int


class ForecastRequest(BaseModel):
    """Request for generating forecasts."""
    sku: str
    horizon: int = Field(default=30, ge=1, le=365)
    models: List[str] = Field(
        default=["naive", "arima", "lightgbm", "deep"],
        description="Models to use for forecasting"
    )
    include_intervals: bool = Field(default=True, description="Include prediction intervals")


class ForecastPoint(BaseModel):
    """Single forecast point."""
    date: date
    predicted: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None


class ForecastResult(BaseModel):
    """Forecast result for a single model."""
    sku: str
    model: str
    horizon: int
    predictions: List[ForecastPoint]
    metrics: Optional[dict] = None


class BacktestRequest(BaseModel):
    """Request for backtesting forecasts."""
    sku: Optional[str] = Field(None, description="SKU to backtest, or all if None")
    n_windows: int = Field(default=5, ge=1, le=20)
    window_size: int = Field(default=30, ge=7, le=90)
    models: List[str] = Field(default=["naive", "arima", "lightgbm", "deep"])


class BacktestMetrics(BaseModel):
    """Backtest metrics for a model."""
    model: str
    mae: float
    rmse: float
    mape: float
    coverage_90: Optional[float] = Field(None, description="90% PI coverage")
    coverage_95: Optional[float] = Field(None, description="95% PI coverage")


class DecisionRequest(BaseModel):
    """Request for optimization decision."""
    sku: str
    forecast: Optional[List[ForecastPoint]] = None
    policy: str = Field(
        default="qubo",
        description="Policy: reorder_point, eoq, qubo, rl"
    )
    service_level: float = Field(default=0.95, ge=0.5, le=0.999)
    holding_cost: Optional[float] = None
    ordering_cost: Optional[float] = None
    stockout_cost: Optional[float] = None


class DecisionResult(BaseModel):
    """Optimization decision result."""
    sku: str
    policy: str
    reorder_point: float
    reorder_quantity: float
    expected_cost: float
    expected_service_level: float
    uncertainty_quantile: Optional[float] = None


class QUBOAblation(BaseModel):
    """QUBO ablation study result."""
    sku: str
    policies: dict[str, DecisionResult]
    qubo_improvement_pct: float
    solver_comparison: dict[str, float]


class DriftMetrics(BaseModel):
    """Distributional drift metrics."""
    sku: str
    timestamp: date
    psi: float
    ks_statistic: float
    ks_pvalue: float
    is_drifted: bool
    feature_drifts: Optional[dict[str, float]] = None


class RLState(BaseModel):
    """RL state representation."""
    inventory_level: float
    demand_forecast: List[float]
    forecast_uncertainty: List[float]
    lead_time: int
    days_since_order: int


class RLAction(BaseModel):
    """RL action."""
    order_quantity: int


class RLTransition(BaseModel):
    """Single RL transition for offline learning."""
    state: RLState
    action: RLAction
    reward: float
    next_state: RLState
    done: bool
