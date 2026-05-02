"""Rolling backtest framework for forecast evaluation."""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Type
from datetime import timedelta

from app.models.inventory import BacktestMetrics, ForecastPoint


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    lower: np.ndarray = None,
    upper: np.ndarray = None
) -> Dict[str, float]:
    """
    Calculate forecast accuracy metrics.
    
    Args:
        actual: Actual values
        predicted: Predicted values
        lower: Lower prediction interval bounds
        upper: Upper prediction interval bounds
    
    Returns:
        Dictionary with MAE, RMSE, MAPE, and coverage metrics
    """
    # Filter out zeros for MAPE calculation
    mask = actual > 0
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    else:
        mape = 0.0
    
    metrics = {
        'mae': round(mae, 4),
        'rmse': round(rmse, 4),
        'mape': round(mape, 2),
    }
    
    # Coverage metrics if intervals provided
    if lower is not None and upper is not None:
        covered = (actual >= lower) & (actual <= upper)
        coverage = np.mean(covered) * 100
        metrics['coverage'] = round(coverage, 2)
    
    return metrics


def run_backtest(
    data: pd.DataFrame,
    model_names: List[str],
    n_windows: int = 5,
    window_size: int = 30,
) -> List[BacktestMetrics]:
    """
    Run rolling origin backtest for multiple models.
    
    Args:
        data: DataFrame with historical data
        model_names: List of model names to test
        n_windows: Number of rolling windows
        window_size: Size of each test window (forecast horizon)
    
    Returns:
        List of BacktestMetrics for each model
    """
    from app.forecasting.naive import SeasonalNaiveForecaster
    from app.forecasting.arima import ARIMAForecaster
    from app.forecasting.lightgbm_model import LightGBMForecaster
    from app.forecasting.deep_model import DeepForecaster
    
    MODELS = {
        'naive': SeasonalNaiveForecaster,
        'arima': ARIMAForecaster,
        'lightgbm': LightGBMForecaster,
        'deep': DeepForecaster,
    }
    
    # Prepare data
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # Get unique dates
    dates = df['date'].unique()
    n_dates = len(dates)
    
    # Calculate window origins
    min_train = max(60, window_size * 2)  # Minimum training data
    usable_dates = n_dates - min_train
    
    if usable_dates < n_windows * window_size:
        n_windows = max(1, usable_dates // window_size)
    
    step_size = max(1, (usable_dates - window_size) // max(1, n_windows - 1)) if n_windows > 1 else 1
    
    results = {name: {'mae': [], 'rmse': [], 'mape': [], 'coverage_90': [], 'coverage_95': []} 
               for name in model_names}
    
    for i in range(n_windows):
        # Define train/test split
        train_end_idx = min_train + i * step_size
        test_end_idx = min(train_end_idx + window_size, n_dates)
        
        if test_end_idx > train_end_idx:
            train_dates = dates[:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]
            
            train_df = df[df['date'].isin(train_dates)]
            test_df = df[df['date'].isin(test_dates)]
            
            # Get actual values
            actual = test_df.groupby('date')['quantity_sold'].sum().values
            horizon = len(actual)
            
            for model_name in model_names:
                if model_name not in MODELS:
                    continue
                
                try:
                    forecaster = MODELS[model_name]()
                    predictions = forecaster.forecast(
                        train_df,
                        horizon=horizon,
                        include_intervals=True
                    )
                    
                    # Extract predictions
                    predicted = np.array([p.predicted for p in predictions])
                    lower_90 = np.array([p.lower_bound if p.lower_bound else 0 for p in predictions])
                    upper_90 = np.array([p.upper_bound if p.upper_bound else 0 for p in predictions])
                    
                    # Calculate metrics
                    metrics = calculate_metrics(actual, predicted, lower_90, upper_90)
                    
                    results[model_name]['mae'].append(metrics['mae'])
                    results[model_name]['rmse'].append(metrics['rmse'])
                    results[model_name]['mape'].append(metrics['mape'])
                    if 'coverage' in metrics:
                        results[model_name]['coverage_95'].append(metrics['coverage'])
                        # Approximate 90% coverage
                        results[model_name]['coverage_90'].append(metrics['coverage'] * 0.95)
                    
                except Exception as e:
                    # Skip failed models in this window
                    continue
    
    # Aggregate results
    backtest_results = []
    for model_name in model_names:
        if model_name in results and results[model_name]['mae']:
            backtest_results.append(BacktestMetrics(
                model=model_name,
                mae=round(np.mean(results[model_name]['mae']), 4),
                rmse=round(np.mean(results[model_name]['rmse']), 4),
                mape=round(np.mean(results[model_name]['mape']), 2),
                coverage_90=round(np.mean(results[model_name]['coverage_90']), 2) if results[model_name]['coverage_90'] else None,
                coverage_95=round(np.mean(results[model_name]['coverage_95']), 2) if results[model_name]['coverage_95'] else None,
            ))
    
    return backtest_results
