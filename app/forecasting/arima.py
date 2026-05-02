"""ARIMA forecaster."""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import timedelta
import warnings

from app.models.inventory import ForecastPoint

# Suppress convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ARIMAForecaster:
    """
    Auto ARIMA forecaster using statsmodels.
    
    Automatically selects optimal (p, d, q) parameters.
    """
    
    def __init__(self, max_p: int = 5, max_d: int = 2, max_q: int = 5):
        """
        Initialize forecaster.
        
        Args:
            max_p: Maximum AR order
            max_d: Maximum differencing order
            max_q: Maximum MA order
        """
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.last_metrics = None
        self._model = None
    
    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        include_intervals: bool = True,
        confidence: float = 0.95
    ) -> List[ForecastPoint]:
        """
        Generate ARIMA forecasts.
        
        Args:
            data: DataFrame with 'date' and 'quantity_sold' columns
            horizon: Forecast horizon in days
            include_intervals: Whether to include prediction intervals
            confidence: Confidence level for intervals
        
        Returns:
            List of ForecastPoint objects
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            # Fallback to naive if statsmodels not available
            from app.forecasting.naive import SeasonalNaiveForecaster
            naive = SeasonalNaiveForecaster()
            self.last_metrics = {'error': 'statsmodels not available, falling back to naive'}
            return naive.forecast(data, horizon, include_intervals, confidence)
        
        # Prepare time series
        df = data.copy()
        df = df.sort_values('date')
        ts = df.groupby('date')['quantity_sold'].sum()
        
        # Ensure we have enough data
        if len(ts) < 10:
            from app.forecasting.naive import SeasonalNaiveForecaster
            naive = SeasonalNaiveForecaster()
            self.last_metrics = {'error': 'Insufficient data for ARIMA'}
            return naive.forecast(data, horizon, include_intervals, confidence)
        
        # Auto-select differencing order using ADF test
        d = 0
        for i in range(self.max_d + 1):
            if i > 0:
                diff_ts = ts.diff(i).dropna()
            else:
                diff_ts = ts
            adf_result = adfuller(diff_ts, autolag='AIC')
            if adf_result[1] < 0.05:  # Stationary at 5% significance
                d = i
                break
        else:
            d = self.max_d
        
        # Find best (p, q) using AIC
        best_aic = float('inf')
        best_order = (1, d, 1)
        
        for p in range(self.max_p + 1):
            for q in range(self.max_q + 1):
                if p == 0 and q == 0:
                    continue
                try:
                    model = ARIMA(ts.values, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    continue
        
        # Fit best model
        try:
            model = ARIMA(ts.values, order=best_order)
            self._model = model.fit()
        except Exception as e:
            from app.forecasting.naive import SeasonalNaiveForecaster
            naive = SeasonalNaiveForecaster()
            self.last_metrics = {'error': f'ARIMA fitting failed: {str(e)}'}
            return naive.forecast(data, horizon, include_intervals, confidence)
        
        # Generate forecast
        forecast_result = self._model.get_forecast(steps=horizon)
        mean_forecast = forecast_result.predicted_mean
        
        if include_intervals:
            conf_int = forecast_result.conf_int(alpha=1 - confidence)
        
        # Get last date
        last_date = ts.index.max()
        
        predictions = []
        for h in range(horizon):
            forecast_date = last_date + timedelta(days=h + 1)
            predicted = max(0, mean_forecast[h])
            
            point = ForecastPoint(
                date=forecast_date.date() if hasattr(forecast_date, 'date') else forecast_date,
                predicted=round(predicted, 2),
                lower_bound=round(max(0, conf_int[h, 0]), 2) if include_intervals else None,
                upper_bound=round(conf_int[h, 1], 2) if include_intervals else None,
            )
            predictions.append(point)
        
        # Store metrics
        self.last_metrics = {
            'model': 'arima',
            'order': best_order,
            'aic': round(best_aic, 2),
        }
        
        return predictions
