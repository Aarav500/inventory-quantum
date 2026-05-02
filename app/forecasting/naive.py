"""Seasonal Naïve forecaster baseline."""

import pandas as pd
import numpy as np
from typing import List, Optional
from datetime import timedelta

from app.models.inventory import ForecastPoint


class SeasonalNaiveForecaster:
    """
    Seasonal Naïve forecaster.
    
    Uses the value from the same day last year (or last available period).
    For shorter histories, uses weekly seasonality.
    """
    
    def __init__(self, seasonality: int = 7):
        """
        Initialize forecaster.
        
        Args:
            seasonality: Seasonal period (7 for weekly, 365 for yearly)
        """
        self.seasonality = seasonality
        self.last_metrics = None
    
    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        include_intervals: bool = True,
        confidence: float = 0.95
    ) -> List[ForecastPoint]:
        """
        Generate seasonal naïve forecasts.
        
        Args:
            data: DataFrame with 'date' and 'quantity_sold' columns
            horizon: Forecast horizon in days
            include_intervals: Whether to include prediction intervals
            confidence: Confidence level for intervals
        
        Returns:
            List of ForecastPoint objects
        """
        # Ensure data is sorted
        df = data.copy()
        df = df.sort_values('date')
        
        # Get historical values
        values = df.groupby('date')['quantity_sold'].sum().values
        
        # Determine seasonality based on data length
        if len(values) >= 365:
            seasonality = 365
        elif len(values) >= 7:
            seasonality = 7
        else:
            seasonality = 1
        
        # Get last date
        last_date = df['date'].max()
        
        predictions = []
        
        # Calculate residual std for intervals
        if len(values) > seasonality:
            residuals = values[seasonality:] - values[:-seasonality]
            residual_std = np.std(residuals)
        else:
            residual_std = np.std(values) if len(values) > 1 else 0
        
        z_score = 1.96 if confidence == 0.95 else 1.645
        
        for h in range(1, horizon + 1):
            forecast_date = last_date + timedelta(days=h)
            
            # Get seasonal index
            seasonal_idx = (len(values) - seasonality + (h - 1) % seasonality) % len(values)
            predicted = max(0, values[seasonal_idx])
            
            # Prediction interval widens with horizon
            interval_width = z_score * residual_std * np.sqrt(h / seasonality + 1)
            
            point = ForecastPoint(
                date=forecast_date.date() if hasattr(forecast_date, 'date') else forecast_date,
                predicted=round(predicted, 2),
                lower_bound=round(max(0, predicted - interval_width), 2) if include_intervals else None,
                upper_bound=round(predicted + interval_width, 2) if include_intervals else None,
            )
            predictions.append(point)
        
        # Store metrics
        self.last_metrics = {
            'model': 'seasonal_naive',
            'seasonality': seasonality,
            'residual_std': round(residual_std, 2),
        }
        
        return predictions
