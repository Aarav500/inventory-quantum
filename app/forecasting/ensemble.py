"""Model ensemble for combining forecasts."""

import numpy as np
from typing import List, Dict, Any, Optional
import pandas as pd

from app.models.inventory import ForecastPoint, ForecastResult
from app.forecasting.naive import SeasonalNaiveForecaster
from app.forecasting.arima import ARIMAForecaster
from app.forecasting.lightgbm_model import LightGBMForecaster
from app.forecasting.deep_model import DeepForecaster


class EnsembleForecaster:
    """
    Ensemble forecaster combining multiple models.
    
    Supports simple averaging, weighted averaging, and stacking.
    """
    
    MODELS = {
        'naive': SeasonalNaiveForecaster,
        'arima': ARIMAForecaster,
        'lightgbm': LightGBMForecaster,
        'deep': DeepForecaster,
    }
    
    def __init__(
        self,
        models: List[str] = None,
        weights: Dict[str, float] = None,
        method: str = 'weighted'
    ):
        """
        Initialize ensemble.
        
        Args:
            models: List of model names to include
            weights: Optional weights for each model
            method: 'simple', 'weighted', or 'median'
        """
        self.models = models or ['naive', 'arima', 'lightgbm']
        self.weights = weights or {m: 1.0 / len(self.models) for m in self.models}
        self.method = method
        self.last_metrics = None
        self._individual_results = {}
    
    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        include_intervals: bool = True,
        confidence: float = 0.95
    ) -> List[ForecastPoint]:
        """
        Generate ensemble forecasts.
        
        Args:
            data: DataFrame with historical data
            horizon: Forecast horizon
            include_intervals: Whether to include prediction intervals
            confidence: Confidence level
        
        Returns:
            List of ForecastPoint with ensemble predictions
        """
        all_predictions = {}
        
        # Get predictions from each model
        for model_name in self.models:
            if model_name not in self.MODELS:
                continue
            
            try:
                forecaster = self.MODELS[model_name]()
                predictions = forecaster.forecast(
                    data, horizon, include_intervals, confidence
                )
                all_predictions[model_name] = predictions
                self._individual_results[model_name] = {
                    'predictions': predictions,
                    'metrics': forecaster.last_metrics
                }
            except Exception as e:
                continue
        
        if not all_predictions:
            # Fallback to naive
            naive = SeasonalNaiveForecaster()
            return naive.forecast(data, horizon, include_intervals, confidence)
        
        # Combine predictions
        ensemble_predictions = []
        
        for h in range(horizon):
            values = []
            weights = []
            lowers = []
            uppers = []
            
            for model_name, preds in all_predictions.items():
                if h < len(preds):
                    values.append(preds[h].predicted)
                    weights.append(self.weights.get(model_name, 1.0))
                    if preds[h].lower_bound is not None:
                        lowers.append(preds[h].lower_bound)
                    if preds[h].upper_bound is not None:
                        uppers.append(preds[h].upper_bound)
            
            if not values:
                continue
            
            # Combine based on method
            if self.method == 'simple':
                predicted = np.mean(values)
            elif self.method == 'median':
                predicted = np.median(values)
            else:  # weighted
                weights = np.array(weights)
                weights = weights / weights.sum()
                predicted = np.average(values, weights=weights)
            
            # Combine intervals (use widest)
            lower_bound = min(lowers) if lowers else None
            upper_bound = max(uppers) if uppers else None
            
            # Get date from first model
            first_model = list(all_predictions.keys())[0]
            forecast_date = all_predictions[first_model][h].date
            
            ensemble_predictions.append(ForecastPoint(
                date=forecast_date,
                predicted=round(predicted, 2),
                lower_bound=round(lower_bound, 2) if lower_bound else None,
                upper_bound=round(upper_bound, 2) if upper_bound else None,
            ))
        
        self.last_metrics = {
            'model': 'ensemble',
            'method': self.method,
            'models_used': list(all_predictions.keys()),
            'weights': {k: round(v, 3) for k, v in self.weights.items() if k in all_predictions},
        }
        
        return ensemble_predictions
    
    def get_individual_results(self) -> Dict[str, Any]:
        """Get individual model results."""
        return self._individual_results
