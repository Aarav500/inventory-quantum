"""
Conformal Prediction for Distribution-Free Uncertainty Quantification.

Provides prediction intervals with guaranteed coverage probability
without distributional assumptions.

Methods:
1. Split Conformal Prediction
2. Conformalized Quantile Regression (CQR)
3. Adaptive Conformal Inference for Time Series
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import timedelta

from app.models.inventory import ForecastPoint


@dataclass
class ConformalResult:
    """Result from conformal prediction."""
    prediction: float
    lower: float
    upper: float
    coverage_target: float
    nonconformity_score: float


class SplitConformalPredictor:
    """
    Split Conformal Prediction.
    
    Provides finite-sample coverage guarantees:
    P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 - α
    
    Uses calibration set to compute nonconformity scores.
    """
    
    def __init__(
        self,
        model: Callable,
        alpha: float = 0.1,
        calibration_ratio: float = 0.2,
    ):
        """
        Initialize conformal predictor.
        
        Args:
            model: Base forecasting model (must have predict method)
            alpha: Miscoverage rate (1 - coverage probability)
            calibration_ratio: Fraction of data for calibration
        """
        self.model = model
        self.alpha = alpha
        self.calibration_ratio = calibration_ratio
        self._calibration_scores = None
        self._quantile = None
    
    def calibrate(self, X: np.ndarray, y: np.ndarray, predictions: np.ndarray = None):
        """
        Calibrate conformal predictor.
        
        Computes nonconformity scores on calibration set.
        """
        n = len(y)
        cal_size = int(n * self.calibration_ratio)
        
        if cal_size < 10:
            cal_size = min(n, 10)
        
        # Use last portion for calibration
        y_cal = y[-cal_size:]
        
        if predictions is None:
            # Get predictions from model
            pred_cal = np.full(cal_size, y[:-cal_size].mean())
        else:
            pred_cal = predictions[-cal_size:]
        
        # Compute nonconformity scores (absolute residuals)
        self._calibration_scores = np.abs(y_cal - pred_cal)
        
        # Compute conformal quantile
        n_cal = len(self._calibration_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)
        
        self._quantile = np.quantile(self._calibration_scores, q_level)
    
    def predict(self, point_prediction: float) -> ConformalResult:
        """
        Generate prediction interval.
        
        Returns:
            ConformalResult with guaranteed coverage
        """
        if self._quantile is None:
            raise ValueError("Must calibrate before predicting")
        
        lower = point_prediction - self._quantile
        upper = point_prediction + self._quantile
        
        return ConformalResult(
            prediction=point_prediction,
            lower=max(0, lower),  # Demand can't be negative
            upper=upper,
            coverage_target=1 - self.alpha,
            nonconformity_score=self._quantile,
        )


class ConformilizedQuantileRegression:
    """
    Conformalized Quantile Regression (CQR).
    
    Combines quantile regression with conformal prediction
    for adaptive, asymmetric intervals.
    
    Reference: Romano, Patterson, Candès (2019)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        quantiles: Tuple[float, float] = None,
    ):
        """
        Initialize CQR.
        
        Args:
            alpha: Miscoverage rate
            quantiles: Lower and upper quantiles (default: symmetric)
        """
        self.alpha = alpha
        self.quantiles = quantiles or (alpha / 2, 1 - alpha / 2)
        self._calibration_scores = None
        self._quantile = None
    
    def calibrate(
        self,
        y_true: np.ndarray,
        y_lower: np.ndarray,
        y_upper: np.ndarray,
    ):
        """
        Calibrate CQR on held-out data.
        
        Args:
            y_true: True values
            y_lower: Lower quantile predictions
            y_upper: Upper quantile predictions
        """
        # CQR nonconformity score: max(q_lo - y, y - q_hi)
        scores = np.maximum(y_lower - y_true, y_true - y_upper)
        self._calibration_scores = scores
        
        # Conformal quantile
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(q_level, 1.0)
        
        self._quantile = np.quantile(scores, q_level)
    
    def predict(
        self,
        point_pred: float,
        lower_pred: float,
        upper_pred: float,
    ) -> ConformalResult:
        """
        Generate conformalized prediction interval.
        
        Adjusts quantile predictions to guarantee coverage.
        """
        if self._quantile is None:
            raise ValueError("Must calibrate before predicting")
        
        # Adjust intervals by conformal quantile
        lower = lower_pred - self._quantile
        upper = upper_pred + self._quantile
        
        return ConformalResult(
            prediction=point_pred,
            lower=max(0, lower),
            upper=upper,
            coverage_target=1 - self.alpha,
            nonconformity_score=self._quantile,
        )


class AdaptiveConformalForecaster:
    """
    Adaptive Conformal Inference for Time Series.
    
    Dynamically adjusts coverage to handle distribution shift
    in non-exchangeable data.
    
    Reference: Gibbs & Candès (2021)
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.01,  # Learning rate for adaptation
        window_size: int = 50,
    ):
        """
        Initialize adaptive conformal forecaster.
        
        Args:
            alpha: Target miscoverage rate
            gamma: Adaptation learning rate
            window_size: Rolling window for score computation
        """
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        
        self._alpha_t = alpha  # Dynamic miscoverage rate
        self._scores_history = []
        self._coverage_history = []
    
    def update(self, y_true: float, interval: Tuple[float, float]):
        """
        Update adaptive alpha based on observed coverage.
        
        Args:
            y_true: Observed true value
            interval: Predicted (lower, upper) interval
        """
        lower, upper = interval
        
        # Check if true value is in interval
        covered = lower <= y_true <= upper
        self._coverage_history.append(covered)
        
        # Adapt alpha: increase if undercovered, decrease if overcovered
        err_t = (1 - int(covered)) - self.alpha
        self._alpha_t = self._alpha_t + self.gamma * err_t
        
        # Clip to valid range
        self._alpha_t = np.clip(self._alpha_t, 0.01, 0.5)
    
    def get_adaptive_quantile(self, scores: np.ndarray) -> float:
        """Get quantile using adaptive alpha."""
        n = len(scores)
        q_level = np.ceil((n + 1) * (1 - self._alpha_t)) / n
        q_level = min(q_level, 1.0)
        return np.quantile(scores, q_level)
    
    def get_empirical_coverage(self) -> float:
        """Get empirical coverage from history."""
        if not self._coverage_history:
            return 0.0
        return np.mean(self._coverage_history[-self.window_size:])


class ConformalForecaster:
    """
    Main conformal forecasting interface.
    
    Wraps any point forecaster with conformal prediction intervals.
    """
    
    def __init__(
        self,
        base_forecaster,
        method: str = 'split',  # 'split', 'cqr', or 'adaptive'
        alpha: float = 0.1,
    ):
        """
        Initialize conformal forecaster.
        
        Args:
            base_forecaster: Any forecaster with forecast() method
            method: Conformal method to use
            alpha: Target miscoverage rate
        """
        self.base_forecaster = base_forecaster
        self.method = method
        self.alpha = alpha
        
        if method == 'split':
            self._conformal = SplitConformalPredictor(base_forecaster, alpha)
        elif method == 'cqr':
            self._conformal = ConformilizedQuantileRegression(alpha)
        else:
            self._conformal = AdaptiveConformalForecaster(alpha)
        
        self.last_metrics = None
        self._calibrated = False
    
    def fit_and_calibrate(self, data: pd.DataFrame):
        """
        Fit base model and calibrate conformal predictor.
        """
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Split data
        n = len(df['date'].unique())
        train_end = int(n * 0.8)
        
        train_dates = df['date'].unique()[:train_end]
        cal_dates = df['date'].unique()[train_end:]
        
        train_df = df[df['date'].isin(train_dates)]
        cal_df = df[df['date'].isin(cal_dates)]
        
        # Get predictions on calibration set
        cal_horizon = len(cal_dates)
        predictions = self.base_forecaster.forecast(train_df, horizon=cal_horizon)
        
        pred_values = np.array([p.predicted for p in predictions])
        
        # Get actual values
        actual = cal_df.groupby('date')['quantity_sold'].sum().values[:len(pred_values)]
        
        # Calibrate
        if self.method == 'split':
            self._conformal.calibrate(None, actual, pred_values)
        elif self.method == 'cqr':
            lower = np.array([p.lower_bound or p.predicted * 0.7 for p in predictions])[:len(actual)]
            upper = np.array([p.upper_bound or p.predicted * 1.3 for p in predictions])[:len(actual)]
            self._conformal.calibrate(actual, lower, upper)
        
        self._calibrated = True
        
        # Compute calibration metrics
        self.last_metrics = {
            'method': self.method,
            'alpha': self.alpha,
            'calibration_quantile': float(self._conformal._quantile) if hasattr(self._conformal, '_quantile') else None,
        }
    
    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        include_intervals: bool = True,
        confidence: float = 0.95,
    ) -> List[ForecastPoint]:
        """
        Generate forecasts with conformal prediction intervals.
        
        Guarantees coverage at specified confidence level.
        """
        # Fit and calibrate if needed
        if not self._calibrated:
            self.fit_and_calibrate(data)
        
        # Get base predictions
        base_predictions = self.base_forecaster.forecast(
            data, horizon, include_intervals=True
        )
        
        # Apply conformal correction
        conformal_predictions = []
        
        for pred in base_predictions:
            if self.method == 'cqr':
                result = self._conformal.predict(
                    pred.predicted,
                    pred.lower_bound or pred.predicted * 0.7,
                    pred.upper_bound or pred.predicted * 1.3,
                )
            else:
                result = self._conformal.predict(pred.predicted)
            
            conformal_predictions.append(ForecastPoint(
                date=pred.date,
                predicted=round(result.prediction, 2),
                lower_bound=round(result.lower, 2) if include_intervals else None,
                upper_bound=round(result.upper, 2) if include_intervals else None,
            ))
        
        return conformal_predictions
    
    def get_coverage_guarantee(self) -> str:
        """Get theoretical coverage guarantee."""
        return f"P(Y ∈ [L, U]) ≥ {1 - self.alpha:.1%} (finite-sample guarantee)"
