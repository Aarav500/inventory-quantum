"""LightGBM forecaster with lag features."""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from datetime import timedelta

from app.models.inventory import ForecastPoint


class LightGBMForecaster:
    """
    LightGBM gradient boosting forecaster with lag features.
    
    Creates extensive lag and rolling features for demand prediction.
    """
    
    def __init__(
        self,
        max_lags: int = 30,
        rolling_windows: List[int] = None,
        n_estimators: int = 100,
    ):
        """
        Initialize forecaster.
        
        Args:
            max_lags: Number of lag features to create
            rolling_windows: Windows for rolling statistics
            n_estimators: Number of boosting iterations
        """
        self.max_lags = max_lags
        self.rolling_windows = rolling_windows or [7, 14, 30]
        self.n_estimators = n_estimators
        self.last_metrics = None
        self._model = None
        self._feature_names = None
    
    def _create_features(self, ts: pd.Series) -> pd.DataFrame:
        """Create lag and rolling features."""
        df = pd.DataFrame({'y': ts.values}, index=ts.index)
        
        # Lag features
        for lag in range(1, self.max_lags + 1):
            df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Rolling mean features
        for window in self.rolling_windows:
            df[f'rolling_mean_{window}'] = df['y'].shift(1).rolling(window=window).mean()
            df[f'rolling_std_{window}'] = df['y'].shift(1).rolling(window=window).std()
            df[f'rolling_min_{window}'] = df['y'].shift(1).rolling(window=window).min()
            df[f'rolling_max_{window}'] = df['y'].shift(1).rolling(window=window).max()
        
        # Time features
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
        
        # Trend feature
        df['trend'] = np.arange(len(df))
        
        return df
    
    def _prepare_forecast_features(
        self,
        ts: pd.Series,
        horizon: int
    ) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Prepare features for forecast horizon."""
        # Create features from history
        historical_features = self._create_features(ts)
        
        # Get future dates
        last_date = ts.index.max()
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=horizon,
            freq='D'
        )
        
        # Initialize future dataframe
        future_df = pd.DataFrame(index=future_dates)
        future_df['y'] = np.nan
        
        # We'll need to predict iteratively
        return historical_features, future_dates
    
    def forecast(
        self,
        data: pd.DataFrame,
        horizon: int = 30,
        include_intervals: bool = True,
        confidence: float = 0.95
    ) -> List[ForecastPoint]:
        """
        Generate LightGBM forecasts.
        
        Args:
            data: DataFrame with 'date' and 'quantity_sold' columns
            horizon: Forecast horizon in days
            include_intervals: Whether to include prediction intervals
            confidence: Confidence level for intervals
        
        Returns:
            List of ForecastPoint objects
        """
        try:
            import lightgbm as lgb
        except ImportError:
            from app.forecasting.naive import SeasonalNaiveForecaster
            naive = SeasonalNaiveForecaster()
            self.last_metrics = {'error': 'LightGBM not available'}
            return naive.forecast(data, horizon, include_intervals, confidence)
        
        # Prepare time series
        df = data.copy()
        df = df.sort_values('date')
        df['date'] = pd.to_datetime(df['date'])
        ts = df.groupby('date')['quantity_sold'].sum()
        
        # Ensure we have enough data
        min_samples = self.max_lags + max(self.rolling_windows) + 10
        if len(ts) < min_samples:
            from app.forecasting.naive import SeasonalNaiveForecaster
            naive = SeasonalNaiveForecaster()
            self.last_metrics = {'error': 'Insufficient data for LightGBM'}
            return naive.forecast(data, horizon, include_intervals, confidence)
        
        # Create features
        features_df = self._create_features(ts)
        features_df = features_df.dropna()
        
        # Split features and target
        X = features_df.drop(columns=['y'])
        y = features_df['y']
        self._feature_names = X.columns.tolist()
        
        # Train model
        train_data = lgb.Dataset(X, label=y)
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1,
        }
        
        self._model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
        )
        
        # Calculate residuals for prediction intervals
        train_preds = self._model.predict(X)
        residuals = y.values - train_preds
        residual_std = np.std(residuals)
        z_score = 1.96 if confidence == 0.95 else 1.645
        
        # Generate forecasts iteratively
        last_date = ts.index.max()
        predictions = []
        
        # Extended history for feature generation
        extended_ts = ts.copy()
        
        for h in range(1, horizon + 1):
            forecast_date = last_date + timedelta(days=h)
            
            # Create features for this step
            extended_features = self._create_features(extended_ts)
            
            # Get the last row's features for prediction
            last_features = extended_features.iloc[-1:].drop(columns=['y'])
            
            # Predict
            predicted = max(0, self._model.predict(last_features)[0])
            
            # Add to extended series for next iteration
            extended_ts = pd.concat([
                extended_ts,
                pd.Series([predicted], index=[forecast_date])
            ])
            
            # Prediction interval
            interval_width = z_score * residual_std * np.sqrt(1 + h / 30)
            
            point = ForecastPoint(
                date=forecast_date.date() if hasattr(forecast_date, 'date') else forecast_date,
                predicted=round(predicted, 2),
                lower_bound=round(max(0, predicted - interval_width), 2) if include_intervals else None,
                upper_bound=round(predicted + interval_width, 2) if include_intervals else None,
            )
            predictions.append(point)
        
        # Store metrics with feature importance
        importance = self._model.feature_importance()
        top_features = sorted(
            zip(self._feature_names, importance),
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        self.last_metrics = {
            'model': 'lightgbm',
            'n_estimators': self.n_estimators,
            'residual_rmse': round(np.sqrt(np.mean(residuals**2)), 2),
            'top_features': {k: int(v) for k, v in top_features},
        }
        
        return predictions
