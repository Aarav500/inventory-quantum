"""Tests for feature generation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_data():
    """Generate sample time series data."""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    demand = 50 + 10 * np.random.randn(100)
    demand = np.maximum(0, demand)
    
    return pd.DataFrame({
        'date': dates,
        'sku': 'SKU001',
        'quantity_sold': demand,
        'quantity_on_hand': np.random.randint(50, 200, 100),
    })


class TestLagFeatures:
    """Tests for lag feature generation."""
    
    def test_lag_features_correct_length(self, sample_data):
        from app.forecasting.lightgbm_model import LightGBMForecaster
        
        forecaster = LightGBMForecaster(max_lags=7)
        ts = sample_data.set_index('date')['quantity_sold']
        features = forecaster._create_features(ts)
        
        # Should have lag columns
        assert 'lag_1' in features.columns
        assert 'lag_7' in features.columns
    
    def test_lag_values_correct(self, sample_data):
        from app.forecasting.lightgbm_model import LightGBMForecaster
        
        forecaster = LightGBMForecaster(max_lags=3)
        ts = sample_data.set_index('date')['quantity_sold']
        features = forecaster._create_features(ts)
        
        # Lag 1 should be previous day's value
        assert features['lag_1'].iloc[1] == features['y'].iloc[0]
    
    def test_rolling_features(self, sample_data):
        from app.forecasting.lightgbm_model import LightGBMForecaster
        
        forecaster = LightGBMForecaster(rolling_windows=[7, 14])
        ts = sample_data.set_index('date')['quantity_sold']
        features = forecaster._create_features(ts)
        
        assert 'rolling_mean_7' in features.columns
        assert 'rolling_std_7' in features.columns
        assert 'rolling_mean_14' in features.columns


class TestTimeFeatures:
    """Tests for time-based features."""
    
    def test_day_of_week_feature(self, sample_data):
        from app.forecasting.lightgbm_model import LightGBMForecaster
        
        forecaster = LightGBMForecaster()
        ts = sample_data.set_index('date')['quantity_sold']
        features = forecaster._create_features(ts)
        
        assert 'day_of_week' in features.columns
        assert features['day_of_week'].min() >= 0
        assert features['day_of_week'].max() <= 6
    
    def test_weekend_feature(self, sample_data):
        from app.forecasting.lightgbm_model import LightGBMForecaster
        
        forecaster = LightGBMForecaster()
        ts = sample_data.set_index('date')['quantity_sold']
        features = forecaster._create_features(ts)
        
        assert 'is_weekend' in features.columns
        assert set(features['is_weekend'].unique()).issubset({0, 1})


class TestFeatureConsistency:
    """Tests for feature consistency across models."""
    
    def test_no_data_leakage(self, sample_data):
        """Ensure no future information leaks into features."""
        from app.forecasting.lightgbm_model import LightGBMForecaster
        
        forecaster = LightGBMForecaster(max_lags=7)
        ts = sample_data.set_index('date')['quantity_sold']
        features = forecaster._create_features(ts)
        
        # After dropping NaN, remaining features should not access future
        features_clean = features.dropna()
        
        for idx in range(len(features_clean)):
            row_date = features_clean.index[idx]
            for col in features_clean.columns:
                if col.startswith('lag_'):
                    lag = int(col.split('_')[1])
                    # Verify lag value comes from past
                    expected_date = row_date - timedelta(days=lag)
                    if expected_date in ts.index:
                        assert features_clean.loc[row_date, col] == ts.loc[expected_date]
                        break  # Just check first valid lag
    
    def test_feature_dimensions(self, sample_data):
        """Ensure consistent feature dimensions."""
        from app.forecasting.lightgbm_model import LightGBMForecaster
        
        forecaster = LightGBMForecaster()
        ts = sample_data.set_index('date')['quantity_sold']
        features = forecaster._create_features(ts)
        
        # All rows should have same number of columns
        assert features.shape[1] > 0
        
        # Features should have proper index
        assert len(features) == len(ts)
