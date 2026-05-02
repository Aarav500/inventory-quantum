"""Golden test on fixture dataset."""

import pytest
import pandas as pd
from pathlib import Path


@pytest.fixture
def fixture_data():
    """Load the golden test fixture."""
    fixture_path = Path(__file__).parent.parent / 'fixtures' / 'tiny_dataset.csv'
    return pd.read_csv(fixture_path)


class TestGoldenFixture:
    """Golden tests on fixture dataset."""
    
    def test_fixture_loads(self, fixture_data):
        assert len(fixture_data) > 0
        assert 'date' in fixture_data.columns
        assert 'sku' in fixture_data.columns
        assert 'quantity_sold' in fixture_data.columns
    
    def test_validation_passes(self, fixture_data):
        from app.services.validation import DataValidator
        
        validator = DataValidator()
        csv_content = fixture_data.to_csv(index=False).encode()
        df, warnings = validator.validate_csv(csv_content)
        
        assert len(df) == len(fixture_data)
        assert len(df['sku'].unique()) == 2  # SKU001 and SKU002
    
    def test_naive_forecast(self, fixture_data):
        from app.forecasting.naive import SeasonalNaiveForecaster
        
        forecaster = SeasonalNaiveForecaster()
        fixture_data['date'] = pd.to_datetime(fixture_data['date'])
        sku_data = fixture_data[fixture_data['sku'] == 'SKU001']
        
        predictions = forecaster.forecast(sku_data, horizon=7)
        
        assert len(predictions) == 7
        assert all(p.predicted >= 0 for p in predictions)
        assert all(p.lower_bound is not None for p in predictions)
    
    def test_arima_forecast(self, fixture_data):
        from app.forecasting.arima import ARIMAForecaster
        
        forecaster = ARIMAForecaster()
        fixture_data['date'] = pd.to_datetime(fixture_data['date'])
        sku_data = fixture_data[fixture_data['sku'] == 'SKU001']
        
        predictions = forecaster.forecast(sku_data, horizon=7)
        
        assert len(predictions) == 7
        # Should produce reasonable values
        avg_demand = sku_data['quantity_sold'].mean()
        assert all(p.predicted < avg_demand * 5 for p in predictions)
    
    def test_reorder_point_policy(self, fixture_data):
        from app.decision.reorder_point import ReorderPointPolicy
        
        policy = ReorderPointPolicy()
        fixture_data['date'] = pd.to_datetime(fixture_data['date'])
        sku_data = fixture_data[fixture_data['sku'] == 'SKU001']
        
        result = policy.optimize(sku_data)
        
        assert result.sku == 'SKU001'
        assert result.reorder_point > 0
        assert result.reorder_quantity > 0
        assert result.expected_service_level > 0
    
    def test_qubo_optimization(self, fixture_data):
        from app.decision.qubo import QUBOOptimizer
        
        optimizer = QUBOOptimizer(n_bits=6)  # Limit bits for speed
        fixture_data['date'] = pd.to_datetime(fixture_data['date'])
        sku_data = fixture_data[fixture_data['sku'] == 'SKU001']
        
        result = optimizer.optimize(sku_data)
        
        assert result.policy == 'qubo'
        assert result.reorder_quantity >= 0
        assert result.expected_cost >= 0
    
    def test_drift_detection(self, fixture_data):
        from app.monitoring.drift import DriftDetector
        
        detector = DriftDetector()
        fixture_data['date'] = pd.to_datetime(fixture_data['date'])
        
        # With this small dataset, we won't have enough data for proper drift
        # detection, but the method should still run
        results = detector.detect_drift(
            fixture_data,
            reference_days=30,
            test_days=10
        )
        
        # May or may not find results depending on data split
        assert isinstance(results, list)
    
    def test_end_to_end_pipeline(self, fixture_data):
        """Test full pipeline from data to decision."""
        from app.services.validation import DataValidator
        from app.forecasting.lightgbm_model import LightGBMForecaster
        from app.decision.eoq import EOQPolicy
        
        # 1. Validate
        validator = DataValidator()
        csv_content = fixture_data.to_csv(index=False).encode()
        df, _ = validator.validate_csv(csv_content)
        
        # 2. Forecast
        sku_data = df[df['sku'] == 'SKU001']
        forecaster = LightGBMForecaster()
        forecast = forecaster.forecast(sku_data, horizon=14)
        
        # 3. Optimize
        policy = EOQPolicy()
        result = policy.optimize(sku_data, forecast)
        
        # 4. Verify results make sense
        assert result.reorder_quantity > 0
        assert result.expected_service_level > 0.5
        assert result.expected_cost < 100000  # Reasonable upper bound
