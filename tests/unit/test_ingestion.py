"""Tests for data ingestion and validation."""

import pytest
import pandas as pd
import numpy as np
from io import BytesIO

from app.services.validation import DataValidator, ValidationError


@pytest.fixture
def validator():
    return DataValidator()


@pytest.fixture
def valid_csv_content():
    return b"""date,sku,quantity_sold,quantity_on_hand,price
2024-01-01,SKU001,45,200,29.99
2024-01-02,SKU001,52,155,29.99
2024-01-03,SKU001,38,103,29.99
2024-01-04,SKU001,61,65,29.99
"""


@pytest.fixture
def minimal_csv_content():
    return b"""date,sku,quantity_sold
2024-01-01,SKU001,45
2024-01-02,SKU001,52
"""


class TestDataIngestion:
    """Tests for CSV ingestion."""
    
    def test_valid_csv_parsing(self, validator, valid_csv_content):
        df, warnings = validator.validate_csv(valid_csv_content)
        
        assert len(df) == 4
        assert 'date' in df.columns
        assert 'sku' in df.columns
        assert 'quantity_sold' in df.columns
    
    def test_minimal_required_columns(self, validator, minimal_csv_content):
        df, warnings = validator.validate_csv(minimal_csv_content)
        
        assert len(df) == 2
        assert df['sku'].iloc[0] == 'SKU001'
    
    def test_missing_required_column_raises(self, validator):
        bad_csv = b"""date,quantity_sold
2024-01-01,45
"""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_csv(bad_csv)
        
        assert 'sku' in str(exc_info.value).lower()
    
    def test_date_parsing(self, validator, valid_csv_content):
        df, _ = validator.validate_csv(valid_csv_content)
        
        assert pd.api.types.is_datetime64_any_dtype(df['date'])
        assert df['date'].min() == pd.Timestamp('2024-01-01')
    
    def test_invalid_date_raises(self, validator):
        bad_csv = b"""date,sku,quantity_sold
not-a-date,SKU001,45
"""
        with pytest.raises(ValidationError):
            validator.validate_csv(bad_csv)
    
    def test_numeric_validation(self, validator):
        csv_with_negatives = b"""date,sku,quantity_sold,price
2024-01-01,SKU001,45,-10
"""
        df, warnings = validator.validate_csv(csv_with_negatives)
        
        # Negative price should be set to 0
        assert df['price'].iloc[0] == 0
        assert any('negative' in w.lower() for w in warnings)
    
    def test_missing_value_handling(self, validator):
        csv_with_missing = b"""date,sku,quantity_sold,quantity_on_hand
2024-01-01,SKU001,,100
2024-01-02,SKU001,45,
"""
        df, warnings = validator.validate_csv(csv_with_missing)
        
        # Missing quantity_sold should be 0
        assert df['quantity_sold'].iloc[0] == 0
    
    def test_column_name_normalization(self, validator):
        csv_with_spaces = b"""Date,SKU,Quantity Sold
2024-01-01,SKU001,45
"""
        df, _ = validator.validate_csv(csv_with_spaces)
        
        assert 'date' in df.columns
        assert 'sku' in df.columns
        assert 'quantity_sold' in df.columns
    
    def test_empty_sku_raises(self, validator):
        bad_csv = b"""date,sku,quantity_sold
2024-01-01,,45
"""
        with pytest.raises(ValidationError):
            validator.validate_csv(bad_csv)
    
    def test_dataset_stats(self, validator, valid_csv_content):
        df, _ = validator.validate_csv(valid_csv_content)
        stats = validator.get_dataset_stats(df)
        
        assert stats['total_records'] == 4
        assert stats['unique_skus'] == 1
        assert 'SKU001' in stats['skus']
        assert stats['date_range'][0] == '2024-01-01'
