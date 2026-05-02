"""Data validation service for CSV uploads."""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List, Optional
from io import BytesIO


class ValidationError(Exception):
    """Custom validation error."""
    pass


class DataValidator:
    """Validates and transforms inventory CSV data."""
    
    REQUIRED_COLUMNS = {'date', 'sku', 'quantity_sold'}
    OPTIONAL_COLUMNS = {
        'quantity_on_hand', 'price', 'lead_time_days',
        'holding_cost', 'ordering_cost', 'stockout_cost'
    }
    
    def __init__(self):
        self.warnings: List[str] = []
    
    def validate_csv(self, file_content: bytes) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate and parse CSV content.
        
        Returns:
            Tuple of (validated DataFrame, list of warnings)
        """
        self.warnings = []
        
        # Parse CSV
        try:
            df = pd.read_csv(BytesIO(file_content))
        except Exception as e:
            raise ValidationError(f"Failed to parse CSV: {e}")
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Check required columns
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")
        
        # Validate and transform date column
        df = self._validate_dates(df)
        
        # Validate SKU column
        df = self._validate_skus(df)
        
        # Validate numeric columns
        df = self._validate_numeric(df)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Sort by date and SKU
        df = df.sort_values(['sku', 'date']).reset_index(drop=True)
        
        return df, self.warnings
    
    def _validate_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and parse date column."""
        try:
            df['date'] = pd.to_datetime(df['date'])
        except Exception:
            raise ValidationError("Failed to parse 'date' column. Use ISO format (YYYY-MM-DD).")
        
        # Check for future dates
        future_dates = df[df['date'] > datetime.now()]
        if len(future_dates) > 0:
            self.warnings.append(f"Found {len(future_dates)} records with future dates")
        
        return df
    
    def _validate_skus(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate SKU column."""
        # Convert to string
        df['sku'] = df['sku'].astype(str).str.strip()
        
        # Check for empty SKUs
        empty_skus = df[df['sku'] == '']
        if len(empty_skus) > 0:
            raise ValidationError(f"Found {len(empty_skus)} records with empty SKU")
        
        n_skus = df['sku'].nunique()
        self.warnings.append(f"Found {n_skus} unique SKUs")
        
        return df
    
    def _validate_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate numeric columns."""
        numeric_cols = ['quantity_sold', 'quantity_on_hand', 'price', 
                       'lead_time_days', 'holding_cost', 'ordering_cost', 'stockout_cost']
        
        for col in numeric_cols:
            if col in df.columns:
                # Convert to numeric
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Check for negative values (except quantity_sold can be returns)
                if col != 'quantity_sold':
                    negatives = df[df[col] < 0]
                    if len(negatives) > 0:
                        self.warnings.append(
                            f"Found {len(negatives)} negative values in '{col}', setting to 0"
                        )
                        df.loc[df[col] < 0, col] = 0
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with sensible defaults."""
        # quantity_sold: fill with 0
        if df['quantity_sold'].isna().any():
            n_missing = df['quantity_sold'].isna().sum()
            self.warnings.append(f"Filled {n_missing} missing quantity_sold with 0")
            df['quantity_sold'] = df['quantity_sold'].fillna(0)
        
        # Optional columns: leave as NaN or fill with defaults
        defaults = {
            'quantity_on_hand': None,  # Keep as NaN
            'price': None,
            'lead_time_days': 7,  # Default 7 days
            'holding_cost': 0.1,  # Default $0.10 per unit per day
            'ordering_cost': 50.0,  # Default $50 per order
            'stockout_cost': 10.0,  # Default $10 per unit stockout
        }
        
        for col, default in defaults.items():
            if col in df.columns and default is not None:
                if df[col].isna().any():
                    n_missing = df[col].isna().sum()
                    self.warnings.append(f"Filled {n_missing} missing {col} with {default}")
                    df[col] = df[col].fillna(default)
        
        return df
    
    def get_dataset_stats(self, df: pd.DataFrame) -> dict:
        """Get statistics about the dataset."""
        return {
            'total_records': len(df),
            'unique_skus': df['sku'].nunique(),
            'date_range': (df['date'].min().isoformat(), df['date'].max().isoformat()),
            'total_days': (df['date'].max() - df['date'].min()).days + 1,
            'avg_daily_demand': df.groupby('date')['quantity_sold'].sum().mean(),
            'skus': df['sku'].unique().tolist(),
        }


# Singleton instance
data_validator = DataValidator()
