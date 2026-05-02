"""
Hierarchical Forecasting with Cross-SKU Learning.

Shares patterns across products in a hierarchy:
- Total -> Category -> SKU

Enables better forecasting for sparse/new products.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class HierarchyLevel:
    """A level in the hierarchy."""
    name: str
    values: List[str]
    parent: Optional[str]


class HierarchicalForecaster:
    """
    Hierarchical forecasting with reconciliation.
    
    Methods:
    - Bottom-up: Aggregate SKU forecasts
    - Top-down: Disaggregate total forecast
    - Middle-out: Start from category level
    - Optimal reconciliation (MinT)
    """
    
    def __init__(self, method: str = 'mint'):
        self.method = method
        self.hierarchy = None
        self.base_forecasts = {}
        self.reconciled_forecasts = {}
    
    def _build_hierarchy(self, data: pd.DataFrame) -> Dict:
        """Build hierarchy from data."""
        hierarchy = {'Total': {}}
        
        if 'category' in data.columns:
            categories = data['category'].unique()
            for cat in categories:
                hierarchy['Total'][cat] = {}
                cat_skus = data[data['category'] == cat]['sku'].unique()
                for sku in cat_skus:
                    hierarchy['Total'][cat][sku] = None
        else:
            skus = data['sku'].unique()
            for sku in skus:
                hierarchy['Total'][sku] = None
        
        self.hierarchy = hierarchy
        return hierarchy
    
    def _get_summing_matrix(self, n_bottom: int, n_total: int) -> np.ndarray:
        """
        Get summing matrix S where y = S * b.
        
        y = all levels, b = bottom level
        """
        # Simplified: just total and bottom
        S = np.zeros((n_total, n_bottom))
        
        # Bottom level identity
        S[-n_bottom:, :] = np.eye(n_bottom)
        
        # Top level sums all
        S[0, :] = 1
        
        return S
    
    def _mint_reconciliation(
        self,
        base_forecasts: np.ndarray,
        S: np.ndarray,
        W: np.ndarray
    ) -> np.ndarray:
        """
        Minimum Trace (MinT) optimal reconciliation.
        
        Minimizes trace of forecast error covariance.
        """
        # G = (S'W^{-1}S)^{-1} S' W^{-1}
        W_inv = np.linalg.inv(W + 1e-6 * np.eye(len(W)))
        G = np.linalg.inv(S.T @ W_inv @ S) @ S.T @ W_inv
        
        # Reconciled bottom forecasts
        b_tilde = G @ base_forecasts
        
        # Full reconciled forecasts
        y_tilde = S @ b_tilde
        
        return y_tilde
    
    def fit(self, data: pd.DataFrame):
        """Fit base forecasters at each level."""
        self._build_hierarchy(data)
        
        # Aggregate to each level
        data['date'] = pd.to_datetime(data['date'])
        
        # Total level
        total_ts = data.groupby('date')['quantity_sold'].sum()
        self.base_forecasts['Total'] = self._fit_simple(total_ts)
        
        # SKU level
        for sku in data['sku'].unique():
            sku_ts = data[data['sku'] == sku].groupby('date')['quantity_sold'].sum()
            self.base_forecasts[sku] = self._fit_simple(sku_ts)
        
        # Category level (if available)
        if 'category' in data.columns:
            for cat in data['category'].unique():
                cat_ts = data[data['category'] == cat].groupby('date')['quantity_sold'].sum()
                self.base_forecasts[cat] = self._fit_simple(cat_ts)
    
    def _fit_simple(self, ts: pd.Series) -> Dict:
        """Simple forecaster for a time series."""
        values = ts.values
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'trend': (values[-1] - values[0]) / len(values) if len(values) > 1 else 0,
            'last': values[-1] if len(values) > 0 else 0
        }
    
    def forecast(self, horizon: int = 30) -> Dict[str, np.ndarray]:
        """Generate reconciled forecasts."""
        # Base forecasts
        base = {}
        for key, params in self.base_forecasts.items():
            trend = params['trend']
            mean = params['mean']
            base[key] = np.array([mean + trend * h for h in range(horizon)])
        
        # Reconciliation
        skus = [k for k in base.keys() if k != 'Total' and not k.startswith('cat_')]
        n_bottom = len(skus)
        
        if self.method == 'bottom_up':
            # Sum bottom level
            reconciled = {}
            for sku in skus:
                reconciled[sku] = base[sku]
            reconciled['Total'] = sum(base[sku] for sku in skus)
        
        elif self.method == 'top_down':
            # Proportionally allocate
            total_mean = sum(self.base_forecasts[sku]['mean'] for sku in skus)
            reconciled = {}
            for sku in skus:
                proportion = self.base_forecasts[sku]['mean'] / (total_mean + 1e-8)
                reconciled[sku] = base['Total'] * proportion
            reconciled['Total'] = base['Total']
        
        else:  # MinT
            # Simplified MinT
            n_total = n_bottom + 1
            S = self._get_summing_matrix(n_bottom, n_total)
            
            # Stack base forecasts
            base_vec = np.vstack([base['Total'].reshape(1, -1)] + 
                                [base[sku].reshape(1, -1) for sku in skus])
            
            # Covariance (simplified - use identity)
            W = np.eye(n_total)
            
            reconciled = {}
            for h in range(horizon):
                y_h = base_vec[:, h]
                y_tilde = self._mint_reconciliation(y_h, S, W)
                
                if h == 0:
                    reconciled['Total'] = [y_tilde[0]]
                    for i, sku in enumerate(skus):
                        reconciled[sku] = [y_tilde[i + 1]]
                else:
                    reconciled['Total'].append(y_tilde[0])
                    for i, sku in enumerate(skus):
                        reconciled[sku].append(y_tilde[i + 1])
            
            reconciled = {k: np.array(v) for k, v in reconciled.items()}
        
        self.reconciled_forecasts = reconciled
        return reconciled
    
    def get_coherency_error(self) -> float:
        """Check if forecasts are coherent (sum to total)."""
        if not self.reconciled_forecasts:
            return 0
        
        skus = [k for k in self.reconciled_forecasts.keys() if k != 'Total']
        bottom_sum = sum(self.reconciled_forecasts[sku] for sku in skus)
        total = self.reconciled_forecasts.get('Total', bottom_sum)
        
        return float(np.mean(np.abs(bottom_sum - total)))
