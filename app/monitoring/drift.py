"""Distributional drift detection for monitoring."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from scipy import stats

from app.models.inventory import DriftMetrics
from app.config import get_settings


class DriftDetector:
    """
    Distributional drift detection for inventory demand.
    
    Uses multiple methods to detect shifts without ground truth:
    - Population Stability Index (PSI)
    - Kolmogorov-Smirnov test
    - Jensen-Shannon divergence for features
    - Prediction interval calibration monitoring
    """
    
    def __init__(
        self,
        psi_threshold: float = None,
        ks_alpha: float = None,
        n_bins: int = 10,
    ):
        """
        Initialize drift detector.
        
        Args:
            psi_threshold: PSI threshold for drift (default: 0.2)
            ks_alpha: Significance level for KS test (default: 0.05)
            n_bins: Number of bins for PSI calculation
        """
        settings = get_settings()
        self.psi_threshold = psi_threshold or settings.drift_psi_threshold
        self.ks_alpha = ks_alpha or settings.drift_ks_alpha
        self.n_bins = n_bins
    
    def _calculate_psi(
        self,
        reference: np.ndarray,
        test: np.ndarray,
    ) -> float:
        """
        Calculate Population Stability Index.
        
        PSI = sum((test_pct - ref_pct) * ln(test_pct / ref_pct))
        
        Rules of thumb:
        - PSI < 0.1: No significant change
        - 0.1 <= PSI < 0.2: Moderate change
        - PSI >= 0.2: Significant change
        """
        # Create bins from reference distribution
        _, bin_edges = np.histogram(reference, bins=self.n_bins)
        
        # Calculate percentages
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        test_counts, _ = np.histogram(test, bins=bin_edges)
        
        # Avoid division by zero
        ref_pct = (ref_counts + 0.0001) / (len(reference) + 0.0001 * self.n_bins)
        test_pct = (test_counts + 0.0001) / (len(test) + 0.0001 * self.n_bins)
        
        # Calculate PSI
        psi = np.sum((test_pct - ref_pct) * np.log(test_pct / ref_pct))
        
        return float(psi)
    
    def _ks_test(
        self,
        reference: np.ndarray,
        test: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Perform two-sample Kolmogorov-Smirnov test.
        
        Returns:
            (statistic, p-value)
        """
        statistic, pvalue = stats.ks_2samp(reference, test)
        return float(statistic), float(pvalue)
    
    def _jensen_shannon(
        self,
        reference: np.ndarray,
        test: np.ndarray,
    ) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        JS divergence is symmetric and bounded [0, 1].
        """
        # Create common bins
        all_data = np.concatenate([reference, test])
        _, bin_edges = np.histogram(all_data, bins=self.n_bins)
        
        # Calculate histograms
        ref_hist, _ = np.histogram(reference, bins=bin_edges, density=True)
        test_hist, _ = np.histogram(test, bins=bin_edges, density=True)
        
        # Normalize
        ref_hist = ref_hist / (ref_hist.sum() + 1e-8)
        test_hist = test_hist / (test_hist.sum() + 1e-8)
        
        # Calculate JS divergence
        m = 0.5 * (ref_hist + test_hist)
        
        # KL divergences
        kl_ref = np.sum(ref_hist * np.log((ref_hist + 1e-8) / (m + 1e-8)))
        kl_test = np.sum(test_hist * np.log((test_hist + 1e-8) / (m + 1e-8)))
        
        js = 0.5 * (kl_ref + kl_test)
        
        return float(js)
    
    def _detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Detect drift in individual features.
        
        Returns dictionary of feature names to JS divergence.
        """
        feature_drifts = {}
        
        # Numeric columns to check
        numeric_cols = reference_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in test_df.columns:
                ref_vals = reference_df[col].dropna().values
                test_vals = test_df[col].dropna().values
                
                if len(ref_vals) > 10 and len(test_vals) > 10:
                    js = self._jensen_shannon(ref_vals, test_vals)
                    feature_drifts[col] = round(js, 4)
        
        return feature_drifts
    
    def detect_drift(
        self,
        data: pd.DataFrame,
        reference_days: int = 90,
        test_days: int = 30,
    ) -> List[DriftMetrics]:
        """
        Detect distributional drift for all SKUs.
        
        Args:
            data: Historical inventory data
            reference_days: Days to use as reference period
            test_days: Days to use as test period
        
        Returns:
            List of DriftMetrics for each SKU
        """
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Get date range
        max_date = df['date'].max()
        test_start = max_date - timedelta(days=test_days)
        reference_start = test_start - timedelta(days=reference_days)
        
        # Split into reference and test
        reference_df = df[(df['date'] >= reference_start) & (df['date'] < test_start)]
        test_df = df[df['date'] >= test_start]
        
        results = []
        
        for sku in df['sku'].unique():
            sku_ref = reference_df[reference_df['sku'] == sku]
            sku_test = test_df[test_df['sku'] == sku]
            
            # Need minimum data
            if len(sku_ref) < 10 or len(sku_test) < 5:
                continue
            
            ref_demand = sku_ref['quantity_sold'].values
            test_demand = sku_test['quantity_sold'].values
            
            # Calculate metrics
            psi = self._calculate_psi(ref_demand, test_demand)
            ks_stat, ks_pvalue = self._ks_test(ref_demand, test_demand)
            
            # Feature drift
            feature_drifts = self._detect_feature_drift(sku_ref, sku_test)
            
            # Determine if drifted
            is_drifted = (psi >= self.psi_threshold) or (ks_pvalue < self.ks_alpha)
            
            results.append(DriftMetrics(
                sku=sku,
                timestamp=max_date.date(),
                psi=round(psi, 4),
                ks_statistic=round(ks_stat, 4),
                ks_pvalue=round(ks_pvalue, 4),
                is_drifted=is_drifted,
                feature_drifts=feature_drifts if feature_drifts else None,
            ))
        
        return results
    
    def monitor_calibration(
        self,
        predictions: List[Dict],
        actuals: np.ndarray,
        confidence_levels: List[float] = None,
    ) -> Dict[str, float]:
        """
        Monitor prediction interval calibration.
        
        Checks if prediction intervals are correctly calibrated
        (e.g., 95% intervals should contain 95% of actuals).
        
        Args:
            predictions: List of dicts with 'lower' and 'upper' bounds
            actuals: Actual values
            confidence_levels: Expected confidence levels
        
        Returns:
            Dictionary of confidence level to actual coverage
        """
        confidence_levels = confidence_levels or [0.90, 0.95]
        
        results = {}
        
        for conf in confidence_levels:
            # Count how many actuals fall within intervals
            in_bound = 0
            total = 0
            
            for pred, actual in zip(predictions, actuals):
                if 'lower' in pred and 'upper' in pred:
                    if pred['lower'] <= actual <= pred['upper']:
                        in_bound += 1
                    total += 1
            
            if total > 0:
                coverage = in_bound / total
                results[f'coverage_{int(conf*100)}'] = round(coverage, 4)
                results[f'miscalibration_{int(conf*100)}'] = round(abs(coverage - conf), 4)
        
        return results


class DriftAlerter:
    """
    Alerting system for drift detection.
    """
    
    def __init__(
        self,
        psi_alert_threshold: float = 0.2,
        consecutive_drift_alert: int = 3,
    ):
        self.psi_alert_threshold = psi_alert_threshold
        self.consecutive_drift_alert = consecutive_drift_alert
        self._drift_history: Dict[str, List[bool]] = {}
    
    def check_and_alert(
        self,
        drift_metrics: List[DriftMetrics],
    ) -> List[Dict]:
        """
        Check drift metrics and generate alerts.
        
        Returns list of alert dictionaries.
        """
        alerts = []
        
        for metric in drift_metrics:
            # Update history
            if metric.sku not in self._drift_history:
                self._drift_history[metric.sku] = []
            
            self._drift_history[metric.sku].append(metric.is_drifted)
            
            # Keep only last N observations
            self._drift_history[metric.sku] = self._drift_history[metric.sku][-10:]
            
            # Check for consecutive drift
            history = self._drift_history[metric.sku]
            if len(history) >= self.consecutive_drift_alert:
                if all(history[-self.consecutive_drift_alert:]):
                    alerts.append({
                        'type': 'consecutive_drift',
                        'sku': metric.sku,
                        'severity': 'high',
                        'message': f"SKU {metric.sku} has shown drift for {self.consecutive_drift_alert} consecutive periods",
                        'psi': metric.psi,
                    })
            
            # Check for severe single drift
            if metric.psi >= self.psi_alert_threshold * 2:
                alerts.append({
                    'type': 'severe_drift',
                    'sku': metric.sku,
                    'severity': 'critical',
                    'message': f"SKU {metric.sku} shows severe distributional shift (PSI={metric.psi:.3f})",
                    'psi': metric.psi,
                })
        
        return alerts
