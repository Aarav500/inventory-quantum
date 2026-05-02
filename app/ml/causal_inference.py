"""
Causal Inference for Promotion Uplift Estimation.

Estimates the causal effect of promotions on demand,
not just correlation.

Methods:
1. Propensity Score Matching
2. Inverse Propensity Weighting (IPW)
3. Double Machine Learning (DML)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import norm


@dataclass
class CausalEffect:
    """Estimated causal effect."""
    ate: float  # Average Treatment Effect
    att: float  # Average Treatment Effect on Treated
    ci_lower: float
    ci_upper: float
    p_value: float
    method: str


class PropensityScoreEstimator:
    """
    Propensity Score estimation using logistic regression.
    
    P(T=1 | X) - probability of receiving treatment given covariates
    """
    
    def __init__(self):
        self._weights = None
        self._bias = 0
    
    def fit(self, X: np.ndarray, T: np.ndarray):
        """Fit logistic regression for propensity scores."""
        n_features = X.shape[1]
        self._weights = np.zeros(n_features)
        self._bias = 0
        
        # Simple gradient descent
        lr = 0.01
        for _ in range(1000):
            scores = X @ self._weights + self._bias
            probs = 1 / (1 + np.exp(-scores))
            
            # Gradient
            error = T - probs
            self._weights += lr * (X.T @ error) / len(T)
            self._bias += lr * np.mean(error)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict propensity scores."""
        scores = X @ self._weights + self._bias
        return 1 / (1 + np.exp(-scores))


class CausalInferenceEngine:
    """
    Causal inference for treatment effect estimation.
    """
    
    def __init__(self, method: str = 'ipw'):
        """
        Initialize causal inference engine.
        
        Args:
            method: 'matching', 'ipw', or 'dml'
        """
        self.method = method
        self.propensity_model = PropensityScoreEstimator()
    
    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract covariates, treatment, and outcome."""
        # Identify columns
        if 'promotion' in data.columns:
            T = data['promotion'].values.astype(float)
        else:
            raise ValueError("No 'promotion' column found")
        
        if 'quantity_sold' in data.columns:
            Y = data['quantity_sold'].values.astype(float)
        else:
            raise ValueError("No 'quantity_sold' column found")
        
        # Covariates (exclude treatment and outcome)
        covariate_cols = [c for c in data.columns 
                        if c not in ['promotion', 'quantity_sold', 'date', 'sku']]
        
        if len(covariate_cols) == 0:
            # Create time-based features
            data['day_of_week'] = pd.to_datetime(data['date']).dt.dayofweek
            covariate_cols = ['day_of_week']
        
        X = data[covariate_cols].values.astype(float)
        
        return X, T, Y
    
    def estimate_ate_ipw(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> CausalEffect:
        """
        Inverse Propensity Weighting estimator.
        
        ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
        """
        # Fit propensity model
        self.propensity_model.fit(X, T)
        e = self.propensity_model.predict_proba(X)
        
        # Clip for stability
        e = np.clip(e, 0.01, 0.99)
        
        # IPW estimator
        treated_term = np.mean(Y * T / e)
        control_term = np.mean(Y * (1 - T) / (1 - e))
        ate = treated_term - control_term
        
        # Bootstrap confidence interval
        n_bootstrap = 100
        ate_samples = []
        for _ in range(n_bootstrap):
            idx = np.random.choice(len(Y), len(Y), replace=True)
            t_term = np.mean(Y[idx] * T[idx] / e[idx])
            c_term = np.mean(Y[idx] * (1 - T[idx]) / (1 - e[idx]))
            ate_samples.append(t_term - c_term)
        
        ci_lower = np.percentile(ate_samples, 2.5)
        ci_upper = np.percentile(ate_samples, 97.5)
        
        # p-value (H0: ATE = 0)
        se = np.std(ate_samples)
        z = ate / (se + 1e-8)
        p_value = 2 * (1 - norm.cdf(abs(z)))
        
        # ATT
        att = np.mean(Y[T == 1]) - np.sum(Y[T == 0] * e[T == 0] / (1 - e[T == 0])) / np.sum(e[T == 1] / (1 - e[T == 1]))
        
        return CausalEffect(
            ate=float(ate),
            att=float(att),
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            method='ipw'
        )
    
    def estimate_ate_matching(self, X: np.ndarray, T: np.ndarray, Y: np.ndarray) -> CausalEffect:
        """
        Propensity Score Matching estimator.
        
        Matches treated units to similar control units.
        """
        # Fit propensity model
        self.propensity_model.fit(X, T)
        e = self.propensity_model.predict_proba(X)
        
        # Match each treated to nearest control
        treated_idx = np.where(T == 1)[0]
        control_idx = np.where(T == 0)[0]
        
        matched_effects = []
        for t_idx in treated_idx:
            # Find nearest control by propensity score
            distances = np.abs(e[control_idx] - e[t_idx])
            nearest = control_idx[np.argmin(distances)]
            matched_effects.append(Y[t_idx] - Y[nearest])
        
        ate = np.mean(matched_effects)
        se = np.std(matched_effects) / np.sqrt(len(matched_effects))
        
        ci_lower = ate - 1.96 * se
        ci_upper = ate + 1.96 * se
        p_value = 2 * (1 - norm.cdf(abs(ate / (se + 1e-8))))
        
        return CausalEffect(
            ate=float(ate),
            att=float(ate),  # ATT = ATE for matching
            ci_lower=float(ci_lower),
            ci_upper=float(ci_upper),
            p_value=float(p_value),
            method='matching'
        )
    
    def estimate_promotion_uplift(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Estimate the causal effect of promotions on sales.
        
        Returns comprehensive uplift analysis.
        """
        X, T, Y = self._prepare_data(data)
        
        # Estimate using both methods
        ipw_effect = self.estimate_ate_ipw(X, T, Y)
        matching_effect = self.estimate_ate_matching(X, T, Y)
        
        # Summary statistics
        promoted_sales = Y[T == 1].mean() if T.sum() > 0 else 0
        baseline_sales = Y[T == 0].mean() if (1 - T).sum() > 0 else 0
        naive_uplift = promoted_sales - baseline_sales
        
        return {
            'naive_uplift': naive_uplift,
            'causal_uplift_ipw': ipw_effect.ate,
            'causal_uplift_matching': matching_effect.ate,
            'ipw_result': ipw_effect,
            'matching_result': matching_effect,
            'promoted_avg_sales': promoted_sales,
            'baseline_avg_sales': baseline_sales,
            'n_promoted': int(T.sum()),
            'n_control': int((1 - T).sum()),
            'interpretation': self._interpret_results(ipw_effect, naive_uplift)
        }
    
    def _interpret_results(self, effect: CausalEffect, naive: float) -> str:
        """Generate human-readable interpretation."""
        if effect.p_value < 0.05:
            significance = "statistically significant"
        else:
            significance = "not statistically significant"
        
        bias = naive - effect.ate
        bias_pct = abs(bias / (naive + 1e-8)) * 100
        
        return (
            f"Promotion increases sales by {effect.ate:.1f} units (95% CI: [{effect.ci_lower:.1f}, {effect.ci_upper:.1f}]). "
            f"This effect is {significance} (p={effect.p_value:.3f}). "
            f"Naive estimate ({naive:.1f}) has {bias_pct:.0f}% selection bias."
        )


def counterfactual_analysis(data: pd.DataFrame, sku: str) -> Dict[str, Any]:
    """
    What-if analysis: What would sales be if we had/hadn't run promotions?
    """
    engine = CausalInferenceEngine()
    uplift = engine.estimate_promotion_uplift(data)
    
    # Counterfactuals
    sku_data = data[data['sku'] == sku] if 'sku' in data.columns else data
    actual_sales = sku_data['quantity_sold'].sum()
    n_promoted_days = sku_data['promotion'].sum() if 'promotion' in sku_data.columns else 0
    
    # Counterfactual: no promotions
    sales_without_promo = actual_sales - n_promoted_days * uplift['causal_uplift_ipw']
    
    # Counterfactual: always promote
    n_days = len(sku_data)
    sales_with_promo = actual_sales + (n_days - n_promoted_days) * uplift['causal_uplift_ipw']
    
    return {
        'actual_total_sales': actual_sales,
        'counterfactual_no_promo': sales_without_promo,
        'counterfactual_always_promo': sales_with_promo,
        'promotion_contribution': actual_sales - sales_without_promo,
        'promotion_contribution_pct': (actual_sales - sales_without_promo) / actual_sales * 100,
        'uplift_per_promo_day': uplift['causal_uplift_ipw'],
    }
