"""
SHAP-based Model Explainability.

Provides feature attribution using Shapley values.
Reference: Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions
"""

import numpy as np
from typing import Dict, List, Callable, Tuple
from dataclasses import dataclass
from itertools import combinations


@dataclass
class ShapExplanation:
    """SHAP explanation for a single prediction."""
    base_value: float
    shap_values: np.ndarray
    feature_names: List[str]
    feature_values: np.ndarray
    prediction: float


class KernelSHAP:
    """
    Kernel SHAP: Model-agnostic Shapley value estimation.
    
    Uses weighted linear regression to efficiently estimate
    Shapley values for any black-box model.
    """
    
    def __init__(self, model: Callable, background_data: np.ndarray, feature_names: List[str] = None):
        """
        Args:
            model: Prediction function (X -> y)
            background_data: Reference dataset for marginal expectations
            feature_names: Names of features
        """
        self.model = model
        self.background = background_data
        self.n_features = background_data.shape[1]
        self.feature_names = feature_names or [f'feature_{i}' for i in range(self.n_features)]
        
        # Base value (expected prediction on background)
        self.base_value = np.mean(model(background_data))
    
    def _shapley_weight(self, M: int, s: int) -> float:
        """
        Shapley kernel weight.
        
        w(s) = (M-1) / (C(M,s) * s * (M-s))
        """
        if s == 0 or s == M:
            return 1e6  # Large weight for empty and full sets
        
        # Binomial coefficient
        from math import comb
        return (M - 1) / (comb(M, s) * s * (M - s))
    
    def explain(self, x: np.ndarray, n_samples: int = 100) -> ShapExplanation:
        """
        Compute SHAP values for a single instance.
        
        Args:
            x: Single instance to explain (1D array)
            n_samples: Number of coalition samples
        
        Returns:
            ShapExplanation with Shapley values
        """
        M = self.n_features
        x = x.flatten()
        
        # Sample coalitions (binary masks indicating which features to include)
        coalitions = []
        weights = []
        predictions = []
        
        # Always include empty and full set
        coalitions.append(np.zeros(M))
        coalitions.append(np.ones(M))
        
        # Sample random coalitions
        for _ in range(n_samples - 2):
            s = np.random.randint(1, M)  # Size of coalition
            mask = np.zeros(M)
            mask[np.random.choice(M, s, replace=False)] = 1
            coalitions.append(mask)
        
        # Evaluate each coalition
        for mask in coalitions:
            # Create masked instance (mix of x and background)
            n_bg = min(10, len(self.background))
            bg_samples = self.background[np.random.choice(len(self.background), n_bg)]
            
            masked_instances = []
            for bg in bg_samples:
                instance = np.where(mask == 1, x, bg)
                masked_instances.append(instance)
            
            masked_instances = np.array(masked_instances)
            pred = np.mean(self.model(masked_instances))
            predictions.append(pred)
            
            s = int(mask.sum())
            weights.append(self._shapley_weight(M, s))
        
        coalitions = np.array(coalitions)
        predictions = np.array(predictions)
        weights = np.array(weights)
        
        # Weighted linear regression to get SHAP values
        # y = base_value + sum(shap_i * mask_i)
        y = predictions - self.base_value
        
        # Weighted least squares: (X'WX)^-1 X'Wy
        W = np.diag(weights)
        XTW = coalitions.T @ W
        
        try:
            shap_values = np.linalg.solve(XTW @ coalitions + 1e-6*np.eye(M), XTW @ y)
        except:
            shap_values = np.linalg.lstsq(coalitions, y, rcond=None)[0]
        
        prediction = float(self.model(x.reshape(1, -1))[0])
        
        return ShapExplanation(
            base_value=self.base_value,
            shap_values=shap_values,
            feature_names=self.feature_names,
            feature_values=x,
            prediction=prediction
        )
    
    def explain_batch(self, X: np.ndarray, n_samples: int = 50) -> List[ShapExplanation]:
        """Explain multiple instances."""
        return [self.explain(x, n_samples) for x in X]


def generate_waterfall_data(explanation: ShapExplanation) -> Dict:
    """
    Generate data for waterfall chart visualization.
    
    Returns sorted features by absolute contribution.
    """
    # Sort by absolute SHAP value
    sorted_idx = np.argsort(-np.abs(explanation.shap_values))
    
    waterfall_data = {
        'base_value': explanation.base_value,
        'prediction': explanation.prediction,
        'features': [],
        'cumulative': [explanation.base_value]
    }
    
    running_total = explanation.base_value
    for idx in sorted_idx:
        running_total += explanation.shap_values[idx]
        waterfall_data['features'].append({
            'name': explanation.feature_names[idx],
            'value': float(explanation.feature_values[idx]),
            'shap_value': float(explanation.shap_values[idx]),
            'cumulative': running_total
        })
        waterfall_data['cumulative'].append(running_total)
    
    return waterfall_data


def generate_summary_plot_data(explanations: List[ShapExplanation]) -> Dict:
    """
    Generate data for SHAP summary plot (beeswarm).
    """
    n_features = len(explanations[0].feature_names)
    
    summary = {
        'feature_names': explanations[0].feature_names,
        'mean_abs_shap': np.zeros(n_features),
        'feature_importance': []
    }
    
    all_shap = np.array([exp.shap_values for exp in explanations])
    summary['mean_abs_shap'] = np.mean(np.abs(all_shap), axis=0)
    
    # Sort by importance
    sorted_idx = np.argsort(-summary['mean_abs_shap'])
    
    for idx in sorted_idx:
        summary['feature_importance'].append({
            'name': explanations[0].feature_names[idx],
            'importance': float(summary['mean_abs_shap'][idx]),
            'values': all_shap[:, idx].tolist()
        })
    
    return summary


class TreeSHAP:
    """
    Tree SHAP for tree-based models (simplified).
    
    More efficient than Kernel SHAP for tree ensembles.
    """
    
    def __init__(self, tree_predictions: Callable, feature_names: List[str]):
        self.predict = tree_predictions
        self.feature_names = feature_names
    
    def explain(self, x: np.ndarray, background: np.ndarray) -> ShapExplanation:
        """Simplified TreeSHAP using interventional approach."""
        n_features = len(x)
        shap_values = np.zeros(n_features)
        
        base_pred = np.mean(self.predict(background))
        full_pred = self.predict(x.reshape(1, -1))[0]
        
        # Approximate: Measure each feature's marginal contribution
        for i in range(n_features):
            # With feature i
            x_with = x.copy()
            # Without feature i (use mean from background)
            x_without = x.copy()
            x_without[i] = np.mean(background[:, i])
            
            pred_with = self.predict(x_with.reshape(1, -1))[0]
            pred_without = self.predict(x_without.reshape(1, -1))[0]
            
            shap_values[i] = pred_with - pred_without
        
        # Normalize to match prediction
        scale = (full_pred - base_pred) / (np.sum(shap_values) + 1e-8)
        shap_values *= scale
        
        return ShapExplanation(
            base_value=base_pred,
            shap_values=shap_values,
            feature_names=self.feature_names,
            feature_values=x,
            prediction=full_pred
        )
