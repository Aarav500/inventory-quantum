"""
Anomaly Detection for Inventory Data.

Detects unusual patterns in demand:
- Isolation Forest for multivariate outliers
- Statistical process control
- Changepoint detection
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Anomaly:
    """Detected anomaly."""
    index: int
    value: float
    score: float
    type: str
    explanation: str


class IsolationForest:
    """
    Isolation Forest for anomaly detection.
    
    Anomalies are isolated quickly (short path length).
    Reference: Liu et al. (2008)
    """
    
    def __init__(self, n_estimators: int = 100, max_samples: int = 256):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.trees = []
    
    def _build_tree(self, X: np.ndarray, depth: int = 0, max_depth: int = 10) -> Dict:
        """Recursively build isolation tree."""
        n_samples, n_features = X.shape
        
        if depth >= max_depth or n_samples <= 1:
            return {'type': 'leaf', 'size': n_samples}
        
        # Random split
        feature = np.random.randint(n_features)
        min_val, max_val = X[:, feature].min(), X[:, feature].max()
        
        if min_val == max_val:
            return {'type': 'leaf', 'size': n_samples}
        
        split_value = np.random.uniform(min_val, max_val)
        
        left_mask = X[:, feature] < split_value
        right_mask = ~left_mask
        
        return {
            'type': 'node',
            'feature': feature,
            'split': split_value,
            'left': self._build_tree(X[left_mask], depth + 1, max_depth),
            'right': self._build_tree(X[right_mask], depth + 1, max_depth)
        }
    
    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Calculate path length to isolate point."""
        if tree['type'] == 'leaf':
            # Adjustment for unbuilt trees
            c = 2 * (np.log(tree['size'] + 1) + 0.5772) - 2 * tree['size'] / (tree['size'] + 1) if tree['size'] > 1 else 0
            return depth + c
        
        if x[tree['feature']] < tree['split']:
            return self._path_length(x, tree['left'], depth + 1)
        else:
            return self._path_length(x, tree['right'], depth + 1)
    
    def fit(self, X: np.ndarray):
        """Fit isolation forest."""
        n_samples = X.shape[0]
        sample_size = min(self.max_samples, n_samples)
        max_depth = int(np.ceil(np.log2(sample_size)))
        
        self.trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, sample_size, replace=False)
            tree = self._build_tree(X[idx], max_depth=max_depth)
            self.trees.append(tree)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores.
        
        Returns: Scores in [-1, 0], more negative = more anomalous
        """
        n_samples = X.shape[0]
        scores = np.zeros(n_samples)
        
        for i, x in enumerate(X):
            avg_path_length = np.mean([self._path_length(x, tree) for tree in self.trees])
            
            # Normalize
            c = 2 * (np.log(self.max_samples) + 0.5772) - 2 * (self.max_samples - 1) / self.max_samples
            scores[i] = -2 ** (-avg_path_length / c)
        
        return scores
    
    def predict(self, X: np.ndarray, threshold: float = -0.5) -> np.ndarray:
        """Predict anomalies (1 = anomaly, 0 = normal)."""
        scores = self.score_samples(X)
        return (scores < threshold).astype(int)


class StatisticalAnomalyDetector:
    """
    Statistical process control for anomaly detection.
    """
    
    def __init__(self, window_size: int = 20, n_sigma: float = 3.0):
        self.window_size = window_size
        self.n_sigma = n_sigma
    
    def detect(self, values: np.ndarray) -> List[Anomaly]:
        """Detect anomalies using rolling statistics."""
        anomalies = []
        
        for i in range(self.window_size, len(values)):
            window = values[i - self.window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std == 0:
                continue
            
            z_score = (values[i] - mean) / std
            
            if abs(z_score) > self.n_sigma:
                anomalies.append(Anomaly(
                    index=i,
                    value=float(values[i]),
                    score=float(abs(z_score)),
                    type='statistical',
                    explanation=f"{z_score:.1f}σ from rolling mean ({mean:.1f})"
                ))
        
        return anomalies


class ChangepointDetector:
    """
    Detect changepoints in time series.
    
    Uses CUSUM algorithm.
    """
    
    def __init__(self, threshold: float = 5.0):
        self.threshold = threshold
    
    def detect(self, values: np.ndarray) -> List[int]:
        """Detect changepoints using CUSUM."""
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return []
        
        normalized = (values - mean) / std
        
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        changepoints = []
        
        for i in range(1, len(values)):
            cusum_pos[i] = max(0, cusum_pos[i-1] + normalized[i] - 0.5)
            cusum_neg[i] = min(0, cusum_neg[i-1] + normalized[i] + 0.5)
            
            if cusum_pos[i] > self.threshold or cusum_neg[i] < -self.threshold:
                changepoints.append(i)
                cusum_pos[i] = 0
                cusum_neg[i] = 0
        
        return changepoints


class AnomalyEngine:
    """
    Combined anomaly detection engine.
    """
    
    def __init__(self):
        self.isolation_forest = IsolationForest()
        self.statistical = StatisticalAnomalyDetector()
        self.changepoint = ChangepointDetector()
    
    def analyze(self, data: np.ndarray) -> Dict:
        """
        Comprehensive anomaly analysis.
        """
        if data.ndim == 1:
            data_2d = data.reshape(-1, 1)
        else:
            data_2d = data
        
        # Isolation Forest
        self.isolation_forest.fit(data_2d)
        if_scores = self.isolation_forest.score_samples(data_2d)
        if_anomalies = np.where(if_scores < -0.5)[0].tolist()
        
        # Statistical
        stat_anomalies = self.statistical.detect(data_2d[:, 0])
        
        # Changepoints
        changepoints = self.changepoint.detect(data_2d[:, 0])
        
        return {
            'isolation_forest_scores': if_scores.tolist(),
            'isolation_forest_anomalies': if_anomalies,
            'statistical_anomalies': [{'index': a.index, 'value': a.value, 'score': a.score, 'explanation': a.explanation} for a in stat_anomalies],
            'changepoints': changepoints,
            'summary': {
                'total_if_anomalies': len(if_anomalies),
                'total_stat_anomalies': len(stat_anomalies),
                'total_changepoints': len(changepoints)
            }
        }
