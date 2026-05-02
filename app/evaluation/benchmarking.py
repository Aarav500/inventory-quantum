"""
Rigorous Benchmarking Module.

Implements proper experimental methodology:
- Statistical significance testing
- Critical difference diagrams
- Shapley values for feature importance
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from scipy import stats


@dataclass
class BenchmarkResult:
    """Result from model benchmarking."""
    model: str
    metric: str
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    ranks: List[float]


class StatisticalTester:
    """
    Statistical significance testing for model comparison.
    """
    
    @staticmethod
    def friedman_test(results_matrix: np.ndarray) -> Tuple[float, float]:
        """
        Friedman test for comparing multiple classifiers.
        
        Non-parametric test for repeated measures.
        
        Args:
            results_matrix: (n_datasets x n_models) matrix of performance scores
        
        Returns:
            (statistic, p-value)
        """
        stat, pvalue = stats.friedmanchisquare(*results_matrix.T)
        return float(stat), float(pvalue)
    
    @staticmethod
    def nemenyi_critical_difference(
        n_models: int,
        n_datasets: int,
        alpha: float = 0.05
    ) -> float:
        """
        Calculate Nemenyi critical difference.
        
        Two models are significantly different if their average ranks
        differ by more than CD.
        """
        # q_alpha values for Nemenyi test
        q_alpha = {
            3: 2.343,
            4: 2.569,
            5: 2.728,
            6: 2.850,
            7: 2.949,
            8: 3.031,
        }
        
        q = q_alpha.get(n_models, 3.0)
        cd = q * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))
        return cd
    
    @staticmethod
    def wilcoxon_test(
        scores1: np.ndarray,
        scores2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Wilcoxon signed-rank test for pairwise comparison.
        """
        stat, pvalue = stats.wilcoxon(scores1, scores2)
        return float(stat), float(pvalue)
    
    @staticmethod
    def bootstrap_ci(
        scores: np.ndarray,
        n_bootstrap: int = 1000,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Bootstrap confidence interval.
        """
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return lower, upper


class ShapleyExplainer:
    """
    Shapley values for feature importance.
    
    Provides fair attribution of feature contributions.
    """
    
    def __init__(self, model, X: np.ndarray, y: np.ndarray):
        self.model = model
        self.X = X
        self.y = y
        self.n_features = X.shape[1]
    
    def _model_predict(self, X: np.ndarray) -> np.ndarray:
        """Get model predictions."""
        if hasattr(self.model, 'predict'):
            return self.model.predict(X)
        return np.zeros(len(X))
    
    def _marginal_contribution(
        self,
        feature_idx: int,
        coalition: List[int],
        x_instance: np.ndarray,
    ) -> float:
        """
        Compute marginal contribution of feature to coalition.
        """
        # Baseline prediction with features marginalized
        n_samples = min(100, len(self.X))
        sample_idx = np.random.choice(len(self.X), n_samples, replace=False)
        
        # With feature
        X_with = self.X[sample_idx].copy()
        for i in coalition + [feature_idx]:
            X_with[:, i] = x_instance[i]
        pred_with = self._model_predict(X_with).mean()
        
        # Without feature
        X_without = self.X[sample_idx].copy()
        for i in coalition:
            X_without[:, i] = x_instance[i]
        pred_without = self._model_predict(X_without).mean()
        
        return pred_with - pred_without
    
    def compute_shapley_values(
        self,
        x_instance: np.ndarray,
        n_samples: int = 100,
    ) -> np.ndarray:
        """
        Compute Shapley values for a single instance.
        
        Uses Monte Carlo sampling for efficiency.
        """
        shapley_values = np.zeros(self.n_features)
        
        for _ in range(n_samples):
            # Random permutation
            perm = np.random.permutation(self.n_features)
            
            coalition = []
            for feature_idx in perm:
                contribution = self._marginal_contribution(
                    feature_idx, coalition, x_instance
                )
                shapley_values[feature_idx] += contribution
                coalition.append(feature_idx)
        
        return shapley_values / n_samples
    
    def explain_all(self, n_instances: int = 50) -> Dict[str, float]:
        """
        Compute average Shapley values across instances.
        """
        indices = np.random.choice(
            len(self.X), 
            min(n_instances, len(self.X)), 
            replace=False
        )
        
        all_shapley = []
        for idx in indices:
            sv = self.compute_shapley_values(self.X[idx])
            all_shapley.append(sv)
        
        mean_shapley = np.mean(all_shapley, axis=0)
        
        return {f'feature_{i}': float(v) for i, v in enumerate(mean_shapley)}


class ModelBenchmark:
    """
    Comprehensive model benchmarking framework.
    """
    
    def __init__(self, models: Dict[str, Any], data_splits: List[Tuple]):
        """
        Args:
            models: Dict of model name -> model instance
            data_splits: List of (train, test) tuples
        """
        self.models = models
        self.data_splits = data_splits
        self.results = {}
    
    def run_benchmark(self, metric_fn: callable) -> Dict[str, BenchmarkResult]:
        """
        Run full benchmark across all models and splits.
        """
        n_models = len(self.models)
        n_splits = len(self.data_splits)
        
        results_matrix = np.zeros((n_splits, n_models))
        model_names = list(self.models.keys())
        
        for split_idx, (train_data, test_data) in enumerate(self.data_splits):
            for model_idx, (name, model) in enumerate(self.models.items()):
                try:
                    # Fit and predict
                    if hasattr(model, 'fit'):
                        model.fit(train_data)
                    
                    predictions = model.forecast(train_data, horizon=len(test_data))
                    pred_values = [p.predicted for p in predictions]
                    
                    # Get actual values
                    actual = test_data.groupby('date')['quantity_sold'].sum().values
                    actual = actual[:len(pred_values)]
                    
                    # Calculate metric
                    score = metric_fn(actual, np.array(pred_values))
                    results_matrix[split_idx, model_idx] = score
                except Exception as e:
                    results_matrix[split_idx, model_idx] = np.nan
        
        # Statistical analysis
        tester = StatisticalTester()
        
        # Friedman test
        valid_results = results_matrix[~np.isnan(results_matrix).any(axis=1)]
        if len(valid_results) >= 3:
            friedman_stat, friedman_p = tester.friedman_test(valid_results)
        else:
            friedman_stat, friedman_p = None, None
        
        # Compute ranks
        ranks = stats.rankdata(results_matrix, axis=1)
        
        # Build results
        benchmark_results = {}
        for model_idx, name in enumerate(model_names):
            scores = results_matrix[:, model_idx]
            valid_scores = scores[~np.isnan(scores)]
            
            if len(valid_scores) > 0:
                ci_lower, ci_upper = tester.bootstrap_ci(valid_scores)
                
                benchmark_results[name] = BenchmarkResult(
                    model=name,
                    metric='rmse',
                    mean=float(np.mean(valid_scores)),
                    std=float(np.std(valid_scores)),
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    ranks=ranks[:, model_idx].tolist(),
                )
        
        self.results = {
            'models': benchmark_results,
            'friedman_statistic': friedman_stat,
            'friedman_pvalue': friedman_p,
            'critical_difference': tester.nemenyi_critical_difference(n_models, n_splits),
        }
        
        return benchmark_results
    
    def generate_report(self) -> str:
        """Generate markdown report of benchmark results."""
        if not self.results:
            return "No benchmark results available."
        
        lines = ["# Model Benchmark Report\n"]
        
        # Friedman test
        if self.results.get('friedman_pvalue'):
            p = self.results['friedman_pvalue']
            sig = "significant" if p < 0.05 else "not significant"
            lines.append(f"## Statistical Significance\n")
            lines.append(f"Friedman test: χ² = {self.results['friedman_statistic']:.2f}, p = {p:.4f} ({sig})\n")
            lines.append(f"Critical Difference (Nemenyi): {self.results['critical_difference']:.3f}\n")
        
        # Model comparison table
        lines.append("\n## Model Performance\n")
        lines.append("| Model | Mean | Std | 95% CI | Avg Rank |")
        lines.append("|-------|------|-----|--------|----------|")
        
        for name, result in self.results['models'].items():
            avg_rank = np.mean(result.ranks)
            lines.append(
                f"| {name} | {result.mean:.4f} | {result.std:.4f} | "
                f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}] | {avg_rank:.2f} |"
            )
        
        return "\n".join(lines)
