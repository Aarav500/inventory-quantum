"""
Bayesian Hyperparameter Optimization.

Uses Gaussian Process surrogate with Expected Improvement acquisition.
More sample-efficient than grid/random search.

Reference: Snoek et al. (2012) - Practical Bayesian Optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
from scipy.stats import norm
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    """Result from Bayesian optimization."""
    best_params: Dict[str, float]
    best_score: float
    all_trials: List[Dict]
    convergence_history: List[float]


class GaussianProcessSurrogate:
    """
    Gaussian Process regression for surrogate modeling.
    
    Uses RBF kernel: k(x, x') = σ² exp(-||x - x'||² / (2l²))
    """
    
    def __init__(self, length_scale: float = 1.0, noise: float = 1e-6):
        self.length_scale = length_scale
        self.noise = noise
        self.X_train = None
        self.y_train = None
        self.K_inv = None
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Radial Basis Function kernel."""
        sq_dist = np.sum(X1**2, axis=1, keepdims=True) + \
                  np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return np.exp(-sq_dist / (2 * self.length_scale**2))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit GP to observed data."""
        self.X_train = X
        self.y_train = y
        
        K = self._rbf_kernel(X, X) + self.noise * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance at new points."""
        if self.X_train is None:
            return np.zeros(len(X)), np.ones(len(X))
        
        K_star = self._rbf_kernel(X, self.X_train)
        K_star_star = self._rbf_kernel(X, X)
        
        mean = K_star @ self.K_inv @ self.y_train
        var = np.diag(K_star_star - K_star @ self.K_inv @ K_star.T)
        var = np.maximum(var, 1e-8)  # Numerical stability
        
        return mean, var


class BayesianOptimizer:
    """
    Bayesian Optimization with Expected Improvement.
    
    Balances exploration (high uncertainty) vs exploitation (high predicted value).
    """
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        n_initial: int = 5,
        n_iterations: int = 20,
        xi: float = 0.01,  # Exploration-exploitation tradeoff
    ):
        self.param_bounds = param_bounds
        self.param_names = list(param_bounds.keys())
        self.n_dims = len(param_bounds)
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.xi = xi
        
        self.gp = GaussianProcessSurrogate()
        self.X_observed = []
        self.y_observed = []
        self.best_score = -np.inf
        self.best_params = None
    
    def _normalize(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1]."""
        x = []
        for name in self.param_names:
            lo, hi = self.param_bounds[name]
            x.append((params[name] - lo) / (hi - lo))
        return np.array(x)
    
    def _denormalize(self, x: np.ndarray) -> Dict[str, float]:
        """Denormalize from [0, 1] to original bounds."""
        params = {}
        for i, name in enumerate(self.param_names):
            lo, hi = self.param_bounds[name]
            params[name] = lo + x[i] * (hi - lo)
        return params
    
    def _expected_improvement(self, x: np.ndarray) -> float:
        """
        Expected Improvement acquisition function.
        
        EI(x) = E[max(f(x) - f(x*), 0)]
        """
        if len(self.X_observed) == 0:
            return 1.0
        
        mean, var = self.gp.predict(x.reshape(1, -1))
        std = np.sqrt(var)
        
        if std < 1e-8:
            return 0.0
        
        z = (mean - self.best_score - self.xi) / std
        ei = (mean - self.best_score - self.xi) * norm.cdf(z) + std * norm.pdf(z)
        
        return float(ei)
    
    def _suggest_next(self) -> Dict[str, float]:
        """Suggest next point to evaluate using EI."""
        if len(self.X_observed) < self.n_initial:
            # Random sampling for initial points
            x = np.random.rand(self.n_dims)
            return self._denormalize(x)
        
        # Fit GP
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)
        
        # Optimize acquisition function
        best_ei = -np.inf
        best_x = None
        
        # Multi-start optimization
        for _ in range(10):
            x0 = np.random.rand(self.n_dims)
            result = minimize(
                lambda x: -self._expected_improvement(x),
                x0,
                bounds=[(0, 1)] * self.n_dims,
                method='L-BFGS-B'
            )
            if -result.fun > best_ei:
                best_ei = -result.fun
                best_x = result.x
        
        return self._denormalize(best_x)
    
    def optimize(self, objective: Callable[[Dict[str, float]], float]) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            objective: Function to maximize (params dict -> score)
        
        Returns:
            OptimizationResult with best parameters found
        """
        trials = []
        convergence = []
        
        for i in range(self.n_iterations):
            # Suggest next point
            params = self._suggest_next()
            
            # Evaluate objective
            score = objective(params)
            
            # Store observation
            x_normalized = self._normalize(params)
            self.X_observed.append(x_normalized)
            self.y_observed.append(score)
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
            
            trials.append({'params': params, 'score': score, 'iteration': i})
            convergence.append(self.best_score)
        
        return OptimizationResult(
            best_params=self.best_params,
            best_score=self.best_score,
            all_trials=trials,
            convergence_history=convergence
        )


def tune_forecaster(forecaster, data, param_bounds: Dict[str, Tuple[float, float]]) -> OptimizationResult:
    """
    Tune forecaster hyperparameters using Bayesian optimization.
    
    Example:
        param_bounds = {
            'learning_rate': (0.001, 0.1),
            'hidden_size': (32, 128),
            'dropout': (0.1, 0.5),
        }
    """
    def objective(params):
        # Set parameters on forecaster
        for key, value in params.items():
            if hasattr(forecaster, key):
                setattr(forecaster, key, value)
        
        # Cross-validation score (simplified)
        try:
            forecaster.fit(data)
            metrics = forecaster.last_metrics or {}
            return -metrics.get('rmse', 100)  # Minimize RMSE
        except:
            return -100
    
    optimizer = BayesianOptimizer(param_bounds)
    return optimizer.optimize(objective)
