"""
Meta-Learning for Few-Shot SKU Adaptation.

Enables forecasting for new products with only 5-10 data points.
Reference: Finn et al. (2017) - Model-Agnostic Meta-Learning (MAML)
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class Task:
    """A meta-learning task (one SKU's data)."""
    sku_id: str
    support_x: np.ndarray  # Few examples for adaptation
    support_y: np.ndarray
    query_x: np.ndarray    # Test examples
    query_y: np.ndarray


class SimpleForecaster:
    """
    Simple neural network forecaster for meta-learning.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Initialize weights
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.1
        self.b2 = np.zeros(1)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        h = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return h @ self.W2 + self.b2
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get model parameters."""
        return {'W1': self.W1, 'b1': self.b1, 'W2': self.W2, 'b2': self.b2}
    
    def set_params(self, params: Dict[str, np.ndarray]):
        """Set model parameters."""
        self.W1 = params['W1'].copy()
        self.b1 = params['b1'].copy()
        self.W2 = params['W2'].copy()
        self.b2 = params['b2'].copy()
    
    def compute_gradients(self, x: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute gradients for a batch."""
        # Forward
        h = np.maximum(0, x @ self.W1 + self.b1)
        y_pred = h @ self.W2 + self.b2
        
        # Loss gradient
        n = len(x)
        d_loss = 2 * (y_pred - y.reshape(-1, 1)) / n
        
        # Backward
        d_W2 = h.T @ d_loss
        d_b2 = np.sum(d_loss, axis=0)
        
        d_h = d_loss @ self.W2.T
        d_h[h <= 0] = 0  # ReLU gradient
        
        d_W1 = x.T @ d_h
        d_b1 = np.sum(d_h, axis=0)
        
        return {'W1': d_W1, 'b1': d_b1, 'W2': d_W2, 'b2': d_b2}


class MAML:
    """
    Model-Agnostic Meta-Learning.
    
    Learns initialization that can quickly adapt to new tasks.
    
    Algorithm:
    1. Sample batch of tasks
    2. For each task, compute adapted parameters (few gradient steps)
    3. Update meta-parameters using adapted parameters' performance
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        inner_lr: float = 0.01,
        outer_lr: float = 0.001,
        inner_steps: int = 5
    ):
        self.model = SimpleForecaster(input_dim, hidden_dim)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.inner_steps = inner_steps
        
        self.meta_params = self.model.get_params()
    
    def _inner_loop(self, task: Task) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Inner loop: Adapt to a specific task.
        
        Returns adapted parameters and query loss.
        """
        # Start from meta-parameters
        self.model.set_params(self.meta_params)
        
        # Gradient descent on support set
        for _ in range(self.inner_steps):
            grads = self.model.compute_gradients(task.support_x, task.support_y)
            params = self.model.get_params()
            
            for key in params:
                params[key] = params[key] - self.inner_lr * grads[key]
            
            self.model.set_params(params)
        
        # Evaluate on query set
        y_pred = self.model.forward(task.query_x)
        query_loss = np.mean((y_pred.flatten() - task.query_y) ** 2)
        
        return self.model.get_params(), query_loss
    
    def meta_train(self, tasks: List[Task], n_iterations: int = 100) -> List[float]:
        """
        Meta-training loop.
        
        Args:
            tasks: List of training tasks (different SKUs)
            n_iterations: Number of meta-updates
        
        Returns:
            List of meta-losses
        """
        meta_losses = []
        
        for iteration in range(n_iterations):
            # Sample task batch
            batch_size = min(4, len(tasks))
            task_batch = np.random.choice(tasks, batch_size, replace=False)
            
            # Accumulate gradients across tasks
            meta_grads = {key: np.zeros_like(val) for key, val in self.meta_params.items()}
            total_loss = 0
            
            for task in task_batch:
                adapted_params, query_loss = self._inner_loop(task)
                total_loss += query_loss
                
                # Compute meta-gradient (simplified: finite difference approximation)
                self.model.set_params(adapted_params)
                grads = self.model.compute_gradients(task.query_x, task.query_y)
                
                for key in meta_grads:
                    meta_grads[key] += grads[key] / batch_size
            
            # Update meta-parameters
            for key in self.meta_params:
                self.meta_params[key] -= self.outer_lr * meta_grads[key]
            
            meta_losses.append(total_loss / batch_size)
        
        return meta_losses
    
    def adapt(self, support_x: np.ndarray, support_y: np.ndarray) -> 'SimpleForecaster':
        """
        Adapt to a new task with few examples.
        
        This is what makes MAML powerful: fast adaptation.
        """
        self.model.set_params(self.meta_params)
        
        for _ in range(self.inner_steps):
            grads = self.model.compute_gradients(support_x, support_y)
            params = self.model.get_params()
            
            for key in params:
                params[key] = params[key] - self.inner_lr * grads[key]
            
            self.model.set_params(params)
        
        return self.model
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predict using adapted model."""
        return self.model.forward(x)


def create_sku_tasks(data_by_sku: Dict[str, Tuple[np.ndarray, np.ndarray]], 
                     k_shot: int = 5) -> List[Task]:
    """
    Create meta-learning tasks from SKU data.
    
    Each SKU becomes a task with k-shot support set.
    """
    tasks = []
    
    for sku_id, (X, y) in data_by_sku.items():
        if len(X) < k_shot + 5:  # Need enough data for support + query
            continue
        
        # Split into support and query
        indices = np.random.permutation(len(X))
        support_idx = indices[:k_shot]
        query_idx = indices[k_shot:]
        
        tasks.append(Task(
            sku_id=sku_id,
            support_x=X[support_idx],
            support_y=y[support_idx],
            query_x=X[query_idx],
            query_y=y[query_idx]
        ))
    
    return tasks


def few_shot_forecast(
    maml: MAML,
    new_sku_data: Tuple[np.ndarray, np.ndarray],
    forecast_horizon: int = 7
) -> Dict:
    """
    Forecast for a new SKU with only a few data points.
    """
    X, y = new_sku_data
    
    # Adapt to new SKU
    adapted_model = maml.adapt(X, y)
    
    # Generate forecast features (simple: just use recent pattern)
    last_features = X[-1].reshape(1, -1)
    predictions = []
    
    for h in range(forecast_horizon):
        pred = adapted_model.forward(last_features)
        predictions.append(float(pred[0, 0]))
        # Shift features (simplified)
        last_features = np.roll(last_features, -1)
        last_features[0, -1] = pred[0, 0]
    
    return {
        'predictions': predictions,
        'n_shots': len(X),
        'adaptation_steps': maml.inner_steps
    }
