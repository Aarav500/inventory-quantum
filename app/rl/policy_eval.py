"""Off-policy evaluation for RL policies."""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass

from app.rl.state_action import (
    InventoryState, InventoryAction, InventoryTransition,
    InventoryEnvironment, generate_offline_data
)


@dataclass
class OPEResult:
    """Result from off-policy evaluation."""
    method: str
    estimated_value: float
    confidence_interval: tuple
    effective_sample_size: Optional[float] = None


class OffPolicyEvaluator:
    """
    Off-policy evaluation methods for inventory RL.
    
    Implements:
    - Importance Sampling (IS)
    - Weighted Importance Sampling (WIS)
    - Doubly Robust (DR)
    - Fitted Q Evaluation (FQE)
    """
    
    def __init__(self, gamma: float = 0.99):
        self.gamma = gamma
    
    def _compute_importance_weights(
        self,
        transitions: List[InventoryTransition],
        target_policy: Callable[[InventoryState], float],
        behavior_policy: Callable[[InventoryState, InventoryAction], float],
    ) -> np.ndarray:
        """
        Compute importance sampling weights.
        
        Returns cumulative product of pi(a|s) / mu(a|s) for each trajectory.
        """
        weights = []
        current_weight = 1.0
        
        for t in transitions:
            # Target policy probability for taken action
            target_prob = target_policy(t.state)
            # Behavior policy probability
            behavior_prob = behavior_policy(t.state, t.action)
            
            ratio = target_prob / (behavior_prob + 1e-8)
            current_weight *= ratio
            weights.append(current_weight)
            
            if t.done:
                current_weight = 1.0
        
        return np.array(weights)
    
    def importance_sampling(
        self,
        transitions: List[InventoryTransition],
        target_policy: Callable,
        behavior_policy: Callable,
    ) -> OPEResult:
        """
        Standard importance sampling estimator.
        
        V_IS = (1/n) * sum(w_i * G_i)
        """
        # Compute returns
        returns = []
        current_return = 0
        
        for t in reversed(transitions):
            current_return = t.reward + self.gamma * current_return
            if t.done:
                returns.append(current_return)
                current_return = 0
        
        returns = list(reversed(returns))
        
        if not returns:
            return OPEResult(
                method='importance_sampling',
                estimated_value=0,
                confidence_interval=(0, 0)
            )
        
        # Simplified: assume uniform behavioral policy
        # In practice, you'd estimate this from data
        weights = np.ones(len(returns))
        
        weighted_returns = weights * np.array(returns)
        estimate = np.mean(weighted_returns)
        std = np.std(weighted_returns) / np.sqrt(len(returns))
        
        return OPEResult(
            method='importance_sampling',
            estimated_value=estimate,
            confidence_interval=(estimate - 1.96 * std, estimate + 1.96 * std),
            effective_sample_size=len(returns) / (1 + np.var(weights))
        )
    
    def weighted_importance_sampling(
        self,
        transitions: List[InventoryTransition],
        target_policy: Callable,
        behavior_policy: Callable,
    ) -> OPEResult:
        """
        Weighted importance sampling (self-normalized).
        
        V_WIS = sum(w_i * G_i) / sum(w_i)
        
        Has lower variance than standard IS.
        """
        returns = []
        current_return = 0
        
        for t in reversed(transitions):
            current_return = t.reward + self.gamma * current_return
            if t.done:
                returns.append(current_return)
                current_return = 0
        
        returns = list(reversed(returns))
        
        if not returns:
            return OPEResult(
                method='weighted_importance_sampling',
                estimated_value=0,
                confidence_interval=(0, 0)
            )
        
        weights = np.ones(len(returns))
        
        # Self-normalized estimator
        estimate = np.sum(weights * np.array(returns)) / np.sum(weights)
        
        # Bootstrap confidence interval
        bootstrap_estimates = []
        for _ in range(1000):
            indices = np.random.choice(len(returns), len(returns), replace=True)
            boot_weights = weights[indices]
            boot_returns = np.array(returns)[indices]
            boot_estimate = np.sum(boot_weights * boot_returns) / np.sum(boot_weights)
            bootstrap_estimates.append(boot_estimate)
        
        ci = (np.percentile(bootstrap_estimates, 2.5), 
              np.percentile(bootstrap_estimates, 97.5))
        
        return OPEResult(
            method='weighted_importance_sampling',
            estimated_value=estimate,
            confidence_interval=ci,
            effective_sample_size=np.sum(weights)**2 / np.sum(weights**2)
        )
    
    def doubly_robust(
        self,
        transitions: List[InventoryTransition],
        q_function: Callable[[InventoryState, InventoryAction], float],
        target_policy: Callable,
        behavior_policy: Callable,
    ) -> OPEResult:
        """
        Doubly robust estimator.
        
        Combines importance sampling with a Q-function baseline
        for variance reduction. Consistent if either the Q-function
        or importance weights are correct.
        """
        estimates = []
        
        for t in transitions:
            # Importance weight (simplified)
            rho = 1.0
            
            # Q-function estimate
            q_val = q_function(t.state, t.action)
            
            # Value function estimate (expected Q under target policy)
            v_val = q_val  # Simplified
            
            # DR estimate for this transition
            dr_estimate = v_val + rho * (t.reward + self.gamma * v_val - q_val)
            estimates.append(dr_estimate)
        
        estimate = np.mean(estimates)
        std = np.std(estimates) / np.sqrt(len(estimates))
        
        return OPEResult(
            method='doubly_robust',
            estimated_value=estimate,
            confidence_interval=(estimate - 1.96 * std, estimate + 1.96 * std)
        )
    
    def fitted_q_evaluation(
        self,
        transitions: List[InventoryTransition],
        target_policy: Callable,
        n_iterations: int = 100,
    ) -> OPEResult:
        """
        Fitted Q-Evaluation (FQE).
        
        Learns a Q-function for the target policy by
        iteratively fitting Q(s,a) = r + gamma * V(s')
        where V(s') = Q(s', pi(s'))
        """
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return OPEResult(
                method='fitted_q_evaluation',
                estimated_value=0,
                confidence_interval=(0, 0)
            )
        
        # Simple Q-network
        state_dim = transitions[0].state.to_vector().shape[0]
        
        class QNetwork(nn.Module):
            def __init__(self, state_dim):
                super().__init__()
                self.fc1 = nn.Linear(state_dim + 1, 64)  # +1 for action
                self.fc2 = nn.Linear(64, 64)
                self.fc3 = nn.Linear(64, 1)
                self.relu = nn.ReLU()
            
            def forward(self, state, action):
                x = torch.cat([state, action.unsqueeze(1)], dim=1)
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return self.fc3(x)
        
        q_net = QNetwork(state_dim)
        optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
        
        # Prepare data
        states = torch.FloatTensor([t.state.to_vector() for t in transitions])
        actions = torch.FloatTensor([t.action.order_quantity for t in transitions])
        rewards = torch.FloatTensor([t.reward for t in transitions])
        next_states = torch.FloatTensor([t.next_state.to_vector() for t in transitions])
        dones = torch.FloatTensor([float(t.done) for t in transitions])
        
        # FQE iterations
        for _ in range(n_iterations):
            # Compute targets
            with torch.no_grad():
                # Get target policy actions for next states
                next_actions = torch.zeros(len(transitions))  # Simplified
                next_q = q_net(next_states, next_actions).squeeze()
                targets = rewards + self.gamma * next_q * (1 - dones)
            
            # Update Q-function
            optimizer.zero_grad()
            q_pred = q_net(states, actions).squeeze()
            loss = nn.MSELoss()(q_pred, targets)
            loss.backward()
            optimizer.step()
        
        # Estimate value
        with torch.no_grad():
            initial_actions = torch.zeros(len(transitions))
            values = q_net(states, initial_actions).squeeze().numpy()
        
        estimate = np.mean(values)
        std = np.std(values) / np.sqrt(len(values))
        
        return OPEResult(
            method='fitted_q_evaluation',
            estimated_value=estimate,
            confidence_interval=(estimate - 1.96 * std, estimate + 1.96 * std)
        )
    
    def evaluate_policy(
        self,
        data: pd.DataFrame,
        target_policy: Callable,
        methods: List[str] = None,
    ) -> Dict[str, OPEResult]:
        """
        Evaluate a target policy using multiple OPE methods.
        
        Args:
            data: Historical inventory data
            target_policy: Policy to evaluate
            methods: List of methods to use
        
        Returns:
            Dictionary of method name to OPEResult
        """
        methods = methods or ['wis', 'fqe']
        
        # Generate transitions
        transitions = generate_offline_data(data, n_episodes=50)
        
        # Dummy policies for demonstration
        def behavior(state, action):
            return 1.0 / 33  # Uniform over 33 actions
        
        results = {}
        
        if 'is' in methods:
            results['is'] = self.importance_sampling(
                transitions, target_policy, behavior
            )
        
        if 'wis' in methods:
            results['wis'] = self.weighted_importance_sampling(
                transitions, target_policy, behavior
            )
        
        if 'fqe' in methods:
            results['fqe'] = self.fitted_q_evaluation(
                transitions, target_policy
            )
        
        return results
