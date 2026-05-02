"""Conservative Q-Learning (CQL) for offline RL."""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from collections import deque
import random

from app.models.inventory import ForecastPoint, DecisionResult
from app.rl.state_action import (
    InventoryState, InventoryAction, InventoryTransition,
    generate_offline_data
)
from app.config import get_settings


class ReplayBuffer:
    """Experience replay buffer for offline RL."""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, transition: InventoryTransition):
        self.buffer.append(transition)
    
    def add_batch(self, transitions: List[InventoryTransition]):
        for t in transitions:
            self.add(t)
    
    def sample(self, batch_size: int) -> List[InventoryTransition]:
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))
    
    def __len__(self):
        return len(self.buffer)


class CQLPolicy:
    """
    Conservative Q-Learning policy for inventory control.
    
    CQL adds a regularization term that penalizes Q-values for
    out-of-distribution actions, preventing overestimation on
    actions not seen in the offline dataset.
    """
    
    def __init__(
        self,
        holding_cost: float = 0.1,
        ordering_cost: float = 50.0,
        stockout_cost: float = 10.0,
        service_level: float = 0.95,
        lead_time: int = 7,
        state_dim: int = 22,
        n_actions: int = 33,
        hidden_dim: int = 128,
        gamma: float = None,
        cql_alpha: float = None,
        learning_rate: float = 1e-3,
    ):
        """
        Initialize CQL policy.
        
        Args:
            state_dim: Dimension of state vector
            n_actions: Number of discrete actions
            hidden_dim: Hidden layer size
            gamma: Discount factor
            cql_alpha: CQL regularization weight
        """
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.service_level = service_level
        self.lead_time = lead_time
        
        settings = get_settings()
        self.gamma = gamma or settings.rl_gamma
        self.cql_alpha = cql_alpha or settings.rl_cql_alpha
        
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        self._q_network = None
        self._target_network = None
        self._optimizer = None
        self._trained = False
        self._replay_buffer = ReplayBuffer()
    
    def _build_network(self):
        """Build Q-network using PyTorch."""
        try:
            import torch
            import torch.nn as nn
            
            class QNetwork(nn.Module):
                def __init__(self, state_dim, n_actions, hidden_dim):
                    super().__init__()
                    self.fc1 = nn.Linear(state_dim, hidden_dim)
                    self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                    self.fc3 = nn.Linear(hidden_dim, n_actions)
                    self.relu = nn.ReLU()
                
                def forward(self, x):
                    x = self.relu(self.fc1(x))
                    x = self.relu(self.fc2(x))
                    return self.fc3(x)
            
            self._q_network = QNetwork(self.state_dim, self.n_actions, self.hidden_dim)
            self._target_network = QNetwork(self.state_dim, self.n_actions, self.hidden_dim)
            self._target_network.load_state_dict(self._q_network.state_dict())
            
            self._optimizer = torch.optim.Adam(
                self._q_network.parameters(), 
                lr=self.learning_rate
            )
            
            return True
        except ImportError:
            return False
    
    def _cql_loss(
        self,
        states: 'torch.Tensor',
        actions: 'torch.Tensor',
        rewards: 'torch.Tensor',
        next_states: 'torch.Tensor',
        dones: 'torch.Tensor',
    ) -> 'torch.Tensor':
        """
        Calculate CQL loss.
        
        CQL loss = Standard TD loss + alpha * (logsumexp(Q) - Q(s,a))
        
        The additional term penalizes high Q-values for unseen actions.
        """
        import torch
        import torch.nn.functional as F
        
        # Current Q-values
        current_q = self._q_network(states)
        current_q_actions = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Target Q-values
        with torch.no_grad():
            next_q = self._target_network(next_states)
            max_next_q = next_q.max(dim=1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Standard TD loss
        td_loss = F.mse_loss(current_q_actions, target_q)
        
        # CQL regularization: penalize Q-values for all actions
        # relative to the Q-value of the taken action
        logsumexp_q = torch.logsumexp(current_q, dim=1)
        cql_loss = (logsumexp_q - current_q_actions).mean()
        
        total_loss = td_loss + self.cql_alpha * cql_loss
        
        return total_loss, td_loss.item(), cql_loss.item()
    
    def train(
        self,
        data: pd.DataFrame,
        n_episodes: int = 100,
        n_iterations: int = 10000,
        batch_size: int = None,
        target_update_freq: int = 100,
    ) -> Dict[str, Any]:
        """
        Train CQL policy on offline data.
        
        Args:
            data: Historical inventory data
            n_episodes: Number of episodes to generate
            n_iterations: Training iterations
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
        
        Returns:
            Training metrics
        """
        import torch
        
        settings = get_settings()
        batch_size = batch_size or settings.rl_batch_size
        
        # Build networks
        if not self._build_network():
            return {'error': 'PyTorch not available'}
        
        # Generate offline data
        transitions = generate_offline_data(data, n_episodes, policy='behavioral')
        self._replay_buffer.add_batch(transitions)
        
        if len(self._replay_buffer) < batch_size:
            return {'error': 'Insufficient data for training'}
        
        # Training loop
        losses = []
        td_losses = []
        cql_losses = []
        
        for iteration in range(n_iterations):
            # Sample batch
            batch = self._replay_buffer.sample(batch_size)
            
            # Convert to tensors
            states = torch.FloatTensor([t.state.to_vector() for t in batch])
            actions = torch.LongTensor([t.action.to_discrete() for t in batch])
            rewards = torch.FloatTensor([t.reward for t in batch])
            next_states = torch.FloatTensor([t.next_state.to_vector() for t in batch])
            dones = torch.FloatTensor([float(t.done) for t in batch])
            
            # Calculate loss and update
            self._optimizer.zero_grad()
            loss, td_loss, cql_loss = self._cql_loss(
                states, actions, rewards, next_states, dones
            )
            loss.backward()
            self._optimizer.step()
            
            losses.append(loss.item())
            td_losses.append(td_loss)
            cql_losses.append(cql_loss)
            
            # Update target network
            if iteration % target_update_freq == 0:
                self._target_network.load_state_dict(self._q_network.state_dict())
        
        self._trained = True
        
        return {
            'n_iterations': n_iterations,
            'final_loss': np.mean(losses[-100:]),
            'final_td_loss': np.mean(td_losses[-100:]),
            'final_cql_loss': np.mean(cql_losses[-100:]),
            'buffer_size': len(self._replay_buffer),
        }
    
    def get_action(self, state: InventoryState) -> InventoryAction:
        """Get action from trained policy."""
        import torch
        
        if not self._trained:
            # Fallback to simple heuristic
            target = np.mean(state.demand_forecast) * state.lead_time * 1.5
            order_qty = max(0, int(target - state.inventory_level))
            return InventoryAction(order_quantity=order_qty)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state.to_vector()).unsqueeze(0)
            q_values = self._q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
        return InventoryAction.from_discrete(action_idx)
    
    def optimize(
        self,
        data: pd.DataFrame,
        forecast: List[ForecastPoint] = None
    ) -> DecisionResult:
        """
        Get optimal decision using trained RL policy.
        
        Args:
            data: Historical/current data
            forecast: Demand forecast
        
        Returns:
            DecisionResult from RL policy
        """
        sku = data['sku'].iloc[0]
        
        # Train if not trained
        if not self._trained:
            self.train(data, n_episodes=50, n_iterations=5000)
        
        # Create current state
        df = data.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Get forecast
        if forecast is None:
            mean_demand = df.groupby('date')['quantity_sold'].sum().mean()
            forecast_values = [mean_demand] * 7
            uncertainty = [mean_demand * 0.3] * 7
        else:
            forecast_values = [f.predicted for f in forecast[:7]]
            uncertainty = [
                (f.upper_bound - f.lower_bound) / 4 if f.upper_bound else f.predicted * 0.3
                for f in forecast[:7]
            ]
        
        # Current inventory
        current_inventory = 0
        if 'quantity_on_hand' in df.columns:
            current_inventory = df['quantity_on_hand'].iloc[-1] or 0
        
        state = InventoryState(
            inventory_level=current_inventory,
            demand_forecast=forecast_values,
            forecast_uncertainty=uncertainty,
            lead_time=self.lead_time,
            days_since_order=0,
            pending_orders=0,
            day_of_week=0,
        )
        
        # Get action
        action = self.get_action(state)
        
        # Estimate costs (simplified)
        mean_demand = np.mean(forecast_values)
        std_demand = np.mean(uncertainty)
        
        expected_inventory = current_inventory + action.order_quantity - mean_demand * self.lead_time
        holding = max(0, expected_inventory) * self.holding_cost * self.lead_time / 2
        ordering = self.ordering_cost if action.order_quantity > 0 else 0
        
        # Service level estimation
        z = (current_inventory + action.order_quantity - mean_demand * self.lead_time) / (
            std_demand * np.sqrt(self.lead_time) + 1e-6
        )
        service_level = min(1, max(0, 0.5 + 0.5 * np.tanh(z / 2)))
        
        total_cost = holding + ordering
        
        # Reorder point
        reorder_point = mean_demand * self.lead_time + 1.645 * std_demand * np.sqrt(self.lead_time)
        
        return DecisionResult(
            sku=sku,
            policy='rl_cql',
            reorder_point=round(reorder_point, 2),
            reorder_quantity=round(action.order_quantity, 2),
            expected_cost=round(total_cost, 2),
            expected_service_level=round(service_level, 4),
            uncertainty_quantile=None,
        )
