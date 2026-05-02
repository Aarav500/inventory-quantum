"""State and action definitions for inventory RL."""

import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class InventoryState:
    """
    State representation for inventory RL.
    
    Captures all relevant information for making ordering decisions.
    """
    inventory_level: float
    demand_forecast: List[float]
    forecast_uncertainty: List[float]
    lead_time: int
    days_since_order: int
    pending_orders: float = 0.0
    day_of_week: int = 0
    
    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for neural network."""
        features = [
            self.inventory_level,
            np.mean(self.demand_forecast),
            np.std(self.demand_forecast),
            np.mean(self.forecast_uncertainty),
            self.lead_time,
            self.days_since_order,
            self.pending_orders,
            self.day_of_week / 7.0,  # Normalize
        ]
        # Add forecast values
        features.extend(self.demand_forecast[:7])  # First 7 days
        features.extend(self.forecast_uncertainty[:7])
        
        return np.array(features, dtype=np.float32)
    
    @property
    def dim(self) -> int:
        """State dimension."""
        return 8 + 14  # Base features + forecast features


@dataclass
class InventoryAction:
    """Action for inventory control."""
    order_quantity: int
    
    @classmethod
    def from_discrete(cls, action_idx: int, max_quantity: int = 256, n_actions: int = 33) -> 'InventoryAction':
        """Convert discrete action index to order quantity."""
        quantity = int(action_idx * max_quantity / (n_actions - 1))
        return cls(order_quantity=quantity)
    
    def to_discrete(self, max_quantity: int = 256, n_actions: int = 33) -> int:
        """Convert order quantity to discrete action index."""
        return int(self.order_quantity * (n_actions - 1) / max_quantity)


@dataclass
class InventoryTransition:
    """Single transition for offline RL."""
    state: InventoryState
    action: InventoryAction
    reward: float
    next_state: InventoryState
    done: bool


class InventoryEnvironment:
    """
    Simulated inventory environment for generating transitions.
    """
    
    def __init__(
        self,
        holding_cost: float = 0.1,
        ordering_cost: float = 50.0,
        stockout_cost: float = 10.0,
        lead_time: int = 7,
        max_inventory: float = 1000.0,
    ):
        self.holding_cost = holding_cost
        self.ordering_cost = ordering_cost
        self.stockout_cost = stockout_cost
        self.lead_time = lead_time
        self.max_inventory = max_inventory
        
        self._inventory = 0.0
        self._pending_orders: List[Tuple[int, float]] = []
        self._day = 0
    
    def reset(self, initial_inventory: float = None) -> InventoryState:
        """Reset environment."""
        self._inventory = initial_inventory if initial_inventory else np.random.uniform(50, 200)
        self._pending_orders = []
        self._day = 0
        return self._get_state()
    
    def _get_state(self) -> InventoryState:
        """Get current state."""
        # Generate simple forecast (would use actual forecaster in practice)
        mean_demand = 50
        forecast = [mean_demand * (1 + 0.1 * np.random.randn()) for _ in range(7)]
        uncertainty = [mean_demand * 0.2 for _ in range(7)]
        
        pending = sum(qty for _, qty in self._pending_orders)
        days_since = min(self._day, 30)
        
        return InventoryState(
            inventory_level=self._inventory,
            demand_forecast=forecast,
            forecast_uncertainty=uncertainty,
            lead_time=self.lead_time,
            days_since_order=days_since,
            pending_orders=pending,
            day_of_week=self._day % 7,
        )
    
    def step(
        self,
        action: InventoryAction,
        demand: float = None
    ) -> Tuple[InventoryState, float, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, done, info.
        """
        # Receive pending orders
        received = 0
        new_pending = []
        for arrival_day, qty in self._pending_orders:
            if arrival_day <= self._day:
                received += qty
            else:
                new_pending.append((arrival_day, qty))
        self._pending_orders = new_pending
        self._inventory += received
        
        # Place new order
        if action.order_quantity > 0:
            arrival = self._day + self.lead_time
            self._pending_orders.append((arrival, action.order_quantity))
        
        # Generate demand if not provided
        if demand is None:
            demand = max(0, 50 + 15 * np.random.randn())
        
        # Calculate reward
        reward = 0
        
        # Ordering cost
        if action.order_quantity > 0:
            reward -= self.ordering_cost
        
        # Holding cost
        reward -= max(0, self._inventory) * self.holding_cost
        
        # Stockout cost
        if demand > self._inventory:
            shortage = demand - self._inventory
            reward -= shortage * self.stockout_cost
        
        # Update inventory
        self._inventory = max(0, self._inventory - demand)
        self._inventory = min(self._inventory, self.max_inventory)
        
        self._day += 1
        done = self._day >= 365  # Episode ends after a year
        
        info = {
            'demand': demand,
            'inventory': self._inventory,
            'stockout': max(0, demand - self._inventory),
        }
        
        return self._get_state(), reward, done, info


def generate_offline_data(
    data: pd.DataFrame,
    n_episodes: int = 100,
    policy: str = 'behavioral'
) -> List[InventoryTransition]:
    """
    Generate offline RL dataset from historical data.
    
    Args:
        data: Historical inventory data
        n_episodes: Number of simulated episodes
        policy: 'behavioral' for random, 'expert' for heuristic
    
    Returns:
        List of transitions
    """
    env = InventoryEnvironment()
    transitions = []
    
    # Calculate demand statistics from data
    df = data.copy()
    df['date'] = pd.to_datetime(df['date'])
    daily_demand = df.groupby('date')['quantity_sold'].sum()
    mean_demand = daily_demand.mean()
    std_demand = daily_demand.std()
    
    for _ in range(n_episodes):
        state = env.reset()
        done = False
        
        while not done:
            # Behavioral policy (mix of random and heuristic)
            if policy == 'behavioral':
                if np.random.random() < 0.3:
                    # Random action
                    order_qty = np.random.randint(0, 200)
                else:
                    # Simple heuristic
                    target = mean_demand * env.lead_time * 1.5
                    order_qty = max(0, int(target - state.inventory_level))
            else:
                # Expert policy (EOQ-like)
                target = mean_demand * env.lead_time + 2 * std_demand * np.sqrt(env.lead_time)
                order_qty = max(0, int(target - state.inventory_level))
            
            action = InventoryAction(order_quantity=order_qty)
            
            # Sample demand from distribution
            demand = max(0, mean_demand + std_demand * np.random.randn())
            
            next_state, reward, done, _ = env.step(action, demand)
            
            transitions.append(InventoryTransition(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            ))
            
            state = next_state
    
    return transitions
