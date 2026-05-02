"""Simulated annealing solver for QUBO problems."""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class SAResult:
    """Result from simulated annealing."""
    solution: np.ndarray
    energy: float
    iterations: int
    temperature_final: float


class SimulatedAnnealingSolver:
    """
    Simulated annealing solver for QUBO optimization.
    
    Implements Metropolis-Hastings algorithm with geometric cooling schedule.
    """
    
    def __init__(
        self,
        initial_temp: float = None,
        cooling_rate: float = None,
        max_iterations: int = None,
        restart_threshold: int = 1000,
    ):
        """
        Initialize solver.
        
        Args:
            initial_temp: Starting temperature
            cooling_rate: Temperature decay factor per iteration
            max_iterations: Maximum iterations
            restart_threshold: Restart if no improvement for this many iterations
        """
        settings = get_settings()
        self.initial_temp = initial_temp or settings.sa_initial_temp
        self.cooling_rate = cooling_rate or settings.sa_cooling_rate
        self.max_iterations = max_iterations or settings.sa_iterations
        self.restart_threshold = restart_threshold
        self._history = []
    
    def _qubo_energy(self, x: np.ndarray, Q: np.ndarray) -> float:
        """Calculate QUBO energy: E = x^T Q x."""
        return float(x.T @ Q @ x)
    
    def _neighbor(self, x: np.ndarray) -> np.ndarray:
        """Generate neighbor by flipping one bit."""
        neighbor = x.copy()
        flip_idx = np.random.randint(len(x))
        neighbor[flip_idx] = 1 - neighbor[flip_idx]
        return neighbor
    
    def _acceptance_probability(
        self,
        current_energy: float,
        new_energy: float,
        temperature: float
    ) -> float:
        """Calculate Metropolis acceptance probability."""
        if new_energy < current_energy:
            return 1.0
        if temperature < 1e-10:
            return 0.0
        return np.exp(-(new_energy - current_energy) / temperature)
    
    def solve(
        self,
        Q: np.ndarray,
        initial_solution: np.ndarray = None
    ) -> np.ndarray:
        """
        Solve QUBO problem using simulated annealing.
        
        Args:
            Q: QUBO matrix (n x n)
            initial_solution: Optional starting point
        
        Returns:
            Best binary solution found
        """
        n = Q.shape[0]
        
        # Initialize
        if initial_solution is None:
            current = np.random.randint(0, 2, n)
        else:
            current = initial_solution.copy()
        
        current_energy = self._qubo_energy(current, Q)
        
        best = current.copy()
        best_energy = current_energy
        
        temperature = self.initial_temp
        iterations_without_improvement = 0
        self._history = [(0, current_energy, best_energy)]
        
        for iteration in range(self.max_iterations):
            # Generate neighbor
            neighbor = self._neighbor(current)
            neighbor_energy = self._qubo_energy(neighbor, Q)
            
            # Accept or reject
            if np.random.random() < self._acceptance_probability(
                current_energy, neighbor_energy, temperature
            ):
                current = neighbor
                current_energy = neighbor_energy
                
                # Update best
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
                    iterations_without_improvement = 0
                else:
                    iterations_without_improvement += 1
            else:
                iterations_without_improvement += 1
            
            # Cool down
            temperature *= self.cooling_rate
            
            # Record history (every 100 iterations)
            if iteration % 100 == 0:
                self._history.append((iteration, current_energy, best_energy))
            
            # Restart if stuck
            if iterations_without_improvement >= self.restart_threshold:
                current = np.random.randint(0, 2, n)
                current_energy = self._qubo_energy(current, Q)
                temperature = self.initial_temp * 0.5  # Restart with lower temp
                iterations_without_improvement = 0
        
        return best
    
    def solve_with_details(
        self,
        Q: np.ndarray,
        initial_solution: np.ndarray = None
    ) -> SAResult:
        """
        Solve QUBO and return detailed result.
        
        Returns:
            SAResult with solution and metadata
        """
        solution = self.solve(Q, initial_solution)
        energy = self._qubo_energy(solution, Q)
        
        return SAResult(
            solution=solution,
            energy=energy,
            iterations=self.max_iterations,
            temperature_final=self.initial_temp * (self.cooling_rate ** self.max_iterations)
        )
    
    def get_history(self) -> list:
        """Get optimization history."""
        return self._history


class TabuSearchSolver:
    """
    Tabu search solver for QUBO as comparison.
    """
    
    def __init__(
        self,
        tabu_size: int = 20,
        max_iterations: int = 5000,
    ):
        self.tabu_size = tabu_size
        self.max_iterations = max_iterations
    
    def _qubo_energy(self, x: np.ndarray, Q: np.ndarray) -> float:
        return float(x.T @ Q @ x)
    
    def solve(self, Q: np.ndarray) -> np.ndarray:
        """Solve using tabu search."""
        n = Q.shape[0]
        current = np.random.randint(0, 2, n)
        current_energy = self._qubo_energy(current, Q)
        
        best = current.copy()
        best_energy = current_energy
        
        tabu_list = []
        
        for _ in range(self.max_iterations):
            # Find best non-tabu neighbor
            best_neighbor = None
            best_neighbor_energy = float('inf')
            best_flip = -1
            
            for i in range(n):
                if i in tabu_list:
                    # Allow if improves global best
                    neighbor = current.copy()
                    neighbor[i] = 1 - neighbor[i]
                    energy = self._qubo_energy(neighbor, Q)
                    if energy < best_energy and energy < best_neighbor_energy:
                        best_neighbor = neighbor
                        best_neighbor_energy = energy
                        best_flip = i
                else:
                    neighbor = current.copy()
                    neighbor[i] = 1 - neighbor[i]
                    energy = self._qubo_energy(neighbor, Q)
                    if energy < best_neighbor_energy:
                        best_neighbor = neighbor
                        best_neighbor_energy = energy
                        best_flip = i
            
            if best_neighbor is not None:
                current = best_neighbor
                current_energy = best_neighbor_energy
                
                # Update tabu list
                tabu_list.append(best_flip)
                if len(tabu_list) > self.tabu_size:
                    tabu_list.pop(0)
                
                if current_energy < best_energy:
                    best = current.copy()
                    best_energy = current_energy
        
        return best
