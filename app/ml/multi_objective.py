"""
Multi-Objective Optimization with Pareto Frontier.

Optimizes multiple conflicting objectives simultaneously:
- Minimize cost
- Maximize service level
- Minimize inventory holding

Uses NSGA-II algorithm for finding Pareto-optimal solutions.
"""

import numpy as np
from typing import List, Tuple, Dict, Callable
from dataclasses import dataclass


@dataclass
class ParetoSolution:
    """A solution on the Pareto frontier."""
    x: np.ndarray
    objectives: np.ndarray
    rank: int
    crowding_distance: float


class NSGA2:
    """
    NSGA-II: Non-dominated Sorting Genetic Algorithm II.
    
    Reference: Deb et al. (2002)
    """
    
    def __init__(
        self,
        n_objectives: int,
        n_variables: int,
        bounds: List[Tuple[float, float]],
        population_size: int = 100,
        n_generations: int = 50,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
    ):
        self.n_objectives = n_objectives
        self.n_variables = n_variables
        self.bounds = bounds
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
    
    def _dominates(self, obj1: np.ndarray, obj2: np.ndarray) -> bool:
        """Check if obj1 dominates obj2 (for minimization)."""
        return np.all(obj1 <= obj2) and np.any(obj1 < obj2)
    
    def _fast_non_dominated_sort(self, objectives: np.ndarray) -> List[List[int]]:
        """Fast non-dominated sorting."""
        n = len(objectives)
        domination_count = np.zeros(n, dtype=int)
        dominated_solutions = [[] for _ in range(n)]
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    if self._dominates(objectives[i], objectives[j]):
                        dominated_solutions[i].append(j)
                    elif self._dominates(objectives[j], objectives[i]):
                        domination_count[i] += 1
            
            if domination_count[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in dominated_solutions[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        return fronts[:-1]  # Remove empty last front
    
    def _crowding_distance(self, objectives: np.ndarray, front: List[int]) -> np.ndarray:
        """Calculate crowding distance for diversity preservation."""
        n = len(front)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        
        for m in range(self.n_objectives):
            sorted_idx = np.argsort(objectives[front, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            obj_range = objectives[front[sorted_idx[-1]], m] - objectives[front[sorted_idx[0]], m]
            if obj_range > 0:
                for i in range(1, n - 1):
                    distances[sorted_idx[i]] += (
                        objectives[front[sorted_idx[i + 1]], m] - 
                        objectives[front[sorted_idx[i - 1]], m]
                    ) / obj_range
        
        return distances
    
    def _tournament_selection(self, population: np.ndarray, ranks: np.ndarray, crowding: np.ndarray) -> int:
        """Binary tournament selection based on rank and crowding distance."""
        i, j = np.random.choice(len(population), 2, replace=False)
        
        if ranks[i] < ranks[j]:
            return i
        elif ranks[j] < ranks[i]:
            return j
        else:
            return i if crowding[i] > crowding[j] else j
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulated Binary Crossover (SBX)."""
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        
        eta = 20
        child1, child2 = parent1.copy(), parent2.copy()
        
        for i in range(self.n_variables):
            if np.random.rand() < 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-10:
                    if parent1[i] < parent2[i]:
                        y1, y2 = parent1[i], parent2[i]
                    else:
                        y1, y2 = parent2[i], parent1[i]
                    
                    beta = 1.0 + (2.0 * (y1 - self.bounds[i][0]) / (y2 - y1))
                    alpha = 2.0 - beta ** (-(eta + 1))
                    
                    rand = np.random.rand()
                    if rand <= 1.0 / alpha:
                        betaq = (rand * alpha) ** (1.0 / (eta + 1))
                    else:
                        betaq = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (eta + 1))
                    
                    child1[i] = 0.5 * ((y1 + y2) - betaq * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + betaq * (y2 - y1))
                    
                    child1[i] = np.clip(child1[i], self.bounds[i][0], self.bounds[i][1])
                    child2[i] = np.clip(child2[i], self.bounds[i][0], self.bounds[i][1])
        
        return child1, child2
    
    def _mutate(self, x: np.ndarray) -> np.ndarray:
        """Polynomial mutation."""
        eta = 20
        mutant = x.copy()
        
        for i in range(self.n_variables):
            if np.random.rand() < self.mutation_prob:
                delta = self.bounds[i][1] - self.bounds[i][0]
                
                rand = np.random.rand()
                if rand < 0.5:
                    deltaq = (2 * rand) ** (1 / (eta + 1)) - 1
                else:
                    deltaq = 1 - (2 * (1 - rand)) ** (1 / (eta + 1))
                
                mutant[i] = x[i] + deltaq * delta
                mutant[i] = np.clip(mutant[i], self.bounds[i][0], self.bounds[i][1])
        
        return mutant
    
    def optimize(self, objective_functions: List[Callable]) -> List[ParetoSolution]:
        """
        Run NSGA-II optimization.
        
        Args:
            objective_functions: List of functions to minimize
        
        Returns:
            Pareto-optimal solutions
        """
        # Initialize population
        population = np.random.rand(self.population_size, self.n_variables)
        for i in range(self.n_variables):
            population[:, i] = self.bounds[i][0] + population[:, i] * (self.bounds[i][1] - self.bounds[i][0])
        
        # Evaluate objectives
        objectives = np.zeros((self.population_size, self.n_objectives))
        for i, x in enumerate(population):
            for j, f in enumerate(objective_functions):
                objectives[i, j] = f(x)
        
        for gen in range(self.n_generations):
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(objectives)
            
            # Assign ranks and crowding distances
            ranks = np.zeros(len(population), dtype=int)
            crowding = np.zeros(len(population))
            
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                cd = self._crowding_distance(objectives, front)
                for i, idx in enumerate(front):
                    crowding[idx] = cd[i]
            
            # Create offspring
            offspring = []
            offspring_obj = []
            
            while len(offspring) < self.population_size:
                p1 = self._tournament_selection(population, ranks, crowding)
                p2 = self._tournament_selection(population, ranks, crowding)
                
                c1, c2 = self._crossover(population[p1], population[p2])
                c1 = self._mutate(c1)
                c2 = self._mutate(c2)
                
                offspring.append(c1)
                offspring.append(c2)
                
                obj1 = [f(c1) for f in objective_functions]
                obj2 = [f(c2) for f in objective_functions]
                offspring_obj.append(obj1)
                offspring_obj.append(obj2)
            
            # Combine parent and offspring
            combined_pop = np.vstack([population, np.array(offspring[:self.population_size])])
            combined_obj = np.vstack([objectives, np.array(offspring_obj[:self.population_size])])
            
            # Select next generation
            fronts = self._fast_non_dominated_sort(combined_obj)
            
            new_pop = []
            new_obj = []
            
            for front in fronts:
                if len(new_pop) + len(front) <= self.population_size:
                    for idx in front:
                        new_pop.append(combined_pop[idx])
                        new_obj.append(combined_obj[idx])
                else:
                    cd = self._crowding_distance(combined_obj, front)
                    sorted_idx = np.argsort(-cd)
                    remaining = self.population_size - len(new_pop)
                    for i in sorted_idx[:remaining]:
                        new_pop.append(combined_pop[front[i]])
                        new_obj.append(combined_obj[front[i]])
                    break
            
            population = np.array(new_pop)
            objectives = np.array(new_obj)
        
        # Return Pareto front
        fronts = self._fast_non_dominated_sort(objectives)
        pareto_solutions = []
        
        for idx in fronts[0]:
            pareto_solutions.append(ParetoSolution(
                x=population[idx],
                objectives=objectives[idx],
                rank=0,
                crowding_distance=self._crowding_distance(objectives, fronts[0])[fronts[0].index(idx)]
            ))
        
        return pareto_solutions


def inventory_multi_objective(
    forecast_mean: float,
    forecast_std: float,
    holding_cost: float = 0.1,
    stockout_cost: float = 10.0,
    ordering_cost: float = 50.0,
) -> List[ParetoSolution]:
    """
    Multi-objective inventory optimization.
    
    Objectives:
    1. Minimize total cost
    2. Maximize service level
    3. Minimize inventory variance
    """
    def total_cost(x):
        order_qty, reorder_point = x
        holding = holding_cost * (reorder_point + order_qty / 2)
        stockout = stockout_cost * max(0, forecast_mean - reorder_point) * 0.1
        ordering = ordering_cost * (forecast_mean / max(order_qty, 1))
        return holding + stockout + ordering
    
    def service_level(x):
        order_qty, reorder_point = x
        safety_stock = reorder_point - forecast_mean
        sl = 1 / (1 + np.exp(-safety_stock / (forecast_std + 1)))
        return 1 - sl  # Minimize (1 - service_level)
    
    def inventory_variance(x):
        order_qty, reorder_point = x
        return (order_qty / 2) ** 2 + forecast_std ** 2
    
    nsga2 = NSGA2(
        n_objectives=3,
        n_variables=2,
        bounds=[(10, 200), (50, 300)],
        population_size=50,
        n_generations=30
    )
    
    return nsga2.optimize([total_cost, service_level, inventory_variance])
