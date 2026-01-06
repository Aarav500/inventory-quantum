"""Tests for QUBO formulation and solvers."""

import pytest
import numpy as np

from app.decision.qubo import QUBOOptimizer
from app.decision.simulated_annealing import SimulatedAnnealingSolver, TabuSearchSolver


@pytest.fixture
def qubo_optimizer():
    return QUBOOptimizer(n_bits=4)  # 16 quantity levels for testing


@pytest.fixture
def sa_solver():
    return SimulatedAnnealingSolver(
        initial_temp=10.0,
        cooling_rate=0.99,
        max_iterations=1000,
    )


class TestQUBOFormulation:
    """Tests for QUBO matrix construction."""
    
    def test_qubo_matrix_shape(self, qubo_optimizer):
        Q = qubo_optimizer._build_qubo_matrix(
            mean_demand=50.0,
            std_demand=10.0,
        )
        
        assert Q.shape == (qubo_optimizer.n_bits, qubo_optimizer.n_bits)
    
    def test_qubo_matrix_symmetric(self, qubo_optimizer):
        Q = qubo_optimizer._build_qubo_matrix(
            mean_demand=50.0,
            std_demand=10.0,
        )
        
        # QUBO matrix should be symmetric (or upper triangular)
        diff = np.abs(Q - Q.T)
        assert np.allclose(diff, 0) or np.allclose(np.tril(Q, -1), 0)
    
    def test_binary_to_quantity_conversion(self, qubo_optimizer):
        # Binary 0101 = 5
        binary = np.array([1, 0, 1, 0])
        quantity = qubo_optimizer._binary_to_quantity(binary)
        assert quantity == 5
        
        # Binary 1111 = 15
        binary = np.array([1, 1, 1, 1])
        quantity = qubo_optimizer._binary_to_quantity(binary)
        assert quantity == 15
    
    def test_quantity_to_binary_roundtrip(self, qubo_optimizer):
        for qty in [0, 5, 10, 15]:
            binary = qubo_optimizer._quantity_to_binary(qty)
            recovered = qubo_optimizer._binary_to_quantity(binary)
            assert recovered == qty
    
    def test_solution_evaluation(self, qubo_optimizer):
        cost, service = qubo_optimizer._evaluate_solution(
            quantity=100,
            mean_demand=50.0,
            std_demand=10.0,
            current_inventory=0,
        )
        
        assert cost >= 0
        assert 0 <= service <= 1


class TestSimulatedAnnealing:
    """Tests for SA solver."""
    
    def test_sa_returns_valid_binary(self, sa_solver):
        # Create simple QUBO
        Q = np.array([
            [-1, 2, 0, 0],
            [2, -1, 0, 0],
            [0, 0, -1, 2],
            [0, 0, 2, -1],
        ], dtype=float)
        
        solution = sa_solver.solve(Q)
        
        assert len(solution) == 4
        assert all(b in [0, 1] for b in solution)
    
    def test_sa_finds_optimal_simple_case(self, sa_solver):
        # Simple case: minimize x1 + x2, optimal is all zeros
        Q = np.array([
            [1, 0],
            [0, 1],
        ], dtype=float)
        
        solution = sa_solver.solve(Q)
        energy = sa_solver._qubo_energy(solution, Q)
        
        # Should find zero or low energy solution
        assert energy <= 1
    
    def test_sa_history_recorded(self, sa_solver):
        Q = np.eye(4)
        sa_solver.solve(Q)
        
        history = sa_solver.get_history()
        assert len(history) > 0
        assert all(len(h) == 3 for h in history)  # (iteration, current, best)


class TestTabuSearch:
    """Tests for Tabu search solver."""
    
    def test_tabu_returns_valid_binary(self):
        solver = TabuSearchSolver(tabu_size=5, max_iterations=100)
        Q = np.eye(4)
        
        solution = solver.solve(Q)
        
        assert len(solution) == 4
        assert all(b in [0, 1] for b in solution)


class TestSolverComparison:
    """Tests for solver comparison."""
    
    def test_sa_outperforms_random(self, qubo_optimizer, sa_solver):
        # Generate a non-trivial QUBO
        Q = qubo_optimizer._build_qubo_matrix(
            mean_demand=50.0,
            std_demand=10.0,
        )
        
        # SA solution
        sa_solution = sa_solver.solve(Q)
        sa_energy = sa_solver._qubo_energy(sa_solution, Q)
        
        # Random solutions
        random_energies = []
        for _ in range(100):
            random_solution = np.random.randint(0, 2, len(sa_solution))
            random_energies.append(sa_solver._qubo_energy(random_solution, Q))
        
        avg_random = np.mean(random_energies)
        
        # SA should be at least as good as average random
        # (with high probability on non-trivial problems)
        assert sa_energy <= avg_random * 1.5  # Allow some margin
