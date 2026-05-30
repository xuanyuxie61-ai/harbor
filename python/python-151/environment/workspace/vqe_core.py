
import numpy as np
from typing import Tuple, Optional, Dict, List

from hamiltonian_builder import MolecularHamiltonian, r8symm_gen
from pauli_operator import PauliString, build_pauli_hamiltonian, extract_independent_paulis
from ansatz_tree import AnsatzTree, initialize_parameters_on_bloch_circle
from optimizer_geodesic import GeodesicVQEOptimizer
from fem_sampler import FEMExpectationSampler
from measurement_sampler import QuantumMeasurementSampler
from molecular_grid import MolecularIntegralGrid
from banded_solver import BandedMatrix, PDEParameters


class VQESolver:
    def __init__(self, n_qubits: int = 2, n_orbitals: int = 4,
                 ansatz_depth: int = 3, n_shots: int = 8192):
        self.n_qubits = n_qubits
        self.n_orbitals = n_orbitals
        self.ansatz_depth = ansatz_depth
        self.n_shots = n_shots


        self.hamiltonian = MolecularHamiltonian(n_orbitals)
        self.pauli_coeffs = self.hamiltonian.to_pauli_strings()
        self.pauli_strings = [PauliString(s, c) for s, c in self.pauli_coeffs.items()]
        self.independent_paulis = extract_independent_paulis(self.pauli_strings)

        self.ansatz = AnsatzTree(n_qubits, ansatz_depth)
        self.ansatz.build_hardware_efficient('CNOT')

        self.fem_sampler = FEMExpectationSampler(n_qubits, n_grid_1d=32)
        self.quantum_sampler = QuantumMeasurementSampler(n_qubits, n_shots)
        self.optimizer = GeodesicVQEOptimizer(max_iter=100, learning_rate=0.1)


        self.H_matrix = build_pauli_hamiltonian(n_qubits,
                                                 {p.string: p.coefficient for p in self.pauli_strings})

    def energy_exact(self, params: np.ndarray) -> float:
        self.ansatz.set_parameters(params)
        psi = self.ansatz.evaluate_statevector()
        E = np.real(np.vdot(psi, self.H_matrix @ psi))
        return float(E)

    def energy_noisy(self, params: np.ndarray) -> float:
        self.ansatz.set_parameters(params)
        psi = self.ansatz.evaluate_statevector()
        E = 0.0
        for p in self.independent_paulis:
            exp_val = self.quantum_sampler.sample_pauli_expectation(psi, p.string)
            E += np.real(p.coefficient) * exp_val
        return float(E)

    def energy_gradient_finite_diff(self, params: np.ndarray,
                                     h: float = 1e-5) -> np.ndarray:
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += h
            params_minus[i] -= h
            grad[i] = (self.energy_exact(params_plus) - self.energy_exact(params_minus)) / (2.0 * h)
        return grad

    def energy_gradient_parameter_shift(self, params: np.ndarray) -> np.ndarray:

        raise NotImplementedError("Hole 3: 请实现参数位移规则梯度计算")


    def run_vqe(self, use_parameter_shift: bool = True,
                exact_energy: bool = False) -> Tuple[np.ndarray, float, dict]:

        initialize_parameters_on_bloch_circle(self.ansatz)
        x0 = self.ansatz.parameters.copy()

        if exact_energy:
            energy_func = self.energy_exact
        else:
            energy_func = self.energy_noisy

        if use_parameter_shift:
            grad_func = self.energy_gradient_parameter_shift
        else:
            grad_func = lambda x: self.energy_gradient_finite_diff(x, h=1e-4)

        optimal_params, optimal_energy = self.optimizer.optimize(
            energy_func, grad_func, x0)

        info = {
            'history': self.optimizer.history,
            'convergence_rate': self.optimizer.get_convergence_rate(),
            'initial_energy': self.optimizer.history[0] if self.optimizer.history else None,
            'n_iterations': len(self.optimizer.history) - 1,
        }
        return optimal_params, optimal_energy, info

    def benchmark_against_exact(self) -> dict:
        E_fci, _ = self.hamiltonian.compute_exact_ground_state()
        opt_params, E_vqe, info = self.run_vqe(exact_energy=True)
        error = abs(E_vqe - E_fci)
        rel_error = error / (abs(E_fci) + 1e-14)
        return {
            'E_fci': E_fci,
            'E_vqe': E_vqe,
            'absolute_error': error,
            'relative_error': rel_error,
            'optimal_params': opt_params,
            'info': info,
        }

    def adapt_vqe_step(self, operator_pool: List[PauliString],
                        threshold: float = 1e-2) -> bool:
        params = self.ansatz.parameters.copy()
        self.ansatz.set_parameters(params)
        psi = self.ansatz.evaluate_statevector()

        gradients = np.zeros(len(operator_pool))
        for i, op in enumerate(operator_pool):



            commutator_val = 0.0
            for p in self.pauli_strings:
                comm = p.commutator(op)
                if abs(comm.coefficient) > 1e-14:
                    mat = comm.to_matrix()
                    val = np.vdot(psi, mat @ psi)
                    commutator_val += np.real(val)
            gradients[i] = abs(commutator_val)

        max_grad = np.max(gradients)
        if max_grad < threshold:
            return False


        self.ansatz.build_adaptive_layer(gradients,
                                         [op.string for op in operator_pool],
                                         threshold)
        return True


class VQEConvergenceAnalysis:
    def __init__(self, solver: VQESolver):
        self.solver = solver

    def analyze_landscape(self, n_points: int = 21) -> Tuple[np.ndarray, np.ndarray]:
        if self.solver.ansatz.param_count != 1:
            raise ValueError("仅支持单参数ansatz")
        thetas = np.linspace(-np.pi, np.pi, n_points)
        energies = np.array([self.solver.energy_exact(np.array([t])) for t in thetas])
        return thetas, energies

    def estimate_spectral_gap(self) -> float:
        eigvals = np.linalg.eigvalsh(self.solver.H_matrix)
        if len(eigvals) >= 2:
            return float(eigvals[1] - eigvals[0])
        return 0.0

    def compute_vqe_error_bound(self) -> float:
        gap = self.estimate_spectral_gap()
        if gap < 1e-14:
            return np.inf

        h_norm = np.linalg.norm(self.solver.H_matrix, 2)

        epsilon_ansatz = 0.1
        bound = h_norm ** 2 / gap * epsilon_ansatz
        return float(bound)
