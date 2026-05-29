"""
vqe_core.py
===========
变分量子本征求解器核心算法

科学功能:
整合前述所有模块，实现完整的VQE流水线:
    1. 哈密顿量构建 (MolecularHamiltonian)
    2. 参数化ansatz状态准备 (AnsatzTree)
    3. 能量期望值估计 (FEMExpectationSampler + QuantumMeasurementSampler)
    4. 经典优化 (GeodesicVQEOptimizer)
    5. 误差分析与收敛诊断
"""

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
    """
    变分量子本征求解器主类。
    """
    def __init__(self, n_qubits: int = 2, n_orbitals: int = 4,
                 ansatz_depth: int = 3, n_shots: int = 8192):
        self.n_qubits = n_qubits
        self.n_orbitals = n_orbitals
        self.ansatz_depth = ansatz_depth
        self.n_shots = n_shots

        # 初始化各组件
        self.hamiltonian = MolecularHamiltonian(n_orbitals)
        self.pauli_coeffs = self.hamiltonian.to_pauli_strings()
        self.pauli_strings = [PauliString(s, c) for s, c in self.pauli_coeffs.items()]
        self.independent_paulis = extract_independent_paulis(self.pauli_strings)

        self.ansatz = AnsatzTree(n_qubits, ansatz_depth)
        self.ansatz.build_hardware_efficient('CNOT')

        self.fem_sampler = FEMExpectationSampler(n_qubits, n_grid_1d=32)
        self.quantum_sampler = QuantumMeasurementSampler(n_qubits, n_shots)
        self.optimizer = GeodesicVQEOptimizer(max_iter=100, learning_rate=0.1)

        # 预计算哈密顿量稠密矩阵（用于经典模拟）
        self.H_matrix = build_pauli_hamiltonian(n_qubits,
                                                 {p.string: p.coefficient for p in self.pauli_strings})

    def energy_exact(self, params: np.ndarray) -> float:
        """
        通过完整状态向量模拟计算精确能量（无统计噪声）。
        E(theta) = <psi(theta)| H |psi(theta)>
        """
        self.ansatz.set_parameters(params)
        psi = self.ansatz.evaluate_statevector()
        E = np.real(np.vdot(psi, self.H_matrix @ psi))
        return float(E)

    def energy_noisy(self, params: np.ndarray) -> float:
        """
        通过有限次测量估计能量（含统计噪声）。
        """
        self.ansatz.set_parameters(params)
        psi = self.ansatz.evaluate_statevector()
        E = 0.0
        for p in self.independent_paulis:
            exp_val = self.quantum_sampler.sample_pauli_expectation(psi, p.string)
            E += np.real(p.coefficient) * exp_val
        return float(E)

    def energy_gradient_finite_diff(self, params: np.ndarray,
                                     h: float = 1e-5) -> np.ndarray:
        """
        使用中心差分计算能量梯度:
            dE/dtheta_i \approx (E(theta + h*e_i) - E(theta - h*e_i)) / (2h)
        """
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += h
            params_minus[i] -= h
            grad[i] = (self.energy_exact(params_plus) - self.energy_exact(params_minus)) / (2.0 * h)
        return grad

    def energy_gradient_parameter_shift(self, params: np.ndarray) -> np.ndarray:
        """
        使用参数位移规则计算精确梯度（对Pauli旋转门）：
            dE/dtheta = 0.5 * (E(theta + pi/2) - E(theta - pi/2))
        """
        # TODO: 实现参数位移规则计算VQE能量梯度
        raise NotImplementedError("Hole 3: 请实现参数位移规则梯度计算")


    def run_vqe(self, use_parameter_shift: bool = True,
                exact_energy: bool = False) -> Tuple[np.ndarray, float, dict]:
        """
        执行VQE优化。

        参数:
            use_parameter_shift: 使用参数位移规则计算梯度
            exact_energy: 使用精确能量（无测量噪声）
        返回:
            optimal_params, optimal_energy, info_dict
        """
        # 初始化参数
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
        """
        与FCI精确基态能量比较，评估VQE性能。
        """
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
        """
        执行一步ADAPT-VQE：计算各候选算符的梯度，选择最大者加入ansatz。
        """
        params = self.ansatz.parameters.copy()
        self.ansatz.set_parameters(params)
        psi = self.ansatz.evaluate_statevector()

        gradients = np.zeros(len(operator_pool))
        for i, op in enumerate(operator_pool):
            # 梯度 = <psi| [H, op] |psi>
            # 近似为能量对对应参数的导数
            # 简化为测量 [H, op] 的期望值
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

        # 添加新层
        self.ansatz.build_adaptive_layer(gradients,
                                         [op.string for op in operator_pool],
                                         threshold)
        return True


class VQEConvergenceAnalysis:
    """
    VQE收敛性分析工具。
    """
    def __init__(self, solver: VQESolver):
        self.solver = solver

    def analyze_landscape(self, n_points: int = 21) -> Tuple[np.ndarray, np.ndarray]:
        """
        对单参数ansatz扫描能量景观。
        """
        if self.solver.ansatz.param_count != 1:
            raise ValueError("仅支持单参数ansatz")
        thetas = np.linspace(-np.pi, np.pi, n_points)
        energies = np.array([self.solver.energy_exact(np.array([t])) for t in thetas])
        return thetas, energies

    def estimate_spectral_gap(self) -> float:
        """
        估计哈密顿量的基态-第一激发态能隙。
        """
        eigvals = np.linalg.eigvalsh(self.solver.H_matrix)
        if len(eigvals) >= 2:
            return float(eigvals[1] - eigvals[0])
        return 0.0

    def compute_vqe_error_bound(self) -> float:
        """
        基于ansatz表达能力估计VQE误差上界。
        使用McClean et al.的误差分析框架简化版。
        """
        gap = self.estimate_spectral_gap()
        if gap < 1e-14:
            return np.inf
        # 误差上界 ~ ||H||^2 / gap * epsilon_ansatz
        h_norm = np.linalg.norm(self.solver.H_matrix, 2)
        # 假设ansatz误差
        epsilon_ansatz = 0.1
        bound = h_norm ** 2 / gap * epsilon_ansatz
        return float(bound)
