"""
scf_solver.py — 自洽场迭代求解器
===================================

本模块实现 Kohn-Sham DFT 的自洽场 (SCF) 迭代。
融合种子项目:
  - 350_fd_predator_prey: 前向 Euler 时间步进 (SCF 动力学类比)
  - 362_fd1d_heat_steady: 稳态热传导求解 (Poisson 方程)
  - 1415_will_you_be_alive: Monte Carlo 概率模拟 (收敛概率分析)

核心物理:
  SCF 循环:
  1. 初始猜测 n⁰(x) 或 V_eff⁰(x)
  2. 对每个 k 点求解 KS 方程 → ε_{nk}, ψ_{nk}
  3. 计算新的电子密度 n^{out}(x)
  4. 构建新的有效势 V_eff^{out}(x)
  5. 混合: V_eff^{new} = α·V_eff^{out} + (1-α)·V_eff^{in}
  6. 检查收敛: ||n^{out} - n^{in}|| < tol
  7. 若不收敛, n^{in} ← n^{new}, 回到步骤 2

  混合方案:
  - 线性混合: V_new = α·V_out + (1-α)·V_in
  - Pulay mixing (DIIS): 使用历史密度构造最优线性组合
  - Kerker mixing: 在 Fourier 空间中过滤长波分量

  收敛判据:
    Δn = max_x |n^{out}(x) - n^{in}(x)| < tol
    或 RMS: √(Σ|Δn|²/N) < tol

  类比 predator-prey 动力学:
  SCF 迭代可视为离散动力系统:
    n^{k+1} = F(n^k)
  其中 F 为 KS 求解算子。稳定性取决于 F 的 Jacobi 矩阵
  谱半径 ρ(J_F) < 1。
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Callable
from physical_constants import TWO_PI


# ============================================================
# SCF 迭代核心
# ============================================================

class SCFSolver:
    """
    自洽场迭代求解器。

    管理 SCF 循环的完整生命周期:
    - 初始密度构建
    - KS 求解调度
    - 密度混合
    - 收敛监控
    - 历史数据存储 (用于 Pulay mixing)
    """

    def __init__(self, mixing_alpha: float = 0.3,
                 max_iterations: int = 100,
                 convergence_tol: float = 1e-8,
                 mixing_scheme: str = 'linear',
                 history_length: int = 8):
        """
        Parameters
        ----------
        mixing_alpha : float
            线性混合参数 α ∈ (0, 1]
            α 小 → 稳定但慢; α 大 → 快但可能发散
        max_iterations : int
            最大 SCF 迭代数
        convergence_tol : float
            收敛阈值 (密度变化)
        mixing_scheme : str
            'linear' 或 'pulay'
        history_length : int
            Pulay 混合的历史步数
        """
        if not 0 < mixing_alpha <= 1:
            raise ValueError(f"混合参数 α 必须在 (0,1], 得到 {mixing_alpha}")
        self.alpha = mixing_alpha
        self.max_iter = max_iterations
        self.tol = convergence_tol
        self.scheme = mixing_scheme
        self.history_length = history_length

        # 历史记录
        self.density_history: List[np.ndarray] = []
        self.residual_history: List[np.ndarray] = []
        self.energy_history: List[float] = []
        self.converged = False
        self.n_iterations = 0

    def reset(self):
        """重置迭代状态"""
        self.density_history.clear()
        self.residual_history.clear()
        self.energy_history.clear()
        self.converged = False
        self.n_iterations = 0

    def initial_density(self, n_grid: int, n_electrons: int,
                         a: float) -> np.ndarray:
        """
        构建初始密度猜测。

        使用均匀电子气密度作为初始猜测:
          n⁰(x) = N_e / a

        加上小幅随机扰动以打破对称性:
          n⁰(x) += 0.01 · rand() · n_avg

        Parameters
        ----------
        n_grid : int
            网格点数
        n_electrons : int
            电子数
        a : float
            晶格常数

        Returns
        -------
        n0 : np.ndarray
            初始密度
        """
        n_avg = n_electrons / a
        n0 = np.ones(n_grid) * n_avg
        # 小幅扰动打破对称性
        np.random.seed(42)  # 可复现
        n0 += 0.01 * n_avg * np.random.randn(n_grid)
        n0 = np.maximum(n0, 1e-10)  # 确保非负
        return n0

    def mix_density(self, n_in: np.ndarray,
                     n_out: np.ndarray) -> np.ndarray:
        """
        密度混合。

        线性混合:
          n_new = α · n_out + (1-α) · n_in

        Pulay mixing (DIIS):
          使用最近 m 步的密度和残差, 构造最优线性组合:
          n_new = Σ c_i (α·n_out^i + (1-α)·n_in^i)
          其中 c_i 最小化 ||Σ c_i R_i||², Σ c_i = 1
          R_i = n_out^i - n_in^i

        Parameters
        ----------
        n_in : np.ndarray
            输入密度
        n_out : np.ndarray
            输出密度 (KS 求解得到的新密度)

        Returns
        -------
        n_mixed : np.ndarray
            混合后的密度
        """
        residual = n_out - n_in

        if self.scheme == 'linear':
            n_mixed = self.alpha * n_out + (1.0 - self.alpha) * n_in

        elif self.scheme == 'pulay' and len(self.density_history) >= 2:
            n_mixed = self._pulay_mix(n_in, n_out, residual)
        else:
            n_mixed = self.alpha * n_out + (1.0 - self.alpha) * n_in

        # 记录历史
        self.density_history.append(n_in.copy())
        self.residual_history.append(residual.copy())

        # 限制历史长度
        if len(self.density_history) > self.history_length:
            self.density_history.pop(0)
            self.residual_history.pop(0)

        # 确保非负
        n_mixed = np.maximum(n_mixed, 1e-20)

        return n_mixed

    def _pulay_mix(self, n_in: np.ndarray, n_out: np.ndarray,
                    residual: np.ndarray) -> np.ndarray:
        """
        Pulay/DIIS 混合 (Direct Inversion in Iterative Subspace)。

        构造混合矩阵:
          A[i,j] = ⟨R_i | R_j⟩
        求解:
          A · c = λ (约束 Σ c_i = 1)
        混合密度:
          n_mixed = Σ c_i · n_mixed^i
        """
        m = len(self.residual_history)
        # 添加当前残差
        all_residuals = self.residuals_with_current(residual, m)

        # 构建 overlap 矩阵
        A = np.zeros((m + 1, m + 1))
        for i in range(m):
            for j in range(i, m):
                A[i, j] = np.sum(all_residuals[i] * all_residuals[j])
                A[j, i] = A[i, j]

        # 添加 Lagrange 乘子行/列
        A[m, :m] = -1.0
        A[:m, m] = -1.0
        A[m, m] = 0.0

        rhs = np.zeros(m + 1)
        rhs[m] = -1.0

        try:
            c = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            # 退化: 回退到线性混合
            return self.alpha * n_out + (1.0 - self.alpha) * n_in

        # 构造混合密度
        n_mixed = np.zeros_like(n_in)
        for i in range(m):
            n_mixed_i = self.alpha * (
                self.density_history[i] + self.residual_history[i]
            ) + (1.0 - self.alpha) * self.density_history[i]
            n_mixed += c[i] * n_mixed_i

        # 当前步的贡献
        n_mixed_current = self.alpha * n_out + (1.0 - self.alpha) * n_in
        n_mixed += c[m] * n_mixed_current if m < len(c) else 0.0

        return n_mixed

    def residuals_with_current(self, residual: np.ndarray,
                                 m: int) -> List[np.ndarray]:
        """返回包含当前残差的完整残差列表"""
        return self.residual_history + [residual]

    def check_convergence(self, n_in: np.ndarray,
                           n_out: np.ndarray) -> Tuple[bool, float, float]:
        """
        检查 SCF 收敛。

        判据:
          1. max |Δn(x)| < tol   (绝对最大偏差)
          2. ||Δn||_RMS < tol     (均方根偏差)
          3. |ΔE| < tol_E         (能量变化)

        Parameters
        ----------
        n_in : np.ndarray
            输入密度
        n_out : np.ndarray
            输出密度

        Returns
        -------
        converged : bool
        max_deviation : float
        rms_deviation : float
        """
        delta_n = n_out - n_in
        max_dev = np.max(np.abs(delta_n))
        rms_dev = np.sqrt(np.mean(delta_n ** 2))

        converged = max_dev < self.tol
        return converged, max_dev, rms_dev

    def run_scf(self, ks_solver: Callable,
                v_ext: np.ndarray,
                n_grid: int, dx: float,
                fd_order: int, a: float,
                n_electrons: int, n_bands: int,
                kpoints: np.ndarray, weights: np.ndarray,
                temperature: float,
                hartree_strength: float = 0.0,
                xc_functional=None,
                verbose: bool = True
                ) -> Dict[str, any]:
        """
        执行完整的 SCF 循环。

        Parameters
        ----------
        ks_solver : callable
            KS 求解函数 (v_eff, kpoints) → (eigenvalues, eigenvectors)
        v_ext : np.ndarray
            外势
        n_grid, dx, fd_order, a :
            网格和物理参数
        n_electrons : int
            电子数
        n_bands : int
            能带数
        kpoints, weights :
            k 点和权重
        temperature : float
            电子温度
        hartree_strength : float
            Hartree 势强度
        xc_functional : XCFunctional1D or None
            XC 泛函
        verbose : bool
            是否打印迭代信息

        Returns
        -------
        result : dict
            包含密度, 有效势, 本征值, 能量等信息
        """
        from potential import compute_hartree_potential
        from xc_functionals import (compute_xc_potential,
                                     compute_xc_energy,
                                     find_fermi_energy)
        from kohn_sham import compute_electron_density

        self.reset()

        # 初始密度
        n_current = self.initial_density(n_grid, n_electrons, a)

        for iteration in range(self.max_iter):
            self.n_iterations = iteration + 1

            # 构建有效势
            V_h = compute_hartree_potential(
                n_current, dx, a, hartree_strength)

            if xc_functional is not None:
                V_xc = compute_xc_potential(xc_functional, n_current)
                E_xc = compute_xc_energy(xc_functional, n_current, dx)
            else:
                V_xc = np.zeros(n_grid)
                E_xc = 0.0

            v_eff = v_ext + V_h + V_xc

            # KS 求解 (所有 k 点)
            eigenvalues, eigenvectors = ks_solver(v_eff, kpoints)

            # Fermi 能
            mu = find_fermi_energy(eigenvalues, weights,
                                    n_electrons, temperature)

            # 新密度
            n_new, n_elec_check = compute_electron_density(
                eigenvalues, eigenvectors, weights, mu, temperature, dx)

            # 收敛检查
            converged, max_dev, rms_dev = self.check_convergence(
                n_current, n_new)

            # 能量
            e_band = np.sum(weights[:, np.newaxis] * eigenvalues) * 2.0
            self.energy_history.append(e_band)

            if verbose:
                print(f"  SCF iter {iteration+1:3d}: "
                      f"max|Δn| = {max_dev:.2e}, "
                      f"RMS = {rms_dev:.2e}, "
                      f"E_band = {e_band:.8f} Ha")

            if converged:
                self.converged = True
                if verbose:
                    print(f"  SCF 收敛! 迭代 {iteration+1} 次")
                break

            # 密度混合
            n_current = self.mix_density(n_current, n_new)

        if not self.converged and verbose:
            print(f"  ⚠ SCF 未收敛 (最大迭代 {self.max_iter})")

        return {
            'density': n_current,
            'v_eff': v_eff,
            'v_hartree': V_h,
            'v_xc': V_xc,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'fermi_energy': mu,
            'energy_band': e_band,
            'energy_xc': E_xc,
            'converged': self.converged,
            'n_iterations': self.n_iterations,
            'energy_history': self.energy_history.copy(),
        }


# ============================================================
# 简化 SCF (非线性 Predator-Prey 类比)
# ============================================================

def scf_as_dynamical_system(n_current: np.ndarray,
                              F_operator: Callable,
                              alpha: float,
                              n_steps: int = 10
                              ) -> Tuple[np.ndarray, List[float]]:
    """
    将 SCF 迭代视为离散动力系统 (类比 350_fd_predator_prey)。

    n^{k+1} = (1-α) n^k + α F(n^k)

    其中 F 为 KS 求解算子。当 α 过大时, 系统可能振荡或发散
    (类似 Euler 方法求解 Lotka-Volterra 方程时的不稳定性)。

    稳定性分析:
    在不动点 n* 附近线性化:
      δn^{k+1} = [(1-α)I + α J_F] δn^k

    收敛条件: ρ[(1-α)I + α J_F] < 1
    其中 J_F = ∂F/∂n 为 KS 算子的 Jacobi 矩阵。

    最优 α 满足:
      α_opt = 2 / (1 + ρ(J_F))  (当 J_F 的特征值全为实时)

    Parameters
    ----------
    n_current : np.ndarray
        初始密度
    F_operator : callable
        KS 求解算子 n → n_new
    alpha : float
        混合参数
    n_steps : int
        迭代步数

    Returns
    -------
    n_final : np.ndarray
        最终密度
    residuals : list of float
        每步的残差
    """
    n = n_current.copy()
    residuals = []

    for step in range(n_steps):
        n_out = F_operator(n)
        residual = np.max(np.abs(n_out - n))
        residuals.append(residual)

        n = (1.0 - alpha) * n + alpha * n_out
        n = np.maximum(n, 1e-20)

    return n, residuals
