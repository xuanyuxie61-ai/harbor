"""
stability_analysis.py — 数值稳定性与误差分析
===============================================

本模块对 FD-DFT 能带计算进行全面的稳定性分析。
融合种子项目:
  - 350_fd_predator_prey: Euler 方法稳定性 (SCF 类比)
  - 271_dg1d_advection: DG 方法的 CFL 条件
  - 1415_will_you_be_alive: Monte Carlo 概率稳定性分析
  - 040_asa121: 特殊函数的数值稳定性

核心分析内容:
  1. FD 色散关系误差分析
  2. SCF 收敛稳定性 (混合参数空间扫描)
  3. 网格收敛性 (dx 外推)
  4. 条件数分析
  5. 能量守恒检验
  6. 概率稳定性 (Monte Carlo 重采样)
"""

import numpy as np
from typing import Tuple, Dict, List
from physical_constants import (fd_coefficients_2nd, PI, TWO_PI,
                                 fd_dispersion_error)
from fd_operators import compute_fd_dispersion, fd_stability_limit


# ============================================================
# FD 色散误差分析
# ============================================================

class FDStabilityAnalyzer:
    """
    有限差分离散化的数值稳定性分析器。

    分析内容:
    1. 修正波数 vs 精确波数的偏差
    2. Nyquist 极限下的最大误差
    3. 稳定性区域 (CFL 条件)
    4. 网格收敛阶数
    """

    def __init__(self, fd_order: int):
        self.fd_order = fd_order

    def dispersion_error_analysis(self, n_samples: int = 1000
                                    ) -> Dict[str, any]:
        """
        完整的色散误差分析。

        返回:
        - kdx_max_1pct: 误差 < 1% 的最大 kdx
        - kdx_max_10pct: 误差 < 10% 的最大 kdx
        - error_at_nyquist: Nyquist 极限 (kdx=π) 的误差
        - convergence_order: 小 kdx 区域的收敛阶数
        """
        kdx, k2_mod, k2_exact = compute_fd_dispersion(
            self.fd_order, n_samples)

        # 相对误差
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.where(
                k2_exact > 1e-10,
                np.abs(k2_mod - k2_exact) / k2_exact,
                0.0)

        # 1% 和 10% 阈值
        kdx_1pct = self._find_threshold(kdx, rel_error, 0.01)
        kdx_10pct = self._find_threshold(kdx, rel_error, 0.10)

        # Nyquist 误差
        error_nyquist = rel_error[-1]

        # 收敛阶数 (小 kdx 区域)
        small_mask = (kdx > 0.01) & (kdx < 0.5)
        if np.sum(small_mask) > 5:
            log_k = np.log(kdx[small_mask])
            log_e = np.log(np.maximum(rel_error[small_mask], 1e-30))
            # 线性拟合 log(error) = p · log(kdx) + const
            coeffs = np.polyfit(log_k, log_e, 1)
            order = coeffs[0]
        else:
            order = 2.0 * self.fd_order

        return {
            'fd_order': self.fd_order,
            'kdx_max_1pct': kdx_1pct,
            'kdx_max_10pct': kdx_10pct,
            'error_at_nyquist': error_nyquist,
            'convergence_order': order,
            'expected_order': 2.0 * self.fd_order,
        }

    def _find_threshold(self, kdx: np.ndarray,
                          error: np.ndarray,
                          threshold: float) -> float:
        """找到误差首次超过阈值的 kdx"""
        valid = kdx > 0.01
        if not np.any(valid):
            return PI

        above = error[valid] > threshold
        if not np.any(above):
            return PI

        idx = np.argmax(above)
        return kdx[valid][idx]

    def required_grid_spacing(self, max_energy: float,
                                max_error_pct: float = 1.0
                                ) -> float:
        """
        计算满足给定能量精度所需的网格间距。

        对于最高能量 E_max, 对应的最大波数:
          k_max = √(2·E_max)   (原子单位)

        要求 k_max·dx < kdx_threshold:
          dx < kdx_threshold / k_max

        Parameters
        ----------
        max_energy : float
            最高感兴趣的能带能量 [Ha]
        max_error_pct : float
            允许的最大色散误差百分比

        Returns
        -------
        dx_max : float
            最大允许的网格间距 [Bohr]
        """
        k_max = np.sqrt(2.0 * max(abs(max_energy), 1e-10))

        if max_error_pct <= 1.0:
            kdx_thresh = fd_stability_limit(self.fd_order)
        else:
            kdx, k2_mod, k2_exact = compute_fd_dispersion(
                self.fd_order, 10000)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(k2_mod - k2_exact) / np.maximum(
                    k2_exact, 1e-30)
            threshold = max_error_pct / 100.0
            above = rel_err > threshold
            if np.any(above):
                idx = np.argmax(above)
                kdx_thresh = kdx[idx]
            else:
                kdx_thresh = PI

        return kdx_thresh / max(k_max, 1e-10)


# ============================================================
# SCF 收敛稳定性分析
# ============================================================

def scf_stability_scan(mixing_alphas: np.ndarray,
                        n_trials: int = 10,
                        n_electrons: int = 2,
                        n_grid: int = 64,
                        fd_order: int = 2,
                        a: float = 10.0) -> Dict[str, any]:
    """
    扫描混合参数空间的 SCF 收敛行为 (概率稳定性分析)。

    对每个 α 值, 进行多次随机初始化的 SCF 运行,
    统计收敛率和迭代次数。

    类比 1415_will_you_be_alive 中的 Monte Carlo 概率模拟:
    "SCF 收敛的概率" 类似于 "存活 10 年的概率"。

    Parameters
    ----------
    mixing_alphas : np.ndarray
        混合参数列表
    n_trials : int
        每个 α 的试验次数
    n_electrons : int
        电子数
    n_grid : int
        网格点数
    fd_order : int
        FD 半带宽
    a : float
        晶格常数

    Returns
    -------
    results : dict
        收敛概率, 平均迭代次数等
    """
    from potential import MathieuPotential
    from scf_solver import SCFSolver
    from kohn_sham import solve_all_bands

    pot = MathieuPotential(a)
    dx = a / n_grid
    x_grid = np.arange(n_grid) * dx
    v_ext = pot.V(x_grid)

    from lattice import monkhorst_pack_grid, kpoint_weights_uniform
    kpoints = monkhorst_pack_grid(8, a)
    weights = kpoint_weights_uniform(len(kpoints))

    convergence_rate = []
    mean_iterations = []
    std_iterations = []

    for alpha in mixing_alphas:
        conv_count = 0
        iter_list = []

        for trial in range(n_trials):
            np.random.seed(trial * 137 + int(alpha * 1000))

            solver = SCFSolver(
                mixing_alpha=alpha,
                max_iterations=50,
                convergence_tol=1e-6)

            def ks_func(v_eff, kpts):
                return solve_all_bands(
                    v_eff, kpts, n_grid, dx, fd_order, a,
                    min(4, n_grid))

            try:
                result = solver.run_scf(
                    ks_func, v_ext, n_grid, dx, fd_order, a,
                    n_electrons, min(4, n_grid),
                    kpoints, weights, temperature=0.01,
                    verbose=False)
                if result['converged']:
                    conv_count += 1
                    iter_list.append(result['n_iterations'])
            except Exception:
                pass

        rate = conv_count / max(n_trials, 1)
        convergence_rate.append(rate)
        mean_iterations.append(
            np.mean(iter_list) if iter_list else float('inf'))
        std_iterations.append(
            np.std(iter_list) if iter_list else 0.0)

    return {
        'alphas': mixing_alphas,
        'convergence_rate': np.array(convergence_rate),
        'mean_iterations': np.array(mean_iterations),
        'std_iterations': np.array(std_iterations),
        'optimal_alpha': mixing_alphas[np.argmax(convergence_rate)],
    }


# ============================================================
# 网格收敛性分析
# ============================================================

def grid_convergence_analysis(n_grids: List[int],
                                fd_order: int,
                                a: float = 10.0,
                                n_bands: int = 3
                                ) -> Dict[str, any]:
    """
    网格收敛性分析 (Richardson 外推)。

    对一系列网格密度, 计算最低能带能量, 然后拟合:
      E(N) = E_∞ + C · dx^p

    其中 p 为 FD 精度阶数 (2p_order)。

    Richardson 外推:
      E_∞ ≈ [N₂^p · E(N₂) - N₁^p · E(N₁)] / (N₂^p - N₁^p)

    Parameters
    ----------
    n_grids : list of int
        网格点数列表
    fd_order : int
        FD 半带宽
    a : float
        晶格常数
    n_bands : int
        能带数

    Returns
    -------
    results : dict
        能量 vs 网格, 外推值, 收敛阶数
    """
    from potential import MathieuPotential
    from kohn_sham import solve_all_bands
    from lattice import monkhorst_pack_grid, kpoint_weights_uniform

    pot = MathieuPotential(a)
    kpoints = monkhorst_pack_grid(16, a)
    weights = kpoint_weights_uniform(len(kpoints))

    energies = []
    dx_values = []

    for N in n_grids:
        dx = a / N
        x_grid = np.arange(N) * dx
        v_ext = pot.V(x_grid)

        evals, _ = solve_all_bands(
            v_ext, kpoints, N, dx, fd_order, a,
            min(n_bands, N))

        # 最低能带的平均能量 (所有 k 点的平均)
        e_avg = np.mean(evals[:, 0])
        energies.append(e_avg)
        dx_values.append(dx)

    energies = np.array(energies)
    dx_values = np.array(dx_values)

    # Richardson 外推 (相邻两点的估计)
    if len(energies) >= 2:
        p_expected = 2.0 * fd_order
        e1, e2 = energies[-2], energies[-1]
        dx1, dx2 = dx_values[-2], dx_values[-1]
        ratio = (dx1 / dx2) ** p_expected
        e_extrapolated = (ratio * e2 - e1) / (ratio - 1.0)
    else:
        e_extrapolated = energies[-1] if len(energies) > 0 else 0.0

    # 收敛阶数估计
    if len(energies) >= 3:
        dE1 = energies[-2] - energies[-3]
        dE2 = energies[-1] - energies[-2]
        if abs(dE2) > 1e-30:
            order_est = np.log(abs(dE1 / dE2)) / np.log(
                dx_values[-3] / dx_values[-2] * dx_values[-2] / dx_values[-1])
        else:
            order_est = float('inf')
    else:
        order_est = 2.0 * fd_order

    return {
        'n_grids': n_grids,
        'dx_values': dx_values,
        'energies': energies,
        'e_extrapolated': e_extrapolated,
        'estimated_order': order_est,
        'expected_order': 2.0 * fd_order,
    }


# ============================================================
# 条件数分析
# ============================================================

def hamiltonian_condition_number(n_grid: int, dx: float,
                                   fd_order: int,
                                   v_eff: np.ndarray,
                                   k_point: float,
                                   a: float) -> Dict[str, float]:
    """
    计算 KS Hamiltonian 矩阵的条件数。

    κ(H) = ||H|| · ||H⁻¹|| = λ_max / λ_min

    条件数反映:
    1. 特征值求解的数值精度: Δε/ε ~ κ(H) · ε_machine
    2. 线性系统的病态程度
    3. FD 网格分辨率是否足够

    对于 FD Laplacian:
      κ(L) ~ O(N²) (2阶), O(N^{2p}) (2p阶)

    Parameters
    ----------
    n_grid, dx, fd_order, v_eff, k_point, a :
        标准参数

    Returns
    -------
    results : dict
        条件数和谱信息
    """
    from fd_operators import build_laplacian_matrix

    L = build_laplacian_matrix(n_grid, dx, fd_order, k_point, a)
    H = -0.5 * L + np.diag(v_eff.astype(complex))

    eigenvalues = np.linalg.eigvalsh(H)
    eig_sorted = np.sort(eigenvalues.real)

    cond = abs(eig_sorted[-1] / eig_sorted[0]) if abs(eig_sorted[0]) > 1e-30 else float('inf')

    return {
        'condition_number': cond,
        'lambda_min': eig_sorted[0],
        'lambda_max': eig_sorted[-1],
        'spectral_gap': eig_sorted[1] - eig_sorted[0] if len(eig_sorted) > 1 else 0.0,
        'n_grid': n_grid,
        'fd_order': fd_order,
    }


# ============================================================
# 概率稳定性分析 (源自 1415_will_you_be_alive)
# ============================================================

def stability_probability_analysis(n_trials: int,
                                     n_grid: int,
                                     fd_order: int,
                                     a: float,
                                     perturbation_scale: float = 0.01
                                     ) -> Dict[str, float]:
    """
    SCF 收敛的概率分析 (源自 1415_will_you_be_alive 的 Monte Carlo)。

    对随机扰动的初始密度进行多次 SCF 运行,
    统计收敛概率和平均迭代次数。

    类比: 就像 "你 10 年后是否还活着" 的概率问题,
    这里问 "SCF 是否能在 N 步内收敛"。

    Parameters
    ----------
    n_trials : int
        试验次数
    n_grid : int
        网格点数
    fd_order : int
        FD 半带宽
    a : float
        晶格常数
    perturbation_scale : float
        初始扰动幅度

    Returns
    -------
    results : dict
        收敛概率, 平均迭代次数, 标准差
    """
    from potential import MathieuPotential
    from scf_solver import SCFSolver
    from kohn_sham import solve_all_bands
    from lattice import monkhorst_pack_grid, kpoint_weights_uniform

    pot = MathieuPotential(a)
    dx = a / n_grid
    x_grid = np.arange(n_grid) * dx
    v_ext = pot.V(x_grid)
    kpoints = monkhorst_pack_grid(8, a)
    weights = kpoint_weights_uniform(len(kpoints))

    converged_count = 0
    iterations_list = []

    for trial in range(n_trials):
        np.random.seed(trial)

        solver = SCFSolver(
            mixing_alpha=0.3,
            max_iterations=50,
            convergence_tol=1e-6)

        def ks_func(v_eff, kpts):
            return solve_all_bands(
                v_eff, kpts, n_grid, dx, fd_order, a, 4)

        try:
            result = solver.run_scf(
                ks_func, v_ext, n_grid, dx, fd_order, a,
                2, 4, kpoints, weights, 0.01, verbose=False)
            if result['converged']:
                converged_count += 1
                iterations_list.append(result['n_iterations'])
        except Exception:
            pass

    p_conv = converged_count / max(n_trials, 1)
    mean_iter = np.mean(iterations_list) if iterations_list else float('inf')
    std_iter = np.std(iterations_list) if iterations_list else 0.0

    return {
        'convergence_probability': p_conv,
        'mean_iterations': mean_iter,
        'std_iterations': std_iter,
        'n_trials': n_trials,
        'n_converged': converged_count,
    }
