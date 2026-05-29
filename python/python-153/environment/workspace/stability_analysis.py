"""
stability_analysis.py
基于项目 1374_unstable_ode (不稳定ODE) 与 353_fd1d_advection_ftcs (FTCS不稳定性)
的数值稳定性分析模块。

核心数学模型:
1. von Neumann 稳定性分析:
   对线性差分格式 u^{n+1} = G(k) u^n,
   稳定性要求放大因子 |G(k)| <= 1 对所有波数 k 成立。

2. FTCS 格式的放大因子 (对流方程):
   G = 1 - i*(c*dt/dx)*sin(k*dx)
   |G|^2 = 1 + (c*dt/dx)^2 * sin^2(k*dx) > 1
   因此 FTCS 对对流方程无条件不稳定。

3. CFL 条件 (Courant-Friedrichs-Lewy):
   对双曲型 PDE，稳定性要求数值依赖域包含物理依赖域。
   c * dt / dx <= 1

4. 矩阵稳定性分析:
   对离散系统 u^{n+1} = A u^n,
   稳定性要求谱半径 rho(A) <= 1。

5. 量子核矩阵的条件数与数值稳定性:
   kappa(K) = lambda_max / lambda_min
   大条件数意味着核矩阵接近奇异，数值求解不稳定。
"""

import numpy as np
from typing import Tuple, Callable


def von_neumann_amplification_ftcs(
    c: float,
    dt: float,
    dx: float,
    k_values: np.ndarray
) -> np.ndarray:
    """
    计算 FTCS 格式在对流方程中的 von Neumann 放大因子。
    G(k) = 1 - i*(c*dt/dx)*sin(k*dx)
    |G(k)| = sqrt(1 + (CFL)^2 * sin^2(k*dx))
    """
    # TODO: Implement the von Neumann amplification factor for FTCS.
    # G(k) = 1 - i*(c*dt/dx)*sin(k*dx)
    # Return the absolute value |G(k)|.
    # Validate inputs and raise appropriate errors.
    pass


def cfl_condition_hyperbolic(
    wave_speed: float,
    dx: float
) -> float:
    """
    计算满足 CFL 条件的最大时间步长。
    dt_max = dx / |wave_speed|
    """
    if dx <= 0:
        raise ValueError("dx must be positive")
    if abs(wave_speed) < 1e-15:
        return np.inf
    return dx / abs(wave_speed)


def diffusion_stability_limit(
    D: float,
    dx: float,
    dimension: int = 1
) -> float:
    """
    扩散方程显式格式的稳定性极限。
    1D: dt <= dx^2 / (2*D)
    2D: dt <= dx^2 / (4*D)
    """
    if D < 0 or dx <= 0:
        raise ValueError("D must be non-negative and dx positive")
    if dimension not in [1, 2, 3]:
        raise ValueError("Dimension must be 1, 2, or 3")

    factor = {1: 2.0, 2: 4.0, 3: 6.0}[dimension]
    return dx * dx / (factor * D + 1e-15)


def matrix_spectral_radius(A: np.ndarray) -> float:
    """
    计算矩阵的谱半径 (特征值最大模)。
    rho(A) = max(|lambda_i|)
    """
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    eigenvalues = np.linalg.eigvals(A)
    return np.max(np.abs(eigenvalues))


def is_stable_matrix(A: np.ndarray, tol: float = 1.0 + 1e-10) -> bool:
    """
    判断矩阵 A 是否稳定 (谱半径 <= 1)。
    """
    return matrix_spectral_radius(A) <= tol


def analyze_kernel_matrix_stability(
    K: np.ndarray,
    reg_values: np.ndarray = None
) -> dict:
    """
    分析量子核矩阵的数值稳定性。

    返回:
        condition_number: 条件数
        smallest_eigenvalue: 最小特征值
        rank_estimate: 有效秩估计
        recommended_reg: 推荐正则化参数
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")

    eigvals = np.linalg.eigvalsh(K)
    pos_eigvals = eigvals[eigvals > 1e-12]

    if len(pos_eigvals) == 0:
        return {
            "condition_number": np.inf,
            "smallest_eigenvalue": 0.0,
            "rank_estimate": 0,
            "recommended_reg": 1e-6,
            "is_well_conditioned": False
        }

    cond = np.max(eigvals) / np.min(pos_eigvals)
    rank = len(pos_eigvals)

    # 推荐正则化参数: 最小特征值的 10 倍
    recommended_reg = max(10.0 * np.min(pos_eigvals), 1e-10)

    # 条件数阈值: 1e12 为 double 精度下的危险边界
    is_well = cond < 1e12

    return {
        "condition_number": cond,
        "smallest_eigenvalue": np.min(pos_eigvals),
        "rank_estimate": rank,
        "recommended_reg": recommended_reg,
        "is_well_conditioned": is_well
    }


def trotter_error_bound(
    H: np.ndarray,
    dt: float,
    order: int = 1
) -> float:
    """
    Trotter-Suzuki 分解误差上界估计。
    对哈密顿量 H = sum_j H_j，
    一阶 Trotter 误差: O(dt^2 * sum_{i<j} ||[H_i, H_j]||)

    参数:
        H: 总哈密顿量
        dt: 时间步长
        order: Trotter 公式阶数
    """
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("H must be a square matrix")

    norm_H = np.linalg.norm(H, ord=2)

    if order == 1:
        # 一阶误差 ~ O(dt^2)
        error = dt * dt * norm_H ** 2 / 2.0
    elif order == 2:
        # 二阶误差 ~ O(dt^3)
        error = dt ** 3 * norm_H ** 3 / 6.0
    else:
        # 高阶误差 ~ O(dt^{order+1})
        error = dt ** (order + 1) * norm_H ** (order + 1)

    return error


def quantum_kernel_robustness_score(
    K: np.ndarray,
    noise_level: float = 0.01
) -> float:
    """
    评估量子核矩阵在噪声下的鲁棒性分数。
    对核矩阵添加随机扰动，检查条件数变化。

    返回: 0 到 1 之间的鲁棒性分数 (1 表示最鲁棒)。
    """
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square matrix")

    n = K.shape[0]
    orig_cond = np.linalg.cond(K)

    # 添加噪声
    noise = noise_level * np.random.randn(n, n)
    noise = (noise + noise.T) / 2.0  # 对称化
    K_noisy = K + noise

    noisy_cond = np.linalg.cond(K_noisy)

    # 条件数变化比
    ratio = noisy_cond / (orig_cond + 1e-15)

    # 映射到 0-1 分数
    score = max(0.0, 1.0 - np.log10(ratio) / 10.0)
    return score
