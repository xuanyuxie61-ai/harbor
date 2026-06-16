"""
hessian_magnon_dos.py
=====================

Hessian 矩阵、Magnon 谱与态密度计算。

物理背景
--------
在自旋玻璃的能量极小点附近, 可以做谐波展开:

    E(S) ≈ E_0 + (1/2) * sum_{ij} H_{ij} * delta_S_i * delta_S_j

其中 Hessian 矩阵:
    H_{ij} = d^2 E / (d S_i d S_j)

对于 Ising 模型, S_i in {-1, +1} 是离散的, Hessian 没有标准定义。
但对连续自旋版本 (S_i in [-1, 1] 或 O(N) 模型):

    H_{ij} = -J_{ij}  (i, j 是邻居)
    H_{ii} = sum_{j in nbr(i)} J_{ij} * S_j * S_i  (与局部场有关)

Hessian 的特征值 {omega_k^2} 对应 magnon (自旋波) 频率:
    omega_k = sqrt(lambda_k)

态密度 (Density of States, DOS):
    g(omega) = (1/N) * sum_k delta(omega - omega_k)

数值上用 Gaussian 展宽:
    g(omega) ≈ (1/N) * sum_k (1/(sigma*sqrt(2*pi))) * exp(-(omega-omega_k)^2/(2*sigma^2))

负特征值的意义
--------------
若 Hessian 有负特征值, 说明当前配置不是局部极小而是鞍点。
负模的数量 (index) 表征鞍点的阶数。

在自旋玻璃中:
- T > T_c: 顺磁态, Hessian 正定 (无负模)
- T ≈ T_c: de Almeida-Thouless 线, 最小特征值趋向 0
- T < T_c: 大量负模, 复杂的能量景观

本模块核心算法来源于 seed project:
- 1242_ce335805_PhotonDosReference: 光子态密度计算
  (映射为 magnon 态密度的计算)
- 1287_keb721_NucleiMorphology: 球谐函数分析
  (映射为 Hessian 的特征向量分析)
"""

import numpy as np
from typing import Dict, Tuple, Optional
from spin_lattice_geometry import CubicLattice3D


# =====================================================================
#  Hessian 矩阵构建
# =====================================================================

def build_hessian(spins: np.ndarray,
                  couplings: Dict[Tuple[int, int], float],
                  lattice: CubicLattice3D,
                  continuous: bool = True) -> np.ndarray:
    """
    构建 Hessian 矩阵 H_{ij} = d^2 E / (d S_i d S_j)。

    对于连续自旋:
        H_{ij} = -J_{ij}  (i != j, 最近邻)
        H_{ii} = 0  (因为 E = -sum J_{ij} S_i S_j 是双线性的)

    注意: 对于 Ising 哈密顿量 H = -sum J_{ij} S_i S_j,
    二阶导数:
        d^2 H / (d S_i d S_j) = -2 * J_{ij}  (i, j 邻居)
        d^2 H / (d S_i^2) = 0

    但如果有 on-site 势 (如连续自旋的 |S|^2 = 1 约束),
    则 H_{ii} != 0。

    参数
    ----
    spins : ndarray (L, L, L)
    couplings : dict
    lattice : CubicLattice3D
    continuous : bool
        是否使用连续自旋近似

    返回
    ----
    H : ndarray (N, N)
        Hessian 矩阵
    """
    N = lattice.N
    H = np.zeros((N, N), dtype=np.float64)

    for (i, j), J_ij in couplings.items():
        # d^2 E / (d S_i d S_j) = -2 * J_{ij} (因为 E = -sum J*S*S)
        H[i, j] = -2.0 * J_ij
        H[j, i] = -2.0 * J_ij

    if continuous:
        # 添加 soft constraint: V(S) = (mu/2) * sum_i (S_i^2 - 1)^2
        # d^2 V / d S_i^2 = 2 * mu * (3 * S_i^2 - 1)
        mu = 1.0  # 约束强度
        for i in range(N):
            s_i = spins.flat[i]
            H[i, i] += 2.0 * mu * (3.0 * s_i ** 2 - 1.0)

    return H


# =====================================================================
#  特征值分析
# =====================================================================

def hessian_eigenvalues(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Hessian 的特征值 (实对称矩阵)。

    返回
    ----
    eigenvalues : ndarray (N,) 升序
    eigenvectors : ndarray (N, N)
    """
    # 确保对称
    H_sym = 0.5 * (H + H.T)
    eigenvalues, eigenvectors = np.linalg.eigh(H_sym)
    return eigenvalues, eigenvectors


def count_negative_modes(eigenvalues: np.ndarray,
                         threshold: float = -1e-10) -> int:
    """
    计算负特征值的数量 (saddle point index)。
    """
    return int(np.sum(eigenvalues < threshold))


def hessian_condition_number(eigenvalues: np.ndarray) -> float:
    """
    Hessian 的条件数:
        kappa = |lambda_max| / |lambda_min|

    大条件数意味着能量景观 ill-conditioned, 优化困难。
    """
    abs_evals = np.abs(eigenvalues)
    lambda_min = np.min(abs_evals)
    lambda_max = np.max(abs_evals)
    if lambda_min < 1e-15:
        return float('inf')
    return lambda_max / lambda_min


# =====================================================================
#  Magnon 态密度 (DOS)
# =====================================================================

def magnon_dos_gaussian_broadening(eigenvalues: np.ndarray,
                                   omega_grid: np.ndarray,
                                   sigma: float = 0.1) -> np.ndarray:
    """
    用 Gaussian 展宽计算 magnon 态密度:
        g(omega) = (1/N) * sum_k (1/(sigma*sqrt(2*pi))) *
                   exp(-(omega - omega_k)^2 / (2*sigma^2))

    其中 omega_k = sqrt(|lambda_k|) (取绝对值以处理负模)

    参数
    ----
    eigenvalues : ndarray (N,)
    omega_grid : ndarray (n_omega,)
    sigma : float
        Gaussian 展宽宽度

    返回
    ----
    dos : ndarray (n_omega,)
    """
    N = len(eigenvalues)
    # 只取正特征值 (稳定模)
    positive_mask = eigenvalues > 0
    if np.sum(positive_mask) == 0:
        return np.zeros_like(omega_grid)

    omega_k = np.sqrt(eigenvalues[positive_mask])

    dos = np.zeros_like(omega_grid)
    prefactor = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    for ok in omega_k:
        dos += prefactor * np.exp(-0.5 * ((omega_grid - ok) / sigma) ** 2)
    dos /= N

    return dos


def magnon_dos_kernel_density(eigenvalues: np.ndarray,
                              omega_grid: np.ndarray,
                              bandwidth: Optional[float] = None) -> np.ndarray:
    """
    用 Silverman 规则的最优核密度估计:
        g(omega) = (1/(N*h)) * sum_k K((omega - omega_k) / h)

    其中 K 是标准正态核, h 是带宽:
        h = 1.06 * sigma_hat * N^{-1/5} (Silverman's rule of thumb)

    参数
    ----
    eigenvalues : ndarray (N,)
    omega_grid : ndarray (n_omega,)
    bandwidth : float or None

    返回
    ----
    dos : ndarray (n_omega,)
    """
    positive_mask = eigenvalues > 0
    omega_k = np.sqrt(eigenvalues[positive_mask])
    N_modes = len(omega_k)

    if N_modes == 0:
        return np.zeros_like(omega_grid)

    if bandwidth is None:
        # Silverman's rule of thumb
        sigma_hat = np.std(omega_k)
        if sigma_hat < 1e-15:
            sigma_hat = 0.1
        bandwidth = 1.06 * sigma_hat * N_modes ** (-0.2)

    h = bandwidth
    dos = np.zeros_like(omega_grid)
    for ok in omega_k:
        u = (omega_grid - ok) / h
        dos += np.exp(-0.5 * u ** 2)
    dos /= (N_modes * h * np.sqrt(2.0 * np.pi))

    return dos


def dos_moment(eigenvalues: np.ndarray, n: int) -> float:
    """
    态密度的 n 阶矩:
        M_n = integral omega^n * g(omega) d omega
            ≈ (1/N) * sum_k |lambda_k|^{n/2}
    """
    positive_mask = eigenvalues > 0
    omega_k = np.sqrt(eigenvalues[positive_mask])
    if len(omega_k) == 0:
        return 0.0
    return float(np.mean(omega_k ** n))


# =====================================================================
#  谱分析综合
# =====================================================================

def spectral_analysis(spins: np.ndarray,
                      couplings: Dict[Tuple[int, int], float],
                      lattice: CubicLattice3D,
                      n_omega: int = 100,
                      sigma_dos: float = 0.2
                      ) -> Dict:
    """
    完整的 Hessian 谱分析。

    返回
    ----
    result : dict
        eigenvalues, eigenvectors, n_negative, condition_number,
        omega_grid, dos, dos_moments
    """
    H = build_hessian(spins, couplings, lattice)
    eigenvalues, eigenvectors = hessian_eigenvalues(H)

    n_negative = count_negative_modes(eigenvalues)
    cond = hessian_condition_number(eigenvalues)

    # Magnon 频率
    omega_max = np.sqrt(np.max(np.abs(eigenvalues)) + 1e-10)
    omega_grid = np.linspace(0, omega_max * 1.2, n_omega)

    dos = magnon_dos_gaussian_broadening(eigenvalues, omega_grid, sigma_dos)

    # 矩
    moments = {n: dos_moment(eigenvalues, n) for n in range(1, 5)}

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "n_negative": n_negative,
        "condition_number": cond,
        "lambda_min": float(np.min(eigenvalues)),
        "lambda_max": float(np.max(eigenvalues)),
        "omega_grid": omega_grid,
        "dos": dos,
        "dos_moments": moments,
    }


# =====================================================================
#  de Almeida-Thouless 稳定性判据
# =====================================================================

def de_almeida_thouless_criterion(eigenvalues: np.ndarray,
                                  beta: float,
                                  J_var: float = 1.0) -> Dict[str, float]:
    """
    de Almeida-Thouless (AT) 线判据:

    在平均场理论中, AT 线由下式给出:
        beta^2 * J_var * integral g(omega) / omega^2 d omega = 1

    当最小特征值 lambda_min -> 0 时, 系统趋近 AT 线。

    参数
    ----
    eigenvalues : ndarray
    beta : float
    J_var : float

    返回
    ----
    result : dict
    """
    lambda_min = float(np.min(eigenvalues))

    # AT 稳定性参数
    positive_mask = eigenvalues > 1e-10
    if np.sum(positive_mask) > 0:
        at_parameter = beta ** 2 * J_var * np.mean(1.0 / eigenvalues[positive_mask])
    else:
        at_parameter = float('inf')

    return {
        "lambda_min": lambda_min,
        "at_parameter": at_parameter,
        "is_stable": at_parameter < 1.0,
        "distance_to_AT": 1.0 - at_parameter if at_parameter != float('inf') else float('-inf'),
    }
