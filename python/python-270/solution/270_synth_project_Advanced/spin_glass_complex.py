"""
spin_glass_complex.py
=====================

复数矩阵运算、特征值分析与谱方法。

物理背景
--------
在自旋玻璃的稳定性分析中, 我们需要处理复数矩阵:

1. 放大矩阵 (Amplification Matrix):
   在 von Neumann 稳定性分析中,  Fourier 模态的演化:
       u_hat^{n+1} = G(k) * u_hat^n
   其中 G(k) 是放大矩阵, 可以是复数的。
   稳定性条件: rho(G) = max |lambda_i(G)| <= 1

2. Hessian 矩阵:
   H_{ij} = d^2 E / (d S_i d S_j)
   是实对称矩阵, 其特征值为 magnon 频率的平方。

3. 复极化率 (Complex Susceptibility):
   chi(omega) = chi'(omega) + i * chi''(omega)
   满足 Kramers-Kronig 关系:
       chi'(omega) = (1/pi) P integral chi''(omega') / (omega' - omega) domega'

本模块核心算法来源于 seed project:
- 131_c8lib: 复数算术库 (c8_abs, c8_mul, c8_exp, c8_log 等)
  (直接映射为复数矩阵运算)
"""

import numpy as np
from typing import Tuple, Optional, List


# =====================================================================
#  复数量子代数 (Complex Algebra)
# =====================================================================

def c8_multiply(z1: complex, z2: complex) -> complex:
    """
    复数乘法:
        (a + bi)(c + di) = (ac - bd) + (ad + bc)i
    """
    a, b = z1.real, z1.imag
    c, d = z2.real, z2.imag
    return complex(a * c - b * d, a * d + b * c)


def c8_exp(z: complex) -> complex:
    """
    复指数:
        exp(a + bi) = exp(a) * (cos(b) + i*sin(b))
    """
    return np.exp(z)


def c8_log(z: complex) -> complex:
    """
    复对数 (主支):
        log(r * exp(i*theta)) = log(r) + i*theta
    其中 theta in (-pi, pi]
    """
    if abs(z) < 1e-300:
        raise ValueError("log(0) 未定义")
    return np.log(z)


def c8_abs(z: complex) -> float:
    """复数模: |z| = sqrt(a^2 + b^2)"""
    return abs(z)


def c8_arg(z: complex) -> float:
    """复数辐角: arg(z) = atan2(b, a)"""
    return np.angle(z)


def c8_power(z: complex, n: int) -> complex:
    """
    复数整数次幂 (De Moivre 公式):
        z^n = r^n * exp(i*n*theta)
    """
    r = abs(z)
    theta = np.angle(z)
    return r ** n * np.exp(1j * n * theta)


# =====================================================================
#  复矩阵运算
# =====================================================================

def c8mat_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    复矩阵乘法: C = A * B
    C_{ij} = sum_k A_{ik} * B_{kj}
    """
    return A @ B


def c8mat_det(A: np.ndarray) -> complex:
    """复矩阵行列式"""
    return np.linalg.det(A)


def c8mat_inv(A: np.ndarray) -> np.ndarray:
    """
    复矩阵逆: A^{-1}
    若奇异则抛出 LinAlgError
    """
    return np.linalg.inv(A)


def c8mat_trace(A: np.ndarray) -> complex:
    """复矩阵的迹: Tr(A) = sum_i A_{ii}"""
    return np.trace(A)


# =====================================================================
#  特征值分析
# =====================================================================

def c8mat_eigen(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    一般复矩阵的特征值分解:
        A * v_i = lambda_i * v_i

    返回
    ----
    eigenvalues : ndarray (n,) complex
    eigenvectors : ndarray (n, n) complex
        列向量 eigenvectors[:, i] 对应特征值 eigenvalues[i]
    """
    return np.linalg.eig(A)


def symmetric_eigen(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    实对称矩阵的特征值分解 (用于 Hessian):
        H * v_i = lambda_i * v_i

    利用 eigh 获得更稳定的结果。

    返回
    ----
    eigenvalues : ndarray (n,) real, 升序
    eigenvectors : ndarray (n, n) real
    """
    if not np.allclose(A, A.T, atol=1e-10):
        # 对称化
        A = 0.5 * (A + A.T)
    return np.linalg.eigh(A)


# =====================================================================
#  谱半径 (Spectral Radius) —— 稳定性分析核心
# =====================================================================

def spectral_radius(A: np.ndarray) -> float:
    """
    矩阵 A 的谱半径:
        rho(A) = max_i |lambda_i(A)|

    稳定性条件: rho(G(k)) <= 1 对所有 k
    """
    eigenvalues = np.linalg.eigvals(A)
    return float(np.max(np.abs(eigenvalues)))


def amplification_factor_symbol(D: float, dt: float,
                                kx: float, ky: float, kz: float,
                                order: int = 2) -> complex:
    """
    计算放大因子 G(k) (标量情况, 无耦合):

    二阶:
        G(k) = 1 + D * dt * [2*(cos(kx)+cos(ky)+cos(kz)) - 6]
    四阶:
        G(k) = 1 + D * dt * sum_mu [-2*cos(2*k_mu)+32*cos(k_mu)-30]/12

    参数
    ----
    D : float
        扩散系数
    dt : float
        时间步长
    kx, ky, kz : float
        波矢分量
    order : int
        有限差分阶数 (2 或 4)

    返回
    ----
    G : complex
        放大因子
    """
    if order == 2:
        laplacian_k = 2.0 * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3.0)
    elif order == 4:
        laplacian_k = sum(
            (-2.0 * np.cos(2.0 * k) + 32.0 * np.cos(k) - 30.0) / 12.0
            for k in (kx, ky, kz)
        )
    else:
        raise ValueError(f"不支持的阶数: {order}")

    G = 1.0 + D * dt * laplacian_k
    return complex(G)


def max_amplification_over_bz(D: float, dt: float,
                               L: int, order: int = 2) -> float:
    """
    在整个布里渊区上扫描放大因子的最大模。

    稳定性判据: max_k |G(k)| <= 1 + epsilon

    参数
    ----
    D : float
        扩散系数
    dt : float
        时间步长
    L : int
        格点数 (决定 k 的离散化)
    order : int
        有限差分阶数

    返回
    ----
    rho_max : float
        max |G(k)| over BZ
    """
    n_modes = np.arange(-L // 2, L // 2)
    k_vals = 2.0 * np.pi * n_modes / L

    rho_max = 0.0
    for kx in k_vals:
        for ky in k_vals:
            for kz in k_vals:
                G = amplification_factor_symbol(D, dt, kx, ky, kz, order)
                rho = abs(G)
                if rho > rho_max:
                    rho_max = rho
    return rho_max


# =====================================================================
#  Gershgorin 圆盘定理
# =====================================================================

def gershgorin_disks(A: np.ndarray) -> List[Tuple[complex, float]]:
    """
    Gershgorin 圆盘定理:
    矩阵 A 的每个特征值至少位于某个圆盘 D(a_ii, R_i) 内, 其中
        R_i = sum_{j != i} |a_{ij}|

    用于估计 Hessian 矩阵的特征值范围。

    返回
    ----
    disks : list of (center, radius)
        每个 Gershgorin 圆盘的中心和半径
    """
    n = A.shape[0]
    disks = []
    for i in range(n):
        center = A[i, i]
        radius = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
        disks.append((complex(center), float(radius)))
    return disks


def gershgorin_eigenvalue_bounds(A: np.ndarray) -> Tuple[float, float]:
    """
    用 Gershgorin 圆盘估计特征值的上下界。

    返回
    ----
    (lambda_min_bound, lambda_max_bound) : (float, float)
    """
    disks = gershgorin_disks(A)
    lambda_min = min(c.real - r for c, r in disks)
    lambda_max = max(c.real + r for c, r in disks)
    return (lambda_min, lambda_max)


# =====================================================================
#  Kramers-Kronig 关系 (复极化率)
# =====================================================================

def kramers_kronig_chi_prime(omega_grid: np.ndarray,
                             chi_double_prime: np.ndarray,
                             omega: float) -> float:
    """
    Kramers-Kronig 关系 (从 chi'' 计算 chi'):
        chi'(omega) = (1/pi) P integral_{-inf}^{inf} chi''(omega') / (omega' - omega) domega'

    使用 Cauchy 主值积分 (Hilbert 变换)。

    参数
    ----
    omega_grid : ndarray
        频率网格
    chi_double_prime : ndarray
        chi''(omega') 的数值
    omega : float
        目标频率

    返回
    ----
    chi_prime : float
    """
    domega = omega_grid[1] - omega_grid[0] if len(omega_grid) > 1 else 1.0
    integrand = chi_double_prime / (omega_grid - omega + 1e-15)
    # 主值积分: 排除奇点
    mask = np.abs(omega_grid - omega) > 1e-10
    return float(np.sum(integrand[mask]) * domega / np.pi)
