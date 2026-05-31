"""
utils.py
通用数值工具模块
提供矩阵操作、数值稳定性处理、物理常数等基础功能。
"""

import numpy as np
from numpy.linalg import cholesky, solve, norm

# =====================================================================
# 物理常数（SI单位制）
# =====================================================================
WATER_DENSITY = 998.0          # ρ [kg/m^3]
WATER_VISCOSITY = 1.002e-3     # μ [Pa·s]
SURFACE_TENSION = 0.0728       # σ [N/m]
SOUND_SPEED_WATER = 1482.0     # c [m/s]
VAPOR_PRESSURE = 2338.0        # p_v [Pa]
ATMOSPHERIC_PRESSURE = 101325.0 # p_∞ [Pa]
GAS_CONSTANT = 8.314           # R [J/(mol·K)]
BOLTZMANN = 1.380649e-23       # k_B [J/K]


def safe_divide(a, b, default=0.0):
    """
    安全除法，避免除以零。
    参数:
        a: 被除数（标量或 ndarray）
        b: 除数（标量或 ndarray）
        default: 除数为零时的返回值
    返回:
        a / b 或 default
    """
    b = np.asarray(b)
    a = np.asarray(a)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-30
    result[mask] = a[mask] / b[mask]
    result[~mask] = default
    return result


def r8po_fa(n, a):
    """
    Cholesky 分解 A = U^T * U，对应 ellipsoid_monte_carlo 中的 r8po_fa。
    参数:
        n: 矩阵维数
        a: 正定对称矩阵（仅存储上三角）
    返回:
        u: Cholesky 因子（上三角）
        info: 0 表示成功，非零表示失败
    """
    a_full = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            a_full[i, j] = a[i, j]
            a_full[j, i] = a[i, j]
    try:
        u = cholesky(a_full).T  # numpy 返回下三角 L，需转置为上三角 U
        return u, 0
    except np.linalg.LinAlgError:
        return np.zeros((n, n)), 1


def r8po_sl(n, u, b):
    """
    利用 Cholesky 因子 U 解线性方程组 U^T U x = b。
    对应 ellipsoid_monte_carlo 中的 r8po_sl。
    """
    A = u.T @ u
    x = solve(A, b)
    return x


def uniform_in_sphere01_map(m, n):
    """
    在单位超球体内均匀采样 n 个点。
    对应 ellipsoid_monte_carlo 中的 uniform_in_sphere01_map。
    """
    x = np.random.randn(m, n)
    norms = np.sqrt(np.sum(x**2, axis=0))
    norms = np.maximum(norms, 1e-15)
    x = x / norms
    r = np.random.uniform(0.0, 1.0, size=n) ** (1.0 / m)
    return x * r


def ellipsoid_sample(m, n, a_mat, v, r):
    """
    从椭球体中均匀采样。
    对应 334_ellipsoid_monte_carlo 的核心算法。
    椭球定义: (X - V)^T A (X - V) <= R^2
    """
    u, info = r8po_fa(m, a_mat)
    if info != 0:
        raise ValueError("矩阵 A 不是正定对称矩阵")
    y = uniform_in_sphere01_map(m, n) * r
    x = np.zeros((m, n))
    for j in range(n):
        x[:, j] = r8po_sl(m, u, y[:, j])
    for i in range(m):
        x[i, :] += v[i]
    return x


def disk01_sample(n):
    """
    在单位圆盘内均匀采样 n 个点。
    对应 301_disk01_monte_carlo 的核心算法。
    """
    x = np.random.randn(2, n)
    norms = np.sqrt(np.sum(x**2, axis=0))
    norms = np.maximum(norms, 1e-15)
    x = x / norms
    r = np.sqrt(np.random.uniform(0.0, 1.0, size=n))
    return x * r


def monomial_value(m, n, e, x):
    """
    计算单项式 x^e 的值。
    对应 ellipsoid_monte_carlo / cube_exactness 中的 monomial_value。
    """
    v = np.ones(n, dtype=float)
    for i in range(m):
        if e[i] != 0:
            v *= x[i, :] ** e[i]
    return v


def print_matrix(mat, title=""):
    """打印矩阵（替代 MATLAB 中的 r8mat_print）"""
    if title:
        print(f"\n{title}")
    print(np.array2string(np.asarray(mat), precision=6, suppress_small=True))


def timestamp():
    """打印时间戳"""
    from datetime import datetime
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
