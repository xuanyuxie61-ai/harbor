"""
high_order_fd.py — 高阶有限差分格式与协变导数
============================================================

本模块实现用于量子霍尔效应哈密顿量离散化的高阶有限差分格式.

有限差分模板 (stencil):
  2阶精度:
    f'(x)  ≈ [-1/2, 0, 1/2] / h           (中心差分)
    f''(x) ≈ [1, -2, 1] / h²             (二阶导数)

  4阶精度:
    f'(x)  ≈ [1/12, -2/3, 0, 2/3, -1/12] / h
    f''(x) ≈ [-1/12, 4/3, -5/2, 4/3, -1/12] / h²

  6阶精度:
    f'(x)  ≈ [-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60] / h
    f''(x) ≈ [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90] / h²

磁场中的有限差分:
  使用 Peierls 替换实现协变导数:
    ∂_x → D_x: 有限差分权重乘以 Peierls 相位 exp(i∫A·dl)

  对于 Landau 规范 A = (0, Bx, 0):
    D_y ψ(x,y) = Σ c_k · exp(iBx·kh) · ψ(x, y+kh) / h
    D_x ψ(x,y) = Σ c_k · ψ(x+kh, y) / h  (A_x = 0, 无相位)

CFL 稳定性条件:
  对于薛定谔方程 i∂_t ψ = Hψ, 显式时间积分要求:
    Δt < 2/||H|| (谱半径倒数)
  其中 ||H|| ~ 4/h² + |B|²·L²/2 (来自动能和势能项)

参考文献:
    [1] Fornberg, B. "Generation of Finite Difference Formulas on Arbitrarily
        Spaced Grids", Math. Comp. 51, 699-706 (1988)
    [2] LeVeque, R. "Finite Difference Methods for Ordinary and Partial
        Differential Equations", SIAM (2007)
"""

import numpy as np
from scipy import sparse
from typing import List, Tuple, Optional


# ──────────────────────────────────────────────────────────────
# 有限差分模板系数
# ──────────────────────────────────────────────────────────────

# 一阶导数模板 (中心差分)
FD_FIRST_DERIVATIVE = {
    2: {'offsets': [-1, 0, 1],
        'coeffs': [-0.5, 0.0, 0.5]},
    4: {'offsets': [-2, -1, 0, 1, 2],
        'coeffs': [1.0/12.0, -2.0/3.0, 0.0, 2.0/3.0, -1.0/12.0]},
    6: {'offsets': [-3, -2, -1, 0, 1, 2, 3],
        'coeffs': [-1.0/60.0, 3.0/20.0, -3.0/4.0, 0.0,
                    3.0/4.0, -3.0/20.0, 1.0/60.0]},
    8: {'offsets': [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        'coeffs': [1.0/280.0, -4.0/105.0, 1.0/5.0, -4.0/5.0, 0.0,
                    4.0/5.0, -1.0/5.0, 4.0/105.0, -1.0/280.0]},
}

# 二阶导数模板 (中心差分)
FD_SECOND_DERIVATIVE = {
    2: {'offsets': [-1, 0, 1],
        'coeffs': [1.0, -2.0, 1.0]},
    4: {'offsets': [-2, -1, 0, 1, 2],
        'coeffs': [1.0/12.0, -4.0/3.0, 5.0/2.0, -4.0/3.0, 1.0/12.0]},
    6: {'offsets': [-3, -2, -1, 0, 1, 2, 3],
        'coeffs': [-1.0/90.0, 3.0/20.0, -3.0/2.0, 49.0/18.0,
                    -3.0/2.0, 3.0/20.0, -1.0/90.0]},
    8: {'offsets': [-4, -3, -2, -1, 0, 1, 2, 3, 4],
        'coeffs': [1.0/560.0, -8.0/315.0, 1.0/5.0, -8.0/5.0,
                    205.0/72.0, -8.0/5.0, 1.0/5.0, -8.0/315.0, 1.0/560.0]},
}


def fd_derivative_1d(values: np.ndarray, h: float, order: int,
                     derivative: int = 1) -> np.ndarray:
    """一维高阶有限差分导数

    Args:
        values: 函数值数组
        h: 网格间距
        order: 精度阶数 (2, 4, 6, 8)
        derivative: 导数阶数 (1 或 2)
    Returns:
        导数值数组 (边界使用降低精度的模板)
    """
    if derivative == 1:
        template = FD_FIRST_DERIVATIVE[order]
    elif derivative == 2:
        template = FD_SECOND_DERIVATIVE[order]
    else:
        raise ValueError(f"不支持的导数阶数: {derivative}")

    offsets = template['offsets']
    coeffs = template['coeffs']
    half_width = max(abs(o) for o in offsets)
    n = len(values)
    result = np.zeros(n, dtype=values.dtype)

    for i in range(n):
        val = 0.0
        for off, c in zip(offsets, coeffs):
            j = i + off
            if 0 <= j < n:
                val += c * values[j]
            else:
                # 边界处理: 使用镜像反射 (Dirichlet BC: ψ = 0)
                if j < 0:
                    j_mirror = -j
                    if j_mirror < n:
                        val += c * values[j_mirror] * (-1) ** (derivative % 2)
                else:
                    j_mirror = 2 * (n - 1) - j
                    if 0 <= j_mirror < n:
                        val += c * values[j_mirror] * (-1) ** (derivative % 2)
        result[i] = val / (h ** derivative)

    return result


def build_1d_laplacian(N: int, h: float, order: int = 4) -> sparse.csr_matrix:
    """构建一维拉普拉斯矩阵 -d²/dx² (带 Dirichlet 边界条件)

    矩阵结构:
        L = (1/h²) · tridiag(-1, 2, -1)  [2阶]
        L = (1/h²) · pentadiag(...)        [4阶]

    对于内部点, 使用 order 阶模板.
    对于边界附近的点, 降级到低阶模板或直接设为 Dirichlet.

    Args:
        N: 内部格点数 (不含边界)
        h: 网格间距
        order: 精度阶数
    Returns:
        稀疏拉普拉斯矩阵 (N×N)
    """
    if order == 2:
        main_diag = 2.0 * np.ones(N)
        off_diag = -1.0 * np.ones(N - 1)
        L = sparse.diags([off_diag, main_diag, off_diag],
                         [-1, 0, 1], shape=(N, N), format='csr')
    elif order == 4:
        d0 = 5.0 / 2.0 * np.ones(N)
        d1 = -4.0 / 3.0 * np.ones(N - 1)
        d2 = 1.0 / 12.0 * np.ones(N - 2)
        L = sparse.diags([d2, d1, d0, d1, d2],
                         [-2, -1, 0, 1, 2], shape=(N, N), format='csr')
    elif order == 6:
        d0 = 49.0 / 18.0 * np.ones(N)
        d1 = -3.0 / 2.0 * np.ones(N - 1)
        d2 = 3.0 / 20.0 * np.ones(N - 2)
        d3 = -1.0 / 90.0 * np.ones(N - 3)
        L = sparse.diags([d3, d2, d1, d0, d1, d2, d3],
                         [-3, -2, -1, 0, 1, 2, 3], shape=(N, N), format='csr')
    else:
        raise ValueError(f"不支持的精度阶数: {order}")

    return L / (h ** 2)


def build_1d_first_derivative(N: int, h: float,
                               order: int = 4) -> sparse.csr_matrix:
    """构建一维一阶导数矩阵 d/dx (反对称矩阵)

    注意: d/dx 是反对称矩阵 (d/dx)† = -d/dx
    因此 -i·d/dx 是厄米矩阵 (对应动量算符 p = -iℏ·d/dx)

    Args:
        N: 格点数
        h: 网格间距
        order: 精度阶数
    Returns:
        稀疏一阶导数矩阵
    """
    if order == 2:
        d1 = -0.5 * np.ones(N - 1)
        D = sparse.diags([d1, -d1], [-1, 1], shape=(N, N), format='csr')
    elif order == 4:
        d1 = -2.0 / 3.0 * np.ones(N - 1)
        d2 = 1.0 / 12.0 * np.ones(N - 2)
        D = sparse.diags([-d2, d1, -d1, d2],
                         [-2, -1, 1, 2], shape=(N, N), format='csr')
    elif order == 6:
        d1 = -3.0 / 4.0 * np.ones(N - 1)
        d2 = 3.0 / 20.0 * np.ones(N - 2)
        d3 = -1.0 / 60.0 * np.ones(N - 3)
        D = sparse.diags([d3, -d2, d1, -d1, d2, -d3],
                         [-3, -2, -1, 1, 2, 3], shape=(N, N), format='csr')
    else:
        raise ValueError(f"不支持的精度阶数: {order}")

    return D / h


def truncation_error(f_func, x0: float, h: float, order: int,
                     derivative: int = 2) -> float:
    """计算有限差分的截断误差

    截断误差分析:
        E_h = |f^{(n)}_exact - f^{(n)}_FD|
        对于 order 阶模板: E_h = O(h^order)

     Richardson 外推可用来估计实际阶数:
        p = log(E_{h}/E_{h/2}) / log(2)

    Args:
        f_func: 解析函数 f(x)
        x0: 计算点
        h: 网格间距
        order: 模板阶数
        derivative: 导数阶数
    Returns:
        截断误差
    """
    eps = h
    if derivative == 2:
        if order == 2:
            fd_approx = (f_func(x0 + eps) - 2 * f_func(x0) +
                         f_func(x0 - eps)) / eps ** 2
        elif order == 4:
            fd_approx = (-f_func(x0 + 2*eps) + 16*f_func(x0 + eps) -
                         30*f_func(x0) + 16*f_func(x0 - eps) -
                         f_func(x0 - 2*eps)) / (12 * eps ** 2)
        else:
            fd_approx = (-f_func(x0 + 2*eps) + 16*f_func(x0 + eps) -
                         30*f_func(x0) + 16*f_func(x0 - eps) -
                         f_func(x0 - 2*eps)) / (12 * eps ** 2)
    elif derivative == 1:
        if order == 2:
            fd_approx = (f_func(x0 + eps) - f_func(x0 - eps)) / (2 * eps)
        elif order == 4:
            fd_approx = (-f_func(x0 + 2*eps) + 8*f_func(x0 + eps) -
                         8*f_func(x0 - eps) + f_func(x0 - 2*eps)) / (12 * eps)
        else:
            fd_approx = (-f_func(x0 + 2*eps) + 8*f_func(x0 + eps) -
                         8*f_func(x0 - eps) + f_func(x0 - 2*eps)) / (12 * eps)
    else:
        raise ValueError

    # 解析导数
    if derivative == 1:
        exact = (f_func(x0 + 1e-12) - f_func(x0 - 1e-12)) / (2e-12)
    else:
        exact = (f_func(x0 + 1e-8) - 2*f_func(x0) +
                 f_func(x0 - 1e-8)) / (1e-8) ** 2

    return abs(fd_approx - exact)


def richardson_extrapolation(E_h1: float, E_h2: float,
                             h1: float, h2: float) -> float:
    """Richardson 外推估计收敛阶数

    如果 E_h = C·h^p, 则:
        p = log(E_{h1}/E_{h2}) / log(h1/h2)

    Args:
        E_h1: 粗网格误差
        E_h2: 细网格误差
        h1: 粗网格间距
        h2: 细网格间距
    Returns:
        估计的收敛阶数 p
    """
    if E_h2 <= 0 or E_h1 <= 0:
        return float('inf')
    return np.log(E_h1 / E_h2) / np.log(h1 / h2)
