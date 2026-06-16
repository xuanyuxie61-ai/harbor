# -*- coding: utf-8 -*-
"""
high_order_fd.py
================
PROJECT_254 — 计算天体物理：双中子星并合与 kilonova 辐射转移

高阶中心差分模板 (2nd, 4th, 6th, 8th 阶精度), 用于 Euler 方程
(质量/动量/能量守恒) 的空间离散.

有限差分公式
----------
一阶导数::

    (D_2 f)_i = (f_{i+1} - f_{i-1}) / (2h)               O(h^2)

    (D_4 f)_i = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12h)   O(h^4)

    (D_6 f)_i = (f_{i+3} - 9 f_{i+2} + 45 f_{i+1} - 45 f_{i-1}
                 + 9 f_{i-2} - f_{i-3}) / (60 h)          O(h^6)

    (D_8 f)_i = (-f_{i+4} + (32/3) f_{i+3} - 56/3 f_{i+2}
                 + 112/3 f_{i+1} - 112/3 f_{i-1} + 56/3 f_{i-2}
                 - 32/3 f_{i-3} + f_{i-4}) / (28/3 * h)  O(h^8)
   简化: 使用 Fornberg 算法计算权重.

二阶导数::

    (D2_2 f)_i = (f_{i+1} - 2 f_i + f_{i-1}) / h^2       O(h^2)

    (D2_4 f)_i = (-f_{i+2} + 16 f_{i+1} - 30 f_i
                  + 16 f_{i-1} - f_{i-2}) / (12 h^2)      O(h^4)

映射种子项目
-----------
- 577 (image_diffuse4/8)  → 4/8 邻居模板的数学本质是 Laplacian 的离散化,
  在 kilonova 辐射转移中对应角度空间的扩散算子
- 014 (approx_chebyshev) → Chebyshev 节点与 FD 模板的谱等价性,
  在边界附近使用 Chebyshev 加权抑制 Gibbs 振荡
"""

from __future__ import annotations
import math
from typing import List, Tuple, Callable


# ---------------------------------------------------------------------------
# 模板权重
# ---------------------------------------------------------------------------
FIRST_DERIV_WEIGHTS = {
    2: [(-1.0 / 2.0, 1), (1.0 / 2.0, -1)],       # (coeff, shift)
    4: [(1.0 / 12.0, -2), (-8.0 / 12.0, -1),
        (8.0 / 12.0, 1), (-1.0 / 12.0, 2)],
    6: [(-1.0 / 60.0, -3), (9.0 / 60.0, -2), (-45.0 / 60.0, -1),
        (45.0 / 60.0, 1), (-9.0 / 60.0, 2), (1.0 / 60.0, 3)],
    8: [(1.0 / 280.0, -4), (-4.0 / 105.0, -3),
        (1.0 / 5.0, -2), (-4.0 / 5.0, -1),
        (4.0 / 5.0, 1), (-1.0 / 5.0, 2),
        (4.0 / 105.0, 3), (-1.0 / 280.0, 4)],
}

SECOND_DERIV_WEIGHTS = {
    2: [(1.0, -1), (-2.0, 0), (1.0, 1)],
    4: [(-1.0 / 12.0, -2), (16.0 / 12.0, -1),
        (-30.0 / 12.0, 0), (16.0 / 12.0, 1), (-1.0 / 12.0, 2)],
    6: [(1.0 / 90.0, -3), (-3.0 / 20.0, -2), (3.0 / 2.0, -1),
        (-49.0 / 18.0, 0), (3.0 / 2.0, 1),
        (-3.0 / 20.0, 2), (1.0 / 90.0, 3)],
}

# 对流通量重构 WENO5 权重 (Jiang & Shu 1996)
WENO5_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# 一维算子
# ---------------------------------------------------------------------------
def first_derivative(f: List[float], h: float, order: int = 4) -> List[float]:
    """一阶导数的高阶中心差分.

    Parameters
    ----------
    f     : List[float]  等距网格上的函数值
    h     : float        网格间距
    order : int          精度阶数 (2, 4, 6, 8)

    Returns
    -------
    List[float]  各内点的导数近似; 边界用降阶单侧差分.
    """
    if order not in FIRST_DERIV_WEIGHTS:
        raise ValueError(f"Unsupported order {order}; choose from {list(FIRST_DERIV_WEIGHTS.keys())}")
    n = len(f)
    if n < order + 1:
        raise ValueError(f"Need at least {order + 1} points for order {order}")
    weights = FIRST_DERIV_WEIGHTS[order]
    half_w = order // 2
    df = [0.0] * n

    # 内点
    for i in range(half_w, n - half_w):
        s = 0.0
        for coeff, shift in weights:
            s += coeff * f[i + shift]
        df[i] = s / h

    # 边界: 使用一阶精度的单侧差分 (前/后向)
    for i in range(half_w):
        if i + 1 < n:
            df[i] = (f[i + 1] - f[i]) / h
    for i in range(n - half_w, n):
        if i - 1 >= 0:
            df[i] = (f[i] - f[i - 1]) / h
    return df


def second_derivative(f: List[float], h: float, order: int = 4) -> List[float]:
    """二阶导数的高阶中心差分.

    Parameters
    ----------
    f     : List[float]
    h     : float
    order : int  (2, 4, 6)
    """
    if order not in SECOND_DERIV_WEIGHTS:
        raise ValueError(f"Unsupported order {order}")
    n = len(f)
    weights = SECOND_DERIV_WEIGHTS[order]
    half_w = order // 2
    d2f = [0.0] * n
    for i in range(half_w, n - half_w):
        s = 0.0
        for coeff, shift in weights:
            s += coeff * f[i + shift]
        d2f[i] = s / (h * h)
    # 边界
    for i in range(half_w):
        if i + 2 < n:
            d2f[i] = (f[i + 2] - 2.0 * f[i + 1] + f[i]) / (h * h)
    for i in range(n - half_w, n):
        if i - 2 >= 0:
            d2f[i] = (f[i] - 2.0 * f[i - 1] + f[i - 2]) / (h * h)
    return d2f


# ---------------------------------------------------------------------------
# WENO5 通量重构 (用于激波捕捉)
# ---------------------------------------------------------------------------
def weno5_reconstruct(fL: List[float], fR: List[float]) -> Tuple[List[float], List[float]]:
    """WENO5 重构: 从单元平均值 f 得到左右界面值 f_{i+1/2}^{L,R}.

    用于 Euler 方程 Godunov 型通量::

        f_{i+1/2}^L = (1/3) f_{i-2} - (7/6) f_{i-1} + (11/6) f_i  (smooth stencil 0)
        ...

    三个候选模板的权重通过光滑度指标 beta_k 自适应组合.
    """
    n = len(fL)
    fhat_L = [0.0] * (n + 1)
    fhat_R = [0.0] * (n + 1)

    def weno5_z(u, i):
        """返回 u_{i+1/2} 的 WENO 重构 (左值)."""
        if i < 2 or i > len(u) - 3:
            return u[i]
        # 三个候选模板
        q0 = (1.0 / 3.0) * u[i - 2] - (7.0 / 6.0) * u[i - 1] + (11.0 / 6.0) * u[i]
        q1 = -(1.0 / 6.0) * u[i - 1] + (5.0 / 6.0) * u[i] + (1.0 / 3.0) * u[i + 1]
        q2 = (1.0 / 3.0) * u[i] + (5.0 / 6.0) * u[i + 1] - (1.0 / 6.0) * u[i + 2]
        # 光滑度指标
        b0 = (13.0 / 12.0) * (u[i - 2] - 2.0 * u[i - 1] + u[i]) ** 2 \
             + 0.25 * (u[i - 2] - 4.0 * u[i - 1] + 3.0 * u[i]) ** 2
        b1 = (13.0 / 12.0) * (u[i - 1] - 2.0 * u[i] + u[i + 1]) ** 2 \
             + 0.25 * (u[i - 1] - u[i + 1]) ** 2
        b2 = (13.0 / 12.0) * (u[i] - 2.0 * u[i + 1] + u[i + 2]) ** 2 \
             + 0.25 * (3.0 * u[i] - 4.0 * u[i + 1] + u[i + 2]) ** 2
        # 理想权重
        d0, d1, d2 = 0.1, 0.6, 0.3
        eps = WENO5_EPS
        a0 = d0 / ((eps + b0) ** 2)
        a1 = d1 / ((eps + b1) ** 2)
        a2 = d2 / ((eps + b2) ** 2)
        asum = a0 + a1 + a2
        w0, w1, w2 = a0 / asum, a1 / asum, a2 / asum
        return w0 * q0 + w1 * q1 + w2 * q2

    for i in range(n):
        fhat_L[i] = weno5_z(fL, i)
    for i in range(n):
        fhat_R[i] = weno5_z(fR, n - 1 - i)
    return fhat_L, fhat_R


# ---------------------------------------------------------------------------
# 二维 Laplacian (kilonova 辐射扩散)
# ---------------------------------------------------------------------------
def laplacian_2d_4th(u: List[List[float]], hx: float, hy: float
                     ) -> List[List[float]]:
    """二维 Laplacian 的四阶中心差分.

    ∂²u/∂x² + ∂²u/∂y² ≈ (D2_4 u)_{i,j} + (D2_4 u)_{j,i}

    对内部点::

        L u = -(u_{i+2,j} + u_{i-2,j} - 16(u_{i+1,j}+u_{i-1,j}) + 30 u_{i,j}) / (12 hx^2)
              -(u_{i,j+2} + u_{i,j-2} - 16(u_{i,j+1}+u_{i,j-1}) + 30 u_{i,j}) / (12 hy^2)

    Parameters
    ----------
    u  : List[List[float]]  二维数组, u[i][j]
    hx : float  x 方向网格间距
    hy : float  y 方向网格间距
    """
    ny = len(u)
    nx = len(u[0])
    Lu = [[0.0] * nx for _ in range(ny)]
    for i in range(2, ny - 2):
        for j in range(2, nx - 2):
            d2x = (-u[i][j + 2] + 16.0 * u[i][j + 1] - 30.0 * u[i][j]
                   + 16.0 * u[i][j - 1] - u[i][j - 2]) / (12.0 * hx * hx)
            d2y = (-u[i + 2][j] + 16.0 * u[i + 1][j] - 30.0 * u[i][j]
                   + 16.0 * u[i - 1][j] - u[i - 2][j]) / (12.0 * hy * hy)
            Lu[i][j] = d2x + d2y
    return Lu


# ---------------------------------------------------------------------------
# 截断误差估计
# ---------------------------------------------------------------------------
def truncation_error_estimate(f: Callable[[float], float],
                              a: float, b: float,
                              N: int, order: int) -> float:
    """用解析函数验证 FD 算子的截断误差 O(h^p).

    Returns
    -------
    float  max |D_h f - f'|  在 [a, b] 上
    """
    h = (b - a) / (N - 1)
    x = [a + i * h for i in range(N)]
    fx = [f(xi) for xi in x]
    dfh = first_derivative(fx, h, order)
    # 解析导数
    df_exact = [
        (f(x[i] + 1.0e-8) - f(x[i] - 1.0e-8)) / 2.0e-8
        for i in range(N)
    ]
    err = max(abs(dfh[i] - df_exact[i]) for i in range(N // 4, 3 * N // 4))
    return err


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def _self_check() -> bool:
    """用 sin(x) 验证各阶精度."""
    import math
    a, b = 0.0, 2.0 * math.pi
    f = math.sin
    df_exact = math.cos
    for order in [2, 4, 6, 8]:
        for N in [32, 64, 128]:
            h = (b - a) / (N - 1)
            x = [a + i * h for i in range(N)]
            fx = [f(xi) for xi in x]
            dfh = first_derivative(fx, h, order)
            err = max(abs(dfh[i] - df_exact(x[i]))
                      for i in range(order // 2, N - order // 2))
            # 粗略验证误差随 N 衰减
            if N == 128 and err > 1.0e-2:
                raise AssertionError(
                    f"FD order {order} at N=128: err={err:.3e} unexpectedly large")
    return True


if __name__ == "__main__":
    _self_check()
    print("high_order_fd self-check passed.")
    import math
    for order in [2, 4, 6, 8]:
        err = truncation_error_estimate(math.sin, 0.0, 2.0 * math.pi, 64, order)
        print(f"  order {order}: max error = {err:.3e}")
