"""
finite_diff.py
==============

高阶有限差分算子 —— Profile Likelihood 数值微分核心。

种子项目 1435 (zoomin) 与 658 (Lebesgue) 启发: 将求根方法的
高阶导数思想推广至有限差分微商计算。

有限差分公式 (中心差分, 各阶):

一阶导数 O(h²):
    f'(x) ≈ [f(x+h) - f(x-h)] / (2h)

一阶导数 O(h⁴):
    f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)

一阶导数 O(h⁶):
    f'(x) ≈ [f(x+3h) - 9f(x+2h) + 45f(x+h) - 45f(x-h) + 9f(x-2h) - f(x-3h)] / (60h)

二阶导数 O(h²):
    f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²

二阶导数 O(h⁴):
    f''(x) ≈ [-f(x+2h) + 16f(x+h) - 30f(x) + 16f(x-h) - f(x-2h)] / (12h²)

三阶导数 O(h²):
    f'''(x) ≈ [f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)] / (2h³)

系数生成 (Fornberg 算法, seed 658 → Lebesgue 稳定节点推广):
    给定节点 z_0, ..., z_n 和导数阶 m,
    求权重 w_0, ..., w_n 使得 Σ_j w_j f(z_j) ≈ f^{(m)}(z_0).

稳定性分析:
    Lebesgue 常数 Λ_n 控制差分算子的放大因子:
    ||D_h f|| ≤ Λ_n · ||f||_∞ / h^m
"""

from __future__ import annotations
import math
import numpy as np
from typing import Callable, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Fornberg 算法: 任意节点、任意阶导数的有限差分权重
# ---------------------------------------------------------------------------
def fornberg_weights(
    z: np.ndarray, x0: float, m: int
) -> np.ndarray:
    """
    Fornberg (1988) 算法计算有限差分权重。

    给定节点 z = [z_0, ..., z_n], 计算权重 w 使得:
        f^{(m)}(x0) ≈ Σ_j w_j · f(z_j)

    精度: O(h^{n+1-m})

    Parameters
    ----------
    z : ndarray, shape (n+1,)
        差分节点 (可不等距)
    x0 : float
        求导点
    m : int
        导数阶数

    Returns
    -------
    ndarray, shape (n+1,)
        有限差分权重

    Algorithm (Fornberg 1988, Math. Comp. 51, 699-706):
        c_00 = 1
        for i = 1, ..., n:
            for j = 0, ..., i-1:  // 递推构造
                ...
    """
    n = len(z) - 1
    if m > n:
        raise ValueError(f"导数阶 m={m} 超过节点数 n={n}")

    # 动态规划表
    d = np.zeros((n + 1, m + 1))
    d[0, 0] = 1.0

    c1 = 1.0
    for i in range(1, n + 1):
        c2 = 1.0
        for j in range(i):
            c3 = z[i] - z[j]
            if abs(c3) < 1e-15:
                raise ValueError(f"节点重合: z[{i}]={z[i]}, z[{j}]={z[j]}")
            c2 *= c3
            if i <= m:
                continue
            for k in range(min(i, m), 0, -1):
                d[i, k] = (
                    (z[i] - x0) * d[i - 1, k] - k * d[i - 1, k - 1]
                ) / c3
            d[i, 0] = (z[i] - x0) * d[i - 1, 0] / c3

        for k in range(min(i, m), 0, -1):
            d[i - 1, k] = (
                c1 * (k * d[i - 1, k - 1] - (z[i - 1] - x0) * d[i - 1, k])
            ) / c2
        d[i - 1, 0] = -c1 * (z[i - 1] - x0) * d[i - 1, 0] / c2
        c1 = c2

    return d[:, m]


# ---------------------------------------------------------------------------
# 预定义中心差分模板
# ---------------------------------------------------------------------------
def fd_coefficients_1st(order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    一阶导数中心差分系数。

    order=2: O(h²), 节点 [-1, 0, 1] → [-1/2, 0, 1/2] / h
    order=4: O(h⁴), 节点 [-2, -1, 0, 1, 2]
    order=6: O(h⁶), 节点 [-3, -2, -1, 0, 1, 2, 3]

    Returns
    -------
    (stencil_offsets, weights) : (ndarray, ndarray)
    """
    if order == 2:
        offsets = np.array([-1, 0, 1], dtype=np.float64)
        weights = np.array([-0.5, 0.0, 0.5])
    elif order == 4:
        offsets = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        weights = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
    elif order == 6:
        offsets = np.array([-3, -2, -1, 0, 1, 2, 3], dtype=np.float64)
        weights = np.array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
    elif order == 8:
        offsets = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float64)
        weights = np.array([1, -8, 56, -224, 0, 224, -56, 8, -1]) / 280.0 / 2.0
        # 校正: 实际 8 阶中心差分系数
        weights = np.array([
            -1.0 / 280, 8.0 / 280, -56.0 / 280, 56.0 * 4.0 / 280,
            0.0,
            -56.0 * 4.0 / 280, 56.0 / 280, -8.0 / 280, 1.0 / 280
        ])
        # 使用 Fornberg 直接计算更可靠
        offsets, weights = _fornberg_centered(1, 4)
    else:
        raise ValueError(f"不支持的精度阶: order={order}")
    return offsets, weights


def fd_coefficients_2nd(order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    二阶导数中心差分系数。

    order=2: [1, -2, 1] / h^2
    order=4: [-1, 16, -30, 16, -1] / (12 h^2)
    order=6: [从 Fornberg]
    """
    if order == 2:
        offsets = np.array([-1, 0, 1], dtype=np.float64)
        weights = np.array([1.0, -2.0, 1.0])
    elif order == 4:
        offsets = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        weights = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0
    elif order == 6:
        offsets, weights = _fornberg_centered(2, 3)
    else:
        raise ValueError(f"不支持的精度阶: order={order}")
    return offsets, weights


def fd_coefficients_3rd(order: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    三阶导数中心差分系数。

    order=2: [1, -2, 0, 2, -1] / (2 h^3)
    """
    if order == 2:
        offsets = np.array([-2, -1, 0, 1, 2], dtype=np.float64)
        weights = np.array([1.0, -2.0, 0.0, 2.0, -1.0]) / 2.0
    else:
        offsets, weights = _fornberg_centered(3, max(order // 2 + 1, 3))
    return offsets, weights


def _fornberg_centered(m: int, half_width: int) -> Tuple[np.ndarray, np.ndarray]:
    """用 Fornberg 算法生成对称中心差分权重。"""
    n_pts = 2 * half_width + 1
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    nodes = offsets.copy()
    weights = fornberg_weights(nodes, 0.0, m)
    return offsets, weights


# ---------------------------------------------------------------------------
# 有限差分求值器
# ---------------------------------------------------------------------------
class FiniteDiff:
    """
    高阶有限差分求导器。

    对给定函数 f 在点 x 处计算 f^{(m)}(x) 的近似。
    支持 Richardson 外推自动提升精度 (seed 1435 → 高阶收敛)。

    公式:
        D_h^{(m)} f(x) = (1/h^m) Σ_j w_j f(x + j·h)
    """

    def __init__(self, m: int = 1, fd_order: int = 4):
        """
        Parameters
        ----------
        m : int
            导数阶数 (1, 2, 3)
        fd_order : int
            差分精度阶 (2, 4, 6, 8)
        """
        self.m = m
        self.fd_order = fd_order
        if m == 1:
            self.offsets, self.weights = fd_coefficients_1st(fd_order)
        elif m == 2:
            self.offsets, self.weights = fd_coefficients_2nd(fd_order)
        elif m == 3:
            self.offsets, self.weights = fd_coefficients_3rd(fd_order)
        else:
            # 使用 Fornberg 通用
            hw = max(m + fd_order // 2, m + 1)
            self.offsets, self.weights = _fornberg_centered(m, hw)

    def __call__(self, f: Callable[[float], float], x: float, h: float) -> float:
        """
        计算 f^{(m)}(x) 的有限差分近似。

        Parameters
        ----------
        f : callable
            一元函数
        x : float
            求导点
        h : float
            步长

        Returns
        -------
        float
            f^{(m)}(x) 的近似值
        """
        if h <= 0.0:
            raise ValueError(f"步长必须为正: h={h}")
        result = 0.0
        for off, w in zip(self.offsets, self.weights):
            result += w * f(x + off * h)
        return result / (h ** self.m)


# ---------------------------------------------------------------------------
# 多元有限差分 (Profile NLL 关于 μ 的导数)
# ---------------------------------------------------------------------------
class FiniteDiffND:
    """
    N 维有限差分求导器, 用于 ∂f/∂x_k 的逐维计算。

    对每个维度使用一维差分模板, 其余维度固定。
    """

    def __init__(self, fd_order: int = 4):
        self.fd1d = FiniteDiff(m=1, fd_order=fd_order)

    def gradient(
        self,
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        """
        梯度 ∇f = (∂f/∂x_1, ..., ∂f/∂x_n)

        Parameters
        ----------
        f : callable, R^n → R
        x : ndarray, shape (n,)
        h : ndarray, shape (n,)
            各维步长

        Returns
        -------
        ndarray, shape (n,)
        """
        x = np.asarray(x, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        n = len(x)
        grad = np.zeros(n)
        for k in range(n):
            def f_1d(t):
                x_shift = x.copy()
                x_shift[k] = t
                return f(x_shift)
            grad[k] = self.fd1d(f_1d, x[k], h[k])
        return grad

    def hessian_diag(
        self,
        f: Callable[[np.ndarray], float],
        x: np.ndarray,
        h: np.ndarray,
    ) -> np.ndarray:
        """
        对角 Hessian (∂²f/∂x_k²):

            H_kk ≈ [f(x + h_k e_k) - 2f(x) + f(x - h_k e_k)] / h_k²
        """
        x = np.asarray(x, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        n = len(x)
        f0 = f(x)
        hdiag = np.zeros(n)
        for k in range(n):
            xp = x.copy(); xp[k] += h[k]
            xm = x.copy(); xm[k] -= h[k]
            hdiag[k] = (f(xp) - 2.0 * f0 + f(xm)) / (h[k] ** 2)
        return hdiag


# ---------------------------------------------------------------------------
# 复步微分 (Lynn's formula, 无相消误差)
# ---------------------------------------------------------------------------
def complex_step_derivative(
    f: Callable[[complex], complex], x: float, h: float = 1e-20
) -> float:
    """
    复步微分: f'(x) ≈ Im[f(x + ih)] / h

    优势: 无相消误差, 精度达机器精度 (~1e-15) 对任意 h ~ O(1)。
    要求 f 可解析延拓至复平面。

    (seed 1435 → 高阶求根方法中导数计算的替代方案)
    """
    if h <= 0.0:
        raise ValueError(f"步长必须为正: h={h}")
    z = complex(x, h)
    fz = f(z)
    return fz.imag / h


# ---------------------------------------------------------------------------
# 快速自检
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 测试 Fornberg 权重
    nodes = np.array([-1.0, 0.0, 1.0])
    w = fornberg_weights(nodes, 0.0, 1)
    print(f"一阶导数权重 (3节点): {w}")  # 应 ≈ [-0.5, 0, 0.5]

    # 测试有限差分
    import math
    f_sin = lambda x: math.sin(x)
    fd1 = FiniteDiff(m=1, fd_order=4)
    df = fd1(f_sin, 0.5, 0.01)
    print(f"d/dx sin(x)|_{{x=0.5}} = {df:.10f}, exact = {math.cos(0.5):.10f}")

    fd2 = FiniteDiff(m=2, fd_order=4)
    d2f = fd2(f_sin, 0.5, 0.01)
    print(f"d²/dx² sin(x)|_{{x=0.5}} = {d2f:.10f}, exact = {-math.sin(0.5):.10f}")

    # 测试复步微分
    f_cx = lambda z: np.sin(z)
    df_cs = complex_step_derivative(f_cx, 0.5)
    print(f"复步微分 d/dx sin(x)|_{{x=0.5}} = {df_cs:.15f}")
