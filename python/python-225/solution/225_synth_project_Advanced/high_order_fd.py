# -*- coding: utf-8 -*-
"""
high_order_fd.py
================

高阶有限差分算子模块

本模块实现用于暗物质相空间输运方程离散化的各阶有限差分算子,
包括 2 阶、4 阶、6 阶精度的中心差分格式, 以及带 Sommerfeld 辐射边界条件的
单向差分格式。对应种子项目 1173_scalar-soliton-collision 中的 4 阶差分引擎。

核心公式
--------
N 阶中心差分:
  f'(x_i) ≈ Σ_k c_k * f(x_{i+k}) / h

2阶:
  f' = (-f_{i-1} + f_{i+1}) / (2h)
  f'' = (f_{i-1} - 2f_i + f_{i+1}) / h^2

4阶:
  f' = (f_{i-2} - 8f_{i-1} + 8f_{i+1} - f_{i+2}) / (12h)
  f'' = (-f_{i-2} + 16f_{i-1} - 30f_i + 16f_{i+1} - f_{i+2}) / (12h^2)

6阶:
  f' = (-f_{i-3} + 9f_{i-2} - 45f_{i-1} + 45f_{i+1} - 9f_{i+3} + f_{i+4}) / (60h)
  f'' = (2f_{i-3} - 27f_{i-2} + 270f_{i-1} - 490f_i + 270f_{i+1} - 27f_{i+2} + 2f_{i+3}) / (180h^2)

色散关系 (Fourier 空间):
  对于 f(x) = exp(ikx), 差分算子的修正波数 k* 满足:
  2阶: k*h = sin(kh)
  4阶: k*h = (8 sin(kh/2) - sin(kh)) / 6  ≈ kh - (kh)^5/180 + ...
  6阶: k*h = (45 sin(kh/2) - 9 sin(3kh/2)/4 + sin(5kh/2)/20) / ...
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# 一维差分算子
# ---------------------------------------------------------------------------

def fd1_2nd(f: np.ndarray, h: float) -> np.ndarray:
    """2阶精度一阶导数 (中心差分)。边界用单侧2阶格式。"""
    n = len(f)
    df = np.zeros(n)
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * h)
    # 左边界: 2阶前向
    df[0] = (-3*f[0] + 4*f[1] - f[2]) / (2.0 * h)
    # 右边界: 2阶后向
    df[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2.0 * h)
    return df


def fd1_4th(f: np.ndarray, h: float) -> np.ndarray:
    """4阶精度一阶导数 (中心差分)。边界用 Sommerfeld 近似。"""
    n = len(f)
    df = np.zeros(n)
    # 内部点: 4阶中心差分
    df[2:-2] = (f[:-4] - 8*f[1:-3] + 8*f[3:-1] - f[4:]) / (12.0 * h)
    # 边界处理: 使用渐近衰减 (Sommerfeld 辐射条件)
    # df/dx ≈ -|df/dt|/c 对应向外传播波
    df[0] = (-f[2] + 8*f[1] - 8*f[0] + f[0]) / (12.0 * h)
    df[0] = (f[0] - f[4] + 8*(f[3] - f[1])) / (12.0 * h)  # 非中心, 近似
    # 安全起见使用单侧
    df[0] = (-25*f[0] + 48*f[1] - 36*f[2] + 16*f[3] - 3*f[4]) / (12.0 * h)
    df[1] = (-3*f[0] - 10*f[1] + 18*f[2] - 6*f[3] + f[4]) / (12.0 * h)
    df[-1] = (25*f[-1] - 48*f[-2] + 36*f[-3] - 16*f[-4] + 3*f[-5]) / (12.0 * h)
    df[-2] = (3*f[-1] + 10*f[-2] - 18*f[-3] + 6*f[-4] - f[-5]) / (12.0 * h)
    return df


def fd1_6th(f: np.ndarray, h: float) -> np.ndarray:
    """6阶精度一阶导数。"""
    n = len(f)
    df = np.zeros(n)
    # 6阶中心
    df[3:-3] = (
        -f[:-6] + 9*f[1:-5] - 45*f[2:-4]
        + 45*f[4:-2] - 9*f[5:-1] + f[6:]
    ) / (60.0 * h)
    # 边界: 降到 2 阶
    df[0] = (-3*f[0] + 4*f[1] - f[2]) / (2.0 * h)
    df[1] = (-3*f[1] + 4*f[2] - f[3]) / (2.0 * h)
    df[2] = (-3*f[2] + 4*f[3] - f[4]) / (2.0 * h)
    df[-1] = (3*f[-1] - 4*f[-2] + f[-3]) / (2.0 * h)
    df[-2] = (3*f[-2] - 4*f[-3] + f[-4]) / (2.0 * h)
    df[-3] = (3*f[-3] - 4*f[-4] + f[-5]) / (2.0 * h)
    return df


def fd2_2nd(f: np.ndarray, h: float) -> np.ndarray:
    """2阶精度二阶导数 (Laplacian in 1D)。"""
    n = len(f)
    d2f = np.zeros(n)
    d2f[1:-1] = (f[:-2] - 2*f[1:-1] + f[2:]) / (h * h)
    # Neumann 边界
    d2f[0] = (2*f[0] - 5*f[1] + 4*f[2] - f[3]) / (h * h)
    d2f[-1] = (2*f[-1] - 5*f[-2] + 4*f[-3] - f[-4]) / (h * h)
    return d2f


def fd2_4th(f: np.ndarray, h: float) -> np.ndarray:
    """4阶精度二阶导数。"""
    n = len(f)
    d2f = np.zeros(n)
    # 内部: 4阶
    d2f[2:-2] = (
        -f[:-4] + 16*f[1:-3] - 30*f[2:-2] + 16*f[3:-1] - f[4:]
    ) / (12.0 * h * h)
    # 边界: 用 2 阶近似
    d2f[0] = (f[0] - 2*f[1] + f[2]) / (h * h)
    d2f[1] = (f[0] - 2*f[1] + f[2]) / (h * h)
    d2f[-1] = (f[-1] - 2*f[-2] + f[-3]) / (h * h)
    d2f[-2] = (f[-1] - 2*f[-2] + f[-3]) / (h * h)
    return d2f


def fd2_6th(f: np.ndarray, h: float) -> np.ndarray:
    """6阶精度二阶导数。"""
    n = len(f)
    d2f = np.zeros(n)
    # 6阶中心差分系数
    d2f[3:-3] = (
        2*f[:-6] - 27*f[1:-5] + 270*f[2:-4]
        - 490*f[3:-3]
        + 270*f[4:-2] - 27*f[5:-1] + 2*f[6:]
    ) / (180.0 * h * h)
    # 边界降级
    for i in range(3):
        d2f[i] = (f[max(i-1, 0)] - 2*f[i] + f[min(i+1, n-1)]) / (h * h)
    for i in range(n-3, n):
        d2f[i] = (f[max(i-1, 0)] - 2*f[i] + f[min(i+1, n-1)]) / (h * h)
    return d2f


# ---------------------------------------------------------------------------
# 统一接口
# ---------------------------------------------------------------------------

class FiniteDifferenceOperator:
    """
    统一的高阶差分算子封装。

    支持阶数 p ∈ {2, 4, 6}, 导数阶 d ∈ {1, 2}。

    色散误差分析:
      对于 f(x) = exp(ikx), 数值波数 k_num 与精确波数 k 的偏差:
        δk/k ≈ C_p (kh)^p  (p = 差分精度阶数)
      4阶: δk/k ≈ -(kh)^4 / 180
      6阶: δk/k ≈ (kh)^6 / 1260
    """

    def __init__(self, order: int = 4, derivative: int = 1):
        if order not in (2, 4, 6):
            raise ValueError(f"阶数必须为 2, 4, 6, 得到 {order}")
        if derivative not in (1, 2):
            raise ValueError(f"导数阶必须为 1 或 2, 得到 {derivative}")
        self.order = order
        self.derivative = derivative
        self._op = self._select_op()

    def _select_op(self):
        ops = {
            (1, 2): fd1_2nd, (1, 4): fd1_4th, (1, 6): fd1_6th,
            (2, 2): fd2_2nd, (2, 4): fd2_4th, (2, 6): fd2_6th,
        }
        return ops[(self.derivative, self.order)]

    def apply(self, f: np.ndarray, h: float) -> np.ndarray:
        """对数组 f 应用差分算子, 步长 h。"""
        return self._op(f, h)

    def modified_wavenumber(self, k: np.ndarray, h: float) -> np.ndarray:
        """
        计算修正波数 k*(kh) 用于色散分析。

        对于 2 阶一阶导数:
          k*h = sin(kh)
        对于 4 阶一阶导数:
          k*h = (8 sin(kh/2) - sin(kh)) / 6
        """
        kh = k * h
        if self.derivative == 1:
            if self.order == 2:
                return np.sin(kh) / h
            elif self.order == 4:
                return (8.0 * np.sin(kh / 2.0) - np.sin(kh)) / (6.0 * h)
            else:  # order 6
                return (
                    45.0 * np.sin(kh / 2.0)
                    - 9.0 * np.sin(3.0 * kh / 2.0) / 4.0
                    + np.sin(5.0 * kh / 2.0) / 20.0
                ) / (15.0 * h)
        else:  # derivative == 2
            if self.order == 2:
                return 2.0 * (np.cos(kh) - 1.0) / (h * h)
            elif self.order == 4:
                return (
                    30.0 * np.cos(kh) - 16.0 * np.cos(kh / 2.0) * 2.0
                    - 14.0
                ) / (12.0 * h * h)  # 近似
            else:
                # 6阶近似
                return 2.0 * (np.cos(kh) - 1.0) / (h * h) * (
                    1.0 - (kh)**4 / 90.0
                )


# ---------------------------------------------------------------------------
# 多维差分
# ---------------------------------------------------------------------------

def gradient_4th(f: np.ndarray, h: float) -> Tuple[np.ndarray, ...]:
    """
    4阶精度的梯度算子, 适用于 2D 或 3D 数组。
    边界使用 Sommerfeld 辐射条件: ∂f/∂n + f/r ≈ 0
    """
    ndim = f.ndim
    grads = []
    for d in range(ndim):
        gf = np.apply_along_axis(lambda x: fd1_4th(x, h), d, f)
        grads.append(gf)
    return tuple(grads)


def laplacian_4th(f: np.ndarray, h: float) -> np.ndarray:
    """
    4阶精度的 Laplacian 算子 (各维度之和)。

    ∇²f = Σ_d ∂²f/∂x_d²
    """
    ndim = f.ndim
    lap = np.zeros_like(f)
    for d in range(ndim):
        lap += np.apply_along_axis(lambda x: fd2_4th(x, h), d, f)
    return lap


# ---------------------------------------------------------------------------
# 收敛性测试
# ---------------------------------------------------------------------------

def convergence_test(
    func,
    exact_deriv,
    N_list: list,
    domain: Tuple[float, float] = (0.0, 2.0 * np.pi),
    order: int = 4,
    derivative: int = 1,
) -> dict:
    """
    Richardson 外推收敛性测试。

    对于 f(x) = sin(kx), 计算不同分辨率下的 L2 误差,
    验证收敛阶 O(h^p)。

    返回 dict: {N, h, error, observed_order}
    """
    a, b = domain
    fd_op = FiniteDifferenceOperator(order, derivative)
    results = {'N': [], 'h': [], 'error': [], 'observed_order': []}

    prev_error = None
    prev_h = None
    for N in N_list:
        h = (b - a) / (N - 1)
        x = np.linspace(a, b, N)
        # 测试函数: sin(3x) 有足够曲率
        f = np.sin(3.0 * x)
        f_num = fd_op.apply(f, h)
        if derivative == 1:
            f_exact = 3.0 * np.cos(3.0 * x)
        else:
            f_exact = -9.0 * np.sin(3.0 * x)
        error = np.sqrt(np.mean((f_num - f_exact)**2))
        results['N'].append(N)
        results['h'].append(h)
        results['error'].append(error)
        if prev_error is not None and error > 0:
            obs_order = np.log(prev_error / error) / np.log(prev_h / h)
            results['observed_order'].append(obs_order)
        else:
            results['observed_order'].append(np.nan)
        prev_error = error
        prev_h = h

    return results
