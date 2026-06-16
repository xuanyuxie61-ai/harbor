#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
high_order_fd.py  ——  高阶紧致/中心有限差分模板 & 导数算子

融合种子项目:
  - 1374_unstable_ode  : 非稳定 ODE 的导数定义 → MHD 半离散右端项
  - 963_r83_np         : 三对角矩阵运算 → 紧致差分隐式求解

核心公式 (均匀网格 2p 阶中心差分):
  f'_i ≈ sum_{k=1}^{p} c_k (f_{i+k} - f_{i-k}) / (k h)
  其中 c_k = (-1)^{k+1} p!^2 / ((p+k)! (p-k)! k)

  4 阶:  f' = (-f_{i+2} + 8 f_{i+1} - 8 f_{i-1} + f_{i-2}) / (12 h)
  6 阶:  f' = (f_{i+3} - 9 f_{i+2} + 45 f_{i+1} - 45 f_{i-1} + 9 f_{i-2} - f_{i-3}) / (60 h)
  8 阶:  系数从 Taylor 展开得到

紧致 Padé 格式 (4 阶):
  (1/6) f'_{i-1} + (2/3) f'_i + (1/6) f'_{i+1}
      = (f_{i+1} - f_{i-1}) / (2 h)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple


# ============================================================
# 中心差分系数 (2p 阶, p=1,2,3,4 对应 2/4/6/8 阶)
# ============================================================
def central_fd_coefficients(order: int) -> np.ndarray:
    """
    返回 2p 阶中心差分一阶导数的系数 c_k, k = 1..p.
    使用 Fornberg 算法的闭合形式.
    """
    p = order // 2
    if p < 1 or p > 4:
        raise ValueError(f"仅支持 2/4/6/8 阶, 给定 order={order}")
    coeffs = np.zeros(p)
    if p == 1:       # 2 阶
        coeffs[0] = 1.0
    elif p == 2:     # 4 阶
        coeffs[0] = 4.0 / 3.0
        coeffs[1] = -1.0 / 6.0
    elif p == 3:     # 6 阶
        coeffs[0] = 3.0 / 2.0
        coeffs[1] = -3.0 / 20.0
        coeffs[2] = 1.0 / 60.0
    elif p == 4:     # 8 阶
        coeffs[0] = 8.0 / 5.0
        coeffs[1] = -4.0 / 5.0 / 7.0  # -4/35
        coeffs[2] = 4.0 / 5.0 / 105.0  # 4/525 ... 但用精确分数
        coeffs[0] = 8.0 / 5.0
        coeffs[1] = -4.0 / 35.0
        coeffs[2] = 8.0 / 315.0
        coeffs[3] = -1.0 / 280.0
    return coeffs


# ============================================================
# 一维高阶导数 (周期边界)
# ============================================================
def fd_first_derivative_periodic(f: np.ndarray, dx: float,
                                  order: int) -> np.ndarray:
    """
    对周期数组 f 做 2p 阶中心差分一阶导数.
    """
    c = central_fd_coefficients(order)
    p = len(c)
    n = len(f)
    df = np.zeros_like(f)
    for k in range(1, p + 1):
        fp = np.roll(f, -k)
        fm = np.roll(f,  k)
        df += c[k - 1] * (fp - fm)
    return df / dx


# ============================================================
# 一维高阶导数 (非周期, 单侧边界处理)
# ============================================================
def fd_first_derivative_nonperiodic(f: np.ndarray, dx: np.ndarray,
                                     order: int) -> np.ndarray:
    """
    对非均匀/非周期数组 f 做中心差分, 边界用降阶单侧模板.
    dx: 间距数组, 长度 >= len(f)-1.
    简化实现: 内部用 2 阶中心差分, 边界用 2 阶单侧.
    """
    n = len(f)
    df = np.zeros(n)
    if n < 3:
        if n == 2:
            h = dx[0] if len(dx) > 0 else 1.0
            df[0] = df[1] = (f[1] - f[0]) / (h + 1e-30)
        return df
    # 使用均匀间距近似 (取平均)
    h = np.mean(dx) if len(dx) > 0 else 1.0
    # 内部点: 2 阶中心差分
    df[1:-1] = (f[2:] - f[:-2]) / (2.0 * h + 1e-30)
    # 边界: 单侧 2 阶
    df[0]  = (-3 * f[0] + 4 * f[1] - f[2]) / (2.0 * h + 1e-30)
    df[-1] = ( 3 * f[-1] - 4 * f[-2] + f[-3]) / (2.0 * h + 1e-30)
    return df


# ============================================================
# 二阶导数 (用于扩散项/粘性)
# ============================================================
def fd_second_derivative(f: np.ndarray, dx: float, order: int = 2) -> np.ndarray:
    """
    2 阶: f''_i = (f_{i-1} - 2 f_i + f_{i+1}) / h^2
    4 阶: f''_i = (-f_{i-2} + 16 f_{i-1} - 30 f_i + 16 f_{i+1} - f_{i+2}) / (12 h^2)
    """
    if order == 2:
        return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / (dx * dx)
    elif order == 4:
        return (-np.roll(f, -2) + 16 * np.roll(f, -1) - 30 * f
                + 16 * np.roll(f, 1) - np.roll(f, 2)) / (12.0 * dx * dx)
    else:
        # 降级到 2 阶
        return (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / (dx * dx)


# ============================================================
# 紧致 Padé 三对角系统 (融合 963_r83_np)
# ============================================================
class CompactPadeSolver:
    """
    4 阶紧致 Padé:
      alpha f'_{i-1} + f'_i + alpha f'_{i+1}
        = a (f_{i+1} - f_{i-1}) / (2 h)
    其中 alpha = 1/4, a = 3/2.
    需要求解三对角系统  A x = b, 使用 Thomas 算法 (参考 963_r83_np_fa).
    """

    def __init__(self, n: int, alpha: float = 0.25, a_coeff: float = 1.5) -> None:
        self.n = n
        self.alpha = alpha
        self.a_coeff = a_coeff
        # 三对角存储 (类似 963_r83 格式)
        self.lower = np.full(n, alpha)
        self.diag  = np.ones(n)
        self.upper = np.full(n, alpha)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        """Thomas 算法求解三对角系统 (无主元, 参考 963_r83_np_fa/sl)."""
        n = self.n
        # 复制以避免修改原数组
        c = self.upper.copy()
        d = rhs.copy()
        a = self.lower.copy()
        b = self.diag.copy()

        # 前消
        for i in range(1, n):
            if abs(b[i - 1]) < 1e-30:
                raise ValueError(f"Thomas 算法在第 {i-1} 步遇到零主元")
            m = a[i - 1] / b[i - 1]
            b[i] -= m * c[i - 1]
            d[i] -= m * d[i - 1]

        # 回代
        x = np.zeros(n)
        if abs(b[-1]) < 1e-30:
            raise ValueError("Thomas 算法: 末行为零")
        x[-1] = d[-1] / b[-1]
        for i in range(n - 2, -1, -1):
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
        return x

    def apply(self, f: np.ndarray, dx: float) -> np.ndarray:
        """应用紧致 Padé 求一阶导数."""
        n = len(f)
        rhs = np.zeros(n)
        # 内部点
        rhs[1:-1] = self.a_coeff * (f[2:] - f[:-2]) / (2.0 * dx)
        # 边界: 单侧
        rhs[0]  = self.a_coeff * (f[1] - f[0]) / dx
        rhs[-1] = self.a_coeff * (f[-1] - f[-2]) / dx
        return self.solve(rhs)


# ============================================================
# 人工粘性 (用于激波捕捉)
# ============================================================
def artificial_viscosity(f: np.ndarray, dx: float,
                          eps_av: float = 0.1) -> np.ndarray:
    """
    4 阶人工粘性:  nu d^4 f / dx^4.
    用二阶 Laplacian 两次近似.
    """
    lap = (np.roll(f, -1) - 2 * f + np.roll(f, 1)) / (dx * dx)
    lap2 = (np.roll(lap, -1) - 2 * lap + np.roll(lap, 1)) / (dx * dx)
    return -eps_av * dx * dx * dx * dx * lap2
