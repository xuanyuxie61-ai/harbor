#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spectral_filter.py  ——  DCT 谱滤波 & 去混叠

融合种子项目:
  - 222_cosine_transform : 离散余弦正/逆变换 → MHD 变量的谱空间滤波

核心公式 (DCT-II / DCT-III):
  正变换 (DCT-II):
    C_k = sqrt(2/N) sum_{j=0}^{N-1} d_j cos(pi (2j+1) k / (2N))
  逆变换 (DCT-III):
    d_j = sqrt(2/N) [ C_0/2 + sum_{k=1}^{N-1} C_k cos(pi (2j+1) k / (2N)) ]

滤波:
  C_k -> sigma_k C_k,   sigma_k = exp(-alpha (k/N_cut)^p)
  其中 p=36 (指数衰减), N_cut = 2N/3 (去混叠).
"""

from __future__ import annotations
import math
import numpy as np
from typing import Optional


def dct_forward(d: np.ndarray) -> np.ndarray:
    """
    DCT-II 正变换 (复刻 222_cosine_transform_data.m).
    d: 输入数据 (N,).
    返回 C (N,).
    """
    n = len(d)
    c = np.zeros(n)
    for i in range(n):
        s = 0.0
        for j in range(n):
            c[i] += math.cos(math.pi * (2 * j + 1) * i / (2.0 * n)) * d[j]
    c *= math.sqrt(2.0 / n)
    return c


def dct_inverse(c: np.ndarray) -> np.ndarray:
    """
    DCT-III 逆变换 (复刻 222_cosine_transform_inverse.m).
    c: DCT 系数 (N,).
    返回 d (N,).
    """
    n = len(c)
    d = np.zeros(n)
    for i in range(n):
        s = c[0] / 2.0
        for j in range(1, n):
            s += math.cos(math.pi * (2 * i + 1) * j / (2.0 * n)) * c[j]
        d[i] = s * math.sqrt(2.0 / n)
    return d


# ============================================================
# 谱滤波器
# ============================================================
class SpectralFilter:
    """
    指数谱滤波器 (用于 MHD 非线性项的去混叠).
    sigma_k = exp(-alpha * (k / N_cut)^{order})
    """

    def __init__(self, n: int, alpha: float = 36.0,
                 order: int = 8, n_cut_frac: float = 2.0 / 3.0) -> None:
        self.n = n
        self.alpha = alpha
        self.order = order
        self.n_cut = max(int(n * n_cut_frac), 1)
        # 预计算滤波因子
        self.sigma = np.zeros(n)
        for k in range(n):
            if k <= self.n_cut:
                self.sigma[k] = math.exp(-alpha * (k / self.n_cut) ** order)
            else:
                self.sigma[k] = 0.0  # 截断高频

    def apply_1d(self, d: np.ndarray) -> np.ndarray:
        """对 1D 数组做 DCT 滤波."""
        c = dct_forward(d)
        c_filtered = c * self.sigma
        return dct_inverse(c_filtered)

    def apply_3d(self, U: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        对 3D 数组沿指定轴做 DCT 滤波.
        U: (Nr, Nt, Np).
        axis: 0=径向, 1=极向, 2=方位角.
        """
        result = U.copy()
        shape = U.shape
        if axis == 0:
            for j in range(shape[1]):
                for k in range(shape[2]):
                    result[:, j, k] = self.apply_1d(U[:, j, k])
        elif axis == 1:
            for i in range(shape[0]):
                for k in range(shape[2]):
                    result[i, :, k] = self.apply_1d(U[i, :, k])
        elif axis == 2:
            for i in range(shape[0]):
                for j in range(shape[1]):
                    result[i, j, :] = self.apply_1d(U[i, j, :])
        return result


# ============================================================
# 去混叠 (3/2 规则)
# ============================================================
def dealias_32_rule(U: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    3/2 去混叠: 将 U 零填充到 3N/2, 做非线性运算, 再截断回 N.
    这里简化为直接截断高频 (融合 222 的 DCT 思想).
    """
    n = U.shape[axis]
    n_cut = int(2 * n / 3)
    result = U.copy()
    if axis == 0:
        for j in range(U.shape[1]):
            for k in range(U.shape[2]):
                c = dct_forward(U[:, j, k])
                c[n_cut:] = 0.0
                result[:, j, k] = dct_inverse(c)
    return result
