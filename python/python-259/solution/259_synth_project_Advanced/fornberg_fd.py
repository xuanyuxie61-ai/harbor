# -*- coding: utf-8 -*-
"""
fornberg_fd.py — Fornberg 算法: 任意阶有限差分权重 (等距 / 非等距网格)

本模块是本项目数值精度的核心. BAO 峰位于 ξ(s) ≈ 150 Mpc/h 处, 要精确解析
这个峰的二阶导数 (用于 "曲率法" 定位峰位), 必须在非均匀共动网格上计算
高阶导数. 我们采用 Fornberg (1988, Math. Comp. 51, 699) 的 O(M^N) 算法
生成任意阶导数在任意离散点集上的有限差分权重.

种子项目映射
----------
- 198 collatz_polynomial : Collatz 多项式的"递推生成"思想被移植到 Fornberg
  权重表的"逐阶构造".
- 990 r8poly             : 多项式代数 (Lagrange 基、求值、微分) 是 Fornberg
  算法的数学对偶.
"""

from __future__ import annotations
from typing import Tuple
import math
import numpy as np


def fornberg_weights(x_center: float, x_nodes: np.ndarray, max_deriv: int
                     ) -> np.ndarray:
    """
    返回形为 W[max_deriv+1, N+1] 的权重矩阵.

    Parameters
    ----------
    x_center  : 待求导数的中心点
    x_nodes   : (N+1,) 离散节点坐标 (可非均匀)
    max_deriv : 需要的最大导数阶数

    Returns
    -------
    W : (max_deriv+1, N+1) 权重矩阵
    """
    N = len(x_nodes) - 1
    if max_deriv > N:
        raise ValueError(f"max_deriv={max_deriv} 不能超过 N={N}")
    M = max_deriv
    c = np.zeros((M + 1, N + 1), dtype=np.float64)
    c1 = 1.0
    c4 = x_nodes[0] - x_center
    c[0, 0] = 1.0
    for i in range(1, N + 1):
        mn = min(i, M)
        c2 = 1.0
        c5 = c4
        c4 = x_nodes[i] - x_center
        for j in range(i):
            c3 = x_nodes[i] - x_nodes[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c


def fornberg_weights_batch(x_centers: np.ndarray, x_nodes: np.ndarray,
                           deriv: int) -> np.ndarray:
    """对一批中心点返回形状为 (len(x_centers), len(x_nodes)) 的权重矩阵."""
    out = np.zeros((len(x_centers), len(x_nodes)), dtype=np.float64)
    for i, xc in enumerate(x_centers):
        W = fornberg_weights(xc, x_nodes, deriv)
        out[i, :] = W[deriv, :]
    return out


def build_derivative_matrix_1d(x_grid: np.ndarray, deriv: int,
                               stencil: int = 5,
                               boundary: str = "one-sided") -> np.ndarray:
    """
    构造 1D 网格 x_grid 上的有限差分矩阵 D.
    """
    N = len(x_grid)
    if stencil % 2 == 0:
        raise ValueError("stencil 必须为奇数")
    half = stencil // 2
    D = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        if i - half < 0:
            lo, hi = 0, stencil
        elif i + half + 1 > N:
            lo, hi = N - stencil, N
        else:
            lo, hi = i - half, i + half + 1
        nodes = x_grid[lo:hi]
        W = fornberg_weights(x_grid[i], nodes, deriv)
        D[i, lo:hi] = W[deriv, :]
    return D


def build_derivative_matrix_2d(x: np.ndarray, y: np.ndarray,
                               deriv_x: int, deriv_y: int,
                               stencil: int = 5
                               ) -> Tuple[np.ndarray, int, int]:
    """构造矩形网格上的张量积差分矩阵."""
    Dx = build_derivative_matrix_1d(x, deriv_x, stencil)
    Dy = build_derivative_matrix_1d(y, deriv_y, stencil)
    Nx, Ny = len(x), len(y)
    if deriv_x == 0:
        D = np.kron(Dy, np.eye(Nx))
    elif deriv_y == 0:
        D = np.kron(np.eye(Ny), Dx)
    else:
        D = np.kron(Dy, Dx)
    return D, Nx, Ny


def truncation_error_estimate(W: np.ndarray, x_nodes: np.ndarray,
                              x_center: float, deriv: int, func) -> float:
    """估计有限差分近似的截断误差."""
    if deriv + 2 > len(x_nodes) - 1:
        return float("nan")
    W_high = fornberg_weights(x_center, x_nodes, deriv + 2)
    fvals = np.array([func(x) for x in x_nodes])
    fd_high = np.dot(W_high[deriv, :], fvals)
    fd_low = np.dot(W[deriv, :], fvals)
    return abs(fd_high - fd_low)


def _self_check() -> None:
    print("[fornberg_fd] 自检: cos(x) 各阶导数")
    x_nodes = np.linspace(-1.0, 1.0, 9)
    xc = 0.0
    W = fornberg_weights(xc, x_nodes, 4)
    fvals = np.cos(x_nodes)
    exact = [math.cos(xc), -math.sin(xc), -math.cos(xc), math.sin(xc), math.cos(xc)]
    for m in range(5):
        approx = np.dot(W[m, :], fvals)
        err = abs(approx - exact[m])
        print(f"  m={m}: approx={approx:+.6e}, exact={exact[m]:+.6e}, err={err:.2e}")


if __name__ == "__main__":
    _self_check()
