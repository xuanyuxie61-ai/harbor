# -*- coding: utf-8 -*-
"""
coronal_basis.py
----------------
高阶基函数库: 6 节点 Lagrange 多项式 + WKB 近似基.

物理动机
--------
在日冕环本征模分析中, 标准有限差分可能不足以解析快速振荡的
Alfvén 波本征函数. 借鉴 WKB (Wentzel-Kramers-Brillouin) 近似,
引入相位修正基函数:

    phi_j(s) = psi_j(s) * exp( i int^s k_A(s') ds' )

其中 k_A(s) = omega / v_A(s) 为局部 Alfvén 波数, psi_j(s) 为
缓慢变化的包络.

6 节点 Lagrange 基 (来自 375_fem_basis_t6_display):
    在参考单元 xi in [-1, 1] 上, 6 个等距节点
    xi_j = -1 + 2 j / 5,  j = 0, ..., 5
    基函数 L_j(xi) = prod_{k != j} (xi - xi_k) / (xi_j - xi_k)
"""
from __future__ import annotations
import numpy as np


def lagrange_6_nodes() -> np.ndarray:
    """6 节点等距节点: xi in [-1, 1]."""
    return np.linspace(-1.0, 1.0, 6)


def lagrange_6_basis(xi: np.ndarray) -> np.ndarray:
    """6 节点 Lagrange 基函数值: L_j(xi), shape (xi.size, 6)."""
    nodes = lagrange_6_nodes()
    n_xi = xi.size
    L = np.ones((n_xi, 6))
    for j in range(6):
        for k in range(6):
            if k != j:
                L[:, j] *= (xi - nodes[k]) / (nodes[j] - nodes[k] + 1.0e-30)
    return L


def lagrange_6_deriv(xi: np.ndarray) -> np.ndarray:
    """6 节点 Lagrange 基导数 dL_j/dxi, shape (xi.size, 6)."""
    nodes = lagrange_6_nodes()
    n_xi = xi.size
    dL = np.zeros((n_xi, 6))
    for j in range(6):
        for m in range(6):
            if m == j:
                continue
            term = np.ones(n_xi) / (nodes[j] - nodes[m] + 1.0e-30)
            for k in range(6):
                if k != j and k != m:
                    term *= (xi - nodes[k]) / (nodes[j] - nodes[k] + 1.0e-30)
            dL[:, j] += term
    return dL


def wkb_phase(s_grid: np.ndarray, v_alfven: np.ndarray,
              omega: float) -> np.ndarray:
    """WKB 相位: phi(s) = int^s omega / v_A(s') ds'."""
    k_local = omega / (v_alfven + 1.0e-12)
    phi = np.zeros_like(s_grid)
    for i in range(1, s_grid.size):
        ds = s_grid[i] - s_grid[i - 1]
        phi[i] = phi[i - 1] + 0.5 * (k_local[i] + k_local[i - 1]) * ds
    return phi


def wkb_basis(s_grid: np.ndarray, v_alfven: np.ndarray,
              omega: float, n_basis: int) -> np.ndarray:
    """构造 WKB 修正基: phi_j(s) = L_j(xi(s)) * cos(phi(s)).

    返回 shape (s_grid.size, n_basis) 的基矩阵.
    """
    from scipy.interpolate import interp1d
    xi_grid = np.linspace(-1, 1, s_grid.size)
    L = lagrange_6_basis(xi_grid)[:, :n_basis]
    phi = wkb_phase(s_grid, v_alfven, omega)
    envelope = np.cos(phi)
    return L * envelope[:, None]


def modal_decomposition(u: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """将场 u(s) 投影到基上: u(s) ~ sum_j c_j phi_j(s).

    最小二乘求解 c = (B^T B)^{-1} B^T u.
    """
    B_T_B = basis.T @ basis + 1.0e-10 * np.eye(basis.shape[1])
    B_T_u = basis.T @ u
    return np.linalg.solve(B_T_B, B_T_u)


def reconstuct_from_modes(coef: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """从模态系数重构场: u(s) = sum c_j phi_j(s)."""
    return basis @ coef
