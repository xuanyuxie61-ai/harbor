"""
adaptive_fem_solver.py
======================
一维序参量自适应有限元求解模块。
融合来源：fem1d_adaptive（自适应网格细化有限元方法）。

物理模型：
    在阻挫磁体的平均场理论中，序参量 m(x) 满足 Ginzburg-Landau 型方程：
        -d/dx ( A(x) dm/dx ) + B(x) m(x) = F(x)
    其中 A(x) 为交换刚度，B(x) 为各向异性/温度相关参数，F(x) 为外场驱动项。

    本模块使用分段线性基函数（P1 元）在 [0,1] 上求解该边值问题，
    通过局部后验误差估计自适应加密网格，以精确捕捉畴壁过渡区。

    能量泛函：
        I[m] = ∫ [ A(x) (m')²/2 + B(x) m²/2 - F(x) m ] dx
    变分导数给出 Euler-Lagrange 方程。
"""

import numpy as np
from typing import Tuple, Callable, Optional
from utils import EPS_MACHINE, rms_norm


def basis_phi(x: float, xL: float, xR: float, derivative: bool = False) -> float:
    """
    一维 P1（分段线性）帽子基函数。
    在单元 [xL, xR] 上：
        φ_L(x) = (xR - x) / h
        φ_R(x) = (x - xL) / h
    此函数返回左端点基函数 φ_L 及其导数。
    """
    h = xR - xL
    if abs(h) < EPS_MACHINE:
        return 0.0
    if derivative:
        return -1.0 / h
    return (xR - x) / h


def basis_psi(x: float, xL: float, xR: float, derivative: bool = False) -> float:
    """右端点 P1 基函数 φ_R。"""
    h = xR - xL
    if abs(h) < EPS_MACHINE:
        return 0.0
    if derivative:
        return 1.0 / h
    return (x - xL) / h


def assemble_tridiagonal_system(
    nodes: np.ndarray,
    A_func: Callable[[float], float],
    B_func: Callable[[float], float],
    F_func: Callable[[float], float],
    m_left: float,
    m_right: float,
    bc_left: str = "dirichlet",
    bc_right: str = "dirichlet",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    组装全局三对角线性系统：
        a_left[i] * m_{i-1} + a_diag[i] * m_i + a_right[i] * m_{i+1} = rhs[i]

    参数
    ----
    nodes : np.ndarray
        节点坐标，长度为 n_nodes。
    A_func, B_func, F_func : callable
        系数函数 A(x), B(x), F(x)。
    m_left, m_right : float
        边界值或边界导数值。
    bc_left, bc_right : str
        边界条件类型："dirichlet" 或 "neumann"。

    返回
    ----
    a_left, a_diag, a_right, rhs : np.ndarray
        三对角系数与右端项。
    """
    n = nodes.size - 1  # 单元数
    N = nodes.size      # 节点数
    a_diag = np.zeros(N, dtype=float)
    a_left = np.zeros(N, dtype=float)
    a_right = np.zeros(N, dtype=float)
    rhs = np.zeros(N, dtype=float)

    for ie in range(n):
        xL = nodes[ie]
        xR = nodes[ie + 1]
        h = xR - xL
        if h <= 0.0:
            continue
        # 两节点高斯积分
        xq1 = 0.5 * (xL + xR) - h / (2.0 * np.sqrt(3.0))
        xq2 = 0.5 * (xL + xR) + h / (2.0 * np.sqrt(3.0))
        wq = 0.5

        for iq, xq in enumerate([xq1, xq2]):
            # TODO: Hole_3 — 实现 P1 有限元单元刚度矩阵组装
            # 需调用 A_func/B_func/F_func 获取系数，利用 basis_phi/basis_psi 计算形函数值与导数，
            # 并按高斯积分权重累加到 a_diag、a_left、a_right 和 rhs。
            # 关键公式（在单元 [xL,xR] 上，h = xR-xL，wq = 0.5）：
            #   a_diag[ie]   += h * wq * (A * phi_i' * phi_i' + B * phi_i * phi_i)
            #   a_diag[ie+1] += h * wq * (A * phi_j' * phi_j' + B * phi_j * phi_j)
            #   aij          += h * wq * (A * phi_i' * phi_j' + B * phi_i * phi_j)
            #   rhs[ie]      += h * wq * F * phi_i
            #   rhs[ie+1]    += h * wq * F * phi_j
            raise NotImplementedError("Hole_3: 请实现 assemble_tridiagonal_system 中的单元刚度矩阵组装")

    # 边界条件处理
    if bc_left == "dirichlet":
        a_diag[0] = 1.0
        a_right[0] = 0.0
        rhs[0] = m_left
        a_left[0] = 0.0
    else:  # neumann: -A m' = m_left at x=0
        rhs[0] += m_left

    if bc_right == "dirichlet":
        a_diag[-1] = 1.0
        a_left[-1] = 0.0
        rhs[-1] = m_right
        a_right[-1] = 0.0
    else:
        rhs[-1] += m_right

    return a_left, a_diag, a_right, rhs


def solve_tridiagonal(
    a_left: np.ndarray, a_diag: np.ndarray, a_right: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """
    Thomas 算法求解三对角系统。
    前向消去 + 回代，O(N) 复杂度。
    """
    n = rhs.size
    c_prime = np.zeros(n - 1, dtype=float)
    d_prime = np.zeros(n, dtype=float)
    c_prime[0] = a_right[0] / a_diag[0]
    d_prime[0] = rhs[0] / a_diag[0]

    for i in range(1, n - 1):
        denom = a_diag[i] - a_left[i] * c_prime[i - 1]
        if abs(denom) < EPS_MACHINE:
            denom = EPS_MACHINE
        c_prime[i] = a_right[i] / denom
        d_prime[i] = (rhs[i] - a_left[i] * d_prime[i - 1]) / denom

    denom = a_diag[n - 1] - a_left[n - 1] * c_prime[n - 2]
    if abs(denom) < EPS_MACHINE:
        denom = EPS_MACHINE
    d_prime[n - 1] = (rhs[n - 1] - a_left[n - 1] * d_prime[n - 2]) / denom

    x = np.zeros(n, dtype=float)
    x[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i + 1]
    return x


def refine_mesh_locally(
    nodes: np.ndarray,
    solution: np.ndarray,
    error_threshold: float = 0.01,
    max_nodes: int = 200,
) -> np.ndarray:
    """
    基于解的局部曲率进行自适应网格加密。
    误差估计器（简化后验估计）：
        η_e ≈ h_e² |m''(x_e)|
    其中二阶导数用有限差分近似。

    参数
    ----
    nodes : np.ndarray
        当前节点。
    solution : np.ndarray
        当前解。
    error_threshold : float
        误差阈值，超过则加密。
    max_nodes : int
        最大节点数限制。

    返回
    ----
    new_nodes : np.ndarray
        加密后的节点坐标。
    """
    n = nodes.size
    if n >= max_nodes:
        return nodes
    new_nodes = [nodes[0]]
    current_n = n
    for i in range(n - 1):
        h = nodes[i + 1] - nodes[i]
        # 二阶导数近似
        if i == 0 or i == n - 2:
            m_pp = 0.0
        else:
            m_pp = (solution[i + 2] - 2.0 * solution[i + 1] + solution[i]) / (h * h)
        eta = abs(h * h * m_pp)
        if eta > error_threshold and current_n < max_nodes:
            mid = 0.5 * (nodes[i] + nodes[i + 1])
            new_nodes.append(mid)
            new_nodes.append(nodes[i + 1])
            current_n += 1
        else:
            new_nodes.append(nodes[i + 1])
    return np.array(new_nodes)


def adaptive_fem_order_parameter(
    A_func: Callable[[float], float],
    B_func: Callable[[float], float],
    F_func: Callable[[float], float],
    m_left: float = 0.0,
    m_right: float = 1.0,
    n_initial: int = 8,
    max_refinements: int = 6,
    error_threshold: float = 0.005,
    max_nodes: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """
    自适应有限元求解一维 Ginzburg-Landau 型序参量方程。

    返回
    ----
    nodes : np.ndarray
        最终节点。
    solution : np.ndarray
        序参量解。
    energy_density : np.ndarray
        单元能量密度估计。
    history : list
        每次迭代的节点数与误差记录。
    """
    nodes = np.linspace(0.0, 1.0, n_initial + 1)
    history = []

    for step in range(max_refinements):
        aL, aD, aR, rhs = assemble_tridiagonal_system(
            nodes, A_func, B_func, F_func, m_left, m_right
        )
        sol = solve_tridiagonal(aL, aD, aR, rhs)
        # 后验误差估计
        n_nodes = nodes.size
        errors = np.zeros(n_nodes - 1)
        energy_density = np.zeros(n_nodes - 1)
        for ie in range(n_nodes - 1):
            h = nodes[ie + 1] - nodes[ie]
            if ie > 0 and ie < n_nodes - 2:
                m_pp = (sol[ie + 1] - 2.0 * sol[ie] + sol[ie - 1]) / ((nodes[ie] - nodes[ie - 1]) ** 2)
            else:
                m_pp = 0.0
            errors[ie] = abs(h * h * m_pp)
            # 单元能量密度 ≈ A (m')²/2 + B m²/2 - F m
            mid = 0.5 * (nodes[ie] + nodes[ie + 1])
            mp = (sol[ie + 1] - sol[ie]) / h
            energy_density[ie] = 0.5 * A_func(mid) * mp * mp + 0.5 * B_func(mid) * sol[ie] ** 2 - F_func(mid) * sol[ie]

        max_err = float(np.max(errors))
        history.append({"step": step, "n_nodes": n_nodes, "max_error": max_err})
        if max_err < error_threshold or n_nodes >= max_nodes:
            break
        nodes = refine_mesh_locally(nodes, sol, error_threshold, max_nodes)
        # 插值旧解到新网格作为初始猜测（可选，此处直接重新求解）

    # 最终求解
    aL, aD, aR, rhs = assemble_tridiagonal_system(
        nodes, A_func, B_func, F_func, m_left, m_right
    )
    solution = solve_tridiagonal(aL, aD, aR, rhs)
    n_nodes = nodes.size
    energy_density = np.zeros(n_nodes - 1)
    for ie in range(n_nodes - 1):
        h = nodes[ie + 1] - nodes[ie]
        mid = 0.5 * (nodes[ie] + nodes[ie + 1])
        mp = (solution[ie + 1] - solution[ie]) / h
        energy_density[ie] = 0.5 * A_func(mid) * mp * mp + 0.5 * B_func(mid) * solution[ie] ** 2 - F_func(mid) * solution[ie]

    return nodes, solution, energy_density, history
