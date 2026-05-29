"""
损伤演化与断裂力学模块
======================
基于种子项目:
  - 350_fd_predator_prey: 有限差分/前向欧拉法

科学背景:
  将经典Lotka-Volterra耦合ODE的思想转化为连续损伤力学(CDM)中的
  损伤-塑性耦合演化方程。在超弹性大变形框架下，引入标量损伤变量
  D ∈ [0, 1]，描述材料刚度退化:
      σ_eff = (1 - D) σ_0
  其中 σ_0 为有效应力，σ_eff 为名义应力。

  损伤演化律 (受L-V耦合思想启发，将损伤D视为"捕食者"，
  塑性应变ε_p视为"猎物"):
      dD/dt = A * D * (ε_p / ε_f - D)       (自饱和增长)
      dε_p/dt = B * ε̇ * (1 - D) * H(σ_vm - σ_y)   (塑性流动)

  其中 H 为Heaviside阶跃函数，σ_y 为屈服应力，ε_f 为断裂应变。
  该耦合系统采用显式前向欧拉法离散求解。

关键公式:
  - 名义应力: σ = (1 - D) σ_eff
  - 损伤能量释放率: Y = -∂ψ/∂D = ψ_e^0
  - 等效塑性应变率: ε̇_p = sqrt(2/3 ε̇_p : ε̇_p)
  - 更新一致性: D_{n+1} = D_n + Δt * f_D(D_n, ε_{p,n})
"""

import numpy as np
from typing import Tuple, Optional


def heaviside(x: float) -> float:
    """平滑Heaviside函数，避免严格阶跃导致的数值不稳定。"""
    if x > 1e-6:
        return 1.0
    elif x < -1e-6:
        return 0.0
    else:
        return 0.5 + x / (2e-6)


def damage_evolution_rate(D: float, eps_p: float,
                           A: float = 2.0, eps_f: float = 0.5) -> float:
    """
    损伤变量的演化率 dD/dt。
    采用Logistic型自饱和增长，受当前塑性应变驱动:
        dD/dt = A * D * (eps_p / eps_f - D)
    当 D → eps_p/eps_f 时增长率趋于零。

    参数:
        D: 当前损伤值 [0, 1]
        eps_p: 等效塑性应变
        A: 损伤增长系数
        eps_f: 参考断裂应变

    返回:
        dDdt: 损伤演化率
    """
    if D < 0:
        D = 0.0
    if D > 1:
        D = 1.0
    if eps_f < 1e-12:
        eps_f = 1e-12
    ratio = eps_p / eps_f
    # 边界处理: 确保饱和值不超过1
    sat = min(ratio, 1.0)
    rate = A * D * (sat - D)
    # 若 D 已接近1，强制减速
    if D > 0.99:
        rate = min(rate, 0.0)
    return rate


def plastic_strain_rate(eps_dot_eq: float, D: float,
                         B: float = 1.0, sigma_vm: float = 0.0,
                         sigma_y: float = 1e6) -> float:
    """
    等效塑性应变演化率 dε_p/dt。
    仅在von Mises应力超过屈服应力时发生塑性流动:
        dε_p/dt = B * ε̇_eq * (1 - D) * H(σ_vm - σ_y)

    参数:
        eps_dot_eq: 等效应变率
        D: 当前损伤值
        B: 塑性流动系数
        sigma_vm: von Mises等效应力
        sigma_y: 屈服应力

    返回:
        depdt: 塑性应变率
    """
    H = heaviside(sigma_vm - sigma_y)
    rate = B * eps_dot_eq * (1.0 - D) * H
    return rate


def forward_euler_damage_step(D_n: float, eps_p_n: float,
                               dt: float, eps_dot_eq: float,
                               sigma_vm: float, sigma_y: float,
                               A: float = 2.0, B: float = 1.0,
                               eps_f: float = 0.5) -> Tuple[float, float]:
    """
    使用前向欧拉法推进损伤-塑性耦合系统一个时间步。

    参数:
        D_n: 当前损伤值
        eps_p_n: 当前等效塑性应变
        dt: 时间步长
        eps_dot_eq: 等效应变率
        sigma_vm: von Mises应力
        sigma_y: 屈服应力
        A, B: 演化系数
        eps_f: 断裂应变

    返回:
        D_next, eps_p_next: 更新后的损伤和塑性应变
    """
    dDdt = damage_evolution_rate(D_n, eps_p_n, A, eps_f)
    depdt = plastic_strain_rate(eps_dot_eq, D_n, B, sigma_vm, sigma_y)

    D_next = D_n + dt * dDdt
    eps_p_next = eps_p_n + dt * depdt

    # 边界处理与物理约束
    D_next = np.clip(D_next, 0.0, 1.0)
    eps_p_next = max(eps_p_next, 0.0)

    return D_next, eps_p_next


def update_element_damage(n_elements: int,
                          D_elements: np.ndarray,
                          eps_p_elements: np.ndarray,
                          dt: float,
                          eps_dot_elements: np.ndarray,
                          sigma_vm_elements: np.ndarray,
                          sigma_y: float = 1e6,
                          A: float = 2.0, B: float = 1.0,
                          eps_f: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    对所有单元并行更新损伤和塑性应变。

    参数:
        n_elements: 单元数
        D_elements: (E,) 当前损伤数组
        eps_p_elements: (E,) 当前塑性应变数组
        dt: 时间步长
        eps_dot_elements: (E,) 等效应变率数组
        sigma_vm_elements: (E,) von Mises应力数组
        sigma_y: 屈服应力
        A, B, eps_f: 材料参数

    返回:
        D_new, eps_p_new: 更新后的数组
    """
    D_new = np.zeros(n_elements, dtype=np.float64)
    eps_p_new = np.zeros(n_elements, dtype=np.float64)

    for e in range(n_elements):
        D_new[e], eps_p_new[e] = forward_euler_damage_step(
            float(D_elements[e]), float(eps_p_elements[e]), dt,
            float(eps_dot_elements[e]), float(sigma_vm_elements[e]),
            sigma_y, A, B, eps_f
        )

    return D_new, eps_p_new


def compute_equivalent_strain_rate(u_current: np.ndarray,
                                    u_prev: np.ndarray,
                                    dt: float,
                                    elements: np.ndarray,
                                    nodes: np.ndarray) -> np.ndarray:
    """
    计算每个单元的等效应变率近似值。
    ε̇_eq = ||ΔE|| / dt，其中 ΔE 为Green-Lagrange应变增量。

    参数:
        u_current, u_prev: 当前和上一时刻的全局位移向量
        dt: 时间步长
        elements: 单元连接表
        nodes: 节点坐标

    返回:
        eps_dot: (E,) 等效应变率数组
    """
    from hyperelastic_constitutive import deformation_gradient, right_cauchy_green, green_lagrange_strain
    from tetrahedral_mesh import tetrahedron_volume

    n_elements = elements.shape[0]
    eps_dot = np.zeros(n_elements, dtype=np.float64)

    # 形函数导数 (常数，仅依赖于参考构型)
    _, dN_dxi = tetrahedron_volume, None
    # 实际上这里我们简化: 用位移差的L2范数除以特征长度
    # 更精确的做法是重新计算每个单元的F和E

    # 简化的等效应变率估计
    for e_idx, e in enumerate(elements):
        nodes_e = nodes[e]
        # 参考Jacobian
        x0, x1, x2, x3 = nodes_e[0], nodes_e[1], nodes_e[2], nodes_e[3]
        mat = np.vstack([x1 - x0, x2 - x0, x3 - x0])
        detJ0 = np.linalg.det(mat)
        if abs(detJ0) < 1e-14:
            eps_dot[e_idx] = 0.0
            continue

        # 计算当前和上一时刻的F
        dN_dxi = np.array([
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        J0 = dN_dxi.T @ nodes_e
        J0_inv = np.linalg.inv(J0)
        dN_dX = dN_dxi @ J0_inv.T

        u_e_cur = u_current[3 * e[:, None] + np.arange(3)].reshape(4, 3)
        u_e_prev = u_prev[3 * e[:, None] + np.arange(3)].reshape(4, 3)

        F_cur = deformation_gradient(dN_dX, u_e_cur)
        F_prev = deformation_gradient(dN_dX, u_e_prev)

        E_cur = green_lagrange_strain(right_cauchy_green(F_cur))
        E_prev = green_lagrange_strain(right_cauchy_green(F_prev))
        dE = E_cur - E_prev
        eps_dot[e_idx] = np.sqrt(2.0 * np.sum(dE * dE)) / max(dt, 1e-12)

    return eps_dot
