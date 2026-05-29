"""
大变形非线性有限元核心模块
==========================
科学背景:
  本模块实现基于Total Lagrangian格式的4节点线性四面体(P1)单元
  大变形非线性有限元分析。核心方程为虚功原理:
      ∫_Ω0 S : δE dV = ∫_Ω0 B^T S dV = F_ext
  其中 S 为第二Piola-Kirchhoff应力，E 为Green-Lagrange应变，
  B 为非线性应变-位移矩阵。

  Newton-Raphson迭代格式:
      K_T^(k) · Δu^(k) = -R^(k)
      u^(k+1) = u^(k) + Δu^(k)
  其中切线刚度矩阵 K_T = K_mat + K_geo (材料刚度 + 几何刚度)。

关键公式:
  - 单元形函数 (重心坐标):
      N1 = 1 - ξ - η - ζ, N2 = ξ, N3 = η, N4 = ζ
  - 形函数导数与参考坐标关系: dN/dX = J0^{-T} dN/dξ
  - Green-Lagrange应变变分: δE = 1/2 (F^T δF + δF^T F)
  - 几何刚度矩阵: K_geo = ∫ G^T S G dV  (G为梯度算子矩阵)
"""

import numpy as np
from typing import Tuple, Optional, List
from hyperelastic_constitutive import (
    deformation_gradient, right_cauchy_green,
    neo_hookean_pk2_stress, neo_hookean_material_tangent,
    green_lagrange_strain, voigt_strain, voigt_stress,
    cauchy_stress_from_pk2, von_mises_cauchy,
    solve_effective_shear_modulus
)
from stiffness_solver import apply_dirichlet_to_system


def tet_p1_shape_derivatives() -> Tuple[np.ndarray, np.ndarray]:
    """
    返回4节点四面体在等参坐标下的形函数值和导数。
    积分点取单元重心 (ξ=η=ζ=1/4)。

    返回:
        N: (4,) 形函数值
        dN_dxi: (4, 3) 形函数对等参坐标的导数
    """
    # 重心积分点
    xi = 1.0 / 4.0
    eta = 1.0 / 4.0
    zeta = 1.0 / 4.0
    N = np.array([1.0 - xi - eta - zeta, xi, eta, zeta], dtype=np.float64)
    dN_dxi = np.array([
        [-1.0, -1.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    return N, dN_dxi


def compute_element_jacobian_and_dNdX(nodes_e: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    计算单元参考Jacobian及其逆，返回形函数对参考坐标的导数。

    参数:
        nodes_e: (4, 3) 单元节点参考坐标

    返回:
        detJ0: 参考Jacobian行列式 (6V)
        J0: (3, 3) 参考Jacobian矩阵
        dN_dX: (4, 3) 形函数对参考坐标的导数
    """
    _, dN_dxi = tet_p1_shape_derivatives()
    J0 = dN_dxi.T @ nodes_e  # (3, 3)
    detJ0 = float(np.linalg.det(J0))
    if abs(detJ0) < 1e-14:
        raise ValueError(f"单元Jacobian行列式接近零: {detJ0}")
    J0_inv = np.linalg.inv(J0)
    dN_dX = dN_dxi @ J0_inv.T  # (4, 3)
    return detJ0, J0, dN_dX


def compute_B_matrix(F: np.ndarray, dN_dX: np.ndarray) -> np.ndarray:
    """
    计算大变形应变-位移矩阵 B (6 x 12)。
    对于Green-Lagrange应变，δE = B · δu_e (Voigt形式)。

    B 的构造基于:
      δE_{IJ} = 1/2 (F_{kI} δu_{k,J} + F_{kJ} δu_{k,I})
    """
    # TODO: Hole 2 - 实现大变形应变-位移矩阵B的构造
    raise NotImplementedError("Hole 2: 请实现compute_B_matrix")


def compute_G_matrix(dN_dX: np.ndarray) -> np.ndarray:
    """
    计算几何刚度矩阵所需的 G 矩阵 (9 x 12)。
    G 将节点位移增量映射到变形梯度增量: δF = G · δu_e
    """
    # TODO: Hole 2 - 实现几何刚度G矩阵的构造
    raise NotImplementedError("Hole 2: 请实现compute_G_matrix")


def assemble_element_stiffness_force(nodes_e: np.ndarray, u_e: np.ndarray,
                                      mu: float, lam: float,
                                      use_damage: bool = False,
                                      gamma_e: float = 0.0,
                                      alpha_damage: float = 0.0) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    组装单元切线刚度矩阵和内部力向量。

    参数:
        nodes_e: (4, 3) 单元参考节点坐标
        u_e: (4, 3) 单元节点位移
        mu, lam: 材料参数
        use_damage: 是否使用损伤耦合
        gamma_e: 单元等效剪应变
        alpha_damage: 损伤耦合系数

    返回:
        k_e: (12, 12) 单元切线刚度矩阵
        f_int: (12,) 单元内部力向量
        stress_data: 应力相关数据字典
    """
    detJ0, _, dN_dX = compute_element_jacobian_and_dNdX(nodes_e)
    F = deformation_gradient(dN_dX, u_e)
    C = right_cauchy_green(F)

    # 损伤耦合: 修正剪切模量
    if use_damage and gamma_e > 1e-12 and alpha_damage > 0:
        mu_eff = solve_effective_shear_modulus(mu, gamma_e, alpha_damage)
    else:
        mu_eff = mu

    try:
        S = neo_hookean_pk2_stress(C, mu_eff, lam)
        C_mat = neo_hookean_material_tangent(C, mu_eff, lam)
    except ValueError:
        # 单元过度扭曲时的安全回退: 返回小刚度
        k_e = np.eye(12, dtype=np.float64) * 1e-6
        f_int = np.zeros(12, dtype=np.float64)
        stress_data = {
            "F": F, "C": C, "S": np.zeros((3,3)),
            "sigma": np.zeros((3,3)), "sigma_vm": 0.0,
            "gamma": 0.0, "mu_eff": mu_eff,
        }
        return k_e, f_int, stress_data

    B = compute_B_matrix(F, dN_dX)
    G = compute_G_matrix(dN_dX)

    # 将PK2应力转换为9x9矩阵形式用于几何刚度
    S_mtx = np.zeros((3, 3), dtype=np.float64)
    S_mtx[0, 0] = S[0, 0]
    S_mtx[1, 1] = S[1, 1]
    S_mtx[2, 2] = S[2, 2]
    S_mtx[0, 1] = S_mtx[1, 0] = S[0, 1]
    S_mtx[0, 2] = S_mtx[2, 0] = S[0, 2]
    S_mtx[1, 2] = S_mtx[2, 1] = S[1, 2]

    S9 = np.zeros((9, 9), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            S9[i * 3 + j, i * 3 + j] = S_mtx[i, j]
            if j != i:
                S9[i * 3 + j, j * 3 + i] = S_mtx[i, j]

    # 积分权重 (单点积分): |detJ0| / 6.0
    w = abs(detJ0) / 6.0

    # 材料刚度
    k_mat = B.T @ C_mat @ B * w
    # 几何刚度
    k_geo = G.T @ S9 @ G * w
    k_e = k_mat + k_geo

    # 内部力
    f_int = B.T @ voigt_stress(S) * w

    # Cauchy应力与von Mises
    sigma = cauchy_stress_from_pk2(F, S)
    sigma_vm = von_mises_cauchy(sigma)

    # 单元等效剪应变 (用于损伤)
    E = green_lagrange_strain(C)
    gamma_calc = float(np.sqrt(2.0 * np.sum(E * E)))

    stress_data = {
        "F": F,
        "C": C,
        "S": S,
        "sigma": sigma,
        "sigma_vm": sigma_vm,
        "gamma": gamma_calc,
        "mu_eff": mu_eff,
    }

    return k_e, f_int, stress_data


def assemble_global_system(nodes: np.ndarray, elements: np.ndarray,
                            u: np.ndarray, mu: float, lam: float,
                            use_damage: bool = False,
                            gamma_elements: Optional[np.ndarray] = None,
                            alpha_damage: float = 0.0) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
    """
    组装全局切线刚度矩阵和残差向量。

    参数:
        nodes: (N, 3) 节点坐标
        elements: (E, 4) 单元连接表
        u: (3*N,) 全局位移向量
        mu, lam: 材料参数
        use_damage: 是否启用损伤
        gamma_elements: (E,) 单元等效剪应变
        alpha_damage: 损伤耦合系数

    返回:
        K_global: (3N, 3N) 全局刚度矩阵
        R: (3N,) 残差向量 (-内部力，未加外力)
        stress_list: 每个单元的应力数据列表
    """
    n_nodes = nodes.shape[0]
    n_dof = 3 * n_nodes
    K_global = np.zeros((n_dof, n_dof), dtype=np.float64)
    R = np.zeros(n_dof, dtype=np.float64)
    stress_list = []

    if gamma_elements is None:
        gamma_elements = np.zeros(elements.shape[0], dtype=np.float64)

    for e_idx, e in enumerate(elements):
        nodes_e = nodes[e]
        u_e = u[3 * e[:, None] + np.arange(3)].reshape(4, 3)
        gamma_e = gamma_elements[e_idx]

        k_e, f_int, sdata = assemble_element_stiffness_force(
            nodes_e, u_e, mu, lam, use_damage, gamma_e, alpha_damage
        )
        stress_list.append(sdata)

        # 组装到全局
        dof_map = []
        for n in e:
            dof_map.extend([3 * n, 3 * n + 1, 3 * n + 2])
        dof_map = np.array(dof_map, dtype=np.int32)

        for i_local in range(12):
            i_global = dof_map[i_local]
            R[i_global] += f_int[i_local]
            for j_local in range(12):
                j_global = dof_map[j_local]
                K_global[i_global, j_global] += k_e[i_local, j_local]

    return K_global, R, stress_list


def compute_external_force(nodes: np.ndarray, elements: np.ndarray,
                            surface_tris: np.ndarray,
                            traction: np.ndarray) -> np.ndarray:
    """
    计算表面力 traction (单位参考面积力) 对应的等效节点力。
    对表面三角形积分: F_ext = ∫ N^T t dA

    参数:
        nodes: (N, 3) 节点坐标
        elements: (E, 4) 单元连接表
        surface_tris: (M, 3) 表面三角形
        traction: (3,) 均匀表面力向量

    返回:
        F_ext: (3N,) 等效外部力向量
    """
    n_nodes = nodes.shape[0]
    F_ext = np.zeros(3 * n_nodes, dtype=np.float64)

    for tri in surface_tris:
        p0, p1, p2 = nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]
        v1 = p1 - p0
        v2 = p2 - p0
        area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        # P1三角形重心积分: 每个节点分配 1/3
        for n in tri:
            F_ext[3 * n:3 * n + 3] += traction * (area / 3.0)

    return F_ext
