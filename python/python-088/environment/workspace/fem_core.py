"""
fem_core.py
有限元核心模块：T6 二次三角形单元

融入种子项目:
  - 380_fem_to_tec: FEM 网格数据结构与处理
  - 375_fem_basis_t6_display: T6 二次三角形基函数

功能:
  - T6 二次三角形形函数与导数
  - 平面应力/平面应变刚度矩阵组装
  - 等效节点力计算
  - 高斯数值积分
  - 考虑蠕变效应的等效刚度
"""

import numpy as np
from typing import Tuple, Optional


# T6 二次三角形单元参考坐标下的形函数（6个节点）
# 节点排布：
#   3
#   |\
#   6 5
#   |  \
#   1-4-2
#
# 参考坐标 (r, s), r \\ge 0, s \\ge 0, r+s \\le 1

def t6_shape_functions(r: float, s: float) -> np.ndarray:
    """
    T6 二次三角形单元的 6 个形函数。

    在参考三角形上，形函数为:
        N_1 = (1 - r - s)(2(1 - r - s) - 1) = 2(1-r-s)^2 - (1-r-s)
        N_2 = r(2r - 1)
        N_3 = s(2s - 1)
        N_4 = 4r(1 - r - s)
        N_5 = 4rs
        N_6 = 4s(1 - r - s)

    参数:
        r, s: 参考坐标

    返回:
        形函数值数组 (6,)
    """
    t = 1.0 - r - s
    N = np.zeros(6)
    N[0] = t * (2.0 * t - 1.0)
    N[1] = r * (2.0 * r - 1.0)
    N[2] = s * (2.0 * s - 1.0)
    N[3] = 4.0 * r * t
    N[4] = 4.0 * r * s
    N[5] = 4.0 * s * t
    return N


def t6_shape_derivatives(r: float, s: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    T6 二次三角形单元的形函数对参考坐标的偏导数。

    dN/dr 和 dN/ds:
        dN_1/dr = -(4(1-r-s) - 1) = -4t + 1
        dN_2/dr = 4r - 1
        dN_3/dr = 0
        dN_4/dr = 4(1 - 2r - s)
        dN_5/dr = 4s
        dN_6/dr = -4s

        dN_1/ds = -(4(1-r-s) - 1) = -4t + 1
        dN_2/ds = 0
        dN_3/ds = 4s - 1
        dN_4/ds = -4r
        dN_5/ds = 4r
        dN_6/ds = 4(1 - r - 2s)

    参数:
        r, s: 参考坐标

    返回:
        (dN_dr, dN_ds)，各为 (6,) 数组
    """
    t = 1.0 - r - s
    dN_dr = np.zeros(6)
    dN_ds = np.zeros(6)

    dN_dr[0] = -4.0 * t + 1.0
    dN_dr[1] = 4.0 * r - 1.0
    dN_dr[2] = 0.0
    dN_dr[3] = 4.0 * (t - r)
    dN_dr[4] = 4.0 * s
    dN_dr[5] = -4.0 * s

    dN_ds[0] = -4.0 * t + 1.0
    dN_ds[1] = 0.0
    dN_ds[2] = 4.0 * s - 1.0
    dN_ds[3] = -4.0 * r
    dN_ds[4] = 4.0 * r
    dN_ds[5] = 4.0 * (t - s)

    return dN_dr, dN_ds


def t6_jacobian(
    nodes: np.ndarray, r: float, s: float
) -> Tuple[np.ndarray, float]:
    """
    计算 T6 单元的 Jacobian 矩阵和行列式。

    Jacobian:
        J = [ dx/dr  dx/ds ]
            [ dy/dr  dy/ds ]

    其中:
        dx/dr = \\\sum_i x_i dN_i/dr, 等等

    行列式 |J| 用于面积元变换:
        dA = |J| dr ds

    参数:
        nodes: 单元节点坐标 (6, 2)
        r, s: 参考坐标

    返回:
        (J, det_J)
    """
    dN_dr, dN_ds = t6_shape_derivatives(r, s)

    J = np.zeros((2, 2))
    J[0, 0] = np.dot(nodes[:, 0], dN_dr)
    J[0, 1] = np.dot(nodes[:, 0], dN_ds)
    J[1, 0] = np.dot(nodes[:, 1], dN_dr)
    J[1, 1] = np.dot(nodes[:, 1], dN_ds)

    det_J = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return J, det_J


def gauss_points_triangle_t6(order: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    三角形区域上的 Gauss 积分点和权重。

    对于参考三角形 (r \\ge 0, s \\ge 0, r+s \\le 1):

    3点公式 (二阶精度):
        w = 1/3, 点位于 (2/3, 1/6), (1/6, 2/3), (1/6, 1/6)

    参数:
        order: 积分阶数 (目前支持 1 或 3)

    返回:
        (points, weights)，points 形状 (n_gp, 2)
    """
    if order == 1:
        points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([0.5])
    elif order == 3:
        points = np.array([
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 6.0, 1.0 / 6.0],
        ])
        weights = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 4:
        # 4点 Hammer 积分（三阶精度）
        a = 1.0 / 3.0
        b = 0.6
        c = 0.2
        w1 = -27.0 / 96.0
        w2 = 25.0 / 96.0
        points = np.array([
            [a, a],
            [b, c],
            [c, b],
            [c, c],
        ])
        weights = np.array([w1, w2, w2, w2])
    else:
        points = np.array([[1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([0.5])

    return points, weights


def plane_stress_constitutive_matrix(
    E: float, nu: float
) -> np.ndarray:
    """
    平面应力本构矩阵 D。

    对于各向同性线弹性材料:
        D = \\frac{E}{1-\\nu^2}
            [ 1    \\nu   0    ]
            [ \\nu  1     0    ]
            [ 0    0     (1-\\nu)/2 ]

    参数:
        E: 弹性模量 [MPa]
        nu: 泊松比

    返回:
        D 矩阵 (3, 3)
    """
    # TODO(Hole_2): 实现平面应力本构矩阵 D
    # 科学公式: D = E/(1-nu^2) * [[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]]
    # 这是线弹性各向同性材料平面应力状态下的 3x3 本构矩阵
    pass


def plane_strain_constitutive_matrix(
    E: float, nu: float
) -> np.ndarray:
    """
    平面应变本构矩阵 D。

    对于各向同性线弹性材料:
        D = \\frac{E(1-\\nu)}{(1+\\nu)(1-2\\nu)}
            [ 1    \\frac{\\nu}{1-\\nu}   0    ]
            [ \\frac{\\nu}{1-\\nu}  1     0    ]
            [ 0    0     \\frac{1-2\\nu}{2(1-\\nu)} ]

    参数:
        E: 弹性模量 [MPa]
        nu: 泊松比

    返回:
        D 矩阵 (3, 3)
    """
    factor = E * (1.0 - nu) / ((1.0 + nu) * (1.0 - 2.0 * nu))
    D = factor * np.array([
        [1.0, nu / (1.0 - nu), 0.0],
        [nu / (1.0 - nu), 1.0, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / (2.0 * (1.0 - nu))],
    ])
    return D


def compute_B_matrix_t6(
    nodes: np.ndarray, r: float, s: float
) -> np.ndarray:
    """
    计算 T6 单元的应变-位移矩阵 B。

    B 矩阵将节点位移与应变联系起来:
        \\varepsilon = [\\varepsilon_{xx}, \\varepsilon_{yy}, \\gamma_{xy}]^T = B \\cdot d

    其中 d 为 12x1 的节点位移向量（每个节点 2 个自由度）。

    参数:
        nodes: 单元节点坐标 (6, 2)
        r, s: 参考坐标

    返回:
        B 矩阵 (3, 12)
    """
    dN_dr, dN_ds = t6_shape_derivatives(r, s)
    J, det_J = t6_jacobian(nodes, r, s)

    if abs(det_J) < 1e-14:
        det_J = 1e-14

    # 逆 Jacobian
    J_inv = np.array([
        [J[1, 1], -J[0, 1]],
        [-J[1, 0], J[0, 0]],
    ]) / det_J

    # dN/dx = dN/dr * dr/dx + dN/ds * ds/dx
    dN_dx = J_inv[0, 0] * dN_dr + J_inv[0, 1] * dN_ds
    dN_dy = J_inv[1, 0] * dN_dr + J_inv[1, 1] * dN_ds

    B = np.zeros((3, 12))
    for i in range(6):
        B[0, 2 * i] = dN_dx[i]       # eps_xx
        B[1, 2 * i + 1] = dN_dy[i]   # eps_yy
        B[2, 2 * i] = dN_dy[i]       # gamma_xy
        B[2, 2 * i + 1] = dN_dx[i]

    return B


def assemble_stiffness_matrix_t6(
    nodes: np.ndarray,
    elements: np.ndarray,
    E: float,
    nu: float,
    thickness: float = 1.0,
    plane_stress: bool = True,
) -> np.ndarray:
    """
    组装全局刚度矩阵（T6 二次三角形单元）。

    单元刚度矩阵:
        k^{(e)} = \\\\int_{\\Omega_e} B^T D B \, dA
                = \\\sum_{gp} w_{gp} B^T D B |J|_{gp}

    全局刚度矩阵通过直接刚度法组装:
        K = \\\bigwedge_e k^{(e)}

    参数:
        nodes: 节点坐标 (n_nodes, 2)
        elements: 单元节点编号 (n_elements, 6)
        E: 弹性模量 [MPa]
        nu: 泊松比
        thickness: 厚度 [m]
        plane_stress: 是否平面应力

    返回:
        全局刚度矩阵 (2*n_nodes, 2*n_nodes)
    """
    n_nodes = len(nodes)
    n_dof = 2 * n_nodes
    K = np.zeros((n_dof, n_dof))

    D = plane_stress_constitutive_matrix(E, nu) if plane_stress else plane_strain_constitutive_matrix(E, nu)
    gp_points, gp_weights = gauss_points_triangle_t6(order=3)

    for elem in elements:
        elem_nodes = nodes[elem]
        ke = np.zeros((12, 12))

        for gp, w in zip(gp_points, gp_weights):
            r, s = gp
            B = compute_B_matrix_t6(elem_nodes, r, s)
            _, det_J = t6_jacobian(elem_nodes, r, s)

            if det_J <= 0:
                det_J = abs(det_J) + 1e-14

            ke += w * det_J * thickness * (B.T @ D @ B)

        # 组装到全局矩阵
        dof_map = []
        for node_idx in elem:
            dof_map.extend([2 * node_idx, 2 * node_idx + 1])

        for i in range(12):
            for j in range(12):
                K[dof_map[i], dof_map[j]] += ke[i, j]

    return K


def apply_dirichlet_boundary(
    K: np.ndarray, F: np.ndarray, bc_nodes: np.ndarray,
    bc_values: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    施加 Dirichlet 边界条件（置大数法）。

    对于边界节点 i 的位移约束 u_i = u_bar:
        K_{ii} \\\leftarrow K_{ii} + 10^{20}
        F_i \\\leftarrow (K_{ii} + 10^{20}) \\cdot u_{bar}

    参数:
        K: 刚度矩阵
        F: 载荷向量
        bc_nodes: 边界约束自由度编号
        bc_values: 约束值（None 时设为 0）

    返回:
        (K_modified, F_modified)
    """
    K_mod = K.copy()
    F_mod = F.copy()
    big_number = 1e20

    if bc_values is None:
        bc_values = np.zeros(len(bc_nodes))

    for idx, node_dof in enumerate(bc_nodes):
        val = bc_values[idx]
        K_mod[node_dof, node_dof] += big_number
        F_mod[node_dof] = (K_mod[node_dof, node_dof]) * val

    return K_mod, F_mod


def compute_nodal_forces_uniform(
    nodes: np.ndarray, elements: np.ndarray,
    qx: float, qy: float, thickness: float = 1.0
) -> np.ndarray:
    """
    计算均匀体力下的等效节点力。

    等效节点力:
        f^{(e)} = \\\\int_{\\Omega_e} N^T q \, dA

    参数:
        nodes: 节点坐标
        elements: 单元
        qx, qy: 体力分量 [N/m^3]
        thickness: 厚度

    返回:
        全局力向量
    """
    n_nodes = len(nodes)
    F = np.zeros(2 * n_nodes)

    gp_points, gp_weights = gauss_points_triangle_t6(order=3)

    for elem in elements:
        elem_nodes = nodes[elem]
        fe = np.zeros(12)

        for gp, w in zip(gp_points, gp_weights):
            r, s = gp
            N = t6_shape_functions(r, s)
            _, det_J = t6_jacobian(elem_nodes, r, s)
            if det_J <= 0:
                det_J = abs(det_J) + 1e-14

            for i in range(6):
                fe[2 * i] += w * det_J * thickness * qx * N[i]
                fe[2 * i + 1] += w * det_J * thickness * qy * N[i]

        dof_map = []
        for node_idx in elem:
            dof_map.extend([2 * node_idx, 2 * node_idx + 1])

        for i in range(12):
            F[dof_map[i]] += fe[i]

    return F


def compute_equivalent_creep_load(
    nodes: np.ndarray,
    elements: np.ndarray,
    epsilon_creep: np.ndarray,
    E_eff: float,
    nu: float,
    thickness: float = 1.0,
    plane_stress: bool = True,
) -> np.ndarray:
    """
    计算蠕变应变对应的等效节点载荷。

    蠕变等效载荷:
        F_{cr} = \\\\int_{\\Omega} B^T D \\varepsilon_{cr} \, dA

    参数:
        nodes: 节点坐标
        elements: 单元
        epsilon_creep: 单元蠕变应变 (假设为常数或按单元给定)
        E_eff: 有效弹性模量
        nu: 泊松比
        thickness: 厚度
        plane_stress: 是否平面应力

    返回:
        等效载荷向量
    """
    n_nodes = len(nodes)
    F_cr = np.zeros(2 * n_nodes)

    D = plane_stress_constitutive_matrix(E_eff, nu) if plane_stress else plane_strain_constitutive_matrix(E_eff, nu)
    gp_points, gp_weights = gauss_points_triangle_t6(order=3)

    for e, elem in enumerate(elements):
        elem_nodes = nodes[elem]
        fecr = np.zeros(12)

        eps_cr = epsilon_creep[e] if len(epsilon_creep.shape) > 1 else epsilon_creep

        for gp, w in zip(gp_points, gp_weights):
            r, s = gp
            B = compute_B_matrix_t6(elem_nodes, r, s)
            _, det_J = t6_jacobian(elem_nodes, r, s)
            if det_J <= 0:
                det_J = abs(det_J) + 1e-14

            sigma_cr = D @ eps_cr
            fecr += w * det_J * thickness * (B.T @ sigma_cr)

        dof_map = []
        for node_idx in elem:
            dof_map.extend([2 * node_idx, 2 * node_idx + 1])

        for i in range(12):
            F_cr[dof_map[i]] += fecr[i]

    return F_cr


def compute_strain_stress_at_nodes(
    nodes: np.ndarray,
    elements: np.ndarray,
    displacements: np.ndarray,
    E: float,
    nu: float,
    plane_stress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算节点处的应变和应力（通过单元平均）。

    参数:
        nodes, elements: 网格数据
        displacements: 位移向量 (2*n_nodes,)
        E, nu: 材料参数
        plane_stress: 是否平面应力

    返回:
        (strains, stresses)，各为 (n_nodes, 3)
    """
    n_nodes = len(nodes)
    strains = np.zeros((n_nodes, 3))
    stresses = np.zeros((n_nodes, 3))
    count = np.zeros(n_nodes)

    D = plane_stress_constitutive_matrix(E, nu) if plane_stress else plane_strain_constitutive_matrix(E, nu)

    for elem in elements:
        elem_nodes = nodes[elem]
        # 在单元重心处计算应变
        r, s = 1.0 / 3.0, 1.0 / 3.0
        B = compute_B_matrix_t6(elem_nodes, r, s)

        d_elem = np.zeros(12)
        for i, node_idx in enumerate(elem):
            d_elem[2 * i] = displacements[2 * node_idx]
            d_elem[2 * i + 1] = displacements[2 * node_idx + 1]

        eps = B @ d_elem
        sig = D @ eps

        for node_idx in elem:
            strains[node_idx] += eps
            stresses[node_idx] += sig
            count[node_idx] += 1

    # 平均
    for i in range(n_nodes):
        if count[i] > 0:
            strains[i] /= count[i]
            stresses[i] /= count[i]

    return strains, stresses
