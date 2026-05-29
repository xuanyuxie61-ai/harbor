"""
em_field.py - 电磁场模式计算与插值模块

融合原项目 380_fem_to_tec（有限元数据读取与转换）与
592_interp_equal（Newton 等距插值）的核心思想，
用于在量子点纳米结构网格上计算与插值电磁场模式分布。

核心物理模型：
    - 微腔中的电磁场满足亥姆霍兹方程：
        nabla^2 E + k^2 epsilon(r) E = 0
    - 品质因子 Q = omega / Delta_omega
    - Purcell 因子（自发辐射增强）：
        F_p = (3/4pi^2) (lambda/n)^3 (Q/V_eff)
    - 有效模式体积：
        V_eff = integral epsilon(r) |E(r)|^2 dV / max[epsilon(r) |E(r)|^2]
"""

import numpy as np
from typing import Tuple, Dict
from utils import validate_array_1d, validate_array_2d
from mesh_generator import reference_to_physical_q4, quadrilateral_area


# 真空光速
C_LIGHT = 2.99792458e8  # m/s
MU0 = 4.0 * np.pi * 1e-7
EPS0 = 8.854187817e-12


def gaussian_mode_profile(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    w0: float,
    amplitude: float = 1.0,
) -> np.ndarray:
    """
    基模高斯光束横向分布：
    
        E(x,y) = E_0 * exp[ - ((x-x0)^2 + (y-y0)^2) / w0^2 ]
    
    其中 w0 为束腰半径。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r2 = (x - x0) ** 2 + (y - y0) ** 2
    E = amplitude * np.exp(-r2 / (w0 ** 2))
    return E


def lorentzian_cavity_mode(
    x: np.ndarray,
    y: np.ndarray,
    x0: float,
    y0: float,
    R_cavity: float,
    n_eff: float = 3.5,
) -> np.ndarray:
    """
    圆盘微腔 whispering-gallery-like 模式近似（柱坐标简化）：
    
        E(r) ~ J_m(k_r r) 在腔内
        E(r) ~ H_m^{(1)}(k_r r) 在腔外（衰减）
    
    此处采用简化高斯-洛伦兹混合近似：
        E(r) = E_0 / (1 + (r/R_cavity)^4)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    r = np.sqrt((x - x0) ** 2 + (y - y0) ** 2)
    E = 1.0 / (1.0 + (r / R_cavity) ** 4)
    return E


def effective_mode_volume_2d(
    nodes: np.ndarray,
    elements: np.ndarray,
    E_field: np.ndarray,
    epsilon_r: np.ndarray,
) -> float:
    """
    计算二维等效模式体积（源自 FEM 积分思想）：
    
        V_eff = sum_e [ epsilon_r(e) |E_e|^2 A_e ] / max[ epsilon_r |E|^2 ]
    
    其中 A_e 为单元面积，E_e 为单元中心场强。
    """
    nodes = validate_array_2d(nodes, "nodes")
    elements = validate_array_2d(elements, "elements")
    E_field = validate_array_1d(E_field, "E_field")
    epsilon_r = validate_array_1d(epsilon_r, "epsilon_r")
    if elements.shape[0] != 4:
        raise ValueError("Only Q4 elements supported")
    n_elem = elements.shape[1]
    if E_field.size != nodes.shape[1]:
        raise ValueError("E_field length must match number of nodes")
    if epsilon_r.size != n_elem and epsilon_r.size != nodes.shape[1]:
        raise ValueError("epsilon_r length mismatch")

    from mesh_generator import quadrilateral_area
    total = 0.0
    max_val = 0.0
    for e in range(n_elem):
        q4 = np.zeros((2, 4), dtype=float)
        for k in range(4):
            q4[:, k] = nodes[:, elements[k, e]]
        area = quadrilateral_area(q4)
        # 单元中心场强取四个节点的平均
        E_center = 0.0
        for k in range(4):
            E_center += E_field[elements[k, e]]
        E_center /= 4.0
        if epsilon_r.size == n_elem:
            eps = epsilon_r[e]
        else:
            eps = 0.0
            for k in range(4):
                eps += epsilon_r[elements[k, e]]
            eps /= 4.0
        val = eps * (E_center ** 2)
        total += val * area
        if val > max_val:
            max_val = val
    if max_val < 1e-20:
        max_val = 1e-20
    return total / max_val


def purcell_factor(
    Q: float,
    V_eff: float,
    wavelength: float,
    n_eff: float = 3.5,
) -> float:
    """
    计算 Purcell 因子（三维近似，对二维结构做等效修正）：
    
        F_p = (3 / (4 pi^2)) * (lambda / n)^3 * (Q / V_eff)
    
    参数:
        Q:          品质因子
        V_eff:      有效模式体积 (m^3)
        wavelength: 真空波长 (m)
        n_eff:      有效折射率
    """
    if Q <= 0 or V_eff <= 0 or wavelength <= 0:
        raise ValueError("Q, V_eff, and wavelength must be positive")
    # TODO Hole 4: 实现 Purcell 因子计算
    # 公式: F_p = (3 / (4 pi^2)) * (lambda / n)^3 * (Q / V_eff)
    raise NotImplementedError("Hole 4: 请实现 purcell_factor 函数体")


def interpolate_field_on_mesh(
    nodes: np.ndarray,
    E_nodes: np.ndarray,
    query_points: np.ndarray,
) -> np.ndarray:
    """
    使用反距离加权插值（IDW）在查询点上估计场强。
    适用于非结构化网格上的场插值。
    
        E(q) = sum_i w_i E_i / sum_i w_i
        w_i = 1 / |q - node_i|^p
    """
    nodes = validate_array_2d(nodes, "nodes")
    E_nodes = validate_array_1d(E_nodes, "E_nodes")
    query_points = validate_array_2d(query_points, "query_points")
    if nodes.shape[0] != 2 or query_points.shape[0] != 2:
        raise ValueError("Coordinates must be 2D")
    n_query = query_points.shape[1]
    n_nodes = nodes.shape[1]
    E_query = np.zeros(n_query, dtype=float)
    p = 2.0  # 幂指数
    for q in range(n_query):
        xq, yq = query_points[0, q], query_points[1, q]
        dist2 = (nodes[0, :] - xq) ** 2 + (nodes[1, :] - yq) ** 2
        dist2 = np.where(dist2 < 1e-20, 1e-20, dist2)
        w = 1.0 / (dist2 ** (p / 2.0))
        E_query[q] = np.sum(w * E_nodes) / np.sum(w)
    return E_query


def fem_mode_solver_1d(
    x: np.ndarray,
    epsilon_profile: np.ndarray,
    target_wavelength: float,
) -> Dict[str, np.ndarray]:
    """
    一维等效折射率法求解微腔基模分布（简化 FEM 思想）。
    
    方程（TE 偏振，一维简化）：
        d^2E/dx^2 + k0^2 epsilon(x) E = beta^2 E
    
    其中 k0 = 2 pi / lambda。
    采用有限差分离散化并求解本征值问题。
    """
    x = validate_array_1d(x, "x")
    epsilon_profile = validate_array_1d(epsilon_profile, "epsilon_profile")
    if x.size != epsilon_profile.size:
        raise ValueError("x and epsilon_profile must have same size")
    n = x.size
    dx = float(x[1] - x[0])
    if abs(dx) < 1e-20:
        raise ValueError("Grid spacing too small")
    k0 = 2.0 * np.pi / target_wavelength

    # 构建有限差分矩阵
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        A[i, i] = -2.0 / (dx ** 2) + k0 ** 2 * epsilon_profile[i]
        if i > 0:
            A[i, i - 1] = 1.0 / (dx ** 2)
        if i < n - 1:
            A[i, i + 1] = 1.0 / (dx ** 2)
    # Dirichlet 边界
    A[0, :] = 0.0
    A[0, 0] = 1.0
    A[n - 1, :] = 0.0
    A[n - 1, n - 1] = 1.0

    eigvals, eigvecs = np.linalg.eigh(A)
    # beta^2 = eigvals，取最大的几个正本征值（对应导模）
    idx = np.argsort(-eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return {
        "beta2": eigvals,
        "beta": np.sqrt(np.maximum(eigvals, 0.0)),
        "mode_profiles": eigvecs,
        "x": x,
    }


def spontaneous_emission_rate(
    dipole_moment: float,
    omega: float,
    local_density_of_states: float,
) -> float:
    """
    Fermi 黄金定则给出的自发辐射速率：
    
        gamma = (omega^3 |d|^2) / (3 pi epsilon_0 hbar c^3) * rho(omega)
    
    其中 rho(omega) 为局域光子态密度（LDOS）。
    """
    if omega <= 0 or local_density_of_states < 0:
        raise ValueError("omega must be positive and LDOS non-negative")
    gamma = (
        (omega ** 3) * (dipole_moment ** 2) * local_density_of_states
        / (3.0 * np.pi * EPS0 * (1.054571817e-34) * (C_LIGHT ** 3))
    )
    return gamma
