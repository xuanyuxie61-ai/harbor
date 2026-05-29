"""
fem_discretization.py
有限元离散化模块
================
对应原项目 377_fem_neumann（一维反应扩散 FEM）与 1318_triangle_symq_rule_original（三角形高精度求积），
实现一维分段线性 hat 函数有限元离散化，并引入三角形数值积分用于二维截面泛函计算。

数学模型：一维反应-扩散方程
    ∂u/∂t = D * ∂²u/∂x² + R(u;θ)    on  Ω = [a,b]
    ∂u/∂n = 0                         on  ∂Ω   (Neumann 边界)

弱形式：求 u∈H¹(Ω) 使得 ∀v∈H¹(Ω)
    (u_t, v) + D*(u_x, v_x) = (R(u), v)
其中 (·,·) 表示 L² 内积。
"""

import numpy as np
from typing import Tuple
from system_utils import EPS, TOL_RANK


# ---------------------------------------------------------------------------
# Hat 函数与局部刚度/质量矩阵
# ---------------------------------------------------------------------------

def hat_basis(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """
    在节点 xi 上计算分段线性 hat 函数基在点 x 处的值。
    返回 shape (len(x), len(xi)) 的稀疏矩阵表示。
    """
    x = np.asarray(x, dtype=float)
    xi = np.asarray(xi, dtype=float)
    n = len(xi)
    phi = np.zeros((len(x), n), dtype=float)
    for i in range(n - 1):
        h = xi[i + 1] - xi[i]
        if abs(h) < EPS:
            continue
        mask = (x >= xi[i]) & (x <= xi[i + 1])
        phi[mask, i] = (xi[i + 1] - x[mask]) / h
        phi[mask, i + 1] = (x[mask] - xi[i]) / h
    return phi


def assemble_fem_matrices_1d(nodes: np.ndarray,
                              diffusion_coeff: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    组装一维 FEM 质量矩阵 M 与刚度矩阵 K。

    单元局部矩阵（参考单元 [-1,1] 映射到 [x_i, x_{i+1}]）
        M_loc = h/6 * [[2, 1], [1, 2]]
        K_loc = D/h * [[ 1, -1], [-1,  1]]

    参数
    ----
    nodes : np.ndarray, shape (n,)
        空间网格节点（已排序）。
    diffusion_coeff : float
        扩散系数 D。

    返回
    ----
    M, K : np.ndarray, shape (n, n)
        质量矩阵与刚度矩阵（均为对称三对角稀疏矩阵）。
    """
    nodes = np.asarray(nodes, dtype=float)
    n = len(nodes)
    if n < 2:
        raise ValueError("Need at least 2 nodes.")
    M = np.zeros((n, n), dtype=float)
    K = np.zeros((n, n), dtype=float)
    # TODO: 实现一维 FEM 质量矩阵 M 与刚度矩阵 K 的组装
    # 提示：
    #   1. 对参考单元 [-1,1] 映射到 [x_i, x_{i+1}]，单元长度 h = x_{i+1} - x_i
    #   2. 局部质量矩阵（科学知识）:
    #          M_loc = h/6 * [[2, 1], [1, 2]]
    #   3. 局部刚度矩阵（科学知识）:
    #          K_loc = D/h * [[ 1, -1], [-1,  1]]
    #      其中 D = diffusion_coeff
    #   4. 将局部矩阵组装到全局 M 和 K 的对应位置
    #   5. 最终返回的 (M, K) 会被 extract_tridiagonal 提取为 R83 格式，
    #      供 tridiagonal_solver.r83_cg 使用
    raise NotImplementedError("Hole 3: FEM 矩阵组装待实现")


def extract_tridiagonal(A: np.ndarray) -> np.ndarray:
    """
    将对称三对角矩阵提取为 R83 紧凑格式 (3, n)。
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    r83 = np.zeros((3, n), dtype=float)
    r83[1, :] = np.diag(A)
    if n > 1:
        r83[0, 1:] = np.diag(A, -1)
        r83[2, :-1] = np.diag(A, 1)
    return r83


# ---------------------------------------------------------------------------
# 二维截面泛函（引入三角形求积）
# ---------------------------------------------------------------------------

def triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    三角形面积（对应原 triangle_area）：
        S = 0.5 * | (v2-v1) × (v3-v1) |
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    cross = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
    return 0.5 * abs(cross)


def integrate_on_2d_section(r_nodes: np.ndarray, z_nodes: np.ndarray,
                            f_values: np.ndarray) -> float:
    """
    在轴对称截面的三角形网格上积分泛函 ∫ f(r,z) dS。
    使用重心坐标下的 3 点 Gauss 求积（对应原 triangle_symq_rule 的低阶规则）。

    求积公式
    --------
    对参考三角形 (0,0),(1,0),(0,1)，3 阶 Gauss 节点为
        g1=(2/3,1/6), g2=(1/6,2/3), g3=(1/6,1/6)
    权重均为 1/6，使得 ∫_T 1 dξdη = 1/2。

    对实际三角形 T，面积坐标变换后：
        ∫_T f dS ≈ S_T * ( f(g1) + f(g2) + f(g3) ) / 3
    """
    r_nodes = np.asarray(r_nodes, dtype=float)
    z_nodes = np.asarray(z_nodes, dtype=float)
    f_values = np.asarray(f_values, dtype=float)
    n = len(r_nodes)
    if n < 3:
        return 0.0
    integral = 0.0
    # 将截面剖分为三角形扇形
    center_r = np.mean(r_nodes)
    center_z = np.mean(z_nodes)
    for i in range(n):
        i1 = i
        i2 = (i + 1) % n
        v1 = np.array([center_r, center_z])
        v2 = np.array([r_nodes[i1], z_nodes[i1]])
        v3 = np.array([r_nodes[i2], z_nodes[i2]])
        area = triangle_area(v1, v2, v3)
        # 3 点重心求积
        g1 = (2.0 * v1 + v2 + v3) / 4.0
        g2 = (v1 + 2.0 * v2 + v3) / 4.0
        g3 = (v1 + v2 + 2.0 * v3) / 4.0
        # 线性插值求 f 在重心处的值
        f_g = (f_values[i1] + f_values[i2]) / 2.0  # 近似
        integral += area * f_g
    return integral


# ---------------------------------------------------------------------------
# 投影与误差估计
# ---------------------------------------------------------------------------

def project_function_to_fem(nodes: np.ndarray, func) -> np.ndarray:
    """
    将连续函数投影到 FEM 节点空间：在节点处取点值（插值投影）。
    对 H² 正则函数，插值误差满足
        ‖f - I_h f‖_{L²} ≤ C h² |f|_{H²}
        ‖f - I_h f‖_{H¹} ≤ C h   |f|_{H²}
    """
    nodes = np.asarray(nodes, dtype=float)
    return func(nodes)


def fem_l2_norm(nodes: np.ndarray, u: np.ndarray, M: np.ndarray = None) -> float:
    """
    通过质量矩阵计算离散 L² 范数：
        ‖u‖_{L²,h} = sqrt( u^T M u )
    """
    u = np.asarray(u, dtype=float)
    if M is None:
        M, _ = assemble_fem_matrices_1d(nodes)
    norm_sq = float(u @ (M @ u))
    return np.sqrt(max(norm_sq, 0.0))
