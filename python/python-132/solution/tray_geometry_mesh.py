"""
tray_geometry_mesh.py
=====================
塔板几何建模与四边形网格划分模块。

本模块包含：
1. 矩形塔板的有符号距离函数（源自项目 305_dist_plot/drectangle）
2. 四边形参考单元到物理单元的等参映射（源自项目 953_quadrilateral_mesh/reference_to_physical_q4）
3. 塔板网格生成与局部面积计算

科学背景
--------
精馏塔塔板通常为圆形或矩形。为计算局部点效率与Murphree效率，需要将塔板
表面离散为网格单元。对于矩形塔板，采用四边形 Q4 等参元映射：

参考单元 (R,S) ∈ [0,1]×[0,1] 映射到物理单元：
    X(R,S) = Σ_{i=1}^{4} N_i(R,S) * X_i
    Y(R,S) = Σ_{i=1}^{4} N_i(R,S) * Y_i

其中形函数：
    N_1 = (1-R)(1-S)
    N_2 = R(1-S)
    N_3 = R S
    N_4 = (1-R) S

有符号距离函数用于判断点是否在塔板内部：
    d(p) = -min( -x1+p_x, x2-p_x, -y1+p_y, y2-p_y )
    d < 0 : 内部；d = 0 : 边界；d > 0 : 外部

Jacobian 行列式给出局部面积微元：
    |J| = | ∂X/∂R  ∂X/∂S |
        | ∂Y/∂R  ∂Y/∂S |
"""

import numpy as np
from utils import clip_with_warning


# ---------------------------------------------------------------------------
# 矩形有符号距离函数（源自项目 305_dist_plot/drectangle）
# ---------------------------------------------------------------------------

def drectangle(p, x1, x2, y1, y2):
    """
    计算点到矩形的有符号距离。

    Parameters
    ----------
    p : ndarray, shape (np, 2)
        点坐标。
    x1, x2 : float
        矩形左右边界。
    y1, y2 : float
        矩形下上边界。

    Returns
    -------
    d : ndarray, shape (np,)
        有符号距离（内部为负，外部为正）。
    """
    p = np.asarray(p, dtype=float)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    dx = np.minimum(-x1 + p[:, 0], x2 - p[:, 0])
    dy = np.minimum(-y1 + p[:, 1], y2 - p[:, 1])
    d = -np.minimum(np.minimum(dx, dy), 0.0)
    # 修正：标准 signed distance 在内部应为负
    inside = (p[:, 0] >= x1) & (p[:, 0] <= x2) & (p[:, 1] >= y1) & (p[:, 1] <= y2)
    d = np.where(inside, -np.minimum(dx, dy), np.maximum(-np.minimum(dx, dy), 0.0))
    return d


# ---------------------------------------------------------------------------
# 四边形参考到物理映射（源自项目 953_quadrilateral_mesh/reference_to_physical_q4）
# ---------------------------------------------------------------------------

def reference_to_physical_q4(q4, n, rs):
    """
    将 Q4 参考单元上的点映射到物理单元。

    Parameters
    ----------
    q4 : ndarray, shape (2, 4)
        物理单元四个顶点坐标。
    n : int
        映射点数。
    rs : ndarray, shape (2, n)
        参考坐标 (R,S)，R,S ∈ [0,1]。

    Returns
    -------
    xy : ndarray, shape (2, n)
        物理坐标。
    """
    rs = np.asarray(rs, dtype=float)
    if rs.shape[1] != n:
        n = rs.shape[1]

    psi = np.zeros((4, n), dtype=float)
    psi[0, :] = (1.0 - rs[0, :]) * (1.0 - rs[1, :])
    psi[1, :] = rs[0, :] * (1.0 - rs[1, :])
    psi[2, :] = rs[0, :] * rs[1, :]
    psi[3, :] = (1.0 - rs[0, :]) * rs[1, :]

    xy = np.dot(q4, psi)
    return xy


def q4_jacobian(q4, rs):
    """
    计算 Q4 映射的 Jacobian 矩阵与行列式。

    Parameters
    ----------
    q4 : ndarray, shape (2, 4)
        顶点坐标。
    rs : ndarray, shape (2,)
        参考坐标。

    Returns
    -------
    detJ : float
        Jacobian 行列式（面积缩放因子）。
    J : ndarray, shape (2, 2)
        Jacobian 矩阵。
    """
    R, S = float(rs[0]), float(rs[1])
    dpsi_dR = np.array([-(1 - S), (1 - S), S, -S], dtype=float)
    dpsi_dS = np.array([-(1 - R), -R, R, (1 - R)], dtype=float)

    J = np.zeros((2, 2), dtype=float)
    J[0, 0] = np.dot(q4[0, :], dpsi_dR)
    J[0, 1] = np.dot(q4[0, :], dpsi_dS)
    J[1, 0] = np.dot(q4[1, :], dpsi_dR)
    J[1, 1] = np.dot(q4[1, :], dpsi_dS)

    detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
    return detJ, J


# ---------------------------------------------------------------------------
# 塔板网格生成
# ---------------------------------------------------------------------------

def generate_tray_mesh(tray_width, tray_height, nx, ny):
    """
    在矩形塔板上生成均匀四边形网格。

    Parameters
    ----------
    tray_width : float
        塔板宽度 [m]。
    tray_height : float
        塔板高度（深度）[m]。
    nx, ny : int
        x, y 方向单元数。

    Returns
    -------
    nodes : ndarray, shape (n_nodes, 2)
        节点坐标。
    elements : ndarray, shape (n_elem, 4)
        单元节点索引（逆时针）。
    areas : ndarray, shape (n_elem,)
        各单元面积。
    """
    if nx < 1:
        nx = 1
    if ny < 1:
        ny = 1

    x = np.linspace(0.0, tray_width, nx + 1)
    y = np.linspace(0.0, tray_height, ny + 1)
    xv, yv = np.meshgrid(x, y)
    nodes = np.column_stack((xv.ravel(), yv.ravel()))

    n_nodes_x = nx + 1
    n_nodes_y = ny + 1
    elements = []
    for j in range(ny):
        for i in range(nx):
            n1 = j * n_nodes_x + i
            n2 = j * n_nodes_x + i + 1
            n3 = (j + 1) * n_nodes_x + i + 1
            n4 = (j + 1) * n_nodes_x + i
            elements.append([n1, n2, n3, n4])
    elements = np.array(elements, dtype=int)

    # 计算各单元面积（精确矩形）
    dx = tray_width / nx
    dy = tray_height / ny
    areas = np.full(len(elements), dx * dy, dtype=float)

    return nodes, elements, areas


def compute_local_efficiency_on_mesh(nodes, elements, x_liq, y_vap, K_eq):
    """
    在网格上计算局部 Murphree 点效率：

        E_OG = (y_{out} - y_{in}) / (y^* - y_{in})

    其中 y^* = K_eq * x_liq 为平衡汽相组成。

    Parameters
    ----------
    nodes : ndarray
        网格节点。
    elements : ndarray
        单元拓扑。
    x_liq : ndarray, shape (nc,)
        液相组成。
    y_vap : ndarray, shape (nc,)
        实际汽相组成。
    K_eq : ndarray, shape (nc,)
        相平衡常数。

    Returns
    -------
    E_local : ndarray, shape (n_elem,)
        各单元局部效率。
    """
    nc = len(x_liq)
    n_elem = len(elements)
    E_local = np.zeros(n_elem, dtype=float)

    y_star = K_eq * x_liq
    y_star = np.clip(y_star, 0.0, 1.0)

    denom = y_star - y_vap
    # 对每个组分取平均效率
    for e in range(n_elem):
        effs = []
        for c in range(nc):
            d = denom[c]
            num = y_star[c] - y_vap[c]
            if abs(d) > 1e-12:
                effs.append(num / d)
            else:
                effs.append(0.0)
        E_local[e] = np.mean(effs)

    E_local = np.clip(E_local, 0.0, 1.0)
    return E_local


def mesh_average_efficiency(nodes, elements, areas, E_local):
    """
    面积加权平均效率：

        E_avg = Σ_e A_e E_e / Σ_e A_e

    Parameters
    ----------
    nodes, elements, areas : ndarray
        网格数据。
    E_local : ndarray
        局部效率。

    Returns
    -------
    E_avg : float
        面积加权平均效率。
    """
    total_area = np.sum(areas)
    if total_area < 1e-15:
        return 0.0
    return float(np.sum(areas * E_local) / total_area)
