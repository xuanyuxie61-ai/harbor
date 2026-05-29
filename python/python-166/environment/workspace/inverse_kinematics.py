"""
inverse_kinematics.py
逆运动学与散乱数据插值模块

融合种子项目:
- 928_pwl_interp_2d_scattered: 散乱数据2D分段线性插值（Delaunay三角剖分 + 行走搜索）

科学应用: 从软体机器人表面散乱传感器读数重构形状（逆运动学）
"""

import numpy as np
from typing import Tuple, Optional, List


def barycentric_coordinates(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[float, float, float]:
    """
    计算点p相对于三角形(a,b,c)的重心坐标

    p = alpha*a + beta*b + gamma*c
    alpha + beta + gamma = 1

    使用面积比公式:
        alpha = area(p,b,c) / area(a,b,c)
        beta  = area(p,a,c) / area(a,b,c)
        gamma = area(p,a,b) / area(a,b,c)
    """
    def tri_area(p1, p2, p3):
        return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))

    total_area = tri_area(a, b, c)
    if total_area < 1e-14:
        return -1.0, -1.0, -1.0

    alpha = tri_area(p, b, c) / total_area
    beta = tri_area(p, a, c) / total_area
    gamma = 1.0 - alpha - beta
    return alpha, beta, gamma


def point_in_triangle(alpha: float, beta: float, gamma: float, tol: float = -1e-10) -> bool:
    """
    判断重心坐标是否表示点在三角形内部
    """
    return alpha >= tol and beta >= tol and gamma >= tol


def delaunay_triangulation_2d(points: np.ndarray) -> np.ndarray:
    """
    简化2D Delaunay三角剖分 — 基于种子项目928_pwl_interp_2d_scattered思想

    对于少量点使用简单方法: scipy.spatial.Delaunay
    若scipy不可用，使用简化实现
    """
    try:
        from scipy.spatial import Delaunay
        tri = Delaunay(points)
        return tri.simplices
    except ImportError:
        # 简化回退: 对凸包进行扇形三角剖分
        from mesh_utils import compute_polygon_area
        # 计算凸包（简化）
        n = points.shape[0]
        if n < 3:
            return np.array([])

        # 使用numpy计算凸包索引（近似）
        center = np.mean(points, axis=0)
        angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
        order = np.argsort(angles)

        triangles = []
        for i in range(1, n - 1):
            triangles.append([order[0], order[i], order[i + 1]])
        return np.array(triangles, dtype=int)


def pwl_interp_2d_scattered(xyd: np.ndarray, zd: np.ndarray,
                            xyi: np.ndarray) -> np.ndarray:
    """
    散乱数据2D分段线性插值 — 基于种子项目928_pwl_interp_2d_scattered

    参数:
        xyd: (nd, 2) 数据点坐标
        zd: (nd,) 数据点值
        xyi: (ni, 2) 查询点坐标

    返回:
        zi: (ni,) 插值结果
    """
    nd = xyd.shape[0]
    ni = xyi.shape[0]

    if len(zd) != nd:
        raise ValueError("zd must have same length as xyd")

    # 构建Delaunay三角剖分
    triangles = delaunay_triangulation_2d(xyd)

    if len(triangles) == 0:
        return np.zeros(ni)

    # 计算邻居关系（简化）
    n_tri = len(triangles)

    zi = np.zeros(ni)
    for qi in range(ni):
        p = xyi[qi]
        found = False

        # 遍历所有三角形查找包含点
        for tri in triangles:
            a = xyd[tri[0]]
            b = xyd[tri[1]]
            c = xyd[tri[2]]
            alpha, beta, gamma = barycentric_coordinates(p, a, b, c)
            if point_in_triangle(alpha, beta, gamma):
                zi[qi] = alpha * zd[tri[0]] + beta * zd[tri[1]] + gamma * zd[tri[2]]
                found = True
                break

        if not found:
            # 回退: 最近邻插值
            dists = np.sum((xyd - p) ** 2, axis=1)
            nearest = np.argmin(dists)
            zi[qi] = zd[nearest]

    return zi


def shape_reconstruction_from_sensors(sensor_positions: np.ndarray,
                                      sensor_readings: np.ndarray,
                                      query_points: np.ndarray,
                                      reconstruction_type: str = 'pwl') -> np.ndarray:
    """
    从散乱传感器读数重构软体机器人形状

    参数:
        sensor_positions: (ns, 2) 传感器在参考构型中的位置
        sensor_readings: (ns,) 传感器测量值（如位移、曲率）
        query_points: (nq, 2) 需要重构的查询点
        reconstruction_type: 'pwl' 分段线性 / 'rbf' 径向基函数
    """
    if reconstruction_type == 'pwl':
        return pwl_interp_2d_scattered(sensor_positions, sensor_readings, query_points)
    elif reconstruction_type == 'rbf':
        # 简化的RBF插值
        return rbf_interp_2d(sensor_positions, sensor_readings, query_points)
    else:
        raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")


def rbf_interp_2d(xyd: np.ndarray, zd: np.ndarray, xyi: np.ndarray,
                  epsilon: float = 1.0) -> np.ndarray:
    """
    径向基函数插值（高斯核）

    f(x) = sum_j w_j * phi(||x - x_j||)
    phi(r) = exp(-(epsilon*r)^2)
    """
    nd = xyd.shape[0]
    ni = xyi.shape[0]

    # 构建插值矩阵
    Phi = np.zeros((nd, nd))
    for i in range(nd):
        for j in range(nd):
            r = np.linalg.norm(xyd[i] - xyd[j])
            Phi[i, j] = np.exp(-(epsilon * r) ** 2)

    # 求解权重
    try:
        w = np.linalg.solve(Phi, zd)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(Phi, zd, rcond=None)[0]

    # 插值
    zi = np.zeros(ni)
    for i in range(ni):
        for j in range(nd):
            r = np.linalg.norm(xyi[i] - xyd[j])
            zi[i] += w[j] * np.exp(-(epsilon * r) ** 2)

    return zi


def inverse_kinematics_soft_robot(target_tip: np.ndarray,
                                  L: float, Ns: int,
                                  material_params: dict,
                                  max_iter: int = 100,
                                  tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    软体机器人逆运动学: 给定末端目标位置，求驱动曲率分布

    使用牛顿-拉夫森迭代:
        kappa^{n+1} = kappa^n - J^{-1} * (r_tip(kappa^n) - target)

    参数:
        target_tip: (3,) 目标末端位置
        L, Ns: 杆长和离散数
        material_params: 材料参数

    返回:
        kappa_opt: (Ns+1, 3) 最优曲率分布
        r_final: (Ns+1, 3) 最终中心线
    """
    from cosserat_core import forward_kinematics_cosserat

    n_nodes = Ns + 1
    # 初始猜测: 零曲率（直杆）
    kappa = np.zeros((n_nodes, 3))

    for iteration in range(max_iter):
        s, r, R = forward_kinematics_cosserat(L, Ns, kappa)
        error = r[-1] - target_tip
        err_norm = np.linalg.norm(error)

        if err_norm < tol:
            break

        # TODO: Hole 2 — 实现逆运动学的Newton-Raphson迭代核心
        # 需要计算数值Jacobian（有限差分）并求解修正量:
        #   J[:, j] = (r_pert[-1] - r[-1]) / h
        #   delta = solve(J, -error)
        #   kappa += damping * delta
        # 注意: 需要调用 forward_kinematics_cosserat 计算扰动后的末端位置
        raise NotImplementedError("Hole 2: 实现Newton迭代核心")

        # 边界约束
        kappa_max = 2.0 * np.pi / L  # 最大曲率限制
        kappa = np.clip(kappa, -kappa_max, kappa_max)

    s, r_final, R = forward_kinematics_cosserat(L, Ns, kappa)
    return kappa, r_final
