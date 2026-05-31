"""
monte_carlo_sampler.py
高维蒙特卡洛采样与球面积分模块
（对应种子项目 554_hyperball_monte_carlo, 1126_sphere_quad, 1113_sphere_cvt）

在光纤光学中，本模块用于：
  1. 光纤参数不确定性量化（高维超球内均匀采样）
  2. 远场辐射模式积分（球面积分）
  3. 光纤端面光场方向分布优化（球面CVT均匀化）

核心物理公式：
  远场辐射强度分布:
    I(θ,φ) = |∫∫ E(x,y) exp[-ik(x sinθ cosφ + y sinθ sinφ)] dxdy|²

  球面积分:
    ∫_{S²} f(Ω) dΩ = ∫_0^{2π} ∫_0^π f(θ,φ) sinθ dθ dφ

  球面三角形面积（由顶点v1,v2,v3确定）:
    利用球面余弦定理，先求边长（弧长）a,b,c：
      cos a = v2·v3, cos b = v1·v3, cos c = v1·v2
    半周长 s = (a+b+c)/2
    面积 = 4 * arctan( sqrt(tan(s/2) tan((s-a)/2) tan((s-b)/2) tan((s-c)/2)) )

  超球均匀采样:
    先生成标准正态分布向量 x ~ N(0,I_m)，归一化到单位球面，
    再乘以 r^{1/m}（r~Uniform[0,1]），得到单位超球内均匀分布。
"""

import numpy as np


def hyperball01_sample(m, n):
    """
    在单位m维超球内均匀采样n个点。
    （对应种子项目 554_hyperball_monte_carlo）
    """
    if m < 1 or n < 1:
        return np.empty((m, n))
    x = np.random.randn(m, n)
    norm = np.sqrt(np.sum(x ** 2, axis=0))
    norm = np.where(norm < 1e-15, 1.0, norm)
    x = x / norm
    r = np.random.rand(1, n) ** (1.0 / m)
    return x * r


def tp_to_xyz(theta, phi):
    """经纬度到笛卡尔坐标（单位球面）。"""
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.array([x, y, z])


def sphere01_triangle_vertices_to_area(v1, v2, v3):
    """
    计算单位球面上由三个顶点确定的球面三角形面积。
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # 边长（弧长）
    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    s = 0.5 * (a + b + c)
    # L'Huilier定理
    tan_s2 = np.tan(s * 0.5)
    tan_sa2 = np.tan(max(0.0, (s - a) * 0.5))
    tan_sb2 = np.tan(max(0.0, (s - b) * 0.5))
    tan_sc2 = np.tan(max(0.0, (s - c) * 0.5))

    if tan_s2 <= 0 or tan_sa2 <= 0 or tan_sb2 <= 0 or tan_sc2 <= 0:
        return 0.0

    area = 4.0 * np.arctan(np.sqrt(tan_s2 * tan_sa2 * tan_sb2 * tan_sc2))
    return area


def sphere01_quad_llm(f, h):
    """
    经纬度网格中点规则球面积分。
    （对应种子项目 1126_sphere_quad）

    参数:
        f: callable, f(x) -> float, x为shape(3,)的单位球面上的点
        h: float, 最大网格边长
    """
    phi_num = max(1, int(np.floor(np.pi / h)))
    theta_num = max(1, int(np.floor(2.0 * np.pi / h)))

    result = 0.0
    n_eval = 0

    if phi_num == 1 and theta_num == 1:
        v = f(np.array([1.0, 0.0, 0.0]))
        return 4.0 * np.pi * v, 1

    # 极帽区域 (phi=0)
    phi1 = 0.0
    phi2 = np.pi / phi_num
    for j in range(theta_num):
        theta1 = j * 2.0 * np.pi / theta_num
        theta2 = (j + 1) * 2.0 * np.pi / theta_num
        x1 = tp_to_xyz(theta1, phi1)
        x12 = tp_to_xyz(theta1, phi2)
        x22 = tp_to_xyz(theta2, phi2)
        area = sphere01_triangle_vertices_to_area(x1, x12, x22)
        m1 = 0.5 * (x1 + x12)
        m2 = 0.5 * (x12 + x22)
        m3 = 0.5 * (x22 + x1)
        for m in [m1, m2, m3]:
            m = m / np.linalg.norm(m)
            result += area * f(m) / 3.0
            n_eval += 1

    # 中间区域
    for i in range(1, phi_num - 1):
        phi1 = i * np.pi / phi_num
        phi2 = (i + 1) * np.pi / phi_num
        for j in range(theta_num):
            theta1 = j * 2.0 * np.pi / theta_num
            theta2 = (j + 1) * 2.0 * np.pi / theta_num
            x11 = tp_to_xyz(theta1, phi1)
            x21 = tp_to_xyz(theta2, phi1)
            x12 = tp_to_xyz(theta1, phi2)
            x22 = tp_to_xyz(theta2, phi2)

            # 第一个三角形
            area = sphere01_triangle_vertices_to_area(x11, x12, x22)
            for m in [0.5 * (x11 + x12), 0.5 * (x12 + x22), 0.5 * (x22 + x11)]:
                m = m / np.linalg.norm(m)
                result += area * f(m) / 3.0
                n_eval += 1

            # 第二个三角形
            area = sphere01_triangle_vertices_to_area(x22, x21, x11)
            for m in [0.5 * (x22 + x21), 0.5 * (x21 + x11), 0.5 * (x11 + x22)]:
                m = m / np.linalg.norm(m)
                result += area * f(m) / 3.0
                n_eval += 1

    # 底极帽 (phi=pi)
    phi1 = (phi_num - 1) * np.pi / phi_num
    phi2 = np.pi
    for j in range(theta_num):
        theta1 = j * 2.0 * np.pi / theta_num
        theta2 = (j + 1) * 2.0 * np.pi / theta_num
        x11 = tp_to_xyz(theta1, phi1)
        x21 = tp_to_xyz(theta2, phi1)
        x2 = tp_to_xyz(theta2, phi2)
        area = sphere01_triangle_vertices_to_area(x11, x2, x21)
        for m in [0.5 * (x11 + x2), 0.5 * (x2 + x21), 0.5 * (x21 + x11)]:
            m = m / np.linalg.norm(m)
            result += area * f(m) / 3.0
            n_eval += 1

    return result, n_eval


def sphere_cvt_step(n_points, xyz):
    """
    球面CVT（Centroidal Voronoi Tessellation）一步迭代。
    （对应种子项目 1113_sphere_cvt）

    物理意义：优化光纤端面输出光场在远场球面上的均匀分布。
    """
    xyz = np.asarray(xyz, dtype=float)
    # 归一化到球面
    for i in range(n_points):
        norm = np.linalg.norm(xyz[:, i])
        if norm > 1e-15:
            xyz[:, i] = xyz[:, i] / norm

    # 简化版CVT：基于最近邻的重心计算
    # 在真实科研代码中会使用Delaunay三角剖分，这里采用蒙特卡洛近似
    centroid = np.zeros_like(xyz)
    n_samples = max(1000, n_points * 50)
    samples = np.random.randn(3, n_samples)
    samples = samples / np.linalg.norm(samples, axis=0)

    # 每个样本分配给最近的点
    for s in range(n_samples):
        dists = np.sum((xyz - samples[:, s:s + 1]) ** 2, axis=0)
        idx = np.argmin(dists)
        centroid[:, idx] += samples[:, s]

    for i in range(n_points):
        norm = np.linalg.norm(centroid[:, i])
        if norm > 1e-15:
            centroid[:, i] = centroid[:, i] / norm
        else:
            centroid[:, i] = xyz[:, i]

    return centroid


def monte_carlo_uncertainty_quantification(param_center, param_std, n_samples=1000):
    """
    对光纤参数进行不确定性量化。

    参数:
        param_center: ndarray shape (m,), 参数中心值
        param_std: ndarray shape (m,), 参数标准差

    返回:
        samples: ndarray shape (m, n_samples), 参数样本
    """
    m = param_center.size
    samples = np.random.randn(m, n_samples) * param_std[:, None] + param_center[:, None]
    return samples
