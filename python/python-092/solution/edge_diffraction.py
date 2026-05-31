"""
edge_diffraction.py
声学边缘衍射建模
基于 edge (阶跃/边缘函数) 与 triangle_symq_rule (三角形求积) 核心算法重构

声学工程应用：
在几何声学中，当射线遇到墙面边缘时，除镜面反射外还存在衍射波。
采用几何衍射理论 (GTD/UTD) 的 Keller 锥模型：
    衍射射线位于入射射线与边缘构成的 Keller 锥上。

衍射系数 D 的计算基于边缘绕射理论：
    D = D(φ, φ', k, β)
其中 φ 为入射角，φ' 为衍射角，k 为波数，β 为边缘阻抗参数。
"""

import numpy as np
from quadrature_rules import integrate_over_triangle


C_AIR = 343.0


def edge_diffraction_coefficient(phi_inc, phi_diff, k, edge_type='hard'):
    """
    计算边缘衍射系数（基于 Keller GTD 简化模型）。

    对于硬边（刚性边界）：
        D = -exp(-iπ/4) / (2n√(2πk)) * [
            1/tan((π + Φ)/2n) + 1/tan((π - Φ)/2n)
            ± (1/tan((π + Φ')/2n) + 1/tan((π - Φ')/2n))
        ]
    其中 n = 2（直边），Φ = φ_inc + φ_diff。

    简化版（来自 edge 检测函数的阶跃思想）：
    在阴影边界附近衍射最强。
    """
    if edge_type == 'hard':
        n_edge = 2.0
        Phi = phi_inc + phi_diff
        # 避免奇点，严格远离 π 的整数倍
        eps = 0.1
        #  Keller 锥条件的有效范围限制
        Phi_mod = Phi % (2.0 * np.pi)
        if abs(Phi_mod) < eps:
            Phi_mod = eps
        if abs(Phi_mod - np.pi) < eps:
            Phi_mod = np.pi + eps
        if abs(Phi_mod - 2.0 * np.pi) < eps:
            Phi_mod = 2.0 * np.pi - eps
        # 简化衍射系数
        D = -np.exp(-1j * np.pi / 4.0) / (2.0 * n_edge * np.sqrt(2.0 * np.pi * max(k, 1e-10)))
        arg1 = (np.pi + Phi_mod) / (2.0 * n_edge)
        arg2 = (np.pi - Phi_mod) / (2.0 * n_edge)
        # 避免 tan ≈ 0 导致的奇点
        tan1 = np.tan(arg1)
        tan2 = np.tan(arg2)
        tan1 = np.where(abs(tan1) < 1e-10, 1e10 * np.sign(tan1) if tan1 != 0 else 1e10, tan1)
        tan2 = np.where(abs(tan2) < 1e-10, 1e10 * np.sign(tan2) if tan2 != 0 else 1e10, tan2)
        term1 = 1.0 / tan1
        term2 = 1.0 / tan2
        D = D * (term1 + term2)
        # 抑制过大值
        if np.abs(D) > 100.0:
            D = 100.0 * np.exp(1j * np.angle(D))
        return D
    else:
        return 0.0 + 0.0j


def keller_cone_direction(edge_point, edge_tangent, incident_dir):
    """
    Keller 锥条件：衍射射线位于以边缘为轴、半顶角为 β 的锥面上，
    其中 cos(β) = |incident_dir · edge_tangent|。
    衍射方向满足：d_dif · edge_tangent = ± d_inc · edge_tangent
    """
    cos_beta = np.dot(incident_dir, edge_tangent)
    # 衍射方向在 Keller 锥上的随机采样
    # 锥轴 = edge_tangent，半顶角 = arccos(|cos_beta|)
    beta = np.arccos(np.clip(abs(cos_beta), 0.0, 1.0))
    # 在锥面上均匀采样角度
    theta = np.random.uniform(0.0, 2.0 * np.pi)
    # 构造锥面上的方向
    # 先找到与 edge_tangent 垂直的基
    if abs(abs(cos_beta) - 1.0) < 1e-10:
        # 入射方向与边缘平行
        perp1 = np.array([1.0, 0.0, 0.0])
        if abs(edge_tangent[0]) > 0.9:
            perp1 = np.array([0.0, 1.0, 0.0])
    else:
        perp1 = incident_dir - cos_beta * edge_tangent
        perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(edge_tangent, perp1)
    perp2 = perp2 / np.linalg.norm(perp2)

    # 锥面上的方向
    d_diff = np.cos(beta) * edge_tangent + np.sin(beta) * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
    # 选择远离入射方向的一侧
    if np.dot(d_diff, incident_dir) > 0:
        d_diff = np.cos(beta) * edge_tangent - np.sin(beta) * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
    d_diff = d_diff / np.linalg.norm(d_diff)
    return d_diff


def detect_room_edges(surfaces):
    """
    检测房间几何中的边缘（墙面交线）。
    基于 edge 检测函数的阶跃/不连续思想。
    返回边缘列表：(point, tangent, surface1, surface2)。
    """
    edges = []
    # 长方体房间的12条边
    room_edges = [
        # 地板边
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 'floor', 'front_wall'),
        (np.array([10.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 'floor', 'front_wall'),
        (np.array([10.0, 8.0, 0.0]), np.array([-1.0, 0.0, 0.0]), 'floor', 'back_wall'),
        (np.array([0.0, 8.0, 0.0]), np.array([0.0, -1.0, 0.0]), 'floor', 'back_wall'),
        # 天花边
        (np.array([0.0, 0.0, 5.0]), np.array([1.0, 0.0, 0.0]), 'ceiling', 'front_wall'),
        (np.array([10.0, 0.0, 5.0]), np.array([0.0, 1.0, 0.0]), 'ceiling', 'front_wall'),
        (np.array([10.0, 8.0, 5.0]), np.array([-1.0, 0.0, 0.0]), 'ceiling', 'back_wall'),
        (np.array([0.0, 8.0, 5.0]), np.array([0.0, -1.0, 0.0]), 'ceiling', 'back_wall'),
        # 垂直边
        (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'left_wall', 'front_wall'),
        (np.array([10.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'right_wall', 'front_wall'),
        (np.array([10.0, 8.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'right_wall', 'back_wall'),
        (np.array([0.0, 8.0, 0.0]), np.array([0.0, 0.0, 1.0]), 'left_wall', 'back_wall'),
    ]
    for point, tangent, s1, s2 in room_edges:
        tangent = tangent / np.linalg.norm(tangent)
        edges.append({'point': point, 'tangent': tangent, 'surf1': s1, 'surf2': s2})
    return edges


def distance_point_to_edge(point, edge_point, edge_tangent):
    """
    点到无限长直线的距离：
        d = ||(p - p0) × t|| / ||t||
    """
    diff = point - edge_point
    cross = np.cross(diff, edge_tangent)
    return np.linalg.norm(cross) / np.linalg.norm(edge_tangent)


def diffraction_edge_response(source_pos, receiver_pos, edge, freq):
    """
    计算单条边缘对声场的衍射贡献（UTD 简化模型）。

    总声压：
        p_dif = D * sqrt(ρ / (s(ρ + s))) * exp(-iks)
    其中 ρ 为从源到边缘的距离，s 为从边缘到接收点的距离。
    """
    k = 2.0 * np.pi * freq / C_AIR
    ep = edge['point']
    et = edge['tangent']

    # 投影到边缘上的最近点
    vec_s = source_pos - ep
    vec_r = receiver_pos - ep
    t_s = np.dot(vec_s, et)
    t_r = np.dot(vec_r, et)
    # 使用边缘中点作为衍射点
    t_edge = (t_s + t_r) / 2.0
    diff_point = ep + t_edge * et

    d_source = np.linalg.norm(source_pos - diff_point)
    d_receiver = np.linalg.norm(receiver_pos - diff_point)

    if d_source < 1e-3 or d_receiver < 1e-3:
        return 0.0

    # 入射角和衍射角（相对于边缘法平面）
    inc_dir = (source_pos - diff_point) / d_source
    rec_dir = (receiver_pos - diff_point) / d_receiver

    # 在边缘法平面上的投影角
    inc_perp = inc_dir - np.dot(inc_dir, et) * et
    rec_perp = rec_dir - np.dot(rec_dir, et) * et
    inc_norm = np.linalg.norm(inc_perp)
    rec_norm = np.linalg.norm(rec_perp)
    if inc_norm < 1e-14 or rec_norm < 1e-14:
        phi_inc = 0.0
        phi_diff = 0.0
    else:
        inc_perp = inc_perp / inc_norm
        rec_perp = rec_perp / rec_norm
        phi_inc = np.arctan2(inc_perp[1], inc_perp[0])
        phi_diff = np.arctan2(rec_perp[1], rec_perp[0])

    D = edge_diffraction_coefficient(abs(phi_inc), abs(phi_diff), k)

    # 传播因子
    rho = d_source
    s = d_receiver
    spread = np.sqrt(rho / (s * (rho + s)))
    phase = np.exp(-1j * k * s)
    amplitude = spread * phase * D

    return amplitude


def integrate_diffraction_over_surface(surfaces, source_pos, edge, freq, precision=5):
    """
    使用三角形求积在房间表面上积分衍射场强。
    基于 triangle_symq_rule 的高阶求积。
    """
    total = 0.0
    for name, tris in surfaces.items():
        for i in range(0, len(tris), 3):
            v0, v1, v2 = tris[i], tris[i + 1], tris[i + 2]
            def func(p):
                return np.abs(diffraction_edge_response(source_pos, p, edge, freq))
            val = integrate_over_triangle(func, v0, v1, v2, precision)
            total += val
    return total


def compute_edge_diffraction_field(source_pos, receiver_pos, edges, freq):
    """
    计算所有边缘对接收点的总衍射声压贡献。
    """
    total = 0.0 + 0.0j
    for edge in edges:
        amp = diffraction_edge_response(source_pos, receiver_pos, edge, freq)
        total += amp
    return total
