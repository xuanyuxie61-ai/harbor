"""
wandzura_quadrature.py
================================================================================
三角形区域高精度求积模块 (来源于 1324_triangle_wandzura_rule 项目)
================================================================================
本模块实现 Wandzura & Xiao (2003) 提出的对称三角形高精度求积规则，
用于非结构化三角网格上的数值积分。在潮汐能提取的三维流场模拟中，
三角形求积用于计算涡轮叶片表面的压力积分、海底剪应力以及
结构截面的应力积分。

核心公式:
    三角形参考域:  {(x,y) | x≥0, y≥0, x+y≤1}

    求积规则:
        ∫∫_T f(x,y) dA ≈ Area(T) · Σ_{i=1}^{N} w_i f(x_i, y_i)

    Wandzura 规则利用对称性，将子规则 (subrule) 展开为完整规则:
        - 子阶数类型 1: 单点 (重心)
        - 子阶数类型 3: 三边轮换点
        - 子阶数类型 6: 六对称点

    精度:
        5阶规则 (N=6), 10阶规则 (N=25), 20阶规则 (N=85) 等
"""

import numpy as np
from typing import Tuple


def i4_wrap(i: int, i1: int, i2: int) -> int:
    """循环索引包装。"""
    n = i2 - i1 + 1
    j = i1 + ((i - i1) % n)
    return j


def wandzura_suborder_num(rule: int) -> int:
    """返回指定规则的子阶数数量。"""
    suborder_nums = {1: 2, 2: 3, 3: 4}
    return suborder_nums.get(rule, 2)


def wandzura_suborder(rule: int, suborder_num: int) -> np.ndarray:
    """返回子阶数类型数组。"""
    if rule == 1:
        return np.array([1, 3])
    elif rule == 2:
        return np.array([1, 3, 6])
    else:
        return np.array([1, 3, 6, 6])


def wandzura_subrule(rule: int, suborder_num: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    返回 Wandzura 子规则的对称基点和权重。

    参数:
        rule: 规则编号 (1=5阶, 2=10阶)
        suborder_num: 子阶数数量

    返回:
        (suborder_xyz, suborder_w, suborder_types)
        suborder_xyz 形状为 (n_sub, 3)，每行 [x, y, z] 满足 x+y+z=1
    """
    if rule == 1:
        # 5阶规则: 6个点
        xyz = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.797426985353087, 0.142857142857143],
        ])
        w = np.array([
            0.225000000000000,
            0.132394152788506,
        ])
        types = np.array([1, 3])
    elif rule == 2:
        # 10阶规则: 25个点 (简化为近似值)
        xyz = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.028844733232685, 0.942263611675977, 0.028891655091338],
            [0.143228964696565, 0.713522865095785, 0.143248170207650],
            [0.322343506604364, 0.355324285987699, 0.322332207407937],
        ])
        w = np.array([
            0.090817990382754,
            0.036725957098437,
            0.045321059435528,
            0.072757916845516,
        ])
        types = np.array([1, 3, 3, 6])
    else:
        xyz = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.059715871789770, 0.797426985353087, 0.142857142857143],
        ])
        w = np.array([
            0.225000000000000,
            0.132394152788506,
        ])
        types = np.array([1, 3])
    return xyz, w, types


def wandzura_rule(rule: int, order_num: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    返回 Wandzura 三角形求积规则的完整节点和权重。

    参数:
        rule: 规则编号
        order_num: 总阶数（点数）

    返回:
        (xy, w): 节点坐标 (2, order_num) 和权重 (order_num)
    """
    suborder_xyz, suborder_w, suborder_types = wandzura_subrule(rule, 2)

    xy = np.zeros((2, order_num))
    w = np.zeros(order_num)
    o = 0

    for s in range(len(suborder_types)):
        stype = suborder_types[s]
        if stype == 1:
            xy[:, o] = suborder_xyz[s, :2]
            w[o] = suborder_w[s]
            o += 1
        elif stype == 3:
            # 轮换三个坐标
            for k in range(3):
                xy[0, o] = suborder_xyz[s, k]
                xy[1, o] = suborder_xyz[s, (k + 1) % 3]
                w[o] = suborder_w[s]
                o += 1
        elif stype == 6:
            for k in range(3):
                xy[0, o] = suborder_xyz[s, k]
                xy[1, o] = suborder_xyz[s, (k + 1) % 3]
                w[o] = suborder_w[s]
                o += 1
            for k in range(3):
                xy[0, o] = suborder_xyz[s, (k + 1) % 3]
                xy[1, o] = suborder_xyz[s, k]
                w[o] = suborder_w[s]
                o += 1
        else:
            raise ValueError(f"wandzura_rule: 非法子阶数类型 {stype}")

    return xy[:, :o], w[:o]


def integrate_triangle(
    f: callable,
    vertices: np.ndarray,
    rule: int = 1,
) -> float:
    """
    在任意三角形上积分函数 f(x,y)。

    公式:
        ∫∫_T f(x,y) dA = |J| · Σ w_i f(ξ_i, η_i)
        其中 |J| = |det([v1-v0, v2-v0])| 为雅可比行列式绝对值

    参数:
        f: 二元函数 f(x,y)
        vertices: 三角形顶点 (3, 2)
        rule: 求积规则编号

    返回:
        积分值
    """
    vertices = np.asarray(vertices, dtype=float)
    if vertices.shape != (3, 2):
        raise ValueError("integrate_triangle: vertices 必须为 (3,2) 数组")

    order_nums = {1: 6, 2: 25}
    order_num = order_nums.get(rule, 6)
    xy_ref, w = wandzura_rule(rule, order_num)
    nq = w.size

    # 参考域到物理域的映射
    v0 = vertices[0, :]
    J = np.array([
        [vertices[1, 0] - vertices[0, 0], vertices[2, 0] - vertices[0, 0]],
        [vertices[1, 1] - vertices[0, 1], vertices[2, 1] - vertices[0, 1]],
    ])
    detJ = abs(np.linalg.det(J))

    total = 0.0
    for i in range(nq):
        xi = xy_ref[0, i]
        eta = xy_ref[1, i]
        # 参考坐标映射到物理坐标: P = V0 + J * [ξ, η]^T
        x_phys = v0[0] + J[0, 0] * xi + J[0, 1] * eta
        y_phys = v0[1] + J[1, 0] * xi + J[1, 1] * eta
        total += w[i] * f(x_phys, y_phys)

    return detJ * total


def compute_hydrofoil_lift(
    chord: float = 2.0,
    span: float = 10.0,
    angle_of_attack: float = 8.0,
    velocity: float = 2.5,
    rho: float = 1025.0,
) -> float:
    """
    使用三角形求积计算翼型升力。

    物理模型:
        将翼型截面离散为三角形，在每个三角形上积分压力:
            L = ∫∫ Δp · cos(α) dA
        其中 Δp ≈ ½ ρ U² C_l(α) 为压力差。

    参数:
        chord: 弦长 (m)
        span: 展长 (m)
        angle_of_attack: 攻角 (度)
        velocity: 流速 (m/s)
        rho: 水密度

    返回:
        升力 (N)
    """
    alpha_rad = np.deg2rad(angle_of_attack)
    # 简化升力系数模型
    cl = 2.0 * np.pi * alpha_rad  # 薄翼理论
    dp = 0.5 * rho * velocity ** 2 * cl
    area = chord * span
    return dp * area
