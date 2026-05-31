"""
section_fem.py
软体机器人横截面有限元分析模块

融合种子项目:
- 375_fem_basis_t6_display: 二次三角形(T6)形函数及导数
- 344_exactness: 高斯求积规则精确性测试与积分计算
- 276_diaphony: 采样点均匀性评估

科学应用: 软体机器人横截面的惯性矩、剪切系数与应力分布的有限元计算
使用6节点二次三角形(T6)单元与高斯-勒让德求积
"""

import numpy as np
from typing import Tuple, List, Callable
from mesh_utils import diaphony_compute


def shape_t6(xi: float, eta: float, node_idx: int) -> float:
    """
    T6二次三角形标准单元上的形函数
    标准三角形顶点: (0,0), (1,0), (0,1)
    节点编号: 1(0,0), 2(1,0), 3(0,1), 4(0.5,0), 5(0.5,0.5), 6(0,0.5)

    形函数:
        N1 = (2xi + 2eta - 1)(xi + eta - 1)   [顶点]
        N2 = xi(2xi - 1)                       [顶点]
        N3 = eta(2eta - 1)                     [顶点]
        N4 = -4xi(xi + eta - 1)                [边中点]
        N5 = 4xi*eta                           [边中点]
        N6 = -4eta(xi + eta - 1)               [边中点]
    """
    if xi < -1e-12 or eta < -1e-12 or xi + eta > 1.0 + 1e-12:
        return 0.0
    if node_idx == 0:
        return (2.0 * xi + 2.0 * eta - 1.0) * (xi + eta - 1.0)
    elif node_idx == 1:
        return xi * (2.0 * xi - 1.0)
    elif node_idx == 2:
        return eta * (2.0 * eta - 1.0)
    elif node_idx == 3:
        return -4.0 * xi * (xi + eta - 1.0)
    elif node_idx == 4:
        return 4.0 * xi * eta
    elif node_idx == 5:
        return -4.0 * eta * (xi + eta - 1.0)
    else:
        raise ValueError("node_idx must be in [0,5]")


def grad_shape_t6(xi: float, eta: float, node_idx: int) -> np.ndarray:
    """
    T6形函数对参考坐标(xi, eta)的梯度
    返回 [dN/dxi, dN/deta]
    """
    if node_idx == 0:
        dxi = 4.0 * xi + 4.0 * eta - 3.0
        deta = 4.0 * xi + 4.0 * eta - 3.0
    elif node_idx == 1:
        dxi = 4.0 * xi - 1.0
        deta = 0.0
    elif node_idx == 2:
        dxi = 0.0
        deta = 4.0 * eta - 1.0
    elif node_idx == 3:
        dxi = -4.0 * (2.0 * xi + eta - 1.0)
        deta = -4.0 * xi
    elif node_idx == 4:
        dxi = 4.0 * eta
        deta = 4.0 * xi
    elif node_idx == 5:
        dxi = -4.0 * eta
        deta = -4.0 * (xi + 2.0 * eta - 1.0)
    else:
        raise ValueError("node_idx must be in [0,5]")
    return np.array([dxi, deta])


def gauss_legendre_triangle(order: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    三角形上的高斯-勒让德求积点与权重
    基于种子项目344_exactness的求积思想

    对于标准三角形，使用Dunavant规则（简化版本）
    order=1: 1点, 精度2
    order=2: 3点, 精度3
    order=3: 4点, 精度4
    order=4: 6点, 精度5
    """
    if order <= 1:
        # 重心点
        xi = np.array([1.0 / 3.0])
        eta = np.array([1.0 / 3.0])
        w = np.array([0.5])
    elif order == 2:
        # 3点规则
        xi = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0])
        eta = np.array([1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0])
        w = np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    elif order == 3:
        # 4点规则
        a = 1.0 / 5.0
        b = 3.0 / 5.0
        xi = np.array([a, a, b, 1.0 / 3.0])
        eta = np.array([a, b, a, 1.0 / 3.0])
        w = np.array([25.0 / 96.0, 25.0 / 96.0, 25.0 / 96.0, -27.0 / 96.0])
    else:
        # 6点规则（order 4）
        a1 = 0.445948490915965
        b1 = 0.091576213509771
        a2 = 0.108103018168070
        b2 = 0.816847572980459
        w1 = 0.111690794839005
        w2 = 0.054975871827661
        xi = np.array([a1, 1.0 - 2.0 * a1, a1, b1, 1.0 - 2.0 * b1, b1])
        eta = np.array([a1, a1, 1.0 - 2.0 * a1, b1, b1, 1.0 - 2.0 * b1])
        w = np.array([w1, w1, w1, w2, w2, w2])
    return np.column_stack([xi, eta]), w


def integrate_triangle(f: Callable[[np.ndarray], np.ndarray],
                       nodes_phys: np.ndarray,
                       order: int = 3) -> float:
    """
    在物理三角形上积分函数f(x,y)
    使用参考三角形上的高斯求积 + 雅可比行列式变换

    坐标变换:
        x = x1 + (x2-x1)*xi + (x3-x1)*eta
        y = y1 + (y2-y1)*xi + (y3-y1)*eta

    雅可比矩阵:
        J = [[x2-x1, x3-x1], [y2-y1, y3-y1]]
        |det(J)| = |(x2-x1)(y3-y1) - (x3-x1)(y2-y1)|
    """
    if nodes_phys.shape != (3, 2):
        raise ValueError("nodes_phys must be (3,2)")

    qp, w = gauss_legendre_triangle(order)
    x1, y1 = nodes_phys[0]
    x2, y2 = nodes_phys[1]
    x3, y3 = nodes_phys[2]
    detJ = abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    result = 0.0
    for i in range(len(w)):
        xi, eta = qp[i]
        x = x1 + (x2 - x1) * xi + (x3 - x1) * eta
        y = y1 + (y2 - y1) * xi + (y3 - y1) * eta
        result += w[i] * f(np.array([x, y]))
    return result * detJ


def compute_section_properties(nodes: np.ndarray, triangles: np.ndarray) -> dict:
    """
    计算横截面的几何属性
    使用T6单元的高斯求积计算面积、形心、惯性矩

    返回字典包含:
        A: 截面积
        cx, cy: 形心坐标
        Ixx: 绕x轴惯性矩  ∫ y^2 dA
        Iyy: 绕y轴惯性矩  ∫ x^2 dA
        Ixy: 惯性积       ∫ xy dA
        J: 极惯性矩       Ixx + Iyy
        diaphony: 节点分布均匀性指标
    """
    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError("nodes must be (N,2)")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must be (M,3)")

    A = 0.0
    Sx = 0.0  # ∫ y dA
    Sy = 0.0  # ∫ x dA
    Ixx = 0.0
    Iyy = 0.0
    Ixy = 0.0

    for tri in triangles:
        tri_nodes = nodes[tri]
        # 使用3节点线性近似（T6顶点）进行积分
        # 面积积分
        A += integrate_triangle(lambda p: 1.0, tri_nodes, order=2)
        # 一阶矩
        Sx += integrate_triangle(lambda p: p[1], tri_nodes, order=2)
        Sy += integrate_triangle(lambda p: p[0], tri_nodes, order=2)
        # 二阶矩
        Ixx += integrate_triangle(lambda p: p[1] ** 2, tri_nodes, order=3)
        Iyy += integrate_triangle(lambda p: p[0] ** 2, tri_nodes, order=3)
        Ixy += integrate_triangle(lambda p: p[0] * p[1], tri_nodes, order=3)

    if abs(A) < 1e-14:
        A = 1e-14
    cx = Sy / A
    cy = Sx / A

    # 移轴定理: 计算关于形心的惯性矩
    Ixx_c = Ixx - A * cy ** 2
    Iyy_c = Iyy - A * cx ** 2
    Ixy_c = Ixy - A * cx * cy

    # Diaphony评估
    diaphony_val = diaphony_compute(nodes)

    return {
        'A': A,
        'cx': cx,
        'cy': cy,
        'Ixx': Ixx_c,
        'Iyy': Iyy_c,
        'Ixy': Ixy_c,
        'J': Ixx_c + Iyy_c,
        'diaphony': diaphony_val
    }


def compute_shear_correction_factor(nodes: np.ndarray, triangles: np.ndarray,
                                   E: float = 1.0, nu: float = 0.35) -> float:
    """
    计算Timoshenko剪切修正系数 kappa
    基于能量等效原理:
        kappa = (V^2 / (2*A*G)) / ∫ (tau^2 / (2*G)) dA

    简化模型: 对于椭圆形截面，kappa ≈ 0.886
    这里使用有限元近似进行数值计算
    """
    props = compute_section_properties(nodes, triangles)
    A = props['A']
    if abs(A) < 1e-14:
        return 0.886

    # 简化计算: 使用形心处剪切应力近似
    # tau = V*Q/(I*b), 这里用数值积分近似
    G = E / (2.0 * (1.0 + nu))

    # 数值积分计算分母
    denom = 0.0
    for tri in triangles:
        tri_nodes = nodes[tri]
        # 近似剪切应力分布: 线性变化
        denom += integrate_triangle(
            lambda p: (p[0] - props['cx']) ** 2 + (p[1] - props['cy']) ** 2,
            tri_nodes, order=2
        )

    if abs(denom) < 1e-14:
        return 0.886

    kappa = A ** 2 / (12.0 * denom)
    # 限制在合理范围
    kappa = max(0.5, min(1.0, kappa))
    return kappa


def assemble_section_stiffness(nodes: np.ndarray, triangles: np.ndarray,
                               E: float, nu: float) -> np.ndarray:
    """
    组装横截面的平面应力刚度矩阵（简化2D弹性）
    用于计算截面的等效弹性模量分布
    """
    nn = nodes.shape[0]
    K = np.zeros((2 * nn, 2 * nn))

    # 平面应力D矩阵
    D_mat = E / (1.0 - nu ** 2) * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ])

    for tri in triangles:
        tri_nodes = nodes[tri]
        x = tri_nodes[:, 0]
        y = tri_nodes[:, 1]

        # 3节点线性三角形B矩阵
        area = 0.5 * abs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 1e-14:
            continue

        # 计算B矩阵 (3x6)
        b1 = y[1] - y[2]
        b2 = y[2] - y[0]
        b3 = y[0] - y[1]
        c1 = x[2] - x[1]
        c2 = x[0] - x[2]
        c3 = x[1] - x[0]

        B = (1.0 / (2.0 * area)) * np.array([
            [b1, 0.0, b2, 0.0, b3, 0.0],
            [0.0, c1, 0.0, c2, 0.0, c3],
            [c1, b1, c2, b2, c3, b3]
        ])

        Ke = B.T @ D_mat @ B * area

        # 组装
        local_dof = []
        for nid in tri:
            local_dof.extend([2 * nid, 2 * nid + 1])
        for i in range(6):
            for j in range(6):
                gi, gj = local_dof[i], local_dof[j]
                if gi < K.shape[0] and gj < K.shape[1]:
                    K[gi, gj] += Ke[i, j]

    return K
