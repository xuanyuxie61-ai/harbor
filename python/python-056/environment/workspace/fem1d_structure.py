"""
fem1d_structure.py
================================================================================
一维有限元结构分析模块 (来源于 395_fem1d_pack 项目)
================================================================================
本模块实现一维有限元方法 (FEM) 用于分析潮汐能提取系统的支撑结构
（塔架、系泊缆索等）。基于拉格朗日插值基函数和Gauss-Legendre求积，
求解欧拉-伯努利梁方程和轴向变形方程。

核心公式:
    欧拉-伯努利梁方程:
        ρA ∂²w/∂t² + EI ∂⁴w/∂x⁴ = q(x,t)

    弱形式:
        ∫ EI w'' v'' dx + ∫ ρA ẅ v dx = ∫ q v dx

    基函数 (局部拉格朗日插值):
        φ_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

    刚度矩阵元素:
        K_{ij} = ∫ EI φ_i'' φ_j'' dx

    质量矩阵元素:
        M_{ij} = ∫ ρA φ_i φ_j dx

    力向量:
        F_i = ∫ q φ_i dx
"""

import numpy as np
from typing import Tuple, Callable


def local_basis_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    """
    计算一维拉格朗日插值基函数。

    公式:
        φ_i(x) = Π_{j≠i} (x - x_j) / (x_i - x_j)

    参数:
        order: 单元阶数
        node_x: 节点坐标，长度 order
        x: 计算点

    返回:
        phi: 基函数值，长度 order
    """
    node_x = np.asarray(node_x, dtype=float).flatten()
    phi = np.ones(order)
    for i in range(order):
        for j in range(order):
            if i != j:
                if abs(node_x[i] - node_x[j]) < 1e-14:
                    raise ValueError("local_basis_1d: 节点坐标重复")
                phi[i] *= (x - node_x[j]) / (node_x[i] - node_x[j])
    return phi


def local_basis_prime_1d(order: int, node_x: np.ndarray, x: float) -> np.ndarray:
    """
    计算一维基函数的一阶导数。

    公式:
        φ_i'(x) = Σ_{k≠i} [1/(x_i - x_k)] · Π_{j≠i,k} (x - x_j)/(x_i - x_j)

    参数:
        order: 单元阶数
        node_x: 节点坐标
        x: 计算点

    返回:
        dphi: 导数值
    """
    node_x = np.asarray(node_x, dtype=float).flatten()
    dphi = np.zeros(order)
    for i in range(order):
        for k in range(order):
            if i != k:
                prod = 1.0
                for j in range(order):
                    if j != i and j != k:
                        prod *= (x - node_x[j]) / (node_x[i] - node_x[j])
                dphi[i] += prod / (node_x[i] - node_x[k])
    return dphi


def legendre_com(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 Gauss-Legendre 求积的节点和权重。

    公式:
        ∫_{-1}^{1} f(x) dx ≈ Σ_{i=1}^{n} w_i f(x_i)

    参数:
        n: 求积阶数

    返回:
        (xtab, weight): 节点和权重数组
    """
    if n < 1:
        raise ValueError("legendre_com: n 必须大于 0")

    xtab = np.zeros(n)
    weight = np.zeros(n)
    e1 = n * (n + 1)
    m = (n + 1) // 2

    for i in range(1, m + 1):
        mp1mi = m + 1 - i
        t = np.pi * (4.0 * i - 1.0) / (4.0 * n + 2.0)
        x0 = np.cos(t) * (1.0 - (1.0 - 1.0 / n) / (8.0 * n * n))

        # Newton 迭代求根
        for _ in range(10):
            pkm1 = 1.0
            pk = x0
            for k in range(2, n + 1):
                pkp1 = 2.0 * x0 * pk - pkm1 - (x0 * pk - pkm1) / k
                pkm1 = pk
                pk = pkp1
            d1 = n * (pkm1 - x0 * pk)
            dpn = d1 / (1.0 - x0 * x0)
            dx = pk / dpn
            x0 = x0 - dx
            if abs(dx) < 1e-14:
                break

        xtab[mp1mi - 1] = x0

        fx = d1
        weight[mp1mi - 1] = 2.0 * (1.0 - x0 * x0) / (fx * fx)

    if n % 2 == 1:
        xtab[m - 1] = 0.0

    # 对称反射负半轴
    for i in range(1, m + 1):
        xtab[n - i] = -xtab[i - 1]
        weight[n - i] = weight[i - 1]

    return xtab, weight


def assemble_beam_system(
    n_elements: int,
    length: float,
    E: float,
    I: float,
    rho: float,
    A: float,
    q_func: Callable[[np.ndarray], np.ndarray],
    order: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    组装欧拉-伯努利梁的刚度矩阵、质量矩阵和力向量。

    参数:
        n_elements: 单元数
        length: 梁长度 (m)
        E: 弹性模量 (Pa)
        I: 截面惯性矩 (m⁴)
        rho: 材料密度 (kg/m³)
        A: 截面积 (m²)
        q_func: 分布载荷函数 q(x)
        order: 单元阶数 (默认线性)

    返回:
        (K, M, F): 刚度矩阵、质量矩阵、力向量
    """
    n_nodes = n_elements * order + 1
    x_nodes = np.linspace(0.0, length, n_nodes)
    h = length / n_elements

    K = np.zeros((n_nodes, n_nodes))
    M = np.zeros((n_nodes, n_nodes))
    F = np.zeros(n_nodes)

    xtab, wgt = legendre_com(4)

    for e in range(n_elements):
        x_e = x_nodes[e * order : (e + 1) * order + 1]
        x_mid = 0.5 * (x_e[0] + x_e[-1])
        jac = 0.5 * (x_e[-1] - x_e[0])

        Ke = np.zeros((order + 1, order + 1))
        Me = np.zeros((order + 1, order + 1))
        Fe = np.zeros(order + 1)

        for iq in range(len(xtab)):
            xi = xtab[iq]
            x_phys = x_mid + jac * xi
            w = wgt[iq] * jac

            phi = local_basis_1d(order + 1, x_e, x_phys)
            dphi = local_basis_prime_1d(order + 1, x_e, x_phys)

            for i in range(order + 1):
                for j in range(order + 1):
                    # 简化: 使用一阶导数近似弯曲刚度 (避免高阶导数复杂性)
                    Ke[i, j] += E * I * dphi[i] * dphi[j] * w / (h * h)
                    Me[i, j] += rho * A * phi[i] * phi[j] * w
                Fe[i] += q_func(np.array([x_phys]))[0] * phi[i] * w

        # 组装到全局矩阵
        for i in range(order + 1):
            gi = e * order + i
            for j in range(order + 1):
                gj = e * order + j
                K[gi, gj] += Ke[i, j]
                M[gi, gj] += Me[i, j]
            F[gi] += Fe[i]

    return K, M, F


def solve_beam_static(
    n_elements: int = 20,
    length: float = 30.0,
    E: float = 2.1e11,
    I: float = 0.5,
    rho: float = 7850.0,
    A: float = 2.0,
    drag_force: float = 5.0e4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    求解潮汐能支撑塔架的静态变形。

    边界条件:
        - 底部固定: w(0) = 0, w'(0) = 0
        - 顶部自由: 承受流体拖曳力

    物理模型:
        塔架简化为悬臂梁，顶部受集中载荷 F_drag。
        分布载荷近似为线性分布 q(x) = q0 (1 - x/L)。

    参数:
        n_elements: 单元数
        length: 塔架高度 (m)
        E: 弹性模量 (Pa)，钢材默认 2.1e11
        I: 截面惯性矩 (m⁴)
        rho: 密度 (kg/m³)
        A: 截面积 (m²)
        drag_force: 顶部拖曳力 (N)

    返回:
        (x, w): 节点坐标和挠度
    """
    def q_func(x_arr: np.ndarray) -> np.ndarray:
        # 线性分布载荷 + 顶部集中力等效分布
        x = np.asarray(x_arr)
        q0 = drag_force / length
        return q0 * (1.0 - x / length)

    K, M, F = assemble_beam_system(n_elements, length, E, I, rho, A, q_func)
    n_nodes = n_elements * 2 + 1

    # 施加固定端边界条件 (节点 0)
    K_reduced = K[1:n_nodes, 1:n_nodes]
    F_reduced = F[1:n_nodes]

    # 数值稳定性检查
    cond_num = np.linalg.cond(K_reduced)
    if cond_num > 1e14:
        # 使用正则化
        K_reduced += 1e-8 * np.eye(n_nodes - 1) * np.max(np.abs(K_reduced))

    w_reduced = np.linalg.solve(K_reduced, F_reduced)
    w = np.concatenate(([0.0], w_reduced))
    x = np.linspace(0.0, length, n_nodes)
    return x, w


def compute_mooring_tension(
    anchor_distance: float = 200.0,
    water_depth: float = 40.0,
    line_density: float = 50.0,
    horizontal_force: float = 1.0e6,
) -> float:
    """
    计算系泊缆索的张力 (悬链线方程的线性化 FEM 近似)。

    公式:
        悬链线方程: y = a cosh(x/a) - a
        其中 a = H / (ρ_line · g)

    参数:
        anchor_distance: 锚点水平距离 (m)
        water_depth: 水深 (m)
        line_density: 缆索线密度 (kg/m)
        horizontal_force: 水平张力 (N)

    返回:
        最大张力 (N)
    """
    g = 9.81
    a = horizontal_force / (line_density * g)
    # 检查是否触底
    y_mid = a * (np.cosh(anchor_distance / (2.0 * a)) - 1.0)
    if y_mid > water_depth:
        # 简化: 线性近似
        slope = water_depth / (0.5 * anchor_distance)
        T_max = horizontal_force * np.sqrt(1.0 + slope ** 2)
    else:
        T_max = horizontal_force * np.cosh(anchor_distance / (2.0 * a))
    return float(T_max)
