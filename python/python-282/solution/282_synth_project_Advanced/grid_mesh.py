"""
grid_mesh.py
============

一维 SEI 层计算网格构造与边界节点识别模块。

融合种子项目：
    1333_triangulation_boundary_nodes：边界节点识别（适配 1D）
    360_fd1d_heat_explicit：等距节点构造
    682_line_lines_packing：随机密堆积生成 SEI 纳米孔隙位置

核心功能：
    1. 构造一维等距网格 x ∈ [0, L_SEI]
    2. 识别 Dirichlet / Neumann / 混合边界节点
    3. 生成 SEI 内纳米孔隙的随机密堆积构型（Rényi 停车问题）
    4. 提供孔隙率分布（局部修正扩散系数）

作者: DA-Synthesis
"""

import math
import random
try:
    from . import sei_parameters as P
except ImportError:
    import sei_parameters as P


def build_equidistant_grid(n_nodes, length):
    """
    构造一维等距网格（参考 360_fd1d_heat_explicit 节点构造）。

    数学描述：
        x_i = i * dx,  i = 0, 1, ..., N-1
        dx  = L / (N - 1)

    Parameters
    ----------
    n_nodes : int
        网格节点数 N ≥ 2。
    length : float
        计算域长度 L > 0 [m]。

    Returns
    -------
    x : list[float]
        节点坐标数组 [m]。
    dx : float
        空间步长 [m]。
    """
    if n_nodes < 2:
        raise ValueError(f"网格节点数必须 >= 2，当前 n_nodes={n_nodes}")
    if length <= 0.0:
        raise ValueError(f"计算域长度必须 > 0，当前 length={length}")
    dx = length / (n_nodes - 1)
    x = [i * dx for i in range(n_nodes)]
    return x, dx


def identify_boundary_nodes(n_nodes):
    """
    识别一维网格的边界节点（参考 1333_triangulation_boundary_nodes 思想）。

    在一维网格中，边界节点为 i = 0 和 i = N-1。
    返回一个整数标记数组 is_boundary[i]：
        1 表示边界节点，0 表示内部节点。

    Parameters
    ----------
    n_nodes : int
        网格节点数。

    Returns
    -------
    is_boundary : list[int]
        边界标记数组。
    """
    if n_nodes < 2:
        raise ValueError("节点数不足")
    is_boundary = [0] * n_nodes
    is_boundary[0] = 1
    is_boundary[-1] = 1
    return is_boundary


def classify_boundary_types(n_nodes, bc_left="dirichlet", bc_right="neumann"):
    """
    对边界节点进行分类：Dirichlet (固定浓度) / Neumann (固定通量) / Robin。

    在 SEI 建模中：
        - 左边界（电极/SEI 界面）通常为 Butler-Volmer 混合边界
        - 右边界（SEI/电解液界面）通常为固定浓度 Dirichlet

    Parameters
    ----------
    n_nodes : int
        网格节点数。
    bc_left : str
        左边界类型。
    bc_right : str
        右边界类型。

    Returns
    -------
    bc_type : list[str]
        每个节点的边界类型描述（内部节点为 'interior'）。
    """
    bc_type = ["interior"] * n_nodes
    bc_type[0] = bc_left
    bc_type[-1] = bc_right
    return bc_type


def renysi_parking_packing(domain_length, car_width, n_attempts_max=50000,
                            rng_seed=None):
    """
    一维随机顺序吸附（RSA）— Rényi 停车问题（参考 682_line_lines_packing）。

    用于模拟 SEI 层中纳米孔隙的随机密堆积结构。
    算法：
        1. 随机选取位置 x ∈ [0, L - w]
        2. 检查与已存在"车辆"不重叠
        3. 若不重叠则放置，否则尝试失败
        4. 重复直到连续失败次数超过阈值

    理论极限：Rényi 停车常数 ≈ 0.7475979202

    Parameters
    ----------
    domain_length : float
        区域长度 [m]。
    car_width : float
        "车辆"（孔隙）宽度 [m]。
    n_attempts_max : int
        最大尝试次数。
    rng_seed : int or None
        随机数种子。

    Returns
    -------
    c_try : int
        总尝试次数。
    c_park : int
        成功放置次数（孔隙数）。
    positions : list[float]
        已放置孔隙中心位置。
    density : float
        堆积密度（孔隙占据比例）。
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    c_rad = car_width / 2.0
    positions = []
    c_park = 0
    c_try = 0
    consecutive_fail = 0

    while c_try < n_attempts_max:
        c_try += 1
        x_try = random.uniform(0.0, domain_length)

        # 检查与所有已放置车辆的间距
        overlap = False
        for x_park in positions:
            if abs(x_try - x_park) < car_width:
                overlap = True
                break

        if not overlap:
            positions.append(x_try)
            c_park += 1
            consecutive_fail = 0
        else:
            consecutive_fail += 1
            # 当连续失败次数远超理论阻塞阈值时终止
            if consecutive_fail > 10 * max(c_park, 1) + 500:
                break

    density = c_park * car_width / domain_length if domain_length > 0 else 0.0
    return c_try, c_park, positions, density


def build_porosity_field(n_nodes, dx, n_pores, pore_positions, pore_width):
    """
    基于密堆积孔隙位置构造局部孔隙率场。

    数学描述：
        phi(x_i) = 1 - sum_p w_p * chi_p(x_i)
    其中 chi_p 为指示函数，在孔隙范围内为 1。

    Parameters
    ----------
    n_nodes : int
        网格节点数。
    dx : float
        空间步长 [m]。
    n_pores : int
        孔隙数。
    pore_positions : list[float]
        孔隙中心位置 [m]。
    pore_width : float
        孔隙宽度 [m]。

    Returns
    -------
    porosity : list[float]
        每个网格节点的局部孔隙率 (0, 1]。
    """
    porosity = [1.0] * n_nodes
    half_w = pore_width / 2.0
    for i in range(n_nodes):
        x_i = i * dx
        for x_c in pore_positions:
            if abs(x_i - x_c) < half_w:
                porosity[i] = 0.0  # 孔隙位置无固相传输
                break
    return porosity


def run_grid_construction_demo():
    """
    演示网格构造、边界识别与孔隙密堆积。

    Returns
    -------
    dict
        包含 x, dx, is_boundary, bc_type, porosity, packing_info。
    """
    x, dx = build_equidistant_grid(P.N_GRID, P.L_DOMAIN)
    is_boundary = identify_boundary_nodes(P.N_GRID)
    bc_type = classify_boundary_types(
        P.N_GRID, bc_left="butler_volmer", bc_right="dirichlet"
    )

    # 纳米孔隙密堆积（典型孔隙直径 ~0.5 nm）
    pore_width = 0.5e-9  # m
    random.seed(P.RNG_SEED)
    c_try, c_park, positions, density = renysi_parking_packing(
        P.L_DOMAIN, pore_width, n_attempts_max=20000,
        rng_seed=P.RNG_SEED
    )
    porosity = build_porosity_field(P.N_GRID, dx, c_park, positions, pore_width)

    packing_info = {
        "n_pores": c_park,
        "n_attempts": c_try,
        "density": density,
        "renyi_theoretical": 0.7475979202,
    }

    return {
        "x": x,
        "dx": dx,
        "is_boundary": is_boundary,
        "bc_type": bc_type,
        "porosity": porosity,
        "packing_info": packing_info,
    }


if __name__ == "__main__":
    result = run_grid_construction_demo()
    print(f"[grid_mesh] 网格节点数: {P.N_GRID}")
    print(f"[grid_mesh] 空间步长 dx = {result['dx']:.4e} m")
    print(f"[grid_mesh] 左边界: {result['bc_type'][0]}")
    print(f"[grid_mesh] 右边界: {result['bc_type'][-1]}")
    pi = result['packing_info']
    print(f"[grid_mesh] 纳米孔隙密堆积: 成功 {pi['n_pores']}/{pi['n_attempts']} 次")
    print(f"[grid_mesh] 堆积密度 = {pi['density']:.4f} "
          f"(理论 Rényi 极限 ≈ {pi['renyi_theoretical']:.4f})")
