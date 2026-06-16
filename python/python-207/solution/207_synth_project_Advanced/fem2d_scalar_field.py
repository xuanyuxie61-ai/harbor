"""
fem2d_scalar_field.py — 2D FEM 标量场求解与不确定性传播

科学背景
========
在 2D 随机域上求解稳态扩散方程:
    -∇·(κ(x,y,ω) ∇u) = f(x,y)

使用三角线性有限元:
    K(κ) · U = F

其中 K 为全局刚度矩阵, 依赖于随机场 κ(x,y,ω).

算法来源 (种子项目 414_fem2d_scalar_display)
============================================
种子 414: 读取节点/单元/值文件, 显示标量场
本项目扩展为:
1. 从 1D 随机场扩展到 2D 随机场
2. 组装并求解 FEM 系统
3. 计算 2D 置信区域

核心公式
========
1. 单元刚度矩阵:
   K_e[i,j] = ∫_e κ·∇φ_i·∇φ_j dA
2. 线性三角形形函数:
   φ_i(x,y) = (a_i + b_i·x + c_i·y) / (2·A_e)
3. 应力强度因子:
   ∇φ_i = [b_i, c_i] / (2·A_e)
"""

import numpy as np
from mesh_vertex_to_element import (
    create_2d_triangulation, compute_triangle_geometry,
    assemble_fem_laplacian_2d, build_vtoe
)
from covariance_cholesky import CovarianceKernel
from kl_expansion import KLExpansion


def solve_fem2d_diffusion(nodes_xy, etov, kappa_field, source_field,
                           dirichlet_nodes=None, dirichlet_values=None):
    """求解 2D FEM 扩散方程.

    K · U = F

    参数
    ----
    nodes_xy : ndarray, shape (v_num, 2)
    etov : ndarray, shape (3, e_num)
    kappa_field : ndarray, shape (v_num,)
    source_field : ndarray, shape (v_num,)
    dirichlet_nodes : list of int or None
    dirichlet_values : list of float or None

    返回
    ----
    U : ndarray, shape (v_num,)
    """
    v_num = nodes_xy.shape[0]
    K = assemble_fem_laplacian_2d(nodes_xy, etov, kappa_field)

    # 载荷向量
    F = source_field.copy()

    # Dirichlet 边界条件 (大数法)
    if dirichlet_nodes is not None and dirichlet_values is not None:
        big = 1.0e20
        for node, val in zip(dirichlet_nodes, dirichlet_values):
            K[node, node] += big
            F[node] += big * val

    # 求解
    try:
        U = np.linalg.solve(K, F)
    except np.linalg.LinAlgError:
        # 退化: 使用伪逆
        U = np.linalg.lstsq(K, F, rcond=None)[0]

    return U


def generate_2d_random_field(x_grid, y_grid, kernel, n_modes=10, rng=None):
    """在 2D 网格上生成随机场的一个实现.

    参数
    ----
    x_grid, y_grid : ndarray
    kernel : CovarianceKernel
    n_modes : int
    rng : Generator

    返回
    ----
    field_2d : ndarray, shape (ny, nx)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    nx = len(x_grid)
    ny = len(y_grid)

    # 简化: 使用可分离核 C(x,y) = C_x(x) · C_y(y)
    kl_x = KLExpansion(kernel, x_grid, n_modes)
    kl_y = KLExpansion(kernel, y_grid, n_modes)

    xi_x = rng.standard_normal(n_modes)
    xi_y = rng.standard_normal(n_modes)

    z_x = kl_x.generate_realization(xi_x)
    z_y = kl_y.generate_realization(xi_y)

    # 张量积
    field_2d = np.outer(z_x, np.ones(ny)) + np.outer(np.ones(nx), z_y)
    field_2d /= np.sqrt(2.0)  # 归一化

    return field_2d


def fem2d_uq_pipeline(x_grid, y_grid, kappa_0, sigma_kappa,
                       length_scale, n_mc=50, n_modes=5):
    """2D FEM 不确定性量化流水线.

    参数
    ----
    x_grid, y_grid : ndarray
    kappa_0 : float
    sigma_kappa : float
    length_scale : float
    n_mc : int
    n_modes : int

    返回
    ----
    results : dict
    """
    rng = np.random.default_rng(42)
    kernel = CovarianceKernel(sigma_kappa, length_scale)

    nodes_xy, etov = create_2d_triangulation(x_grid, y_grid)
    v_num = nodes_xy.shape[0]

    # 边界节点
    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    tol = 1e-10

    left_nodes = [i for i in range(v_num) if abs(nodes_xy[i, 0] - x_min) < tol]
    right_nodes = [i for i in range(v_num) if abs(nodes_xy[i, 0] - x_max) < tol]
    bottom_nodes = [i for i in range(v_num) if abs(nodes_xy[i, 1] - y_min) < tol]
    top_nodes = [i for i in range(v_num) if abs(nodes_xy[i, 1] - y_max) < tol]

    dirichlet_nodes = list(set(left_nodes + right_nodes + bottom_nodes + top_nodes))
    dirichlet_values = [100.0 if nodes_xy[i, 0] < x_min + tol else 25.0
                         for i in dirichlet_nodes]

    solutions = np.zeros((n_mc, v_num))

    for m in range(n_mc):
        # 生成 2D 随机扩散系数场
        field_2d = generate_2d_random_field(
            x_grid, y_grid, kernel, n_modes, rng
        )
        kappa_field = kappa_0 * np.exp(
            (sigma_kappa / kappa_0) * field_2d.ravel() / np.sqrt(np.var(field_2d) + 1e-30)
        )
        kappa_field = np.maximum(kappa_field, kappa_0 * 0.1)

        # 源项
        source = np.zeros(v_num)

        # 求解
        U = solve_fem2d_diffusion(
            nodes_xy, etov, kappa_field, source,
            dirichlet_nodes, dirichlet_values
        )
        solutions[m] = U

    # 统计
    mean_field = np.mean(solutions, axis=0)
    std_field = np.std(solutions, axis=0, ddof=1) if n_mc > 1 else np.zeros(v_num)

    return {
        'nodes_xy': nodes_xy,
        'etov': etov,
        'solutions': solutions,
        'mean_field': mean_field,
        'std_field': std_field,
        'v_num': v_num,
        'e_num': etov.shape[1],
    }
