"""
Shepard 插值解传递模块

融合自:
- 1073_shepard_interp_nd: n维 Shepard 插值

Shepard 插值公式:
    u(x) = Σ_{i=1}^{N} w_i(x) * u_i

其中权重 w_i(x) 采用逆距离加权:
    w_i(x) = ||x - x_i||^{-p} / Σ_{j=1}^{N} ||x - x_j||^{-p}

当 p = 0 时，退化为简单平均:
    w_i(x) = 1/N

当插值点恰好与数据点重合时，直接返回该数据点的值:
    w_i(x_i) = 1,  w_j(x_i) = 0 (j ≠ i)

数学性质:
    - 精确再生常数: Shepard 插值精确再现零次多项式
    - 局部性: 大 p 值使插值更局部化
    - 正权重: 保持解的极值性质（无振荡）
"""

import numpy as np


def shepard_interp_nd(m, xd, zd, p, xi):
    """
    n维 Shepard 插值。
    
    对应原 shepard_interp_nd.m 的核心功能。
    
    Parameters
    ----------
    m : int
        空间维度
    xd : ndarray, shape (nd, m)
        数据点坐标
    zd : ndarray, shape (nd,)
        数据点处的函数值
    p : float
        幂指数 (通常 p = 2 或 p = 3)
    xi : ndarray, shape (ni, m)
        插值点坐标
    
    Returns
    -------
    zi : ndarray, shape (ni,)
        插值点处的函数值
    """
    nd = xd.shape[0]
    ni = xi.shape[0]

    if xd.shape[1] != m or xi.shape[1] != m:
        raise ValueError("shepard_interp_nd: 维度不匹配")

    zi = np.zeros(ni)

    for i in range(ni):
        if p == 0.0:
            w = np.ones(nd) / nd
        else:
            w = np.zeros(nd)
            exact_match = -1

            for j in range(nd):
                dist = np.linalg.norm(xi[i] - xd[j])
                if dist < 1e-14:
                    exact_match = j
                    break
                w[j] = dist

            if exact_match >= 0:
                w = np.zeros(nd)
                w[exact_match] = 1.0
            else:
                w = 1.0 / (w ** p)
                s = np.sum(w)
                if s > 1e-14:
                    w = w / s
                else:
                    w = np.ones(nd) / nd

        zi[i] = np.dot(w, zd)

    return zi


def prolongation_shepard(coarse_nodes, coarse_solution, fine_nodes, p=2.0):
    """
    将粗网格解 prolongate 到细网格。
    
    在自适应多重网格方法中，prolongation 算子 I_{2h}^h 将
    粗网格空间 V_{2h} 的解映射到细网格空间 V_h:
        u_h = I_{2h}^h u_{2h}
    
    Parameters
    ----------
    coarse_nodes : ndarray, shape (n_coarse, 2)
        粗网格节点坐标
    coarse_solution : ndarray, shape (n_coarse,)
        粗网格上的解
    fine_nodes : ndarray, shape (n_fine, 2)
        细网格节点坐标
    p : float
        Shepard 插值幂指数
    
    Returns
    -------
    fine_solution : ndarray, shape (n_fine,)
        细网格上的插值解
    """
    if len(coarse_nodes) == 0 or len(fine_nodes) == 0:
        raise ValueError("prolongation_shepard: 空网格")

    m = coarse_nodes.shape[1]
    fine_solution = shepard_interp_nd(m, coarse_nodes, coarse_solution, p, fine_nodes)
    return fine_solution


def restriction_integral(coarse_nodes, coarse_triangles, fine_nodes, fine_solution):
    """
    通过局部积分将细网格解 restrict 到粗网格。
    
    限制算子 I_h^{2h} 通常取 prolongation 的转置（在 Galerkin 意义下）:
        u_{2h} = I_h^{2h} u_h
    
    本实现采用积分平均:
        u_{2h}(P_i) = (∫_{Ω_i} u_h dx) / |Ω_i|
    其中 Ω_i 是粗节点 P_i 的 Voronoi 单元或支撑域。
    
    简化实现：对每个粗节点，找到附近的细节点并加权平均。
    
    Parameters
    ----------
    coarse_nodes : ndarray, shape (n_coarse, 2)
    coarse_triangles : ndarray, shape (n_tri, 3)
    fine_nodes : ndarray, shape (n_fine, 2)
    fine_solution : ndarray, shape (n_fine,)
    
    Returns
    -------
    coarse_solution : ndarray, shape (n_coarse,)
    """
    n_coarse = len(coarse_nodes)
    coarse_solution = np.zeros(n_coarse)

    for i in range(n_coarse):
        # 找到距离粗节点最近的细节点（简单策略）
        dists = np.linalg.norm(fine_nodes - coarse_nodes[i], axis=1)
        # 使用高斯权重
        sigma = np.mean(dists[dists > 1e-14]) if np.any(dists > 1e-14) else 1.0
        weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))
        weights_sum = np.sum(weights)
        if weights_sum > 1e-14:
            coarse_solution[i] = np.sum(weights * fine_solution) / weights_sum
        else:
            coarse_solution[i] = fine_solution[np.argmin(dists)]

    return coarse_solution


def solution_transfer_between_meshes(
    old_nodes, old_triangles, old_solution,
    new_nodes, new_triangles, transfer_type='shepard'
):
    """
    在自适应网格细化/粗化后，将解从旧网格传递到新网格。
    
    Parameters
    ----------
    old_nodes : ndarray
    old_triangles : ndarray
    old_solution : ndarray
    new_nodes : ndarray
    new_triangles : ndarray
    transfer_type : str
        'shepard' 或 'integral'
    
    Returns
    -------
    new_solution : ndarray
    """
    n_new = len(new_nodes)

    if transfer_type == 'shepard':
        new_solution = prolongation_shepard(old_nodes, old_solution, new_nodes, p=2.0)
    elif transfer_type == 'integral':
        new_solution = restriction_integral(old_nodes, old_triangles, new_nodes, old_solution)
        # 如果新网格更细，需要反过来
        if len(new_nodes) > len(old_nodes):
            new_solution = prolongation_shepard(old_nodes, old_solution, new_nodes, p=2.0)
    else:
        raise ValueError(f"solution_transfer_between_meshes: 未知的 transfer_type={transfer_type}")

    return new_solution
