"""
mesh3d_generator.py
===================
基于距离函数的3D燃烧室非结构网格生成器。

核心算法源自 distmesh_3d (Project 309)，并改造用于生成湍流燃烧
数值模拟所需的3D计算域网格。

核心原理（Persson & Strang, SIAM Review 2004）：
----------------------------------------------
1. 在包围盒内均匀撒点，根据距离函数 fd(x,y,z) 筛选内部点；
2. 使用 Delaunay 三角剖分生成初始四面体网格；
3. 基于边长力平衡迭代优化节点位置：

   对每条边 e = (p_i, p_j)，定义归一化边长：
       L_e = |p_i - p_j| / h(x_m)
   其中 h(x_m) 为中点处的目标边长。

   边力（弹簧力模型）：
       F_e = max(L_e - L0, 0) * (p_j - p_i) / |p_j - p_i|

   节点更新：
       p_i^{new} = p_i + Δt * Σ F_e

4. 将越界点投影回边界（距离函数梯度步进）。

距离函数示例（圆柱形燃烧室）：
    fd_cylinder(x,y,z) = max( sqrt(x²+y²) - R, |z| - H/2 )

目标边长函数：
    h(x,y,z) = h0 * (1 + 0.5 * |fd(x,y,z)| / R)
"""

import numpy as np


def distance_cylinder(p, R=0.05, H=0.20):
    """
    圆柱形燃烧室的有向距离函数。

    Parameters
    ----------
    p : ndarray, shape (N, 3)
        节点坐标。
    R : float
        圆柱半径，单位 m。
    H : float
        圆柱高度，单位 m。

    Returns
    -------
    d : ndarray, shape (N,)
        有向距离，d < 0 表示内部。
    """
    if p.ndim == 1:
        p = p.reshape(1, -1)
    r = np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2)
    d1 = r - R
    d2 = np.abs(p[:, 2]) - H / 2.0
    return np.maximum(d1, d2)


def target_edge_length(p, h0, R=0.05):
    """
    目标边长函数：边界附近加密。

    Parameters
    ----------
    p : ndarray, shape (N, 3)
        节点坐标。
    h0 : float
        基准边长。
    R : float
        特征长度（圆柱半径）。

    Returns
    -------
    h : ndarray, shape (N,)
        目标边长。
    """
    if p.ndim == 1:
        p = p.reshape(1, -1)
    d = distance_cylinder(p, R)
    # 边界附近加密
    h = h0 * (1.0 + 0.5 * np.clip(np.abs(d) / R, 0.0, 2.0))
    return h


def generate_mesh_3d(h0=0.015, R=0.05, H=0.20, iteration_max=20, pfix=None):
    """
    生成3D圆柱形燃烧室的四面体网格。

    Parameters
    ----------
    h0 : float
        初始目标边长。
    R : float
        圆柱半径。
    H : float
        圆柱高度。
    iteration_max : int
        最大迭代次数。
    pfix : ndarray or None
        固定节点坐标。

    Returns
    -------
    p : ndarray, shape (N, 3)
        节点坐标。
    t : ndarray, shape (NT, 4)
        四面体单元索引（基于0）。
    """
    dim = 3
    ptol = 0.001
    ttol = 0.1
    L0mult = 1.0 + 0.4 / 2.0 ** (dim - 1)
    deltat = 0.1
    geps = 0.1 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0

    # 包围盒
    box = np.array([[-R, -R, -H / 2.0],
                    [R, R, H / 2.0]])

    # 1. 初始均匀撒点
    grids = [np.arange(box[0, i], box[1, i] + h0, h0) for i in range(dim)]
    mesh = np.meshgrid(*grids, indexing='ij')
    p = np.vstack([m.ravel() for m in mesh]).T

    # 2. 筛选内部点
    p = p[distance_cylinder(p, R, H) < geps]

    # 3. 应用拒绝采样使点分布更均匀
    r0 = target_edge_length(p, h0, R)
    if len(p) > 0:
        prob = np.min(r0) ** dim / (r0 ** dim)
        p = p[np.random.rand(len(p)) < prob]

    # 添加固定节点
    if pfix is not None and len(pfix) > 0:
        p = np.vstack([pfix, p])
    else:
        pfix = np.zeros((0, 3))

    # 去重
    p = np.unique(np.round(p / (h0 * 1.0e-6)) * (h0 * 1.0e-6), axis=0)

    if iteration_max <= 0:
        try:
            from scipy.spatial import Delaunay
            t = Delaunay(p).simplices
        except Exception:
            t = np.zeros((0, 4), dtype=int)
        return p, t

    N = len(p)
    count = 0
    p0 = np.inf * np.ones_like(p)
    t = np.zeros((0, 4), dtype=int)

    for iteration in range(iteration_max):
        # 3. 重三角化（如果节点移动显著）
        if ttol * h0 < np.max(np.sqrt(np.sum((p - p0) ** 2, axis=1))):
            p0 = p.copy()
            try:
                from scipy.spatial import Delaunay
                t_new = Delaunay(p).simplices
            except Exception:
                break

            # 筛选质心在内部的四面体
            pmid = np.zeros((len(t_new), dim))
            for ii in range(dim + 1):
                pmid += p[t_new[:, ii], :] / (dim + 1)
            t_new = t_new[distance_cylinder(pmid, R, H) < -geps]
            t = t_new
            count += 1

        if len(t) == 0:
            break

        # 4. 提取唯一边
        edges = []
        for i in range(dim + 1):
            for j in range(i + 1, dim + 1):
                edges.append(t[:, [i, j]])
        edges = np.vstack(edges)
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

        # 6. 边长力平衡
        bars = p[edges[:, 0], :] - p[edges[:, 1], :]
        L = np.sqrt(np.sum(bars ** 2, axis=1))

        mid = (p[edges[:, 0], :] + p[edges[:, 1], :]) / 2.0
        L0 = target_edge_length(mid, h0, R)
        L0 = L0 * L0mult * (np.sum(L ** dim) / np.sum(L0 ** dim)) ** (1.0 / dim)

        F = np.maximum(L0 - L, 0.0)
        Fbar = np.hstack([bars, -bars]) * np.tile(F / np.maximum(L, 1.0e-12), (1, 2 * dim)).reshape(-1, 2 * dim)

        # 组装节点力
        dp = np.zeros((N, dim))
        for idx in range(len(edges)):
            for d in range(dim):
                dp[edges[idx, 0], d] += Fbar[idx, d]
                dp[edges[idx, 1], d] += Fbar[idx, dim + d]

        # 固定节点不移动
        if len(pfix) > 0:
            dp[:len(pfix), :] = 0.0

        p = p + deltat * dp

        # 7. 将越界点投影回边界
        for _ in range(2):
            d = distance_cylinder(p, R, H)
            ix = d > 0
            if not np.any(ix):
                break
            gradd = np.zeros((np.sum(ix), dim))
            for ii in range(dim):
                a = np.zeros((1, dim))
                a[0, ii] = deps
                d1x = distance_cylinder(p[ix, :] + np.ones((np.sum(ix), 1)) * a, R, H)
                gradd[:, ii] = (d1x - d[ix]) / deps

            grad_norm = np.sqrt(np.sum(gradd ** 2, axis=1))
            grad_norm = np.where(grad_norm < 1.0e-12, 1.0, grad_norm)
            p[ix, :] -= d[ix][:, None] * gradd / grad_norm[:, None]

        # 8. 终止准则
        maxdp = np.max(deltat * np.sqrt(np.sum(dp[d < -geps] ** 2, axis=1))) if np.any(d < -geps) else 0.0
        if maxdp < ptol * h0:
            break

    return p, t
