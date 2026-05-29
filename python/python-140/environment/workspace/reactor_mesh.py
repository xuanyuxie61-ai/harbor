"""
reactor_mesh.py
反应器网格生成与自适应细化模块
基于有向距离函数（SDF）生成二维/三维非结构化网格，并支持局部自适应细化。
原项目映射: 
  - 308_distmesh (基于距离函数的网格生成)
  - 067_ball_grid (球体内网格点生成)
  - 1351_triangulation_refine_local (局部三角剖分细化)
"""

import numpy as np
from scipy.spatial import Delaunay


def huniform(p):
    """
    均匀网格尺寸函数。
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:
        return 1.0
    else:
        return np.ones(p.shape[0], dtype=np.float64)


def distmesh_2d(fd, fh, h0, bbox, pfix=None, max_iter=100):
    """
    二维 DistMesh 网格生成器（Python 实现）。
    基于 Persson & Strang, SIAM Review 2004。
    
    参数:
        fd: 有向距离函数 fd(p) -> scalar/array
        fh: 网格尺寸函数 fh(p) -> scalar/array
        h0: 初始边长
        bbox: [[xmin, ymin], [xmax, ymax]] 包围盒
        pfix: 固定节点坐标 (Nfix, 2)
        max_iter: 最大迭代次数
    
    返回:
        p: 节点坐标 (N, 2)
        t: 三角形单元索引 (NT, 3)
    """
    if pfix is None:
        pfix = np.zeros((0, 2), dtype=np.float64)
    else:
        pfix = np.asarray(pfix, dtype=np.float64)

    dptol = 0.001
    ttol = 0.1
    Fscale = 1.2
    deltat = 0.2
    geps = 0.001 * h0
    deps = np.sqrt(np.finfo(float).eps) * h0

    # 1. 初始矩形网格
    x = np.arange(bbox[0, 0], bbox[1, 0] + h0, h0)
    y = np.arange(bbox[0, 1], bbox[1, 1] + h0 * np.sqrt(3.0) / 2.0, h0 * np.sqrt(3.0) / 2.0)
    xx, yy = np.meshgrid(x, y)
    xx[1::2, :] += h0 / 2.0
    p = np.column_stack((xx.ravel(), yy.ravel()))

    # 2. 保留区域内点
    d_val = fd(p)
    p = p[d_val < geps]

    # 根据密度函数筛选
    r0 = 1.0 / fh(p) ** 2
    max_r0 = np.max(r0) if len(r0) > 0 else 1.0
    if len(p) > 0:
        keep = np.random.rand(len(p)) < r0 / max_r0
        p = p[keep]

    # 去重并固定边界点
    if len(pfix) > 0:
        p = np.vstack((pfix, p))
        p_unique = []
        seen = set()
        for pt in p:
            key = (round(pt[0], 10), round(pt[1], 10))
            if key not in seen:
                seen.add(key)
                p_unique.append(pt)
        p = np.array(p_unique, dtype=np.float64)
        # 确保固定点在前
        pfix_set = set((round(pt[0], 10), round(pt[1], 10)) for pt in pfix)
        p_nonfix = [pt for pt in p if (round(pt[0], 10), round(pt[1], 10)) not in pfix_set]
        p = np.vstack((pfix, np.array(p_nonfix, dtype=np.float64))) if len(p_nonfix) > 0 else pfix

    N = p.shape[0]
    pold = np.inf * np.ones_like(p)
    triangulation_count = 0

    bars = None
    for iteration in range(max_iter):
        # 3. 重新三角化
        movement = np.max(np.sqrt(np.sum((p - pold) ** 2, axis=1)) / h0) if np.isfinite(pold).all() else np.inf
        if movement > ttol:
            pold = p.copy()
            if N >= 3:
                tri = Delaunay(p)
                t = tri.simplices
                triangulation_count += 1
                # 保留质心在区域内的三角形
                pmid = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
                t = t[fd(pmid) < -geps]
                if len(t) == 0:
                    break
                # 4. 提取边
                bars = np.vstack((t[:, [0, 1]], t[:, [0, 2]], t[:, [1, 2]]))
                bars = np.unique(np.sort(bars, axis=1), axis=0)
            else:
                break

        if bars is None or len(bars) == 0:
            break

        # 6. 基于边长力的节点移动
        barvec = p[bars[:, 0]] - p[bars[:, 1]]
        L = np.sqrt(np.sum(barvec ** 2, axis=1))
        hbars = fh((p[bars[:, 0]] + p[bars[:, 1]]) / 2.0)
        L0 = hbars * Fscale * np.sqrt(np.sum(L ** 2) / np.sum(hbars ** 2))
        F = np.maximum(L0 - L, 0.0)
        # 避免除以零
        L_safe = np.where(L < 1e-14, 1.0, L)
        Fvec = (F / L_safe)[:, None] * barvec

        Ftot = np.zeros_like(p)
        for i in range(len(bars)):
            Ftot[bars[i, 0]] += Fvec[i]
            Ftot[bars[i, 1]] -= Fvec[i]

        if len(pfix) > 0:
            Ftot[:len(pfix)] = 0.0
        p = p + deltat * Ftot

        # 7. 将外部点投影回边界
        d = fd(p)
        ix = d > 0
        if np.any(ix):
            # 数值梯度
            dx = np.zeros(np.sum(ix), dtype=np.float64)
            dy = np.zeros(np.sum(ix), dtype=np.float64)
            p_ix = p[ix]
            for idx in range(len(p_ix)):
                dgradx = (fd(np.array([[p_ix[idx, 0] + deps, p_ix[idx, 1]]])) - d[ix][idx]) / deps
                dgrady = (fd(np.array([[p_ix[idx, 0], p_ix[idx, 1] + deps]])) - d[ix][idx]) / deps
                dx[idx] = dgradx[0] if hasattr(dgradx, '__len__') else dgradx
                dy[idx] = dgrady[0] if hasattr(dgrady, '__len__') else dgrady
            grad_norm = np.sqrt(dx ** 2 + dy ** 2)
            grad_norm = np.where(grad_norm < 1e-14, 1.0, grad_norm)
            p[ix, 0] -= d[ix] * dx / grad_norm
            p[ix, 1] -= d[ix] * dy / grad_norm

        # 8. 终止条件
        interior = d < -geps
        if np.any(interior):
            max_move = np.max(np.sqrt(np.sum((deltat * Ftot[interior]) ** 2, axis=1)) / h0)
            if max_move < dptol:
                break

    # 最终三角化
    if N >= 3 and p.shape[0] >= 3:
        tri = Delaunay(p)
        t = tri.simplices
        pmid = (p[t[:, 0]] + p[t[:, 1]] + p[t[:, 2]]) / 3.0
        t = t[fd(pmid) < -geps]
    else:
        t = np.zeros((0, 3), dtype=np.int64)

    return p, t


def ball_grid(n, r, c):
    """
    在三维球体内生成规则网格点。
    映射自 ball_grid.m，用于催化剂颗粒内部的离散化。
    
    参数:
        n: 半径方向的分段数
        r: 球半径
        c: 球心坐标 (3,)
    返回:
        bg: 网格点坐标 (NG, 3)
    """
    c = np.asarray(c, dtype=np.float64)
    bg = []
    for i in range(n + 1):
        x = c[0] + r * 2.0 * i / (2.0 * n + 1.0)
        for j in range(n + 1):
            y = c[1] + r * 2.0 * j / (2.0 * n + 1.0)
            for k in range(n + 1):
                z = c[2] + r * 2.0 * k / (2.0 * n + 1.0)
                if r * r < (x - c[0]) ** 2 + (y - c[1]) ** 2 + (z - c[2]) ** 2:
                    break
                bg.append([x, y, z])
                if i > 0:
                    bg.append([2.0 * c[0] - x, y, z])
                if j > 0:
                    bg.append([x, 2.0 * c[1] - y, z])
                if k > 0:
                    bg.append([x, y, 2.0 * c[2] - z])
                if i > 0 and j > 0:
                    bg.append([2.0 * c[0] - x, 2.0 * c[1] - y, z])
                if i > 0 and k > 0:
                    bg.append([2.0 * c[0] - x, y, 2.0 * c[2] - z])
                if j > 0 and k > 0:
                    bg.append([x, 2.0 * c[1] - y, 2.0 * c[2] - z])
                if i > 0 and j > 0 and k > 0:
                    bg.append([2.0 * c[0] - x, 2.0 * c[1] - y, 2.0 * c[2] - z])
    return np.array(bg, dtype=np.float64)


def triangulation_refine_local(node_xy, element_node, element_to_refine):
    """
    局部三角剖分细化。
    映射自 triangulation_refine_local.m。
    对指定单元进行 1:4 细分（中点分割）。
    
    参数:
        node_xy: (N, 2) 节点坐标
        element_node: (NE, 3) 单元节点索引（0-based）
        element_to_refine: 要细化的单元索引
    返回:
        new_node_xy, new_element_node
    """
    node_xy = np.asarray(node_xy, dtype=np.float64)
    element_node = np.asarray(element_node, dtype=np.int64)
    e = element_to_refine

    if e < 0 or e >= len(element_node):
        raise ValueError("element_to_refine 超出范围")

    n1, n2, n3 = element_node[e]
    # 新节点为中点
    n12 = len(node_xy)
    n23 = len(node_xy) + 1
    n31 = len(node_xy) + 2

    p12 = 0.5 * (node_xy[n1] + node_xy[n2])
    p23 = 0.5 * (node_xy[n2] + node_xy[n3])
    p31 = 0.5 * (node_xy[n3] + node_xy[n1])

    new_node_xy = np.vstack((node_xy, p12, p23, p31))

    # 原单元替换为中心小三角形
    new_element_node = element_node.copy()
    new_element_node[e] = [n23, n31, n12]

    # 添加三个角上的小三角形
    new_triangles = np.array([
        [n1, n12, n31],
        [n2, n23, n12],
        [n3, n31, n23]
    ], dtype=np.int64)
    new_element_node = np.vstack((new_element_node, new_triangles))

    return new_node_xy, new_element_node


def compute_mesh_quality(p, t):
    """
    计算三角网格的质量指标（内切圆半径/外接圆半径之比）。
    质量接近 1.0 表示等边三角形。
    """
    if len(t) == 0:
        return np.array([])
    p = np.asarray(p, dtype=np.float64)
    t = np.asarray(t, dtype=np.int64)
    qualities = []
    for tri in t:
        a = np.linalg.norm(p[tri[1]] - p[tri[0]])
        b = np.linalg.norm(p[tri[2]] - p[tri[1]])
        c = np.linalg.norm(p[tri[0]] - p[tri[2]])
        s = 0.5 * (a + b + c)
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-30))
        r_in = area / s
        r_circ = a * b * c / (4.0 * max(area, 1e-30))
        qualities.append(2.0 * r_in / max(r_circ, 1e-30))
    return np.array(qualities, dtype=np.float64)
