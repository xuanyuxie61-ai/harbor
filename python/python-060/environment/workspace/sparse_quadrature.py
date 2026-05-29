# -*- coding: utf-8 -*-
"""
sparse_quadrature.py
稀疏网格 Clenshaw-Curtis 求积模块，用于计算平流层光化辐射通量的
高维积分（波长-天顶角-方位角空间）。

融合来源：
  - 1103_sparse_grid_cc: Smolyak 构造的稀疏网格求积
"""

import numpy as np


def cc_abscissa(order, index):
    r"""
    Clenshaw-Curtis 一维节点：

        x_i = \cos\left( \frac{(i-1)\pi}{n-1} \right), \quad i = 1,\dots,n

    当 n=1 时，x_1 = 0。

    Parameters
    ----------
    order : int
        节点总数 n。
    index : int
        1-based 节点索引。

    Returns
    -------
    x : float
        节点坐标，已映射到 [0, 1]。
    """
    if order < 1:
        raise ValueError("order 必须 ≥ 1")
    if order == 1:
        return 0.5
    if index < 1 or index > order:
        raise ValueError("index 超出范围")
    x = np.cos((index - 1) * np.pi / (order - 1))
    return 0.5 * (x + 1.0)


def cc_weights_1d(order):
    r"""
    一维 Clenshaw-Curtis 权重（闭区间 [0,1]）。
    使用基于余弦变换的显式公式：

        w_i = \frac{c_i}{n-1} \sum_{j=0}^{\lfloor (n-1)/2 \rfloor}
              \frac{b_j}{4j^2 - 1} \cos\left( \frac{2ij\pi}{n-1} \right)

    其中 c_0 = c_{n-1} = 1/2，其余 c_i = 1；
    b_0 = 1，b_{\lfloor(n-1)/2\rfloor} = 1（若 n-1 偶），其余 b_j = 2。

    Parameters
    ----------
    order : int
        节点数。

    Returns
    -------
    w : ndarray, shape (order,)
    """
    if order == 1:
        return np.array([1.0])
    n = order
    w = np.zeros(n)
    c = np.ones(n)
    c[0] = 0.5
    c[-1] = 0.5
    m = (n - 1) // 2
    b = np.ones(m + 1)
    if (n - 1) % 2 == 0:
        b[-1] = 1.0
    else:
        b[-1] = 2.0
    b[1:-1] = 2.0
    for i in range(n):
        s = 0.0
        for j in range(m + 1):
            s += b[j] / (4.0 * j * j - 1.0) * np.cos(2.0 * j * i * np.pi / (n - 1))
        w[i] = c[i] * s / (n - 1)
    return w


def sparse_grid_cc_size(dim_num, level_max):
    r"""
    计算稀疏网格的节点数（Smolyak 构造）：

        |Q_{\ell}^{(d)}| = \sum_{\ell-d+1 \le |\mathbf{i}|_1 \le \ell}
                           \prod_{k=1}^{d} n(i_k),
        n(i) = \begin{cases} 1 & i=0 \\ 2^{i-1}+1 & i \ge 1 \end{cases}

    Parameters
    ----------
    dim_num : int
        空间维数 d。
    level_max : int
        最大层级 ℓ。

    Returns
    -------
    point_num : int
        节点总数。
    """
    if dim_num < 1 or level_max < 0:
        return 0
    point_num = 0
    def order_1d(level):
        if level == 0:
            return 1
        return 2 ** (level - 1) + 1

    def comp_next(k, a, more, h, t):
        if not more:
            a[:] = 0
            t = k
            h = 0
            more = True if k > 0 else False
            return a, more, h, t
        if 1 < t:
            h = 0
        h += 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] += 1
        if a[k - 1] == k:
            more = False
        return a, more, h, t

    a = np.zeros(dim_num, dtype=int)
    more = False
    h = 0
    t = 0
    while True:
        a, more, h, t = comp_next(dim_num, a, more, h, t)
        level = np.sum(a)
        if level_max - dim_num + 1 <= level <= level_max:
            prod = 1
            for i in range(dim_num):
                prod *= order_1d(a[i])
            point_num += prod
        if not more:
            break
    return point_num


def multigrid_index0(dim_num, level_max):
    r"""
    生成满足 |i|_1 ≤ ℓ 的多层索引。

    Parameters
    ----------
    dim_num : int
    level_max : int

    Returns
    -------
    indices : list of ndarray
        每一层的索引列表。
    """
    def comp_next(k, a, more, h, t):
        if not more:
            a[:] = 0
            t = k
            h = 0
            more = True if k > 0 else False
            return a, more, h, t
        if 1 < t:
            h = 0
        h += 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] += 1
        more_out = not (a[k - 1] == k)
        return a, more_out, h, t

    result = []
    a = np.zeros(dim_num, dtype=int)
    more = False
    h = 0
    t = 0
    while True:
        a, more, h, t = comp_next(dim_num, a, more, h, t)
        if np.sum(a) <= level_max:
            result.append(a.copy())
        if not more:
            break
    return result


def sparse_grid_cc(dim_num, level_max):
    r"""
    构造 d 维稀疏网格 Clenshaw-Curtis 求积规则。
    返回节点（映射到 [0,1]^d）和权重。

    Parameters
    ----------
    dim_num : int
        维数 d（≥1）。
    level_max : int
        最大层级 ℓ（≥0）。

    Returns
    -------
    points : ndarray, shape (point_num, dim_num)
    weights : ndarray, shape (point_num,)
    """
    if dim_num < 1:
        raise ValueError("dim_num 必须 ≥ 1")
    if level_max < 0:
        raise ValueError("level_max 必须 ≥ 0")

    point_num = sparse_grid_cc_size(dim_num, level_max)
    points = np.zeros((point_num, dim_num))
    weights = np.zeros(point_num)

    def order_1d(level):
        if level == 0:
            return 1
        return 2 ** (level - 1) + 1

    idx = 0
    for level in range(level_max + 1):
        indices = multigrid_index0(dim_num, level)
        for ind in indices:
            if np.sum(ind) != level:
                continue
            orders = [order_1d(ind[d]) for d in range(dim_num)]
            total = int(np.prod(orders))
            sub_pts = np.zeros((total, dim_num))
            sub_w = np.ones(total)
            for d in range(dim_num):
                od = orders[d]
                w1d = cc_weights_1d(od)
                repeats = int(np.prod(orders[d + 1:]))
                tiles = int(np.prod(orders[:d]))
                for i in range(od):
                    x = cc_abscissa(od, i + 1)
                    for t in range(tiles):
                        for r in range(repeats):
                            pos = t * od * repeats + i * repeats + r
                            sub_pts[pos, d] = x
                            sub_w[pos] *= w1d[i]
            # 组合系数
            coeff = (-1) ** (level_max - level)
            from math import comb
            coeff *= comb(dim_num - 1, level_max - level)
            points[idx:idx + total, :] = sub_pts
            weights[idx:idx + total] = coeff * sub_w
            idx += total

    # 合并重复点（简单去重：若两点距离小于容差则合并）
    tol = 1e-12
    unique_pts = []
    unique_w = []
    for i in range(points.shape[0]):
        p = points[i]
        w = weights[i]
        found = False
        for j, up in enumerate(unique_pts):
            if np.linalg.norm(p - up) < tol:
                unique_w[j] += w
                found = True
                break
        if not found:
            unique_pts.append(p.copy())
            unique_w.append(w)
    return np.array(unique_pts), np.array(unique_w)


def integrate_sparse_grid(f, dim_num, level_max):
    r"""
    使用稀疏网格数值积分：

        I = \int_{[0,1]^d} f(\mathbf{x}) \, d\mathbf{x}
          \approx \sum_{i=1}^{N} w_i f(\mathbf{x}_i)

    Parameters
    ----------
    f : callable
        被积函数 f(x) -> float，x 为 ndarray。
    dim_num : int
    level_max : int

    Returns
    -------
    integral : float
    """
    pts, wts = sparse_grid_cc(dim_num, level_max)
    s = 0.0
    for i in range(pts.shape[0]):
        s += wts[i] * f(pts[i])
    return s
