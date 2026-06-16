"""
quadrature_rules.py
===================
高阶数值积分规则模块，整合三种正交规则:

  1. 楔形积分规则 (源自 1406_wedge_exactness)
     - 用于三棱柱单元的体积分
     - 三角形 (底) x 线段 (高) 的张量积

  2. 金字塔积分规则 (源自 937_pyramid_witherden_rule)
     - Witherden & Vincent (2015) 的高精度金字塔规则
     - 用于金字塔形过渡单元

  3. Clenshaw-Curtis 稀疏网格 (源自 143_cc_display)
     - Smolyak 算法构造高维稀疏网格
     - 用于高通量参数空间采样

数学背景:
  楔形单元 (三角形 x 线段):
    W = {(x,y,z) : (x,y) in Triangle, z in [0,1]}
    积分: I = int_0^1 int_T f(x,y,z) dA dz
    规则: 三角形 Gauss 规则 (n_T 点) x Gauss-Legendre (n_z 点)

  金字塔单元:
    P = {(x,y,z) : |x|+|y| <= 1-z, z in [0,1]}
    体积 V = 4/3

  Clenshaw-Curtis 节点 (1D):
    x_j = -cos(pi*j/N), j = 0, ..., N
    权重通过 FFT 或递推计算

  Smolyak 稀疏网格 (Smolyak 1963):
    A(q, d) = sum_{q-|alpha|<=q} (-1)^{q-|alpha|} * C(d-1, q-|alpha|)
              * (Q^{alpha_1} x ... x Q^{alpha_d})
    其中 |alpha| = alpha_1 + ... + alpha_d, q 为精度层级
    点数远少于全张量网格 (维数灾难缓解)
"""

import numpy as np
from itertools import product as iterproduct
from material_constants import SMALL_NUMBER


# ============================================================================
# 1D Gauss-Legendre 和 Clenshaw-Curtis
# ============================================================================
def gauss_legendre_1d(n_points):
    """
    n 点 Gauss-Legendre 规则在 [-1, 1] 上:

    节点: P_n(x_j) = 0 的根 (Legendre 多项式)
    权重: w_j = 2 / ((1-x_j^2) * [P_n'(x_j)]^2)

    精度: 对 2n-1 次多项式精确
    """
    x, w = np.polynomial.legendre.leggauss(n_points)
    return x, w


def clenshaw_curtis_1d(n_points):
    """
    Clenshaw-Curtis 求积规则在 [-1, 1]:

    节点: x_j = -cos(pi * j / (n-1)), j = 0, ..., n-1
    权重: 通过 Chebyshev 展开系数计算

    对 n 点规则, 精度 ~ n-1 次多项式 (嵌套优势)

    权重公式:
      w_j = (c_j / (n-1)) * sum_{k=0}^{(n-1)/2} b_k * cos(2*pi*k*j/(n-1))
    其中 c_0 = c_{n-1} = 1, 其他为 2
          b_0 = 1, b_{(n-1)/2} = 1, 其他为 2/(1-4k^2)
    """
    n = n_points
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    theta = np.pi * np.arange(n) / (n - 1)
    x = -np.cos(theta)
    w = np.zeros(n)
    N = n - 1
    for j in range(n):
        s = 0.0
        for k in range(N // 2 + 1):
            bk = 1.0 if k == 0 else (1.0 if 2*k == N else 2.0 / (1.0 - 4.0*k*k))
            s += bk * np.cos(2.0 * k * theta[j])
        cj = 1.0 if (j == 0 or j == N) else 2.0
        w[j] = cj * s / N
    return x, w


# ============================================================================
# 楔形积分规则
# ============================================================================
def triangle_gauss(order):
    """
    三角形上的 Gauss 求积规则 (参考 Lyness & Jespersen 1975).

    对精度 order (多项式次数):
      order=1: 1 点 (重心)
      order=2: 3 点
      order=3: 4 点 (Hammer-Marlowe-Stroud)
      order=5: 7 点 (Lyness-Jespersen)

    标准三角形 T = {(x,y): x>=0, y>=0, x+y<=1}
    面积 = 0.5

    返回: (bary_coords [n_pts, 3], weights [n_pts])
    """
    if order <= 1:
        return np.array([[1/3, 1/3, 1/3]]), np.array([0.5])
    elif order == 2:
        pts = np.array([
            [1/6, 1/6, 2/3],
            [1/6, 2/3, 1/6],
            [2/3, 1/6, 1/6],
        ])
        w = np.array([1/6, 1/6, 1/6])
        return pts, w
    elif order <= 3:
        a1 = 0.25
        a2 = 1.0 / 6.0
        w0 = -27.0 / 96.0
        w1 = 25.0 / 96.0
        pts = np.array([
            [1/3, 1/3, 1/3],
            [a1, a1, 1-2*a1],
            [a1, 1-2*a1, a1],
            [1-2*a1, a1, a1],
        ])
        w = np.array([w0, w1, w1, w1]) * 0.5
        return pts, w
    else:  # order >= 5, 使用 7 点规则
        a1 = 0.059715871789770
        b1 = 0.470142064105115
        a2 = 0.797426985353087
        b2 = 0.101286507323456
        w0 = 0.225000000000000
        w1 = 0.132394152788506
        w2 = 0.125939180544827
        pts = np.array([
            [1/3, 1/3, 1/3],
            [a1, b1, b1], [b1, a1, b1], [b1, b1, a1],
            [a2, b2, b2], [b2, a2, b2], [b2, b2, a2],
        ])
        w = np.array([w0, w1, w1, w1, w2, w2, w2]) * 0.5
        return pts, w


def wedge_quadrature(tri_order=3, z_order=3):
    """
    楔形单元 (三角形 x 线段) 上的积分规则。

    张量积: W = T x [0, 1]
    节点: (xi, eta, zeta) = (bary_to_xy(tri_pts), z_gl)
    权重: w = w_tri * w_gl

    总精度: min(tri_order, 2*z_order - 1)

    返回:
        points: [n_pts, 3] (x, y, z) in 参考楔形
        weights: [n_pts]
    """
    bary, w_tri = triangle_gauss(tri_order)
    z_gl, w_gl = gauss_legendre_1d(z_order)

    pts = []
    wts = []
    for i in range(len(bary)):
        for j in range(len(z_gl)):
            # 重心坐标转 (x, y) 在标准三角形
            L1, L2, L3 = bary[i]
            x = L2  # 顶点 (0,0), (1,0), (0,1)
            y = L3
            z = 0.5 * (z_gl[j] + 1.0)  # [-1,1] -> [0,1]
            pts.append([x, y, z])
            wts.append(w_tri[i] * w_gl[j] * 0.5)  # Jacobian 1/2

    return np.array(pts), np.array(wts)


# ============================================================================
# 金字塔积分规则 (Witherden & Vincent 2015 简化版)
# ============================================================================
def pyramid_witherden_rule(order=3):
    """
    金字塔单元上的高精度积分规则。

    金字塔 P = {(x,y,z): |x|+|y| <= 1-z, z in [0,1]}
    顶点: (+-1, 0, 0), (0, +-1, 0), (0, 0, 1)
    体积 V = 4/3

    对给定 order, 使用分层方法:
    将金字塔按 z 切片为正方形, 使用 Gauss-Legendre 在每层上积分。

    变换: (u, v, w) in [-1,1]^2 x [0,1] -> (x, y, z)
      z = w
      x = (1-w) * u
      y = (1-w) * v
      Jacobian: |J| = (1-w)^2

    积分: I = int_0^1 int_{-1}^1 int_{-1}^1 f(x(u,v,w),...) * (1-w)^2 du dv dw

    返回:
        points: [n_pts, 3] (x, y, z) 在物理金字塔内
        weights: [n_pts]
    """
    # z 方向使用 n 点 Gauss-Legendre
    n_z = max(order, 2)
    z_gl, w_gl = gauss_legendre_1d(n_z)
    # u, v 方向各使用 n 点
    n_uv = max(order, 2)
    u_gl, w_u = gauss_legendre_1d(n_uv)

    pts = []
    wts = []
    for iz in range(n_z):
        w_coord = 0.5 * (z_gl[iz] + 1.0)  # [0, 1]
        for iu in range(n_uv):
            for iv in range(n_uv):
                u = u_gl[iu]
                v = u_gl[iv]
                x = (1.0 - w_coord) * u
                y = (1.0 - w_coord) * v
                z = w_coord
                jac = (1.0 - w_coord)**2 * 0.5  # 0.5 for [-1,1]->[0,1]
                wt = w_u[iu] * w_u[iv] * w_gl[iz] * jac
                pts.append([x, y, z])
                wts.append(wt)

    return np.array(pts), np.array(wts)


def pyramid_volume():
    """
    标准金字塔体积:
      V = int_0^1 (2*(1-z))^2 dz = 4 * int_0^1 (1-z)^2 dz = 4/3
    """
    return 4.0 / 3.0


# ============================================================================
# Smolyak 稀疏网格 (Clenshaw-Curtis)
# ============================================================================
def smolyak_indices(d, q):
    """
    Smolyak 组合稀疏网格的多指标集合:

    A(q, d) = sum_{alpha: |alpha| in [q-d+1, q]} (-1)^{q-|alpha|}
              * C(d-1, q-|alpha|) * (Q^a1 x ... x Q^ad)

    其中 alpha = (a1, ..., ad), ai >= 1, |alpha| = a1 + ... + ad
    """
    from math import comb
    multi_indices = []
    coeffs = []
    for total in range(max(d, q - d + 1), q + 1):
        # 枚举所有 |alpha| = total, ai >= 1
        for alpha in _compositions(total, d):
            sign = (-1) ** (q - total)
            c = comb(d - 1, q - total)
            multi_indices.append(tuple(alpha))
            coeffs.append(sign * c)
    return multi_indices, coeffs


def _compositions(n, k):
    """生成所有 k 个正整数之和为 n 的组合 (ai >= 1)"""
    if k == 1:
        yield (n,)
        return
    for i in range(1, n - k + 2):
        for rest in _compositions(n - i, k - 1):
            yield (i,) + rest


def clenshaw_curtis_sparse_grid(d, q, max_points_per_dim=20):
    """
    d 维 Clenshaw-Curtis 稀疏网格, 精度层级 q.

    构造:
      对每个多指标 alpha, 取 1D CC 规则的张量积:
        Q^alpha = Q^{a1} x Q^{a2} x ... x Q^{ad}
      其中 Q^{ai} 为 ai 点的 CC 规则

    总点数远小于全张量网格 (缓解维数灾难)

    返回:
        points: [N_total, d]
        combined_weights: [N_total]

    注意: 重复点会被累加权重
    """
    multi_indices, coeffs = smolyak_indices(d, q)

    point_weight_map = {}
    for alpha, coeff in zip(multi_indices, coeffs):
        if coeff == 0:
            continue
        # 每个维度生成 CC 节点
        grids_1d = []
        for a_i in alpha:
            n_pts = min(max(a_i, 2), max_points_per_dim)
            xi, wi = clenshaw_curtis_1d(n_pts)
            grids_1d.append((xi, wi))
        # 张量积
        all_pts = list(iterproduct(*[g[0] for g in grids_1d]))
        all_wts = list(iterproduct(*[g[1] for g in grids_1d]))
        for pt, wt in zip(all_pts, all_wts):
            pt_key = tuple(round(p, 14) for p in pt)
            w_prod = np.prod(wt)
            if pt_key in point_weight_map:
                point_weight_map[pt_key] += coeff * w_prod
            else:
                point_weight_map[pt_key] = coeff * w_prod

    pts = np.array(list(point_weight_map.keys()))
    wts = np.array(list(point_weight_map.values()))
    # 过滤权重接近零的点
    mask = np.abs(wts) > SMALL_NUMBER
    return pts[mask], wts[mask]


def sparse_grid_integrate(func, d, q):
    """
    使用稀疏网格计算 d 维积分:

    I ≈ sum_i w_i * f(x_i)

    积分区域: [-1, 1]^d

    返回: (integral_estimate, n_points_used)
    """
    pts, wts = clenshaw_curtis_sparse_grid(d, q)
    integral = 0.0
    for i in range(len(pts)):
        integral += wts[i] * func(pts[i])
    return integral, len(pts)


def quadrature_exactness_test(rule_points, rule_weights, max_degree=5):
    """
    测试求积规则的精确度 (多项式精确度)。

    对 1D 或张量积规则, 测试 int x^k 对 k=0,...,max_degree:

    1D: int_{-1}^{1} x^k dx = 2/(k+1) if k even, 0 if k odd
    """
    exactness = 0
    for k in range(max_degree + 1):
        # 精确值
        if k % 2 == 0:
            exact = 2.0 / (k + 1)
        else:
            exact = 0.0
        # 数值值 (假设 1D)
        if rule_points.ndim == 1:
            numerical = np.sum(rule_weights * rule_points**k)
        else:
            numerical = np.sum(rule_weights * rule_points[:, 0]**k)
        err = abs(numerical - exact)
        if err < 1e-10:
            exactness = k
        else:
            break
    return exactness
