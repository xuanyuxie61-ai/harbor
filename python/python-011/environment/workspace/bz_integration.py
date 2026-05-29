# -*- coding: utf-8 -*-
"""
bz_integration.py
-----------------
布里渊区数值积分模块：楔形积分、四面体高斯求积、Jacobi 多项式谱积分。

对应种子项目：
  - 1409_wedge_integrals：单位楔形上的单形积分与 Monte Carlo 采样
  - 1244_tetrahedron_arbq_rule：四面体高阶数值积分规则
  - 607_jacobi_polynomial：Jacobi 多项式与 Gauss-Jacobi 求积

物理背景：
  高温超导体中大量物理量需要在布里渊区（BZ）上积分，如
  电子态密度、超导能隙方程中的配对积分、热力学势等。
  对于具有 C4v 对称性的二维方格晶格，可将 BZ 约化为不可约楔形
  (irreducible wedge)，从而将积分域缩小为 0 ≤ kx, 0 ≤ ky, kx+ky ≤ π。

核心公式：
  积分 I = ∫_BZ f(k) dk = N_sym × ∫_wedge f(k) dk
  其中 N_sym = 8 为对称操作数。
"""

import numpy as np
from scipy.special import gamma as GammaFunc
from scipy.special import jacobi
from numpy.polynomial.legendre import leggauss


# ---------------------------------------------------------------------------
# 1. 楔形积分（来自 1409_wedge_integrals）
# ---------------------------------------------------------------------------

def wedge01_volume():
    """单位楔形体积 = 1.0。"""
    return 1.0


def wedge01_monomial_integral(e):
    """
    精确计算单位楔形上的单项式积分：
        ∫_wedge x^e1 y^e2 z^e3 dV
    其中楔形定义为：0 ≤ x, 0 ≤ y, x+y ≤ 1, -1 ≤ z ≤ 1。

    解析公式：
      XY 部分：对二维单形，积分 = e2! / [(e1+e2+1)(e1+e2+2) * C(e1+e2, e1)]
             更简洁地：
                 value_xy = ∏_{i=1}^{e2} [i / (e1 + i)] / [(e1+e2+1)(e1+e2+2)]
      Z 部分：若 e3 为奇数 → 0；若偶数 → 2/(e3+1)。
    """
    e = np.asarray(e, dtype=int).flatten()
    if e.size != 3:
        raise ValueError("e 必须是长度为 3 的整数向量。")
    e1, e2, e3 = e[0], e[1], e[2]

    # XY 单形积分
    if e1 < 0 or e2 < 0:
        return 0.0
    value_xy = 1.0
    for i in range(1, e2 + 1):
        value_xy *= float(i) / float(e1 + i)
    denom = float((e1 + e2 + 1) * (e1 + e2 + 2))
    value_xy /= denom

    # Z 区间积分
    if e3 % 2 == 1:
        value_z = 0.0
    else:
        value_z = 2.0 / float(e3 + 1)

    return value_xy * value_z


def wedge01_sample(n):
    """
    在单位楔形内均匀随机采样 n 个点。

    算法：Dirichlet 采样——取 3 个独立指数分布变量并归一化得到 (x,y) 在单形上的均匀分布；
          z 由 U(-1,1) 映射。
    """
    if n < 1:
        return np.zeros((3, 0))
    # 指数分布：-log(U)
    E = -np.log(np.random.rand(3, n))
    S = np.sum(E, axis=0)
    xy = E[:2, :] / S[np.newaxis, :]
    z = 2.0 * np.random.rand(1, n) - 1.0
    return np.vstack([xy, z])


def monomial_value(m, n_pts, e, x):
    """
    在 N 个点上计算单项式值 ∏_{d=1}^M x_d^{e_d}。

    Parameters
    ----------
    m : int
        空间维数。
    n_pts : int
        点数。
    e : ndarray, shape (m,)
        指数向量。
    x : ndarray, shape (m, n_pts)
        坐标矩阵。

    Returns
    -------
    v : ndarray, shape (n_pts,)
    """
    e = np.asarray(e, dtype=int)
    x = np.asarray(x, dtype=float)
    if x.shape != (m, n_pts):
        raise ValueError("x 形状不匹配。")
    v = np.ones(n_pts, dtype=float)
    for d in range(m):
        if e[d] == 0:
            continue
        v *= np.power(x[d, :], e[d])
    return v


# ---------------------------------------------------------------------------
# 2. 四面体高斯求积（来自 1244_tetrahedron_arbq_rule）
# ---------------------------------------------------------------------------

def tetrahedron_arbq_size(degree):
    """
    四面体高斯求积规则点数查找表（Xiao-Gimbutas, 2010）。
    degree 0..15 对应点数。
    """
    sizes = [1, 1, 4, 6, 11, 14, 23, 31, 44, 57, 74, 95, 122, 146, 177, 214]
    d = int(degree)
    if d < 0 or d >= len(sizes):
        raise ValueError("degree 必须在 0..15 范围内。")
    return sizes[d]


def tetrahedron_ref():
    """参考四面体顶点（中心在原点，体积 sqrt(8)/3）。"""
    return np.array([
        [-1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0, -1.0,  1.0]
    ], dtype=float)


def ref_to_koorn(r):
    """
    将参考四面体坐标映射到 Koornwinder 坐标系。
    仿射变换。
    """
    r = np.asarray(r, dtype=float)
    if r.ndim == 1:
        # 单点
        x, y, z = r[0], r[1], r[2]
        return np.array([x, y, z], dtype=float)
    return r.copy()


def ortho3eva(degree, xyz):
    """
    在参考四面体上求值正交多项式基（Koornwinder 多项式）。

    使用 Jacobi 多项式的乘积表示：
        K_{mnk}(x,y,z) = P_m^{(0,0)}(ξ1) * [(1-η1)/2]^m
                       * P_n^{(2m+1,0)}(ξ2) * [(1-ζ1)/2]^n
                       * P_k^{(2m+2n+2,0)}(ζ1)
    其中 ξ1, η1, ζ1 为 Duffy 变换后的局部坐标。
    """
    xyz = np.asarray(xyz, dtype=float)
    single = xyz.ndim == 1
    if single:
        xyz = xyz.reshape(1, 3)
    npts = xyz.shape[0]

    # Duffy 变换到标准单形坐标
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    # 避免除零
    eps = 1e-15
    # 局部坐标 (这里做简化：假设 xyz 已在标准四面体内)
    # 标准单形: x>=0, y>=0, z>=0, x+y+z<=1
    xi1 = np.where(np.abs(1.0 - y - z) > eps,
                   (2.0 * x + 2.0 + y + z) / (-y - z + eps), 0.0)
    xi2 = np.where(np.abs(1.0 - z) > eps,
                   (2.0 * y + 1.0 + z) / (1.0 - z + eps), 0.0)
    zeta = z.copy()

    # 为简化，使用 scipy 的 jacobi 多项式
    max_m = degree
    fvals = []
    for m in range(max_m + 1):
        for n in range(max_m + 1 - m):
            for k in range(max_m + 1 - m - n):
                # 计算 P_m^{0,0}(xi1) * [(1-eta)/2]^m
                Pm = np.ones(npts)
                if m > 0:
                    # 使用 numpy 的 legendre 多项式
                    leg_m = np.polynomial.legendre.legval(xi1, [0] * m + [1])
                    Pm = leg_m
                factor1 = Pm * np.power(np.maximum((-y - z + 1.0) * 0.5, eps), m)

                # P_n^{2m+1, 0}(xi2)
                Pn = np.ones(npts)
                if n > 0:
                    jac = jacobi(n, 2 * m + 1, 0)
                    Pn = np.polyval(jac, xi2)
                factor2 = Pn * np.power(np.maximum((1.0 - zeta) * 0.5, eps), n)

                # P_k^{2m+2n+2, 0}(zeta)
                Pk = np.ones(npts)
                if k > 0:
                    jac = jacobi(k, 2 * m + 2 * n + 2, 0)
                    Pk = np.polyval(jac, zeta)
                factor3 = Pk

                fvals.append(factor1 * factor2 * factor3)
    res = np.column_stack(fvals) if len(fvals) > 0 else np.ones((npts, 1))
    if single:
        return res[0, :]
    return res


def _gauss_tetrahedron_nodes_weights(degree):
    """
    生成四面体上的高斯积分节点和权重（简化实现）。
    对于低阶（degree<=3），使用精确公式；高阶时回退到 Monte Carlo。
    """
    if degree <= 1:
        # 1 点规则：重心，权重 = 体积 / 1
        nodes = np.array([[0.25, 0.25, 0.25]])
        weights = np.array([1.0 / 6.0])
        return nodes, weights
    elif degree <= 2:
        # 4 点规则（二阶精确）
        a = 0.58541020
        b = 0.13819660
        nodes = np.array([
            [a, b, b],
            [b, a, b],
            [b, b, a],
            [b, b, b]
        ])
        weights = np.ones(4) / 24.0
        return nodes, weights
    elif degree <= 3:
        # 5 点规则（三阶精确）
        nodes = np.array([
            [0.25, 0.25, 0.25],
            [0.5, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 0.5, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 0.5],
            [1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0]
        ])
        weights = np.array([-4.0 / 30.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0, 3.0 / 40.0])
        return nodes, weights
    else:
        # 回退：Monte Carlo 采样
        n_mc = tetrahedron_arbq_size(min(degree, 15))
        nodes = np.random.rand(n_mc, 3)
        # 投影到单形
        nodes = -np.log(np.maximum(nodes, 1e-15))
        s = np.sum(nodes, axis=1, keepdims=True)
        nodes = nodes / s
        weights = np.ones(n_mc) / (6.0 * n_mc)
        return nodes, weights


def integrate_tetrahedron(f, degree=3):
    """
    在标准四面体（x,y,z>=0, x+y+z<=1）上数值积分函数 f。

    Parameters
    ----------
    f : callable
        接受 ndarray (n, 3) 返回 ndarray (n,)。
    degree : int
        期望的代数精度。

    Returns
    -------
    result : float
        积分近似值。
    """
    nodes, weights = _gauss_tetrahedron_nodes_weights(degree)
    vals = f(nodes)
    return np.dot(weights, vals)


# ---------------------------------------------------------------------------
# 3. Jacobi 多项式与 Gauss-Jacobi 求积（来自 607_jacobi_polynomial）
# ---------------------------------------------------------------------------

def jacobi_polynomial_eval(m, n, alpha, beta, x):
    """
    在 m 个点上计算 Jacobi 多项式 P_k^{(α,β)}(x)，k=0..n。

    使用三项递推：
        P_0(x) = 1
        P_1(x) = (α-β)/2 + (α+β+2)/2 * x
        对 k >= 1:
          a1 = 2(k+1)(k+α+β+1)(2k+α+β)
          a2 = (2k+α+β+1)(α^2-β^2)
          a3 = (2k+α+β)(2k+α+β+1)(2k+α+β+2)
          a4 = 2(k+α)(k+β)(2k+α+β+2)
          P_{k+1} = [(a2 + a3 x) P_k - a4 P_{k-1}] / a1
    """
    x = np.asarray(x, dtype=float).flatten()
    m_pts = x.size
    if n < 0:
        return np.zeros((m_pts, 0))
    v = np.ones((m_pts, n + 1), dtype=float)
    if n >= 1:
        v[:, 1] = (alpha - beta) * 0.5 + (alpha + beta + 2.0) * 0.5 * x
    for k in range(1, n):
        a1 = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * (2.0 * k + alpha + beta)
        a2 = (2.0 * k + alpha + beta + 1.0) * (alpha ** 2 - beta ** 2)
        a3 = (2.0 * k + alpha + beta) * (2.0 * k + alpha + beta + 1.0) * (2.0 * k + alpha + beta + 2.0)
        a4 = 2.0 * (k + alpha) * (k + beta) * (2.0 * k + alpha + beta + 2.0)
        if abs(a1) < 1e-15:
            break
        v[:, k + 1] = ((a2 + a3 * x) * v[:, k] - a4 * v[:, k - 1]) / a1
    return v


def jacobi_polynomial_zeros(n, alpha, beta):
    """
    计算 Jacobi 多项式 P_n^{(α,β)}(x) 的 n 个零点。
    通过构造对称三对角 Jacobi 矩阵并用 numpy 求解本征值。
    """
    if n < 1:
        return np.array([])
    # Jacobi 矩阵对角元与次对角元
    d = np.zeros(n)
    e = np.zeros(n - 1)
    for i in range(n):
        d[i] = (beta ** 2 - alpha ** 2) / ((2.0 * i + alpha + beta) * (2.0 * i + alpha + beta + 2.0)) if (2.0 * i + alpha + beta) > 0 else 0.0
    for i in range(1, n):
        ab = alpha + beta
        denom = (2.0 * i + ab - 1.0) * (2.0 * i + ab + 1.0)
        if denom > 1e-15:
            num = i * (i + ab) * (i + alpha) * (i + beta)
            e[i - 1] = 2.0 / (2.0 * i + ab) * np.sqrt(num / denom)
    # 本征值
    J = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    zeros = np.sort(np.linalg.eigvalsh(J))
    return zeros


def gauss_jacobi_quadrature(n, alpha, beta):
    """
    n 点 Gauss-Jacobi 求积：返回节点 x 和权重 w。

    节点为 P_n^{(α,β)}(x) 的零点。
    权重通过 Jacobi 矩阵本征向量第一分量平方计算：
        w_i = μ_0 * (v_i[0])^2
    其中 μ_0 = ∫_{-1}^{1} (1-x)^α (1+x)^β dx
             = 2^{α+β+1} Γ(α+1) Γ(β+1) / Γ(α+β+2)。
    """
    if n < 1:
        return np.array([]), np.array([])
    # 构造 Jacobi 矩阵并求本征分解
    d = np.zeros(n)
    e = np.zeros(n - 1)
    for i in range(n):
        ab = alpha + beta
        if (2.0 * i + ab) > 1e-15:
            d[i] = (beta ** 2 - alpha ** 2) / ((2.0 * i + ab) * (2.0 * i + ab + 2.0))
    for i in range(1, n):
        ab = alpha + beta
        denom = (2.0 * i + ab - 1.0) * (2.0 * i + ab + 1.0)
        if denom > 1e-15:
            num = i * (i + alpha + beta) * (i + alpha) * (i + beta)
            e[i - 1] = 2.0 / (2.0 * i + ab) * np.sqrt(num / denom)
    J = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    w, v = np.linalg.eigh(J)
    zeros = np.sort(w)
    # 重排本征向量
    idx = np.argsort(w)
    v = v[:, idx]
    # μ_0
    mu0 = (2.0 ** (alpha + beta + 1.0) *
           GammaFunc(alpha + 1.0) * GammaFunc(beta + 1.0) /
           GammaFunc(alpha + beta + 2.0))
    weights = mu0 * v[0, :] ** 2
    return zeros, weights


def jacobi_double_product_integral(i, j, alpha, beta):
    """
    计算加权内积
        ∫_{-1}^{1} P_i^{(α,β)}(x) P_j^{(α,β)}(x) (1-x)^α (1+x)^β dx
    利用正交性：i≠j 时为 0；i=j 时 = 2^{α+β+1} / (2i+α+β+1) * Γ(i+α+1)Γ(i+β+1) / [i! Γ(i+α+β+1)]。
    """
    if i != j:
        return 0.0
    if i < 0:
        return 0.0
    val = (2.0 ** (alpha + beta + 1.0) / (2.0 * i + alpha + beta + 1.0) *
           GammaFunc(i + alpha + 1.0) * GammaFunc(i + beta + 1.0) /
           (GammaFunc(i + 1.0) * GammaFunc(i + alpha + beta + 1.0)))
    return float(val)


# ---------------------------------------------------------------------------
# 4. 布里渊区积分封装
# ---------------------------------------------------------------------------

def integrate_irreducible_wedge(func, n_sample=20000):
    """
    对二维方格晶格的不可约楔形（0<=kx, 0<=ky, kx+ky<=π）进行 Monte Carlo 积分。

    物理量 I 的 BZ 积分为：
        I = 8 * ∫_wedge f(k) dk
    其中因子 8 来自 C4v 对称操作数。

    Parameters
    ----------
    func : callable
        接受 ndarray (n, 2) 返回 ndarray (n,)。
    n_sample : int
        Monte Carlo 采样数。

    Returns
    -------
    result : float
        BZ 积分近似值。
    """
    if n_sample < 1:
        return 0.0
    # 楔形体积在 (kx,ky) 空间：面积 = π^2 / 8
    wedge_area = np.pi ** 2 / 8.0
    # 使用 Dirichlet 采样生成楔形内的点：kx = π * u1, ky = π * u2 * (1-u1)
    # 但更简单：直接拒绝采样在三角形内
    samples = []
    batch = min(n_sample * 5, 200000)
    while len(samples) < n_sample:
        s = np.random.rand(batch, 2) * np.pi
        mask = s[:, 0] >= 0
        mask &= s[:, 1] >= 0
        mask &= (s[:, 0] + s[:, 1]) <= np.pi
        valid = s[mask]
        needed = n_sample - len(samples)
        samples.extend(valid[:needed].tolist())
    samples = np.array(samples[:n_sample])
    vals = func(samples)
    # 因子 8 对应全 BZ
    return 8.0 * wedge_area * np.mean(vals)


def integrate_bz_gauss_legendre_2d(func, n_per_dim=40):
    """
    使用张量积 Gauss-Legendre 求积在整个 BZ [-π,π]^2 上积分。

    Parameters
    ----------
    func : callable
        接受 ndarray (n, 2) 返回 ndarray (n,)。
    n_per_dim : int
        每维 Gauss 点数。

    Returns
    -------
    result : float
    """
    # TODO: Hole_3 - implement 2D BZ integration using Gauss-Legendre tensor product
    raise NotImplementedError("Hole_3: implement integrate_bz_gauss_legendre_2d with Gauss-Legendre quadrature")
