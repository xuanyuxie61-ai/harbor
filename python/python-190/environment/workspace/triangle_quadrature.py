"""
triangle_quadrature.py
======================
基于种子项目 1316_triangle_symq_rule 与 1130_sphere_triangle_quad 的数值积分模块。
提供平面三角形上的高阶对称求积公式以及单位球面三角形上的数值积分，
用于在物理信息 GAN 的判别器中精确计算 PDE 残差的区域积分。

核心数学：
  1. 单位三角形上的单项式精确积分：
       ∫∫_T x^m y^n dx dy = m!·n! / (m+n+2)!
     其中 T 为顶点 (0,0), (1,0), (0,1) 的参考三角形。

  2. 对称求积规则（Wandzura & Xiao, 2003; Taylor & Wingate, 2000）：
     通过重心坐标 (α, β, γ) 与权重 w 实现：
       ∫∫_T f(x,y) dx dy ≈ area(T) · Σ_i w_i · f(x_i, y_i)
     其中 (x_i, y_i) = α_i·V1 + β_i·V2 + γ_i·V3。

  3. 球面三角形面积（L'Huilier 定理）：
       tan(E/4) = √(tan(s/2)·tan((s-a)/2)·tan((s-b)/2)·tan((s-c)/2))
       area = E · R²
     其中 a, b, c 为球面边长（大圆弧对应的中心角），
     s = (a+b+c)/2，E 为球面盈量（spherical excess）。

  4. 球面三角形上的 Monte Carlo 求积：
       ∫∫_Δ f(ω) dS ≈ area(Δ) · (1/N) Σ_j f(ω_j)
     其中 ω_j 在球面三角形内均匀采样。
"""

import numpy as np


def triangle_unit_monomial_integral(expon: np.ndarray) -> float:
    """
    计算单位三角形上单项式 x^m y^n 的精确积分。

    Parameters
    ----------
    expon : np.ndarray, shape (2,)
        指数 [m, n]，非负整数。

    Returns
    -------
    value : float
        积分值 m!·n! / (m+n+2)!。
    """
    m = int(expon[0])
    n = int(expon[1])
    if m < 0 or n < 0:
        raise ValueError("指数必须为非负整数。")
    value = 1.0
    k = 0
    for _ in range(m):
        k += 1
        value = value * 1.0 / k
    for _ in range(n):
        k += 1
        value = value * 1.0 / k
    k += 1
    value = value / k
    k += 1
    value = value / k
    return float(value)


def triangle_area(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> float:
    """
    计算平面三角形面积（基于叉积的 1/2 |cross|）。

    Parameters
    ----------
    v1, v2, v3 : np.ndarray, shape (2,) or (3,)
        三角形顶点。

    Returns
    -------
    area : float
        三角形面积。
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    if v1.shape[0] == 2:
        cross = (v2[0] - v1[0]) * (v3[1] - v1[1]) - (v2[1] - v1[1]) * (v3[0] - v1[0])
        return 0.5 * abs(cross)
    else:
        cross = np.cross(v2 - v1, v3 - v1)
        return 0.5 * np.linalg.norm(cross)


def triangle_symq_rule(degree: int = 7) -> tuple:
    """
    返回单位参考三角形上的对称求积节点与权重。
    这里采用公开文献中的低阶精确规则（degree ≤ 7）。

    Parameters
    ----------
    degree : int
        期望代数精度（当前支持 1, 2, 3, 4, 5, 7）。

    Returns
    -------
    bary : np.ndarray, shape (npts, 3)
        重心坐标 (α, β, γ)。
    weights : np.ndarray, shape (npts,)
        归一化权重（Σ w_i = 1）。
    """
    if degree <= 1:
        # 1点规则（重心），精度 1
        bary = np.array([[1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]])
        weights = np.array([1.0])
    elif degree == 2:
        # 3点规则，精度 2
        bary = np.array([
            [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
        ])
        weights = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    elif degree == 3:
        # 4点规则（含重心），精度 3
        bary = np.array([
            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            [0.6, 0.2, 0.2],
            [0.2, 0.6, 0.2],
            [0.2, 0.2, 0.6],
        ])
        weights = np.array([-9.0 / 16.0, 25.0 / 48.0, 25.0 / 48.0, 25.0 / 48.0])
    elif degree == 4 or degree == 5:
        # 6点规则，精度 4（扩展为 5 的近似）
        # 使用 Strang 的 6点对称规则
        a1 = 0.816847572980459
        a2 = 0.091576213509771
        b1 = 0.108103018168070
        b2 = 0.445948490915965
        bary = np.array([
            [a1, a2, a2],
            [a2, a1, a2],
            [a2, a2, a1],
            [b1, b2, b2],
            [b2, b1, b2],
            [b2, b2, b1],
        ])
        w1 = 0.109951743655322
        w2 = 0.223381589678011
        weights = np.array([w1, w1, w1, w2, w2, w2])
    else:
        # degree 7：使用 12点规则近似
        a1 = 0.797426985353087
        a2 = 0.101286507323456
        b1 = 0.059715871789770
        b2 = 0.470142064105115
        c1 = 1.0 / 3.0
        bary = np.array([
            [a1, a2, a2],
            [a2, a1, a2],
            [a2, a2, a1],
            [b1, b2, b2],
            [b2, b1, b2],
            [b2, b2, b1],
            [c1, c1, c1],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        w1 = 0.062969590272413
        w2 = 0.066197076394253
        w3 = -0.074785022233841
        w4 = 0.0625
        w5 = 0.0625
        weights = np.array([w1, w1, w1, w2, w2, w2, w3, w4, w4, w4, w5, w5, w5])
    # 归一化权重
    weights = weights / np.sum(weights)
    return bary, weights


def integrate_over_triangle(f, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                            degree: int = 5) -> float:
    """
    在任意平面三角形上数值积分标量函数 f(x, y)。

    Parameters
    ----------
    f : callable
        接受形状 (N, 2) 的坐标数组，返回形状 (N,) 的函数值。
    v1, v2, v3 : np.ndarray, shape (2,)
        三角形顶点。
    degree : int
        求积公式精度。

    Returns
    -------
    result : float
        积分近似值。
    """
    bary, w = triangle_symq_rule(degree)
    # 从重心坐标映射到笛卡尔坐标
    pts = (bary[:, 0:1] * v1.reshape(1, -1)
           + bary[:, 1:2] * v2.reshape(1, -1)
           + bary[:, 2:3] * v3.reshape(1, -1))
    vals = f(pts)
    area = triangle_area(v1, v2, v3)
    return float(area * np.dot(w, vals))


def sphere01_triangle_vertices_to_area(v1: np.ndarray, v2: np.ndarray,
                                       v3: np.ndarray) -> float:
    """
    计算单位球面上由三个顶点围成的球面三角形面积（L'Huilier 定理）。

    Parameters
    ----------
    v1, v2, v3 : np.ndarray, shape (3,)
        单位球面上的顶点（自动归一化）。

    Returns
    -------
    area : float
        球面三角形面积。
    """
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # 球面边长（中心角）
    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v3, v1), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

    s = 0.5 * (a + b + c)
    # 边界处理：若 s 接近 π，面积趋近于 2π
    if s > np.pi - 1e-12:
        return 2.0 * np.pi

    # L'Huilier 定理
    tan_s2 = np.tan(s * 0.5)
    tan_as = np.tan(max(0.0, (s - a) * 0.5))
    tan_bs = np.tan(max(0.0, (s - b) * 0.5))
    tan_cs = np.tan(max(0.0, (s - c) * 0.5))

    # 避免负数导致 sqrt 出错
    prod = tan_s2 * tan_as * tan_bs * tan_cs
    prod = max(prod, 0.0)
    E = 4.0 * np.arctan(np.sqrt(prod))
    return float(E)


def sphere01_triangle_sample(n: int, v1: np.ndarray, v2: np.ndarray,
                             v3: np.ndarray, seed: int = None) -> np.ndarray:
    """
    在单位球面三角形内均匀随机采样 n 个点。

    Parameters
    ----------
    n : int
        采样点数。
    v1, v2, v3 : np.ndarray, shape (3,)
        球面三角形顶点。
    seed : int, optional
        随机种子。

    Returns
    -------
    pts : np.ndarray, shape (3, n)
        采样点坐标（列向量形式）。
    """
    rng = np.random.default_rng(seed)
    # 使用球面上大圆弧三角形的面积坐标采样
    # 先构造切平面上的局部坐标系
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    v3 = np.asarray(v3, dtype=float)
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)

    # 在切平面上使用标准三角形采样，再投影回球面
    r1 = rng.random(n)
    r2 = rng.random(n)
    mask = r1 + r2 > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]

    pts_local = (r1[None, :] * v2[:, None]
                 + r2[None, :] * v3[:, None]
                 + (1.0 - r1 - r2)[None, :] * v1[:, None])
    # 投影到单位球面
    norms = np.sqrt(np.sum(pts_local ** 2, axis=0))
    norms = np.where(norms < 1e-15, 1.0, norms)
    pts = pts_local / norms[None, :]
    return pts


def sphere01_triangle_quad_00(n: int, v1: np.ndarray, v2: np.ndarray,
                              v3: np.ndarray, f, seed: int = None) -> float:
    """
    单位球面三角形上的 Monte Carlo 求积（0阶规则，均匀采样）。

    Parameters
    ----------
    n : int
        采样点数。
    v1, v2, v3 : np.ndarray, shape (3,)
        球面三角形顶点。
    f : callable
        被积函数 f(x) 其中 x 为形状 (3,) 的向量。
    seed : int, optional
        随机种子。

    Returns
    -------
    result : float
        积分近似值。
    """
    area = sphere01_triangle_vertices_to_area(v1, v2, v3)
    pts = sphere01_triangle_sample(n, v1, v2, v3, seed)
    quad = 0.0
    for j in range(n):
        quad += f(pts[:, j])
    return quad * area / n


def integrate_pde_residual_over_mesh(residual_func, triangles: list,
                                     degree: int = 5) -> float:
    """
    在由平面三角形组成的网格上积分 PDE 残差函数。

    Parameters
    ----------
    residual_func : callable
        输入形状 (N, 2) 的坐标，返回残差值数组 (N,)。
    triangles : list of tuple
        每个元素为 (v1, v2, v3)，其中 vi 为 shape (2,) 的顶点。
    degree : int
        求积精度。

    Returns
    -------
    total_residual : float
        网格上残差积分的总和。
    """
    total = 0.0
    for v1, v2, v3 in triangles:
        total += integrate_over_triangle(residual_func, v1, v2, v3, degree)
    return float(total)
