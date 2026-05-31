"""
quadrature_rules.py
===================
高斯数值积分规则生成器与催化反应体积分的精确度验证。

基于种子项目 467_gen_laguerre_rule 与 461_gegenbauer_exactness 重构：
- gen_laguerre_rule 生成广义 Gauss-Laguerre 积分规则（IQPACK 算法）；
- gegenbauer_exactness 验证 Gegenbauer 权函数下积分的多项式精确度。

在本系统中用于：
1. 催化剂颗粒径向积分（广义 Laguerre 映射到有限球域）；
2. 反应速率在孔径分布上的加权平均（Gegenbauer 权重模拟孔径分布）；
3. 有限元质量矩阵与刚度矩阵的单元积分；
4. 数值积分精确度的系统验证。
"""

import numpy as np
from scipy.special import gamma as scipy_gamma
from scipy.special import roots_genlaguerre, roots_legendre, roots_jacobi


class QuadratureError(Exception):
    """数值积分异常。"""
    pass


def gauss_legendre_rule(n, a=-1.0, b=1.0):
    r"""
    生成 [a, b] 上的 n 点 Gauss-Legendre 积分规则。

    积分公式：
        \int_a^b f(x) dx \approx \sum_{i=1}^n w_i f(x_i)

    对于多项式 f(x) 次数 ≤ 2n-1，积分精确成立。

    Parameters
    ----------
    n : int
        积分节点数，n ≥ 1。
    a, b : float
        积分区间端点。

    Returns
    -------
    x : ndarray, shape (n,)
        积分节点。
    w : ndarray, shape (n,)
        积分权重。
    """
    if n < 1:
        raise QuadratureError("n 必须 ≥ 1")
    x, w = roots_legendre(n)
    # 线性映射到 [a, b]
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    x = scale * x + shift
    w = scale * w
    return x, w


def gauss_genlaguerre_rule(n, alpha, a=0.0, b=1.0):
    r"""
    生成广义 Gauss-Laguerre 积分规则，并缩放至 [a, +∞)。

    标准权重：
        w(x) = (x-a)^{\alpha} \exp(-b (x-a))

    积分：
        \int_a^{\infty} (x-a)^{\alpha} e^{-b(x-a)} f(x) dx
        \approx \sum_{i=1}^n w_i f(x_i)

    基于 gen_laguerre_rule 的 IQPACK 算法思想，使用 scipy 的
    roots_genlaguerre 获取节点与权重，再进行指数缩放。

    Parameters
    ----------
    n : int
    alpha : float
        α > -1。
    a : float
        左端点。
    b : float
        指数缩放因子，b > 0。

    Returns
    -------
    x : ndarray
    w : ndarray
    """
    if n < 1:
        raise QuadratureError("n 必须 ≥ 1")
    if alpha <= -1.0:
        raise QuadratureError("alpha 必须 > -1")
    if b <= 0.0:
        raise QuadratureError("b 必须 > 0")

    # 标准 Laguerre 节点 (0, +inf)，权重 x^alpha exp(-x)
    t, wts = roots_genlaguerre(n, alpha)
    # 缩放：令 x = a + t / b
    x = a + t / b
    w = wts / (b ** (alpha + 1.0))
    return x, w


def radial_quadrature_sphere(n, R):
    r"""
    生成球坐标径向积分的高斯积分规则。

    球内体积分：
        \int_0^R f(r) 4\pi r^2 dr

    通过变量替换 r = R (t+1)/2，将 [0, R] 映射到 [-1, 1]，
    使用 Gauss-Legendre 规则：

        = 4\pi R^3 \int_{-1}^{1} f(R(t+1)/2) \left(\frac{t+1}{2}\right)^2 \frac{dt}{2}

    Parameters
    ----------
    n : int
        积分节点数。
    R : float
        球半径，R > 0。

    Returns
    -------
    r_nodes : ndarray
        径向节点（位于 [0, R] 内）。
    r_weights : ndarray
        已包含 4πr² 雅可比因子的权重。
    """
    if R <= 0:
        raise QuadratureError("R 必须为正")

    t, wt = roots_legendre(n)
    r_nodes = R * (t + 1.0) / 2.0
    jacobian = R / 2.0
    # 权重包含 4πr² 和 dr/dt = R/2
    r_weights = 4.0 * np.pi * (r_nodes ** 2) * wt * jacobian
    return r_nodes, r_weights


def gegenbauer_quadrature_exactness(alpha, n_points, degree_max):
    """
    验证 Gegenbauer 积分规则的单项式精确度。

    对于 n_points 个节点的 Gauss-Gegenbauer 规则，理论上可精确积分
    次数 ≤ 2n_points - 1 的单项式。

    计算：
        E(p) = | I_{quad}(x^p) - I_{exact}(x^p) | / | I_{exact}(x^p) |

    Parameters
    ----------
    alpha : float
        Gegenbauer 参数。
    n_points : int
        积分节点数。
    degree_max : int
        测试的最高单项式次数。

    Returns
    -------
    errors : dict
        键为单项式次数，值为相对误差。
    """
    from special_functions import gegenbauer_integral

    # 获取 Gegenbauer 节点和权重（Jacobi(alpha+0.5, alpha+0.5) 的变形）
    # 标准 Gegenbauer 权重: (1-x^2)^alpha on [-1,1]
    # 使用 scipy.special.roots_jacobi 变换得到
    x, w = roots_jacobi(n_points, alpha, alpha)
    # 注意 scipy 的 Jacobi 权重是 (1-x)^a (1+x)^b，这里 a=b=alpha
    # 此时节点和权重对应于 Jacobi 积分；对于 Gegenbauer 需要归一化
    # 精确归一化：将权重除以 sum(w) * 2^{2a+1} * B(a+1,a+1) ...
    # 为简化，直接用数值验证

    # 计算零阶矩以归一化
    moment0 = np.sum(w)
    # 目标零阶矩：\int_{-1}^1 (1-x^2)^alpha dx
    target_moment0 = (2.0 ** (2.0 * alpha + 1.0)) * (scipy_gamma(alpha + 1.0) ** 2) \
                     / scipy_gamma(2.0 * alpha + 2.0)
    scale = target_moment0 / moment0
    w = w * scale

    errors = {}
    for p in range(degree_max + 1):
        exact = gegenbauer_integral(p, alpha)
        quad_val = np.sum(w * (x ** p))
        if abs(exact) < np.finfo(float).eps:
            err = abs(quad_val - exact)
        else:
            err = abs(quad_val - exact) / abs(exact)
        errors[p] = err
    return errors


def integrate_reaction_rate_radial(reaction_rate_func, R, n_quad=16):
    r"""
    在球形催化剂颗粒内积分反应速率。

    计算总体反应速率：
        R_{total} = \int_0^R r_{local}(r) \cdot 4\pi r^2 dr

    Parameters
    ----------
    reaction_rate_func : callable
        函数签名 f(r) -> float，返回径向位置 r 处的局部反应速率 [mol/(m³·s)]。
    R : float
        颗粒半径 [m]。
    n_quad : int, default 16
        高斯积分节点数。

    Returns
    -------
    total_rate : float
        总体反应速率 [mol/s]。
    """
    r_nodes, r_weights = radial_quadrature_sphere(n_quad, R)
    rates = np.array([reaction_rate_func(r) for r in r_nodes])
    total_rate = np.sum(rates * r_weights)
    return total_rate


def pore_size_moment_quadrature(pore_diameters, weights, moment_order,
                                alpha_param=0.5):
    r"""
    使用 Gegenbauer 权重对孔径分布进行矩计算。

    将孔径分布归一化到 [-1, 1] 区间，使用 Gegenbauer 权函数
    计算加权矩：
        \mu_k = \int_{-1}^{1} d^k (1-d^2)^{\alpha} p(d) dd

    这里采用离散近似：
        \mu_k \approx \sum_j w_j d_j^k (1-d_j^2)^{\alpha}

    Parameters
    ----------
    pore_diameters : ndarray
        归一化孔径（范围 [-1, 1]）。
    weights : ndarray
        各孔径对应的权重/概率。
    moment_order : int
        矩的阶数。
    alpha_param : float
        Gegenbauer 参数。

    Returns
    -------
    moment : float
    """
    pore_diameters = np.asarray(pore_diameters, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if pore_diameters.size == 0:
        raise QuadratureError("孔径分布为空")
    # 归一化权重
    wsum = np.sum(weights)
    if wsum == 0:
        raise QuadratureError("权重之和为零")
    weights = weights / wsum

    # 限制孔径在 [-1, 1] 内
    d_clip = np.clip(pore_diameters, -1.0 + 1e-12, 1.0 - 1e-12)
    weighted = (d_clip ** moment_order) * ((1.0 - d_clip ** 2) ** alpha_param)
    moment = np.sum(weights * weighted)
    return moment
