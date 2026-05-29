"""
integration_validator.py
========================
数值积分规则的精确度验证与催化剂反应-扩散方程的积分守恒校验。

基于种子项目 1143_square_exactness 与 098_black_scholes 重构：
- square_exactness 验证 2D Legendre 积分规则对单项式的精确度；
- black_scholes 提供偏微分方程（抛物型）的解析解框架，
  可转化为扩散方程的 Green 函数解析解，用于验证数值积分。

在本系统中：
1. 验证高斯积分规则在催化剂径向/体积积分中的多项式精确度；
2. 利用 Black-Scholes PDE 的解析解结构（热核形式），
   构造扩散方程的精确解，校验有限差分/有限元离散后的积分守恒性；
3. 确保总体反应速率的数值积分误差在可控范围内。
"""

import numpy as np
from scipy.special import roots_legendre


class IntegrationValidatorError(Exception):
    """积分验证异常。"""
    pass


def legendre_2d_monomial_integral(a, b, p):
    r"""
    计算 2D Legendre 型单项式积分的精确值：

        I = \int_{a_x}^{b_x} \int_{a_y}^{b_y} x^{p_0} y^{p_1} dx dy

    解析解：
        I = \frac{b_x^{p_0+1} - a_x^{p_0+1}}{p_0+1}
          * \frac{b_y^{p_1+1} - a_y^{p_1+1}}{p_1+1}

    Parameters
    ----------
    a, b : ndarray, shape (2,)
        积分区域下界与上界。
    p : ndarray, shape (2,)
        单项式指数 [p0, p1]。

    Returns
    -------
    value : float
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    p = np.asarray(p, dtype=int)

    val = 1.0
    for dim in range(2):
        if p[dim] < 0:
            raise IntegrationValidatorError("指数必须非负")
        exp = p[dim]
        if exp == 0:
            val *= (b[dim] - a[dim])
        else:
            val *= (b[dim] ** (exp + 1) - a[dim] ** (exp + 1)) / (exp + 1)
    return val


def validate_2d_quadrature_rule(n_points, degree_max, a=None, b=None):
    r"""
    验证 n_points × n_points 的二维张量积 Gauss-Legendre 规则
    对总次数 ≤ degree_max 的单项式的积分精确度。

    二维张量积规则：
        \iint f(x,y) dx dy \approx \sum_{i=1}^n \sum_{j=1}^n w_i w_j f(x_i, y_j)

    精确度：可精确积分总次数 ≤ 2n-1 的单项式。

    Parameters
    ----------
    n_points : int
        每维积分节点数。
    degree_max : int
        测试的最高总次数。
    a, b : ndarray, optional
        积分区域，默认 [-1, -1] 到 [1, 1]。

    Returns
    -------
    max_error : float
        所有测试单项式中的最大相对误差。
    error_dict : dict
        各总次数下的最大误差。
    """
    if a is None:
        a = np.array([-1.0, -1.0])
    if b is None:
        b = np.array([1.0, 1.0])

    x1d, w1d = roots_legendre(n_points)
    # 线性映射
    scale = (b - a) / 2.0
    shift = (a + b) / 2.0
    x_nodes = scale[0] * x1d + shift[0]
    y_nodes = scale[1] * x1d + shift[1]
    w_x = scale[0] * w1d
    w_y = scale[1] * w1d

    max_error = 0.0
    error_dict = {}

    for total_degree in range(degree_max + 1):
        max_err_td = 0.0
        for py in range(total_degree + 1):
            px = total_degree - py
            exact = legendre_2d_monomial_integral(a, b, [px, py])

            # 数值积分
            X, Y = np.meshgrid(x_nodes, y_nodes, indexing='ij')
            W = np.outer(w_x, w_y)
            vals = (X ** px) * (Y ** py)
            quad = np.sum(vals * W)

            if abs(exact) < np.finfo(float).eps:
                err = abs(quad - exact)
            else:
                err = abs(quad - exact) / abs(exact)
            max_err_td = max(max_err_td, err)

        error_dict[total_degree] = max_err_td
        max_error = max(max_error, max_err_td)

    return max_error, error_dict


def diffusion_green_function_integral(r, t, D, R):
    r"""
    球坐标下扩散方程 Green 函数的体积分校验。

    考虑催化剂颗粒内瞬时点源扩散，Green 函数满足：
        \frac{\partial G}{\partial t} = D \nabla^2 G

    在球坐标径向对称情况下，自由空间 Green 函数为：
        G(r, t) = \frac{1}{(4\pi D t)^{3/2}} \exp\left(-\frac{r^2}{4Dt}\right)

    其体积分应守恒：
        \int_0^{\infty} G(r,t) 4\pi r^2 dr = 1

    该函数用于验证数值积分规则是否保持守恒性。

    Parameters
    ----------
    r : ndarray
        径向节点 [m]。
    t : float
        时间 [s]，t > 0。
    D : float
        扩散系数 [m²/s]。
    R : float
        积分上限 [m]。

    Returns
    -------
    integral_value : float
        数值积分结果。
    exact_value : float
        解析值 1 - erf(R / sqrt(4Dt)) + ...（对于截断球域）。
    """
    if t <= 0 or D <= 0 or R <= 0:
        raise IntegrationValidatorError("t, D, R 必须为正")

    r = np.asarray(r, dtype=float)
    G = (1.0 / (4.0 * np.pi * D * t) ** 1.5) * np.exp(-r ** 2 / (4.0 * D * t))
    integrand = G * 4.0 * np.pi * r ** 2

    # 梯形积分
    integral_value = np.trapezoid(integrand, r)

    # 解析值（全空间为1，截断到R使用不完全Gamma）
    from scipy.special import erf
    exact_full = 1.0
    # 截断误差估计
    truncation = erf(R / np.sqrt(4.0 * D * t))
    exact_truncated = truncation  # 近似
    return integral_value, exact_full, exact_truncated


def black_scholes_diffusion_analogy(S, K, T, r, sigma):
    r"""
    Black-Scholes 公式与催化剂扩散-反应的数学类比。

    Black-Scholes PDE：
        \frac{\partial V}{\partial t} + \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2}
        + r S \frac{\partial V}{\partial S} - r V = 0

    通过变量替换 x = \ln(S/K)，可转化为标准热传导方程：
        \frac{\partial u}{\partial \tau} = \frac{\partial^2 u}{\partial x^2}

    这与催化剂孔道内的一维非稳态扩散方程形式一致：
        \frac{\partial C}{\partial t} = D \frac{\partial^2 C}{\partial z^2}

    此处保留 Black-Scholes 的解析解结构，用于构造测试问题的精确解。

    欧洲看涨期权的解析解：
        C = S_0 N(d_1) - K e^{-rT} N(d_2)
    其中
        d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)T}{\sigma\sqrt{T}}
        d_2 = d_1 - \sigma\sqrt{T}

    Parameters
    ----------
    S, K, T, r, sigma : float
        Black-Scholes 参数。

    Returns
    -------
    call_price : float
        看涨期权价格（类比为扩散通量）。
    d1, d2 : float
        中间变量。
    """
    from scipy.stats import norm
    if T <= 0:
        return max(S - K, 0.0), 0.0, 0.0
    if sigma <= 0:
        raise IntegrationValidatorError("sigma 必须为正")

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    N1 = norm.cdf(d1)
    N2 = norm.cdf(d2)
    call_price = S * N1 - K * np.exp(-r * T) * N2
    return call_price, d1, d2


def validate_reaction_diffusion_conservation(C, R, r_nodes, reaction_rates,
                                              D_eff=1e-6):
    r"""
    验证催化剂颗粒内反应-扩散的积分守恒性。

    稳态下，颗粒表面的扩散通量应等于内部总反应消耗速率：
        4\pi R^2 D \left.\frac{dC}{dr}\right|_{r=R}
        = \int_0^R R_{local}(r) 4\pi r^2 dr

    Parameters
    ----------
    C : ndarray
        径向浓度分布。
    R : float
        颗粒半径。
    r_nodes : ndarray
        径向节点。
    reaction_rates : ndarray
        各节点上的反应速率 [mol/(m³·s)]。
    D_eff : float
        有效扩散系数 [m²/s]。

    Returns
    -------
    flux_surface : float
        表面扩散通量 [mol/s]。
    total_reaction : float
        总反应速率 [mol/s]。
    relative_error : float
        相对守恒误差。
    """
    if r_nodes.size != C.size or r_nodes.size != reaction_rates.size:
        raise IntegrationValidatorError("数组长度不一致")

    # 表面浓度梯度（一阶后向差分）
    dr_last = r_nodes[-1] - r_nodes[-2]
    dCdr = (C[-1] - C[-2]) / dr_last
    flux_surface = 4.0 * np.pi * R ** 2 * D_eff * dCdr

    # 总体反应速率（梯形积分）
    total_reaction = np.trapezoid(reaction_rates * 4.0 * np.pi * r_nodes ** 2,
                                   r_nodes)

    denom = max(abs(flux_surface), abs(total_reaction), np.finfo(float).eps)
    relative_error = abs(flux_surface - total_reaction) / denom
    return flux_surface, total_reaction, relative_error
