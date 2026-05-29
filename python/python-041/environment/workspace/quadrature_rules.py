"""
 quadrature_rules.py
 
 融合种子项目:
   - 804_nint_exactness_mixed: 多维混合高斯求积规则的精确度测试
 
 科学应用:
   在全波形反演的谱元法（Spectral Element Method, SEM）中，需要高精度的数值积分
   来计算质量矩阵和刚度矩阵。本模块提供 Legendre、Jacobi、Laguerre、Hermite 
   等正交多项式的积分公式及其精确度验证，用于地震波谱元离散化中的 Gauss-Lobatto-Legendre
   (GLL) 积分点权重计算。
"""

import numpy as np
from scipy.special import gamma, factorial, factorial2, hyp2f1


def monomial_integral_legendre(expon):
    """
    计算 Legendre 权下的单项式积分:
      I = integral_{-1}^{+1} x^n dx
    
    解析公式:
      n 为奇数: I = 0
      n 为偶数: I = 2 / (n + 1)
    
    Parameters
    ----------
    expon : int
        单项式指数，必须 >= 0。
    
    Returns
    -------
    value : float
        积分值。
    """
    if expon < 0:
        return -np.inf
    if expon % 2 == 1:
        return 0.0
    return 2.0 / (expon + 1)


def monomial_integral_jacobi(expon, alpha, beta):
    """
    计算 Jacobi 权下的单项式积分:
      I = integral_{-1}^{+1} x^n (1-x)^alpha (1+x)^beta dx
    
    解析解由超几何函数给出:
      I = gamma(1+n) * [ s*gamma(1+beta)*2F1(-alpha, 1+n; 2+beta+n; -1) / gamma(2+beta+n)
                         + gamma(1+alpha)*2F1(-beta, 1+n; 2+alpha+n; -1) / gamma(2+alpha+n) ]
    其中 s = (-1)^n。
    
    Parameters
    ----------
    expon : int
        单项式指数。
    alpha, beta : float
        Jacobi 参数，需满足 > -1。
    
    Returns
    -------
    value : float
        积分值。
    """
    if alpha <= -1.0 or beta <= -1.0:
        return -np.inf
    s = 1.0 if (expon % 2 == 0) else -1.0
    arg1 = -alpha
    arg2 = 1.0 + expon
    arg3 = 2.0 + beta + expon
    arg4 = -1.0
    value1 = hyp2f1(arg1, arg2, arg3, arg4)
    value2 = hyp2f1(-beta, arg2, 2.0 + alpha + expon, arg4)
    value = (
        gamma(1.0 + expon) * (
            s * gamma(1.0 + beta) * value1 / gamma(2.0 + beta + expon)
            + gamma(1.0 + alpha) * value2 / gamma(2.0 + alpha + expon)
        )
    )
    return value


def monomial_integral_laguerre(expon):
    """
    计算 Laguerre 权下的单项式积分:
      I = integral_{0}^{+oo} x^n exp(-x) dx
    
    解析公式:
      I = n!
    
    Parameters
    ----------
    expon : int
        单项式指数，>= 0。
    
    Returns
    -------
    value : float
        积分值。
    """
    if expon < 0:
        return -np.inf
    return float(factorial(expon))


def monomial_integral_generalized_laguerre(expon, alpha):
    """
    计算广义 Laguerre 权下的单项式积分:
      I = integral_{0}^{+oo} x^n x^alpha exp(-x) dx
    
    解析公式:
      I = gamma(alpha + n + 1)
    
    Parameters
    ----------
    expon : int
        单项式指数，>= 0。
    alpha : float
        参数，需满足 > -1。
    
    Returns
    -------
    value : float
        积分值。
    """
    if alpha <= -1.0:
        return -np.inf
    arg = alpha + expon + 1.0
    return gamma(arg)


def monomial_integral_hermite(expon):
    """
    计算 Hermite 权下的单项式积分:
      I = integral_{-oo}^{+oo} x^n exp(-x^2) dx
    
    解析公式:
      n 为奇数: I = 0
      n 为偶数: I = (n-1)!! * sqrt(pi) / 2^{n/2}
    
    Parameters
    ----------
    expon : int
        单项式指数，>= 0。
    
    Returns
    -------
    value : float
        积分值。
    """
    if expon < 0:
        return -np.inf
    if expon % 2 == 1:
        return 0.0
    return float(factorial2(expon - 1)) * np.sqrt(np.pi) / (2.0 ** (expon / 2))


def monomial_integral_generalized_hermite(expon, alpha):
    """
    计算广义 Hermite 权下的单项式积分:
      I = integral_{-oo}^{+oo} x^n |x|^alpha exp(-x^2) dx
    
    解析公式:
      n 为奇数: I = 0
      n 为偶数: I = gamma((alpha + n + 1)/2)
    
    Parameters
    ----------
    expon : int
        单项式指数，>= 0。
    alpha : float
        参数，需满足 > -1。
    
    Returns
    -------
    value : float
        积分值。
    """
    if alpha <= -1.0:
        return -np.inf
    if expon % 2 == 1:
        return 0.0
    arg = (alpha + expon) / 2.0
    if arg <= -1.0:
        return -np.inf
    return gamma((alpha + expon + 1.0) / 2.0)


def monomial_integral_mixed(dim_num, rule, alpha, beta, expon):
    """
    计算多维混合权下的单项式积分:
      I = integral_{R} prod_{d=1}^{D} x_d^{expon_d} w_d(x_d) dx
    
    各维度权函数类型由 rule 指定:
      1 = Gauss-Legendre on [-1,+1]
      2 = Gauss-Jacobi on [-1,+1]
      3 = Gauss-Laguerre on [0,+oo)
      4 = Generalized Gauss-Laguerre on [0,+oo)
      5 = Gauss-Hermite on (-oo,+oo)
      6 = Generalized Gauss-Hermite on (-oo,+oo)
    
    Parameters
    ----------
    dim_num : int
        空间维度。
    rule : ndarray, shape (dim_num,)
        每维的积分规则类型。
    alpha, beta : ndarray, shape (dim_num,)
        Jacobi/广义 Laguerre/广义 Hermite 参数。
    expon : ndarray, shape (dim_num,)
        单项式指数。
    
    Returns
    -------
    value : float
        多维积分值。
    """
    value = 1.0
    for dim in range(dim_num):
        r = int(rule[dim])
        if r == 1:
            value *= monomial_integral_legendre(int(expon[dim]))
        elif r == 2:
            value *= monomial_integral_jacobi(int(expon[dim]), alpha[dim], beta[dim])
        elif r == 3:
            value *= monomial_integral_laguerre(int(expon[dim]))
        elif r == 4:
            value *= monomial_integral_generalized_laguerre(int(expon[dim]), alpha[dim])
        elif r == 5:
            value *= monomial_integral_hermite(int(expon[dim]))
        elif r == 6:
            value *= monomial_integral_generalized_hermite(int(expon[dim]), alpha[dim])
        else:
            raise ValueError(f"Unknown rule type: {r}")
    return value


def gauss_lobatto_legendre_points_weights(n):
    """
    计算 Gauss-Lobatto-Legendre (GLL) 积分点和权重。
    
    GLL 积分在谱元法中广泛使用，其公式为:
      xi_i 为 P'_{N-1}(xi) = 0 的根，加上端点 xi = +/-1
      wi_i = 2 / [N(N-1) * P_{N-1}(xi_i)^2]
    
    其中 P_{N-1} 为 N-1 阶 Legendre 多项式，内部点为 P'_{N-1} 的零点。
    
    Parameters
    ----------
    n : int
        积分点数（多项式阶数为 n-1）。
    
    Returns
    -------
    points : ndarray, shape (n,)
        GLL 积分点。
    weights : ndarray, shape (n,)
        GLL 权重。
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    if n == 2:
        points = np.array([-1.0, 1.0])
        weights = np.array([1.0, 1.0])
        return points, weights
    # 计算 P_{n-1} 的导数多项式，并求其零点作为内部点
    from numpy.polynomial.legendre import legder, legroots
    # P_{n-1} 的系数表示（numpy 格式）
    coeffs = np.zeros(n)
    coeffs[-1] = 1.0
    dp_coeffs = legder(coeffs)
    inner_pts = np.sort(legroots(dp_coeffs))
    points = np.concatenate([[-1.0], inner_pts, [1.0]])
    # GLL 权重（解析公式）
    from scipy.special import eval_legendre
    weights = np.zeros(n)
    for i in range(n):
        xi = points[i]
        p_n1 = eval_legendre(n - 1, xi)
        # 边界处理
        if abs(p_n1) < 1e-15:
            weights[i] = 2.0 / (n * (n - 1))
        else:
            weights[i] = 2.0 / (n * (n - 1) * p_n1 ** 2)
    # 重新归一化以保持总权重为2（处理数值误差）
    weights = weights / np.sum(weights) * 2.0
    return points, weights


def test_quadrature_exactness(max_degree=6):
    """
    测试混合求积规则的精确度。
    
    对二维 Legendre x Legendre 乘积区域，验证单项式积分的精确度。
    
    Parameters
    ----------
    max_degree : int
        最大测试多项式阶数。
    
    Returns
    -------
    errors : list of float
        各阶数最大误差。
    """
    dim_num = 2
    rule = np.array([1, 1])
    alpha = np.zeros(dim_num)
    beta = np.zeros(dim_num)
    errors = []
    for degree in range(max_degree + 1):
        max_err = 0.0
        # GLL 对 2N-3 阶多项式精确，因此需要 N >= (degree+3)/2
        n_quad = max(2, int(np.ceil((degree + 3) / 2.0)) + 1)
        pts, wts = gauss_lobatto_legendre_points_weights(n_quad)
        for i in range(degree + 1):
            expon = np.array([i, degree - i])
            exact = monomial_integral_mixed(dim_num, rule, alpha, beta, expon)
            # 使用 GLL 求积近似
            approx = 0.0
            for ix in range(n_quad):
                for iy in range(n_quad):
                    fval = (pts[ix] ** expon[0]) * (pts[iy] ** expon[1])
                    approx += wts[ix] * wts[iy] * fval
            err = abs(approx - exact)
            if err > max_err:
                max_err = err
        errors.append(max_err)
    return errors
