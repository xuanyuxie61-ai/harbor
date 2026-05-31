"""
polynomial_chaos_uq.py
======================
博士级不确定性量化：广义 Wiener-Hermite 多项式混沌展开

本模块实现了概率论 Hermite 多项式（He_n(x)）及其乘积多项式，
用于对 HJM 框架中随机前向利率曲线进行多项式混沌（Polynomial Chaos, PC）展开。

数学理论
--------
对于定义在 Wiener 空间上的随机前向利率 f(t,T;ξ)，其 Chaos 展开为：

    f(t,T;ξ) = Σ_{|α|=0}^{∞} f_α(t,T) * He_α(ξ)

其中 α = (α_1, α_2, ..., α_d) 为多维指标，
He_α(ξ) = He_{α_1}(ξ_1) * He_{α_2}(ξ_2) * ... * He_{α_d}(ξ_d)
为概率论 Hermite 乘积多项式，满足正交性：

    E[ He_α(ξ) * He_β(ξ) ] = α! * δ_{α,β}

概率论 Hermite 多项式的三项递推关系：
    He_0(x) = 1
    He_1(x) = x
    He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

该展开在金融工程中用于：
  1. 随机利率模型的高效蒙特卡洛替代方案
  2. 敏感性分析（Sobol 指标）
  3. 模型降阶与代理模型构造
"""

import numpy as np


def hep_coefficients(n):
    """
    计算概率论 Hermite 多项式 He_n(x) 的系数。

    递推公式:
        He_{n+1}(x) = x * He_n(x) - n * He_{n-1}(x)

    返回按 x 的升幂排列的系数数组。

    Parameters
    ----------
    n : int
        多项式次数，n >= 0。

    Returns
    -------
    c : np.ndarray, shape (n+1,)
        系数数组，c[k] 为 x^k 的系数。
    """
    if n < 0:
        raise ValueError("hep_coefficients: n 必须非负")
    if n == 0:
        return np.array([1.0])

    # ct[i,j] 存储 He_i 中 x^j 的系数
    ct = np.zeros((n + 1, n + 1), dtype=float)
    ct[0, 0] = 1.0
    ct[1, 1] = 1.0

    for i in range(1, n):
        # He_{i+1} = x * He_i - i * He_{i-1}
        ct[i + 1, 1:i + 2] = ct[i, 0:i + 1]  # x * He_i 移位
        ct[i + 1, 0:i] -= i * ct[i - 1, 0:i]  # -i * He_{i-1}

    return ct[n, 0:n + 1]


def hep_value(x, degree):
    """
    使用 Clenshaw 递推计算概率论 Hermite 多项式 He_n(x)。

    递推:
        v_0 = 1
        v_1 = x
        v_{j+1} = x * v_j - j * v_{j-1}

    Parameters
    ----------
    x : float or np.ndarray
        自变量。
    degree : int
        多项式次数，degree >= 0。

    Returns
    -------
    float or np.ndarray
        He_degree(x) 的值。
    """
    if degree < 0:
        raise ValueError("hep_value: degree 必须非负")
    x = np.asarray(x, dtype=float)
    if degree == 0:
        return np.ones_like(x)
    if degree == 1:
        return x.copy()

    v_prev2 = np.ones_like(x)
    v_prev1 = x.copy()
    v_curr = None
    for j in range(1, degree):
        v_curr = x * v_prev1 - j * v_prev2
        v_prev2 = v_prev1
        v_prev1 = v_curr
    return v_curr


def hep_values(x, max_degree):
    """
    计算从 He_0(x) 到 He_{max_degree}(x) 的所有值。

    Parameters
    ----------
    x : np.ndarray, shape (n,)
        自变量数组。
    max_degree : int
        最大次数。

    Returns
    -------
    v : np.ndarray, shape (n, max_degree + 1)
        v[:, j] = He_j(x)。
    """
    if max_degree < 0:
        raise ValueError("hep_values: max_degree 必须非负")
    x = np.asarray(x, dtype=float).reshape(-1)
    n = x.shape[0]
    v = np.zeros((n, max_degree + 1), dtype=float)
    v[:, 0] = 1.0
    if max_degree >= 1:
        v[:, 1] = x
    for j in range(1, max_degree):
        v[:, j + 1] = x * v[:, j] - j * v[:, j - 1]
    return v


def hermite_product_polynomial_value(m, degrees, x):
    """
    计算 Hermite 乘积多项式 He_{l_1,...,l_m}(x_1,...,x_m)。

    公式:
        He_{l}(x) = Π_{i=1}^{m} He_{l_i}(x_i)

    Parameters
    ----------
    m : int
        空间维度。
    degrees : sequence of int, length m
        各维度的多项式次数。
    x : np.ndarray, shape (n, m)
         evaluation points.

    Returns
    -------
    np.ndarray, shape (n,)
        乘积多项式的值。
    """
    if len(degrees) != m:
        raise ValueError("hermite_product_polynomial_value: degrees 长度必须与 m 一致")
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    if x.shape[1] != m:
        raise ValueError("hermite_product_polynomial_value: x 的列数必须与 m 一致")

    result = np.ones(x.shape[0], dtype=float)
    for i in range(m):
        result *= hep_value(x[:, i], degrees[i])
    return result


def polynomial_chaos_expand(coeffs, multi_indices, xi_samples):
    """
    执行多项式混沌展开求值。

    公式:
        Y(ξ) = Σ_{α} c_α * He_α(ξ)

    Parameters
    ----------
    coeffs : np.ndarray, shape (n_terms,)
        混沌系数 c_α。
    multi_indices : np.ndarray, shape (n_terms, d)
        多维指标 α。
    xi_samples : np.ndarray, shape (n_samples, d)
        标准正态随机变量样本。

    Returns
    -------
    np.ndarray, shape (n_samples,)
        展开后的随机输出。
    """
    coeffs = np.asarray(coeffs, dtype=float)
    multi_indices = np.asarray(multi_indices, dtype=int)
    xi_samples = np.asarray(xi_samples, dtype=float)

    n_terms = coeffs.shape[0]
    n_samples = xi_samples.shape[0]
    d = multi_indices.shape[1]

    if xi_samples.shape[1] != d:
        raise ValueError("polynomial_chaos_expand: xi_samples 维度与 multi_indices 不匹配")

    result = np.zeros(n_samples, dtype=float)
    for k in range(n_terms):
        alpha = multi_indices[k]
        val = np.ones(n_samples, dtype=float)
        for j in range(d):
            val *= hep_value(xi_samples[:, j], alpha[j])
        result += coeffs[k] * val
    return result


def generate_multi_indices(d, p):
    """
    生成总次数不超过 p 的 d 维多项式混沌多重指标。

    指标数: N = (d + p)! / (d! * p!)

    Parameters
    ----------
    d : int
        随机维度。
    p : int
        总次数上限。

    Returns
    -------
    np.ndarray, shape (N, d)
        多重指标数组。
    """
    if d <= 0 or p < 0:
        raise ValueError("generate_multi_indices: d > 0 且 p >= 0")

    indices = []
    def recurse(current, dim, remaining):
        if dim == d - 1:
            current.append(remaining)
            indices.append(current.copy())
            current.pop()
            return
        for k in range(remaining + 1):
            current.append(k)
            recurse(current, dim + 1, remaining - k)
            current.pop()

    for total in range(p + 1):
        recurse([], 0, total)
    return np.array(indices, dtype=int)


def sobol_sensitivity(coeffs, multi_indices):
    """
    基于多项式混沌系数计算 Sobol 敏感性指标。

    总方差:
        Var(Y) = Σ_{|α|>0} α! * c_α²
    第 i 个变量的主效应 Sobol 指标:
        S_i = (1/Var(Y)) * Σ_{α: α_i>0, α_j=0 (j≠i)} α! * c_α²

    Parameters
    ----------
    coeffs : np.ndarray, shape (n_terms,)
        混沌系数。
    multi_indices : np.ndarray, shape (n_terms, d)
        多重指标。

    Returns
    -------
    total_variance : float
        总方差。
    sobol_main : np.ndarray, shape (d,)
        主效应 Sobol 指标。
    """
    coeffs = np.asarray(coeffs, dtype=float)
    multi_indices = np.asarray(multi_indices, dtype=int)

    n_terms = coeffs.shape[0]
    d = multi_indices.shape[1]

    from scipy.special import factorial as sp_factorial
    # 计算阶乘权重 α!
    factorial_alpha = np.prod(sp_factorial(multi_indices, exact=True), axis=1)

    total_variance = np.sum(factorial_alpha[1:] * coeffs[1:] ** 2)
    if total_variance < 1e-30:
        total_variance = 1e-30

    sobol_main = np.zeros(d, dtype=float)
    for i in range(d):
        mask = (multi_indices[:, i] > 0) & (np.sum(multi_indices[:, :i], axis=1) + np.sum(multi_indices[:, i+1:], axis=1) == 0)
        mask[0] = False
        sobol_main[i] = np.sum(factorial_alpha[mask] * coeffs[mask] ** 2) / total_variance

    return total_variance, sobol_main
