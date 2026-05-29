"""
不确定性量化模块 (多项式混沌展开)
==================================
基于种子项目:
  - 854_pce_ode_hermite: 多项式混沌展开与Hermite多项式
  - 1360_truncated_normal: 截断正态分布

科学背景:
  在结构力学中，材料参数(弹性模量、屈服强度等)往往具有不确定性。
  多项式混沌展开(Polynomial Chaos Expansion, PCE)通过将随机变量
  投影到正交多项式基上，将随机问题转化为一组确定性方程。

  对于高斯随机变量 ξ ~ N(0,1)，采用Hermite多项式 {He_n(ξ)} 作为基:
      He_0(ξ) = 1
      He_1(ξ) = ξ
      He_2(ξ) = ξ^2 - 1
      He_3(ξ) = ξ^3 - 3ξ
      ...

  随机材料参数展开:
      μ(ξ) = Σ_{i=0}^{P} μ_i He_i(ξ)

  Galerkin投影将随机有限元方程转化为一组耦合的确定性方程:
      Σ_j K_{ij} u_j = F_i
  其中 K_{ij} = E[He_i(ξ) K(ξ) He_j(ξ)] / E[He_i^2]

关键公式:
  - Hermite多项式递推: He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
  - 正交性: E[He_m(ξ) He_n(ξ)] = n! δ_{mn}
  - 三重积: E[He_i He_j He_k] = i! j! k! / (s! (s-i)! (s-j)! (s-k)!)
    其中 s = (i+j+k)/2 为整数，否则为0
  - 截断正态分布逆CDF: F^{-1}(p) = μ + σ Φ^{-1}(Φ(α) + p(Φ(β)-Φ(α)))
    其中 α=(a-μ)/σ, β=(b-μ)/σ
"""

import numpy as np
from typing import Tuple, List, Optional
from scipy.special import factorial


# ========================================================================
# Hermite 多项式 (概率学家形式 He_n)
# ========================================================================

def hermite_polynomial(n: int, x: float) -> float:
    """
    计算概率学家Hermite多项式 He_n(x)。
    递推关系: He_0(x) = 1, He_1(x) = x,
              He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)
    """
    if n < 0:
        return 0.0
    if n == 0:
        return 1.0
    if n == 1:
        return x
    H_prev2 = 1.0
    H_prev1 = x
    for k in range(1, n):
        H_curr = x * H_prev1 - k * H_prev2
        H_prev2, H_prev1 = H_prev1, H_curr
    return H_prev1


def hermite_basis_vector(x: float, degree: int) -> np.ndarray:
    """
    计算前 degree+1 阶Hermite多项式在 x 处的值。
    返回: [He_0(x), He_1(x), ..., He_degree(x)]
    """
    vals = np.zeros(degree + 1, dtype=np.float64)
    vals[0] = 1.0
    if degree >= 1:
        vals[1] = x
    for n in range(1, degree):
        vals[n + 1] = x * vals[n] - n * vals[n - 1]
    return vals


def hermite_double_product(i: int, j: int) -> float:
    """
    Hermite多项式正交内积: E[He_i(ξ) He_j(ξ)] = i! δ_{ij}
    """
    if i == j:
        return float(factorial(i))
    return 0.0


def hermite_triple_product(i: int, j: int, k: int) -> float:
    """
    三重乘积期望 E[He_i He_j He_k]。
    仅当 i+j+k 为偶数且满足三角不等式时非零。
    公式: i! j! k! / (s! (s-i)! (s-j)! (s-k)!)
    其中 s = (i+j+k)/2
    """
    total = i + j + k
    if total % 2 != 0:
        return 0.0
    s = total // 2
    if s < i or s < j or s < k:
        return 0.0
    num = float(factorial(i) * factorial(j) * factorial(k))
    den = float(factorial(s) * factorial(s - i) * factorial(s - j) * factorial(s - k))
    return num / den


# ========================================================================
# 截断正态分布采样 (基于1360_truncated_normal)
# ========================================================================

def standard_normal_cdf(x: float) -> float:
    """标准正态分布CDF，基于误差函数。"""
    from math import erf, sqrt
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def standard_normal_cdf_inv(p: float) -> float:
    """
    标准正态分布逆CDF (分位数函数)。
    使用Wichura AS 241算法的近似实现，精度约 10^{-16}。
    """
    if p <= 0:
        return -1e12
    if p >= 1:
        return 1e12
    from math import log, sqrt
    #  rational approximation for lower tail
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        num = (((((((2.5090809287301226727e+3 * r +
                     3.3430575583588128105e+4) * r +
                     6.7265770927008700853e+4) * r +
                     4.5921953931549871457e+4) * r +
                     1.3731693765509461125e+4) * r +
                     1.9715909503065514427e+3) * r +
                     1.3314136788658342429e+2) * r +
                     3.3871328727963666080e+0) * q
        den = (((((((5.2264952788528545610e+3 * r +
                     2.8729085735721942674e+4) * r +
                     3.9307895800092710610e+4) * r +
                     2.1213794301586595867e+4) * r +
                     5.3941960214247511077e+3) * r +
                     6.8718700749205790830e+2) * r +
                     4.2313330701600911252e+1) * r +
                     1.0)
        return num / den
    else:
        r = p if q <= 0 else 1.0 - p
        r = -log(r)
        if r <= 5.0:
            r = r - 1.6
            num = (((((((7.7454501427834140764e-4 * r +
                         2.27238449892691845833e-2) * r +
                         2.41780725177450611770e-1) * r +
                         1.27045825245236838258e+0) * r +
                         3.64784832476320460504e+0) * r +
                         5.76949722146069140550e+0) * r +
                         4.63033784615654529590e+0) * r +
                         1.42343711074968357734e+0)
            den = (((((((1.05075007164441684324e-9 * r +
                         5.47593808499534494600e-4) * r +
                         1.51986665636164571966e-2) * r +
                         1.48103976427480074590e-1) * r +
                         6.89767334985100004550e-1) * r +
                         1.67638483018380384940e+0) * r +
                         2.05319162663775882187e+0) * r +
                         1.0)
        else:
            r = sqrt(r) - 3.0
            num = (((((((2.01033439929228813265e-7 * r +
                         2.71155556874348757815e-5) * r +
                         1.24266094738807843860e-3) * r +
                         2.65321895265761230930e-2) * r +
                         2.96560571828504891230e-1) * r +
                         1.78482653991729133580e+0) * r +
                         5.46378491116411436990e+0) * r +
                         6.65790464350110377720e+0)
            den = (((((((2.04426310338993978564e-15 * r +
                         1.42151175831644588870e-7) * r +
                         1.84631831751005468180e-5) * r +
                         7.86869131145613259100e-4) * r +
                         1.48753612908506148525e-2) * r +
                         1.36929880922735805310e-1) * r +
                         5.99832206555887937690e-1) * r +
                         1.0)
        x = num / den
        return -x if q < 0 else x


def truncated_normal_sample(mu_param: float, sigma_param: float,
                             a: float, b: float,
                             n_samples: int = 1,
                             rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    从截断正态分布 N(μ, σ^2; a, b) 中采样。
    方法: 逆变换采样。
      X = μ + σ * Φ^{-1}( Φ(α) + U * (Φ(β) - Φ(α)) )
    其中 α=(a-μ)/σ, β=(b-μ)/σ, U~Uniform(0,1)。

    参数:
        mu_param: 均值 μ
        sigma_param: 标准差 σ
        a, b: 截断区间 [a, b]
        n_samples: 采样数
        rng: 随机数生成器

    返回:
        samples: (n_samples,) 采样数组
    """
    if sigma_param <= 0:
        raise ValueError("标准差必须为正")
    if rng is None:
        rng = np.random.default_rng(seed=42)
    alpha = (a - mu_param) / sigma_param
    beta = (b - mu_param) / sigma_param
    Phi_alpha = standard_normal_cdf(alpha)
    Phi_beta = standard_normal_cdf(beta)
    U = rng.random(n_samples)
    Z = Phi_alpha + U * (Phi_beta - Phi_alpha)
    # 避免边界
    Z = np.clip(Z, 1e-12, 1.0 - 1e-12)
    samples = mu_param + sigma_param * np.array([standard_normal_cdf_inv(z) for z in Z])
    # 再次截断到 [a, b]
    samples = np.clip(samples, a, b)
    return samples


# ========================================================================
# PCE 展开与统计矩
# ========================================================================

def pce_coefficients_from_samples(samples: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    从样本数据估计PCE系数 (基于Galerkin投影的非侵入式方法)。
    使用数值积分近似:
      c_k = E[y(ξ) He_k(ξ)] / E[He_k^2]

    参数:
        samples: (n_samples,) 样本值(假设输入为标准高斯变量)
        degree: PCE阶数

    返回:
        coeffs: (degree+1,) PCE系数
    """
    coeffs = np.zeros(degree + 1, dtype=np.float64)
    for k in range(degree + 1):
        basis_vals = np.array([hermite_polynomial(k, xi) for xi in samples])
        # 蒙特卡洛估计期望
        numerator = np.mean(samples * basis_vals)
        denominator = hermite_double_product(k, k)
        coeffs[k] = numerator / denominator
    return coeffs


def pce_mean(coeffs: np.ndarray) -> float:
    """PCE展开均值 = c_0 (因为 E[He_k]=0 for k>0)"""
    return float(coeffs[0])


def pce_variance(coeffs: np.ndarray) -> float:
    """PCE展开方差 = Σ_{k=1}^{P} c_k^2 * k!"""
    var = 0.0
    for k in range(1, len(coeffs)):
        var += coeffs[k] ** 2 * float(factorial(k))
    return var


def pce_standard_deviation(coeffs: np.ndarray) -> float:
    """PCE标准差"""
    return np.sqrt(pce_variance(coeffs))


def generate_hermite_quadrature_points(n_points: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成Gauss-Hermite积分点与权重 (用于标准正态空间)。
    利用numpy的hermite_e多项式根。

    返回:
        xi: (n_points,) 积分点
        w: (n_points,) 权重
    """
    try:
        from numpy.polynomial.hermite_e import hermegauss
        xi, w = hermegauss(n_points)
        return xi.astype(np.float64), w.astype(np.float64)
    except Exception:
        # 备选: 使用少量预定义点
        if n_points == 3:
            xi = np.array([-1.7320508075688772, 0.0, 1.7320508075688772])
            w = np.array([0.16666666666666666, 0.6666666666666666, 0.16666666666666666])
        elif n_points == 5:
            xi = np.array([-2.8569700138728, -1.3556261799743, 0.0,
                           1.3556261799743, 2.8569700138728])
            w = np.array([0.011257411327721, 0.11723990766176, 0.24300470558030,
                          0.11723990766176, 0.011257411327721])
        else:
            xi = np.array([0.0])
            w = np.array([1.0])
        return xi, w
