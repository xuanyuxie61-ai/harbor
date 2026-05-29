"""
quadrature_integrator.py
高精度数值积分模块
提供广义 Gauss-Hermite 求积、Monte Carlo 积分等方法，
用于计算热解反应中的活化能分布积分与反应器截面上的统计量。
原项目映射:
  - 464_gen_hermite_exactness (广义 Gauss-Hermite 求积)
  - 1147_square_integrals (单位正方形上的 Monte Carlo 积分)
"""

import numpy as np
from scipy.special import gamma as scipy_gamma
from utils import safe_exp


def generalized_hermite_integral(expon, alpha):
    """
    计算广义 Hermite 积分的精确值:
        H(n, α) = ∫_{-∞}^{+∞} x^n |x|^α exp(-x²) dx
    
    精确解:
        若 n 为奇数: 0
        若 n 为偶数: Γ((α + n + 1) / 2)
    
    映射自 gen_hermite_exactness.m 中的 gen_hermite_integral。
    """
    if expon % 2 == 1:
        return 0.0
    a = alpha + expon
    if a <= -1.0:
        return -np.inf
    return scipy_gamma((a + 1.0) / 2.0)


def gauss_hermite_nodes_weights(n, alpha=0.0):
    """
    构造广义 Gauss-Hermite 求积节点与权重。
    使用 scipy 的 hermite 多项式零点。
    
    标准 Gauss-Hermite 求积:
        ∫_{-∞}^{+∞} exp(-x²) f(x) dx ≈ Σ w_i f(x_i)
    
    广义形式 (|x|^α exp(-x²) 权函数):
        w_i^{gen} = w_i^{std} * |x_i|^α
    """
    from numpy.polynomial.hermite import hermgauss
    x, w = hermgauss(n)
    # 广义权重
    if alpha != 0.0:
        w = w * (np.abs(x) ** alpha)
    return x.astype(np.float64), w.astype(np.float64)


def integrate_hermite_quadrature(f, n, alpha=0.0):
    """
    使用广义 Gauss-Hermite 求积计算积分:
        I = ∫_{-∞}^{+∞} |x|^α exp(-x²) f(x) dx
    
    参数:
        f: 被积函数
        n: 求积阶数
        alpha: 权函数指数
    返回:
        I: 积分近似值
    """
    x, w = gauss_hermite_nodes_weights(n, alpha)
    fx = np.array([f(xi) for xi in x], dtype=np.float64)
    return np.sum(w * fx)


def integrate_daem_activation_energy(E, sigma, T, n_quad=16):
    """
    使用 Gauss-Hermite 求积计算分布式活化能模型（DAEM）中的积分。
    
    DAEM 模型中，反应速率:
        k(T) = A * ∫_{0}^{∞} exp(-E'/(RT)) f(E') dE'
    
    其中 f(E') 通常取 Gaussian 分布:
        f(E') = 1/(σ√(2π)) * exp(-(E'-E₀)²/(2σ²))
    
    通过变量替换 x = (E' - E₀) / (√2 σ)，积分变为:
        k(T) = A / √π * ∫_{-∞}^{+∞} exp(-x²) * exp(-(E₀ + √2 σ x)/(RT)) dx
    
    参数:
        E: 平均活化能 [J/mol]
        sigma: 活化能标准差 [J/mol]
        T: 温度 [K]
        n_quad: 求积节点数
    返回:
        k_eff: 有效反应速率因子
    """
    if T < 1e-6:
        return 0.0
    R = 8.314
    x, w = gauss_hermite_nodes_weights(n_quad, alpha=0.0)
    # 被积函数: exp(-(E + sqrt(2)*sigma*x) / (R*T))
    integrand = safe_exp(-(E + np.sqrt(2.0) * sigma * x) / (R * T))
    return np.sum(w * integrand) / np.sqrt(np.pi)


def square01_sample(n):
    """
    在单位正方形 [0,1]² 上均匀随机采样 n 个点。
    映射自 square01_sample.m。
    """
    return np.random.rand(2, n).astype(np.float64)


def square01_monte_carlo_integrate(f, n_samples):
    """
    Monte Carlo 积分计算单位正方形上的积分:
        I = ∫_{0}^{1}∫_{0}^{1} f(x, y) dx dy ≈ (1/N) Σ f(x_i, y_i)
    
    映射自 square_integrals 系列函数。
    """
    points = square01_sample(n_samples)
    values = np.array([f(points[0, i], points[1, i]) for i in range(n_samples)], dtype=np.float64)
    return np.mean(values), np.std(values) / np.sqrt(n_samples)


def reactor_cross_section_average(f, radius=1.0, n_samples=10000):
    """
    在圆形反应器截面上使用 Monte Carlo 方法计算函数平均值。
    
    采用拒绝采样在圆内均匀撒点:
        x = r * sqrt(u1) * cos(2π u2)
        y = r * sqrt(u1) * sin(2π u2)
    其中 u1, u2 ~ U(0,1)。
    """
    u1 = np.random.rand(n_samples)
    u2 = np.random.rand(n_samples)
    r = radius * np.sqrt(u1)
    theta = 2.0 * np.pi * u2
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    values = np.array([f(x[i], y[i]) for i in range(n_samples)], dtype=np.float64)
    return np.mean(values), np.std(values) / np.sqrt(n_samples)


def quadrature_error_analysis(f, f_exact, max_degree=10, alpha=0.0):
    """
    求积误差分析。
    对单项式 x^degree 测试求积规则的精确度。
    映射自 gen_hermite_exactness.m 的误差分析部分。
    """
    errors = []
    for degree in range(max_degree + 1):
        exact = generalized_hermite_integral(degree, alpha)
        x, w = gauss_hermite_nodes_weights(degree + 2, alpha)
        quad = np.sum(w * (x ** degree))
        if abs(exact) < 1e-15:
            err = abs(quad)
        else:
            err = abs((quad - exact) / exact)
        errors.append((degree, exact, quad, err))
    return errors
