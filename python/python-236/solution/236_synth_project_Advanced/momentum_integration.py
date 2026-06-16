"""
momentum_integration.py — 动量空间高精度积分
==============================================
融合种子项目:
  [916_prism_jaskowiec_rule] : 高阶求积规则 → Brillouin 区积分
  [945_quad_trapezoid] : 复合梯形 → 一维动量积分

物理背景:
  格点 QCD 中许多物理量需要在 Brillouin 区上积分:
  I = int_{-pi}^{pi} d^4p / (2*pi)^4  f(p)

  例如:
  - 传播子的圈图修正: Sigma(p) = int d^4k/(2pi)^4 G(k) V(k,p)
  - 有限体积修正的积分表示
  - 动量分布函数

  高阶求积规则 (源自 [916] 的 Jaskowiec 规则):
  - Gauss-Legendre 求积: 对 N 点精确到 2N-1 次多项式
  - 复合 Simpson: O(h^4) 精度
  - 张量积规则: d 维 = 1d 规则的张量积

核心公式:
  1D Gauss-Legendre:  int_{-1}^{1} f(x)dx ≈ sum_i w_i f(x_i)
  复合 Simpson:        int_a^b f(x)dx ≈ (h/3)[f(a) + 4f(a+h) + 2f(a+2h) + ... + f(b)]
  周期函数积分:        梯形法则对周期函数是指数收敛的!
"""

import numpy as np
from typing import Tuple, Callable, Optional


def gauss_legendre_1d(n_points: int) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre 求积节点和权重.

    int_{-1}^{1} f(x) dx ≈ sum_{i=1}^{N} w_i * f(x_i)

    精确到 2N-1 次多项式.
    节点为 Legendre 多项式 P_N(x) 的零点.

    参数
    ----
    n_points : int
        求积点数.

    返回
    ----
    nodes : ndarray, shape (N,)
    weights : ndarray, shape (N,)
    """
    if n_points < 1:
        raise ValueError(f"求积点数 {n_points} 至少为 1")
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return nodes, weights


def composite_simpson_1d(f: Callable[[np.ndarray], np.ndarray],
                         a: float, b: float,
                         n_intervals: int) -> float:
    """复合 Simpson 求积 (源自 [945_quad_trapezoid] 的梯形法则推广).

    int_a^b f(x) dx ≈ (h/3) [f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + ... + f(x_n)]
    其中 h = (b-a)/n,  n 必须为偶数.

    精度: O(h^4).

    参数
    ----
    f : callable
        被积函数.
    a, b : float
        积分区间.
    n_intervals : int
        子区间数 (必须为偶数).

    返回
    ----
    result : float
        积分近似值.
    """
    if n_intervals % 2 != 0:
        n_intervals += 1
    if n_intervals < 2:
        n_intervals = 2

    h = (b - a) / n_intervals
    x = np.linspace(a, b, n_intervals + 1)
    fx = np.array([f(xi) for xi in x])

    # Simpson 权重: 1, 4, 2, 4, 2, ..., 4, 1
    weights = np.ones(n_intervals + 1)
    weights[1:-1:2] = 4.0
    weights[2:-2:2] = 2.0

    return float(h / 3.0 * np.sum(weights * fx))


def trapezoid_periodic_1d(f: Callable[[np.ndarray], np.ndarray],
                          a: float, b: float,
                          n_points: int) -> float:
    """梯形法则 (对周期函数指数收敛).

    源自 [945_quad_trapezoid]: 复合梯形公式.

    对周期函数 f(x+L) = f(x), 梯形法则的误差为 O(exp(-c*N)).

    参数
    ----
    f : callable
    a, b : float
        积分区间 (应为一个完整周期).
    n_points : int
        采样点数.

    返回
    ----
    result : float
    """
    x = np.linspace(a, b, n_points + 1)[:-1]  # 不包含右端点 (周期)
    fx = np.array([f(xi) for xi in x])
    return float((b - a) / n_points * np.sum(fx))


def brillouin_zone_integral_1d(integrand: Callable[[float], float],
                               n_points: int = 64,
                               method: str = 'gauss') -> float:
    """一维 Brillouin 区积分.

    I = (1/2pi) * int_{-pi}^{pi} dp  f(p)

    参数
    ----
    integrand : callable
    n_points : int
    method : str
        'gauss' (Gauss-Legendre), 'simpson', 'trapezoid'.

    返回
    ----
    result : float
    """
    if method == 'gauss':
        nodes, weights = gauss_legendre_1d(n_points)
        # 变换 [-1,1] → [-pi, pi]
        x = np.pi * nodes
        w = np.pi * weights
        vals = np.array([integrand(xi) for xi in x])
        return float(np.sum(w * vals) / (2 * np.pi))
    elif method == 'simpson':
        def scaled_f(x):
            return integrand(x) / (2 * np.pi)
        return composite_simpson_1d(scaled_f, -np.pi, np.pi, n_points)
    elif method == 'trapezoid':
        def scaled_f(x):
            return integrand(x) / (2 * np.pi)
        return trapezoid_periodic_1d(scaled_f, -np.pi, np.pi, n_points)
    else:
        raise ValueError(f"未知积分方法: {method}")


def lattice_tadpole_integral(mass_sq: float,
                             n_points: int = 32,
                             ndim: int = 4) -> float:
    """格点 tadpole 积分.

    I_tad = int_{BZ} d^dp/(2pi)^d  1/(hat{p}^2 + m^2)
    其中 hat{p}_mu = 2*sin(p_mu/2) 为格点动量.

    这是格点微扰论中的基本积分, 出现在质量重整化中.

    参数
    ----
    mass_sq : float
        质量平方 m^2.
    n_points : int
        每个方向的积分点数.
    ndim : int
        维度.
    """
    if mass_sq < 0:
        raise ValueError(f"质量平方 {mass_sq} 不能为负")

    # Gauss-Legendre 节点
    nodes, weights = gauss_legendre_1d(n_points)
    # 变换到 [-pi, pi]
    p = np.pi * nodes
    w = np.pi * weights

    if ndim == 1:
        # 1D: 直接求和
        hat_p_sq = 4.0 * np.sin(p / 2.0) ** 2
        vals = 1.0 / (hat_p_sq + mass_sq)
        return float(np.sum(w * vals) / (2 * np.pi))
    elif ndim == 2:
        # 2D 张量积
        result = 0.0
        for i in range(n_points):
            for j in range(n_points):
                hp2 = (4.0 * np.sin(p[i] / 2) ** 2
                       + 4.0 * np.sin(p[j] / 2) ** 2)
                result += w[i] * w[j] / (hp2 + mass_sq)
        return float(result / (2 * np.pi) ** 2)
    elif ndim == 4:
        # 4D: 使用 Monte Carlo 近似 (完全张量积太大)
        rng = np.random.default_rng(236)
        n_mc = min(n_points ** 2, 10000)
        p_mc = rng.uniform(-np.pi, np.pi, size=(n_mc, 4))
        hp2 = np.sum(4.0 * np.sin(p_mc / 2.0) ** 2, axis=1)
        vals = 1.0 / (hp2 + mass_sq)
        volume = (2 * np.pi) ** 4
        return float(np.mean(vals))
    else:
        raise ValueError(f"维度 {ndim} 不支持 (仅 1,2,4)")


def spectral_representation_integral(rho: np.ndarray,
                                     omega: np.ndarray,
                                     t_values: np.ndarray) -> np.ndarray:
    """谱表示积分: C(t) = int_0^inf d_omega rho(omega) * exp(-omega*t).

    使用复合 Simpson 或梯形法则 (源自 [945]).

    参数
    ----
    rho : ndarray
        谱函数 rho(omega).
    omega : ndarray
        频率网格.
    t_values : ndarray
        时间值.

    返回
    ----
    C_t : ndarray
        关联函数.
    """
    C_t = np.zeros(len(t_values))
    for i, t in enumerate(t_values):
        integrand = rho * np.exp(-omega * t)
        # 梯形法则 (源自 [945])
        C_t[i] = np.trapz(integrand, omega)
    return C_t
