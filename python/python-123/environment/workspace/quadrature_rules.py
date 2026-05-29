"""
quadrature_rules.py

高阶数值积分规则模块

本模块融合以下种子项目的核心算法：
  - 940_quad_gauss: Gauss-Legendre 数值积分
  - 1056_sandia_sparse: Clenshaw-Curtis 求积权重与节点

科学背景：
  肿瘤治疗响应的定量评估需要对多个物理量进行高精度数值积分，例如：
    - 肿瘤内平均药物浓度
    - 累积氧气消耗量
    - 治疗诱导的凋亡细胞总量

  Gauss-Legendre 求积在 [-1,1] 上对 2n-1 次多项式精确：
      integral_{-1}^{1} f(x) dx = sum_{i=1}^{n} w_i * f(x_i)
    其中 x_i 为 n 阶 Legendre 多项式 P_n(x) 的根，
    w_i = 2 / [ (1-x_i^2) * (P_n'(x_i))^2 ]

  Clenshaw-Curtis 求积基于 Chebyshev 节点：
      x_i = cos( (i-1)*pi / (n-1) ),  i = 1..n
    权重通过离散余弦变换（DCT）计算，对解析函数具有谱收敛性。

  治疗响应指数（Therapeutic Response Index, TRI）定义为：
      TRI = integral_{Omega} C_drug(x) * rho(x) * S(sigma_vm(x)) dx
    其中 S(sigma) 为应力依赖的药物渗透函数。
"""

import numpy as np
from typing import Tuple, Callable


def clenshaw_curtis_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Clenshaw-Curtis 求积的节点和权重。

    节点:
        x_i = cos( (i-1) * pi / (n-1) ),  i = 1, ..., n

    权重（基于 Trefethen 的 DCT 公式）:
        w = DCT-I( g )，其中 g = [2; 0; -2/(3*1); 0; -2/(3*3); ...]

    参数:
        n: 求积阶数

    返回:
        x: (n,) 节点数组，位于 [-1, 1]
        w: (n,) 权重数组
    """
    if n < 1:
        raise ValueError("clenshaw_curtis_nodes_weights: n >= 1")

    if n == 1:
        return np.array([0.0]), np.array([2.0])

    theta = np.linspace(0.0, np.pi, n)
    x = np.cos(theta)

    # 内部权重计算
    w = np.zeros(n)
    for i in range(n):
        w[i] = 1.0
        for j in range(1, (n - 1) // 2 + 1):
            b = 1.0 if 2 * j == n - 1 else 2.0
            w[i] -= b * np.cos(2.0 * j * theta[i]) / (4.0 * j * j - 1.0)

    w[0] = w[0] / (n - 1)
    w[1:n - 1] = 2.0 * w[1:n - 1] / (n - 1)
    w[n - 1] = w[n - 1] / (n - 1)

    return x, w


def gauss_legendre_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算 n 点 Gauss-Legendre 求积节点和权重。

    采用 numpy 的 Legendre 多项式根计算（内部使用特征值法）。
    """
    if n < 1:
        raise ValueError("gauss_legendre_nodes_weights: n >= 1")

    # numpy.polynomial.legendre.leggauss 返回 [-1,1] 上的节点和权重
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def integrate_1d(f: Callable[[np.ndarray], np.ndarray],
                 a: float, b: float,
                 rule: str = "gauss", n: int = 16) -> float:
    """
    一维数值积分通用接口。

    参数:
        f: 被积函数
        a, b: 积分区间
        rule: "gauss" 或 "clenshaw_curtis"
        n: 求积阶数

    返回:
        integral: 积分估计值
    """
    if b <= a:
        raise ValueError("integrate_1d: 需要 b > a")

    if rule == "gauss":
        t, w = gauss_legendre_nodes_weights(n)
    elif rule == "clenshaw_curtis":
        t, w = clenshaw_curtis_nodes_weights(n)
    else:
        raise ValueError("integrate_1d: rule 必须是 'gauss' 或 'clenshaw_curtis'")

    # 变量变换: x = (b-a)/2 * t + (a+b)/2
    x = 0.5 * ((b - a) * t + (a + b))
    fx = f(x)
    fx = np.asarray(fx, dtype=float).ravel()

    if fx.shape[0] != n:
        raise ValueError("integrate_1d: f 输出维度与求积阶数不匹配")

    return float(np.sum(w * fx) * (b - a) / 2.0)


def compute_therapy_response_index(
    drug_concentration: np.ndarray,
    cell_density: np.ndarray,
    stress_field: np.ndarray,
    dx: float, dy: float,
    stress_sensitivity: float = 2.0
) -> float:
    """
    计算治疗响应指数（Therapeutic Response Index, TRI）。

    公式:
        TRI = integral Omega C_drug(x) * rho(x) * exp(-stress_sensitivity * sigma_vm(x)) dA

    数值实现采用中点法则（与网格数据兼容）。

    参数:
        drug_concentration: (H, W) 药物浓度场
        cell_density: (H, W) 细胞密度场
        stress_field: (H, W) 冯·米塞斯应力场
        dx, dy: 网格步长
        stress_sensitivity: 应力敏感系数

    返回:
        tri: 治疗响应指数
    """
    if drug_concentration.shape != cell_density.shape or drug_concentration.shape != stress_field.shape:
        raise ValueError("compute_therapy_response_index: 输入场维度不匹配")

    # Sigmoid-like stress penetration penalty
    penalty = np.exp(-stress_sensitivity * np.maximum(stress_field, 0.0))
    integrand = drug_concentration * cell_density * penalty
    tri = float(np.sum(integrand) * dx * dy)
    return tri


def compute_cumulative_oxygen_consumption(
    oxygen_field: np.ndarray,
    cell_density: np.ndarray,
    dx: float, dy: float,
    Vmax: float = 1.0, Km: float = 0.1
) -> float:
    """
    计算整个肿瘤域的累积氧气消耗量。

    公式:
        Q_total = integral Omega Vmax * rho * C / (Km + C) dA
    """
    # === HOLE 3 START ===
    # 请根据 Michaelis-Menten 消耗动力学实现累积氧消耗量的积分计算
    raise NotImplementedError("Hole_3: compute_cumulative_oxygen_consumption 待实现")
    # === HOLE 3 END ===


def integrate_radial_profile(
    r_vals: np.ndarray, f_vals: np.ndarray, dim: int = 2
) -> float:
    """
    积分径向函数 profile。

    2D:  integral_0^R f(r) * 2*pi*r dr
    3D:  integral_0^R f(r) * 4*pi*r^2 dr

    使用梯形法则。
    """
    if r_vals.shape[0] < 2:
        return 0.0
    if f_vals.shape != r_vals.shape:
        raise ValueError("integrate_radial_profile: r_vals 与 f_vals 形状不匹配")

    if dim == 2:
        integrand = f_vals * 2.0 * np.pi * r_vals
    elif dim == 3:
        integrand = f_vals * 4.0 * np.pi * r_vals ** 2
    else:
        raise ValueError("integrate_radial_profile: dim 必须是 2 或 3")

    return float(np.trapezoid(integrand, r_vals))


def estimate_quadrature_error(
    f: Callable[[np.ndarray], np.ndarray],
    a: float, b: float,
    rule: str = "gauss",
    n_coarse: int = 8, n_fine: int = 32
) -> float:
    """
    通过 Richardson 外推估计数值积分误差。

        err_est = |Q_{fine} - Q_{coarse}|

    参数:
        f, a, b: 被积函数与区间
        rule: 求积规则
        n_coarse, n_fine: 粗细网格阶数

    返回:
        err_est: 误差估计
    """
    q_coarse = integrate_1d(f, a, b, rule, n_coarse)
    q_fine = integrate_1d(f, a, b, rule, n_fine)
    return abs(q_fine - q_coarse)
