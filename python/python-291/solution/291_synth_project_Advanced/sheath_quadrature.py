"""
sheath_quadrature.py
====================
速度空间高斯求积模块。

本模块实现用于等离子体动理学计算的高斯求积方法：
    1. Gauss-Legendre 求积（有限区间）
    2. Gauss-Hermite 求积（无穷区间，速度分布函数积分）
    3. 三角形单元 Gauss 求积
    4. 双曲立方体求积（多维速度空间）
    5. 等离子体矩量计算

核心公式：
    等离子体矩量：
        n(x) = ∫ f(x,v) dv        (密度)
        u(x) = (1/n) ∫ v f dv     (平均速度)
        T(x) = (m/n) ∫ (v-u)² f dv (温度)
    Maxwell 分布：
        f_M(v) = n / sqrt(2π v_th²) * exp(-v² / (2 v_th²))
    其中 v_th = sqrt(k_B T / m)
"""

import numpy as np
from scipy.special import roots_legendre, roots_hermite
from scipy import integrate
import math
from typing import Tuple, Optional, Callable


def gauss_legendre_nodes_weights(n_quad: int, a: float = -1.0, b: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Legendre 求积节点和权重

    ∫_a^b f(x) dx ≈ Σ w_i f(x_i)

    参数：
        n_quad: 求积点数
        a, b: 积分区间

    返回：
        nodes: shape (n_quad,) 求积节点
        weights: shape (n_quad,) 求积权重
    """
    nodes_ref, weights_ref = roots_legendre(n_quad)
    # 映射到 [a, b]
    nodes = 0.5 * (b - a) * nodes_ref + 0.5 * (a + b)
    weights = 0.5 * (b - a) * weights_ref
    return nodes, weights


def gauss_hermite_nodes_weights(n_quad: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gauss-Hermite 求积节点和权重

    ∫_{-∞}^{∞} exp(-x²) f(x) dx ≈ Σ w_i f(x_i)

    参数：
        n_quad: 求积点数

    返回：
        nodes: shape (n_quad,) 求积节点
        weights: shape (n_quad,) 求积权重
    """
    nodes, weights = roots_hermite(n_quad)
    return nodes, weights


def maxwellian_1d(v: np.ndarray, n: float, T: float, m: float) -> np.ndarray:
    """
    一维 Maxwell 速度分布函数
        f_M(v) = n / sqrt(2π k_B T / m) * exp(-m v² / (2 k_B T))
    在无量纲单位 (v_th = 1) 下：
        f_M(v) = n / sqrt(2π) * exp(-v² / 2)

    参数：
        v: 速度数组
        n: 数密度
        T: 温度 [eV]
        m: 粒子质量 [kg]

    返回：
        f: 分布函数值
    """
    from sheath_constants import E_CHARGE
    v_th = math.sqrt(T * E_CHARGE / m)
    if v_th < 1e-30:
        # 冷等离子体极限
        f = np.zeros_like(v)
        f[np.abs(v) < 1e-10] = n / 1e-10
        return f
    return n / (math.sqrt(2.0 * math.pi) * v_th) * np.exp(-0.5 * (v / v_th)**2)


def compute_moments(
    f_dist: np.ndarray,
    v_nodes: np.ndarray,
    v_weights: np.ndarray,
) -> Tuple[float, float, float]:
    """
    计算等离子体矩量

    给定分布函数 f(v) 在求积节点上的值：
        n = ∫ f(v) dv ≈ Σ w_i f_i
        n*u = ∫ v f(v) dv ≈ Σ w_i v_i f_i
        (3/2) n k_B T = ∫ (1/2) m (v-u)² f(v) dv

    参数：
        f_dist: shape (n_quad,) 分布函数值
        v_nodes: shape (n_quad,) 速度节点
        v_weights: shape (n_quad,) 求积权重

    返回：
        n: 密度
        u: 平均速度
        T_eff: 有效温度 (能量单位 eV)
    """
    n = np.sum(v_weights * f_dist)
    if abs(n) < 1e-30:
        return 0.0, 0.0, 0.0

    nu = np.sum(v_weights * v_nodes * f_dist)
    u = nu / n

    # 温度 (动能矩)
    from sheath_constants import E_CHARGE
    v_shifted = v_nodes - u
    KE = 0.5 * np.sum(v_weights * v_shifted**2 * f_dist)
    T_eff = KE / n  # 简化单位

    return float(n), float(u), float(T_eff)


def triangle_quadrature(
    order: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    三角形单元 Gauss 求积规则
    （源自种子项目 1305_triangle_grid: 三角形网格求积点）

    在参考三角形 T = {(ξ,η): 0≤ξ≤1, 0≤η≤1-ξ} 上：
        ∫∫_T f(ξ,η) dξ dη ≈ Σ w_k f(ξ_k, η_k)

    使用 Duffy 变换 + Gauss-Legendre:
        ξ = s
        η = t * (1 - s)
        dξ dη = (1-s) ds dt

    参数：
        order: 求积精度阶数

    返回：
        xi: shape (n_pts,) ξ 坐标
        eta: shape (n_pts,) η 坐标
        weights: shape (n_pts,) 求积权重
    """
    n_gl = max(2, (order + 1) // 2)
    gl_nodes, gl_weights = roots_legendre(n_gl)
    # 映射到 [0, 1]
    gl_nodes = 0.5 * (gl_nodes + 1.0)
    gl_weights = 0.5 * gl_weights

    pts_xi = []
    pts_eta = []
    pts_w = []

    for i in range(n_gl):
        s = gl_nodes[i]
        ws = gl_weights[i]
        for j in range(n_gl):
            t = gl_nodes[j]
            wt = gl_weights[j]

            xi = s
            eta = t * (1.0 - s)
            w = ws * wt * (1.0 - s)  # Duffy 变换雅可比

            pts_xi.append(xi)
            pts_eta.append(eta)
            pts_w.append(w)

    return np.array(pts_xi), np.array(pts_eta), np.array(pts_w)


def hypercube_monomial_integral(dimension: int, exponents: np.ndarray) -> float:
    """
    单位超立方体上的单项式积分
    （源自种子项目 559_hypercube_integrals）

    ∫_{[0,1]^d} Π x_i^{e_i} dx = Π (1 / (e_i + 1))

    在鞘层问题中的应用：
        多维速度空间的矩量积分

    参数：
        dimension: 空间维度 d
        exponents: shape (d,) 各维度的幂次

    返回：
        integral: 积分值
    """
    if len(exponents) != dimension:
        raise ValueError("exponents 长度必须等于 dimension")

    result = 1.0
    for e in exponents:
        if e < 0:
            raise ValueError("幂次必须非负")
        result /= (e + 1.0)
    return result


def hypercube_sample(dim: int, n_sample: int, seed: int = 42) -> np.ndarray:
    """
    单位超立方体均匀采样
    （源自种子项目 559_hypercube_integrals）

    参数：
        dim: 维度
        n_sample: 采样点数
        seed: 随机种子

    返回：
        samples: shape (n_sample, dim)
    """
    rng = np.random.RandomState(seed)
    return rng.rand(n_sample, dim)


def velocity_space_integral(
    integrand: Callable[[np.ndarray], np.ndarray],
    v_th: float,
    n_quad: int = 32,
    domain_factor: float = 6.0,
) -> float:
    """
    速度空间积分（使用 Gauss-Hermite）

    ∫ f(v) dv = v_th ∫ f(v_th * u) du
              ≈ v_th * Σ w_i f(v_th * u_i) * exp(u_i²)

    其中 u_i, w_i 为 Gauss-Hermite 节点/权重

    参数：
        integrand: 被积函数 f(v)
        v_th: 热速度
        n_quad: 求积点数
        domain_factor: 积分域半径（以 v_th 为单位）

    返回：
        integral: 积分值
    """
    nodes, weights = gauss_hermite_nodes_weights(n_quad)

    # 变换到物理速度空间
    v = v_th * nodes

    # 计算被积函数
    f_vals = integrand(v)

    # Gauss-Hermite 求积：
    # ∫ f(v) dv = v_th * ∫ f(v_th * u) du
    # ≈ v_th * Σ w_i * f(v_th * u_i) * exp(u_i²) / sqrt(π) * sqrt(π)
    # 因为 GH 权函数是 exp(-u²)
    integral = v_th * np.sum(weights * f_vals * np.exp(nodes**2))

    return float(integral)


def plasma_dispersion_function(z: complex, n_quad: int = 64) -> complex:
    """
    等离子体色散函数 (Fried-Conte 函数)
        Z(ζ) = (1/√π) ∫_{-∞}^{∞} exp(-t²) / (t - ζ) dt

    对于 Im(ζ) > 0 可直接数值积分。
    对于 Im(ζ) ≤ 0 需要加上 Landau 极点贡献：
        Z(ζ) = Z_principal(ζ) + i√π exp(-ζ²)

    参数：
        z: 复数参数 ζ
        n_quad: 求积点数

    返回：
        Z: 等离子体色散函数值
    """
    nodes, weights = gauss_hermite_nodes_weights(n_quad)

    # 主值积分
    denom = nodes - z
    # 避免除零
    denom[np.abs(denom) < 1e-15] = 1e-15

    Z_principal = (1.0 / math.sqrt(math.pi)) * np.sum(weights / denom)

    # Landau 阻尼项 (当 Im(z) ≤ 0)
    if z.imag <= 0:
        Z_landau = 1j * math.sqrt(math.pi) * np.exp(-z**2)
    else:
        Z_landau = 0.0 + 0.0j

    return complex(Z_principal + Z_landau)


def bohman_velocity_integral(
    T_e: float,
    T_i: float,
    m_i: float,
    n_quad: int = 32,
) -> Tuple[float, float]:
    """
    验证 Bohm 判据的速度空间积分

    离子通量：Γ_i = n_i u_i = ∫ v f_i(v) dv
    电子通量：Γ_e = n_e u_e = ∫ v f_e(v) dv

    在鞘层边缘 (s)：
        Γ_i(s) ≥ n_s c_s

    参数：
        T_e: 电子温度 [eV]
        T_i: 离子温度 [eV]
        m_i: 离子质量 [kg]
        n_quad: 求积点数

    返回：
        c_s: Bohm 速度
        Gamma_ratio: Γ_i / (n_s c_s)
    """
    from sheath_constants import E_CHARGE

    c_s = math.sqrt(T_e * E_CHARGE / m_i)
    v_th_i = math.sqrt(T_i * E_CHARGE / m_i)

    # 离子漂移速度（设为 c_s）
    u_i = c_s

    # 偏移 Maxwell 分布的动量矩
    def ion_flux_integrand(v):
        v_th = v_th_i
        if v_th < 1e-30:
            return v * (1.0 if abs(v - u_i) < 1e-10 else 0.0)
        f = 1.0 / (math.sqrt(2 * math.pi) * v_th) * np.exp(-0.5 * ((v - u_i) / v_th)**2)
        return v * f

    gamma_i = velocity_space_integral(ion_flux_integrand, max(v_th_i, 1e-6), n_quad)

    # 比值
    ratio = gamma_i / c_s if c_s > 1e-30 else 0.0

    return c_s, ratio
