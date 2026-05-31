"""
sphere_quadrature.py
球面积分与角度离散化模块。

融合原始项目：1116_sphere_exactness（球面积分规则精确性测试）

在天体物理辐射传输中，球面方向积分是核心计算：
    ∫_{4π} I(Ω) dΩ
需要在单位球面上构造高效的数值积分规则。
"""

import numpy as np
from typing import Tuple, List


def spherical_to_cartesian(theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    球面坐标 (θ, φ) 转笛卡尔坐标 (x, y, z)。

    公式:
        x = sin(θ) cos(φ)
        y = sin(θ) sin(φ)
        z = cos(θ)

    其中 θ ∈ [0, π] 是极角（与z轴夹角），φ ∈ [0, 2π) 是方位角。
    """
    st = np.sin(theta)
    x = st * np.cos(phi)
    y = st * np.sin(phi)
    z = np.cos(theta)
    return x, y, z


def cartesian_to_spherical(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    笛卡尔坐标转球面坐标。

    边界处理: 当 r ≈ 0 时，任意返回 θ=0, φ=0。
    当 x=y=0 时，φ 取 0。
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.zeros_like(r)
    phi = np.zeros_like(r)
    mask = r > 1e-15
    theta[mask] = np.arccos(np.clip(z[mask] / r[mask], -1.0, 1.0))
    phi[mask] = np.arctan2(y[mask], x[mask])
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return theta, phi


def gauss_legendre_angles(n_polar: int, n_azimuth: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    构造 Gauss-Legendre × 均匀方位角 的球面角度积分网格。

    极向使用 Gauss-Legendre 节点 {μ_k} 和权重 {w_k}，其中 μ = cos(θ)。
    由于 dΩ = sin(θ) dθ dφ = -dμ dφ，积分公式为:
        ∫_{4π} f(Ω) dΩ = ∫_0^{2π} ∫_{-1}^{1} f(μ, φ) dμ dφ
                       ≈ Σ_{k=1}^{n_polar} Σ_{l=1}^{n_azimuth} w_k (2π/n_azimuth) f(μ_k, φ_l)

    参数:
        n_polar: 极向 Gauss-Legendre 节点数
        n_azimuth: 方位角均匀分割数

    返回:
        mu: 极向 cos(θ) 节点，形状 (n_polar,)
        w_mu: 极向权重，形状 (n_polar,)
        phi: 方位角节点，形状 (n_azimuth,)
        w_phi: 方位角权重（常数 2π/n_azimuth）
    """
    if n_polar <= 0 or n_azimuth <= 0:
        raise ValueError("节点数必须为正")

    mu, w_mu = np.polynomial.legendre.leggauss(n_polar)
    phi = np.linspace(0, 2 * np.pi, n_azimuth, endpoint=False)
    w_phi = np.full(n_azimuth, 2 * np.pi / n_azimuth)
    return mu, w_mu, phi, w_phi


def integrate_sphere_function(f_values: np.ndarray, w_mu: np.ndarray, w_phi: np.ndarray) -> float:
    """
    在球面上数值积分标量函数。

    公式:
        I = Σ_{i,j} f(μ_i, φ_j) * w_mu[i] * w_phi[j]

    参数:
        f_values: 函数值数组，形状 (n_polar, n_azimuth)
        w_mu: 极向权重
        w_phi: 方位角权重

    返回:
        积分值
    """
    if f_values.shape[0] != w_mu.shape[0] or f_values.shape[1] != w_phi.shape[0]:
        raise ValueError("函数值维度与权重不匹配")
    return float(np.dot(w_mu, np.dot(f_values, w_phi)))


def henyey_greenstein_phase_function(cos_scatter: np.ndarray, g: float) -> np.ndarray:
    """
    Henyey-Greenstein 相函数，描述大气散射的角分布。

    公式:
        P_HG(cos Θ) = (1 / 4π) * (1 - g^2) / (1 + g^2 - 2g cos Θ)^{3/2}

    其中 Θ 是散射角，g ∈ (-1, 1) 是非对称参数:
        g > 0: 前向散射主导
        g < 0: 后向散射主导
        g = 0: 各向同性散射

    归一化条件:
        ∫_{4π} P_HG(cos Θ) dΩ = 1

    参数:
        cos_scatter: cos(Θ) 值
        g: 非对称参数

    返回:
        相函数值
    """
    cos_scatter = np.asarray(cos_scatter, dtype=np.float64)
    if abs(g) >= 1.0:
        raise ValueError(f"Henyey-Greenstein参数 g 必须在 (-1, 1) 内，得到 g={g}")

    denom = 1.0 + g**2 - 2.0 * g * cos_scatter
    denom = np.maximum(denom, 1e-15)
    p = (1.0 - g**2) / (4.0 * np.pi * denom**1.5)
    return p


def compute_scatter_angles(mu_in: np.ndarray, phi_in: np.ndarray,
                           mu_out: np.ndarray, phi_out: np.ndarray) -> np.ndarray:
    """
    计算两个方向之间的散射角余弦。

    设入射方向 Ω' = (sin θ' cos φ', sin θ' sin φ', cos θ')
    散射方向 Ω = (sin θ cos φ, sin θ sin φ, cos θ)

    散射角余弦:
        cos Θ = Ω' · Ω
              = sin θ' sin θ cos(φ' - φ) + cos θ' cos θ
              = √(1-μ'^2) √(1-μ^2) cos(φ'-φ) + μ' μ

    参数:
        mu_in: 入射方向 cos(θ')
        phi_in: 入射方向 φ'
        mu_out: 散射方向 cos(θ)
        phi_out: 散射方向 φ

    返回:
        cos(Θ) 值
    """
    dphi = phi_in - phi_out
    term1 = np.sqrt(np.maximum(1.0 - mu_in**2, 0.0)) * np.sqrt(np.maximum(1.0 - mu_out**2, 0.0)) * np.cos(dphi)
    term2 = mu_in * mu_out
    return np.clip(term1 + term2, -1.0, 1.0)


def delta_eddington_approximation(tau: np.ndarray, omega: np.ndarray, g: float,
                                  mu0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Delta-Eddington 近似求解辐射传输的反射率和透射率。

    这是求解平面平行大气辐射传输的经典近似方法。

    参数变换:
        τ* = (1 - ω f) τ
        ω* = (1 - f) ω / (1 - ω f)
        g* = (g - f) / (1 - f)
    其中 f = g^3 是 Delta 函数截断参数。

    然后使用 Eddington 近似:
        反射率 R = [ (r_∞ - r_0) (r_∞ + r_0) (1 - e^{-2k τ*}) ] /
                   [ (r_∞ + r_0)^2 - (r_∞ - r_0)^2 e^{-2k τ*} ]

    其中:
        k = √[3 (1 - ω*) (1 - ω* g*)]
        r_∞ = (1 + 2k/3) / (1 - 2k/3)   (渐近反射率)
        r_0   = (1 + 2γ_3) / (1 - 2γ_3)
        γ_3 = √(1 - ω*) / 3

    参数:
        tau: 光学厚度数组
        omega: 单次散射反照率数组
        g: 原始终端不对称参数
        mu0: 入射天顶角余弦

    返回:
        R: 反射率数组
        T: 透射率数组
    """
    tau = np.asarray(tau, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)

    if np.any(tau < 0):
        raise ValueError("光学厚度不能为负")
    if np.any((omega < 0) | (omega > 1)):
        raise ValueError("单次散射反照率必须在 [0, 1] 内")
    if abs(g) >= 1.0:
        raise ValueError("不对称参数 g 必须在 (-1, 1) 内")
    if mu0 <= 0 or mu0 > 1:
        raise ValueError("入射角余弦 mu0 必须在 (0, 1] 内")

    f = g**3
    tau_star = tau * (1.0 - omega * f)
    omega_star = np.where(1.0 - omega * f > 1e-15,
                          (1.0 - f) * omega / (1.0 - omega * f),
                          0.0)
    g_star = np.where(1.0 - f > 1e-15, (g - f) / (1.0 - f), 0.0)

    k = np.sqrt(3.0 * (1.0 - omega_star) * (1.0 - omega_star * g_star))
    k = np.maximum(k, 1e-15)

    gamma3 = np.sqrt(1.0 - omega_star) / 3.0

    r_inf = (1.0 + 2.0 * k / 3.0) / np.maximum(1.0 - 2.0 * k / 3.0, 1e-15)
    r_0 = (1.0 + 2.0 * gamma3) / np.maximum(1.0 - 2.0 * gamma3, 1e-15)

    exp_term = np.exp(-2.0 * k * tau_star)
    denom = (r_inf + r_0)**2 - (r_inf - r_0)**2 * exp_term
    denom = np.maximum(denom, 1e-15)

    R = (r_inf - r_0) * (r_inf + r_0) * (1.0 - exp_term) / denom

    direct_trans = np.exp(-tau_star / mu0)
    T = direct_trans + (1.0 - R) * (1.0 - direct_trans)

    R = np.clip(R, 0.0, 1.0)
    T = np.clip(T, 0.0, 1.0)
    return R, T
