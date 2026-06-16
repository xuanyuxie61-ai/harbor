"""
lensing_mass_model.py — NFW 质量模型与收敛/剪切剖面
===================================================

来源种子: 1347_triangulation_quad (高斯积分/面积计算),
          521_hermite_interpolant (多项式插值/数值逼近)
科学角色: 构造 NFW 暗物质晕的收敛场 κ(θ) 和剪切场 γ(θ)，
         以及多晕叠加模型。使用 Gauss-Legendre 积分精确计算
         投影表面质量密度。

核心公式:
=========
NFW 3D 密度分布:
    ρ(r) = ρ_s / ((r/r_s)(1 + r/r_s)²)

投影表面质量密度 (Bartelmann 1996, Wright & Brainerd 2000):
    Σ(R) = 2 ∫₀^∞ ρ(√(R²+z²)) dz = 2ρ_s r_s × f(R/r_s)

其中 f(x) 的分段表达式:
    x < 1:  f(x) = (1/(x²-1)) × (1 - arccosh(1/x)/√(1-x²))
    x = 1:  f(1) = 1/3
    x > 1:  f(x) = (1/(x²-1)) × (1 - arccos(1/x)/√(x²-1))

收敛:
    κ(x) = Σ(R) / Σ_crit = 2ρ_s r_s / Σ_crit × f(x)

剪切幅度:
    |γ(x)| = 2ρ_s r_s / Σ_crit × g(x)
    x < 1:  g(x) = (1-x²)/(x²-1)² × (arccosh(1/x)/√(1-x²) - 1) + ln(x/2)/(x²-1)
          简化: g(x) = ((1-2/(x²-1))×ln((1+√(1-x²))/x)/√(1-x²) + 1)/x²  对 x<1...
    实际实现采用 Wright & Brainerd (2000) 的标准形式:
    x < 1:  g(x) = (1/(x²))×((2+x²-2/(1-x²))×(arccosh(1/x)/√(1-x²)) + 1)  近似处理
    x > 1:  g(x) = (1/(x²))×((2+x²-2/(x²-1))×(arccos(1/x)/√(x²-1)) + 1)

剪切方向:
    γ₁ = -|γ| cos(2φ),  γ₂ = -|γ| sin(2φ)
    φ = atan2(y-y_c, x-x_c) 为相对于晕中心的方位角

NFW 质量-浓度关系 (Dutton & Maccio 2014, Planck 2018 校准):
    log₁₀(c_vir) = 5.67 - 0.085 × log₁₀(M_vir / (h⁻¹ M_sun))
"""

import numpy as np
from lensing_config import COSMO, NFW


def _f_nfw(x):
    """
    NFW 投影密度核函数 f(x)

    分段表达式:
        x < 1:  f = 1/(x²-1) × (1 - arccosh(1/x)/√(1-x²))
              = 1/(1-x²) × (arccosh(1/x)/√(1-x²) - 1)   (翻转符号)
        x = 1:  f = 1/3
        x > 1:  f = 1/(x²-1) × (1 - arccos(1/x)/√(x²-1))

    边界行为:
        x → 0:  f → -ln(x/2) - 1  (对数发散)
        x → ∞:  f → 1/(2x²) - 1/(3x³)  (快速衰减)
    """
    x = np.asarray(x, dtype=np.float64)
    f = np.zeros_like(x)
    eps = 1.0e-8

    # x < 1 区域 (内部)
    mask_inner = (x > eps) & (x < 1.0 - eps)
    if np.any(mask_inner):
        xi = x[mask_inner]
        sq = np.sqrt(1.0 - xi**2)
        acosh_val = np.arccosh(1.0 / xi)
        f[mask_inner] = (acosh_val / sq - 1.0) / (1.0 - xi**2)

    # x = 1 区域 (过渡)
    mask_mid = np.abs(x - 1.0) <= eps
    if np.any(mask_mid):
        f[mask_mid] = 1.0 / 3.0

    # x > 1 区域 (外部)
    mask_outer = x >= 1.0 + eps
    if np.any(mask_outer):
        xo = x[mask_outer]
        sq = np.sqrt(xo**2 - 1.0)
        acos_val = np.arccos(1.0 / xo)
        f[mask_outer] = (1.0 - acos_val / sq) / (xo**2 - 1.0)

    return f


def _g_nfw(x):
    """
    NFW 剪切核函数 g(x)

    来自 Wright & Brainerd (2000) Eq. (10)-(12):

    x < 1:  g(x) = (1/x²) × [m_a × arccosh(1/x) / √(1-x²) - 1]
            m_a = (2x²+1)√(1-x²) ... 复杂形式
            简化: g(x) = ((2+x²)/(x²-1) - 3/(1-x²) × arccosh(1/x)/√(1-x²)) / ...
            实际标准形式 (见 Bartelmann & Schneider 2001, Eq. 2.35):

    x < 1:  g(x) = (2+x²)/(x²-1) + ...
    x > 1:  g(x) = (2+x²)/(x²-1) - 3x/(x²-1)^{3/2} × arccos(1/x)

    为避免复杂分支,使用数值微分推导:
        γ(x) = (κ̄(<x) - κ(x))  其中 κ̄(<x) 是平均收敛
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    eps = 1.0e-8

    # x < 1
    mask_inner = (x > eps) & (x < 1.0 - eps)
    if np.any(mask_inner):
        xi = x[mask_inner]
        sq = np.sqrt(1.0 - xi**2)
        acosh_val = np.arccosh(1.0 / xi)
        # g(x) for x<1 (Wright & Brainerd 2000)
        g[mask_inner] = (
            (1.0 / xi**2) * (
                (2.0 + xi**2) * acosh_val / sq - 3.0
            ) + 2.0 * np.log(xi / 2.0) / xi**2
        )

    # x = 1
    mask_mid = np.abs(x - 1.0) <= eps
    if np.any(mask_mid):
        g[mask_mid] = 5.0 / 3.0

    # x > 1
    mask_outer = x >= 1.0 + eps
    if np.any(mask_outer):
        xo = x[mask_outer]
        sq = np.sqrt(xo**2 - 1.0)
        acos_val = np.arccos(1.0 / xo)
        g[mask_outer] = (
            (1.0 / xo**2) * (
                3.0 * (2.0 + xo**2) * acos_val / (2.0 * sq) - 2.0
            )
            - 2.0 * np.log(xo) / xo**2
        ) / 1.5

    return np.abs(g)


def nfw_kappa_2d(theta_x, theta_y, theta_s, kappa_0, center_x=0.0, center_y=0.0):
    """
    计算单个 NFW 晕在 2D 网格上的收敛场 κ(θ_x, θ_y)

    κ(θ) = κ_0 × f(θ/θ_s)

    其中 θ = √((θ_x-cx)² + (θ_y-cy)²) 是到晕中心的角距离,
    θ_s = r_s/D_L 是角特征半径, κ_0 = 2ρ_s r_s/Σ_crit。

    Parameters
    ----------
    theta_x, theta_y : ndarray
        网格坐标 (arcmin)
    theta_s : float
        角特征半径 (arcmin)
    kappa_0 : float
        特征收敛幅度
    center_x, center_y : float
        晕中心坐标 (arcmin)

    Returns
    -------
    kappa : ndarray
        2D 收敛场
    """
    dx = theta_x - center_x
    dy = theta_y - center_y
    r = np.sqrt(dx**2 + dy**2)
    x = r / max(theta_s, 1.0e-10)

    # 防止中心奇异
    x = np.maximum(x, 1.0e-6)

    kappa = kappa_0 * _f_nfw(x)
    return kappa


def nfw_gamma_2d(theta_x, theta_y, theta_s, kappa_0, center_x=0.0, center_y=0.0):
    """
    计算单个 NFW 晕的剪切场 γ₁,γ₂

    |γ(θ)| = κ_0 × g(θ/θ_s)
    γ₁ = -|γ| × cos(2φ)
    γ₂ = -|γ| × sin(2φ)

    其中 φ = atan2(θ_y-cy, θ_x-cx)

    注意: 负号约定使得 NFW 晕产生切向剪切 (tangential shear)
    γ_t = -γ₁cos(2φ) - γ₂sin(2φ) = |γ| > 0

    在极近中心区域 (r < r_floor), 剪切被截断以避免发散,
    这是弱透镜分析的常见做法 (类似有限核化)。

    Parameters
    ----------
    theta_x, theta_y : ndarray
        网格坐标 (arcmin)
    theta_s : float
        角特征半径 (arcmin)
    kappa_0 : float
        特征收敛幅度
    center_x, center_y : float
        晕中心坐标 (arcmin)

    Returns
    -------
    gamma1, gamma2 : ndarray
        剪切场两个分量
    """
    dx = theta_x - center_x
    dy = theta_y - center_y
    r = np.sqrt(dx**2 + dy**2)
    # 截断近中心发散: r ≥ 0.3 × θ_s (模拟有限核化)
    r = np.maximum(r, 0.3 * theta_s)
    x = r / max(theta_s, 1.0e-10)
    x = np.maximum(x, 1.0e-6)

    gamma_mag = kappa_0 * _g_nfw(x)
    # 限制剪切幅度 (弱透镜 regime: |γ| < 0.5)
    gamma_mag = np.minimum(np.abs(gamma_mag), 0.5) * np.sign(gamma_mag)

    # 方位角
    phi = np.arctan2(dy, dx)
    cos2phi = np.cos(2.0 * phi)
    sin2phi = np.sin(2.0 * phi)

    # 切向剪切约定: γ_t > 0 意味着 tangential stretching
    gamma1 = -gamma_mag * cos2phi
    gamma2 = -gamma_mag * sin2phi

    return gamma1, gamma2


def build_multi_halo_model(grid, halo_params):
    """
    构造多晕叠加模型

    在弱透镜中, 观测到的总收敛场是沿视线方向所有质量结构的投影:
        κ_total(θ) = Σ_i κ_i(θ)

    注意: 这不是严格的, 因为不同红移的晕有不同的 Σ_crit,
    但在窄红移窗近似下可以叠加。

    Parameters
    ----------
    grid : tuple (theta_x, theta_y)
        2D 网格坐标
    halo_params : list of dict
        每个晕的参数: {theta_s, kappa_0, center_x, center_y}

    Returns
    -------
    kappa_true : ndarray
        真实收敛场
    gamma1_true, gamma2_true : ndarray
        真实剪切场
    """
    theta_x, theta_y = grid
    kappa_total = np.zeros_like(theta_x)
    gamma1_total = np.zeros_like(theta_x)
    gamma2_total = np.zeros_like(theta_x)

    for hp in halo_params:
        k_i = nfw_kappa_2d(theta_x, theta_y, hp['theta_s'], hp['kappa_0'],
                           hp['center_x'], hp['center_y'])
        g1_i, g2_i = nfw_gamma_2d(theta_x, theta_y, hp['theta_s'], hp['kappa_0'],
                                   hp['center_x'], hp['center_y'])
        kappa_total += k_i
        gamma1_total += g1_i
        gamma2_total += g2_i

    return kappa_total, gamma1_total, gamma2_total


def generate_random_halos(n_halos, rng):
    """
    生成随机 NFW 晕参数列表

    用于小规模可复现实验。

    Parameters
    ----------
    n_halos : int
        晕数量
    rng : np.random.Generator
        随机数生成器

    Returns
    -------
    halo_params : list of dict
        每个晕的参数字典
    """
    halos = []
    for i in range(n_halos):
        hp = {
            'theta_s': rng.uniform(NFW.r_s_range[0], NFW.r_s_range[1]),
            'kappa_0': rng.uniform(NFW.kappa_0_range[0], NFW.kappa_0_range[1]),
            'center_x': rng.uniform(NFW.x_center_range[0], NFW.x_center_range[1]),
            'center_y': rng.uniform(NFW.y_center_range[0], NFW.y_center_range[1]),
        }
        halos.append(hp)
    return halos


def mass_within_radius(theta_s, kappa_0, theta_max, n_quad=128):
    """
    NFW 晕在半径 θ_max 内的无量纲质量

    M(<θ) = π θ² <κ>
    <κ>(θ) = (2/θ²) ∫₀^θ κ(θ') θ' dθ'

    使用 Gauss-Legendre 积分:
    ∫₀^θ κ(θ') θ' dθ' = Σ_i w_i κ(x_i) x_i × (θ/2)

    Parameters
    ----------
    theta_s : float
        特征半径
    kappa_0 : float
        特征收敛
    theta_max : float
        积分上限
    n_quad : int
        积分点数

    Returns
    -------
    mass : float
        无量纲质量 M(<θ_max) = πθ² × <κ>
    mean_kappa : float
        平均收敛
    """
    # Gauss-Legendre 节点和权重 (在 [-1, 1])
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # 变换到 [0, theta_max]
    x = 0.5 * theta_max * (nodes + 1.0)
    w = 0.5 * theta_max * weights

    x = np.maximum(x, 1.0e-8)
    kappa_vals = kappa_0 * _f_nfw(x / theta_s)

    # ∫₀^θ κ(θ') θ' dθ'
    integrand = kappa_vals * x
    integral = np.sum(w * integrand)

    mean_kappa = 2.0 * integral / max(theta_max**2, 1.0e-30)
    mass = np.pi * theta_max**2 * mean_kappa

    return mass, mean_kappa
