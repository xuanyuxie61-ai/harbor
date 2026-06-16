"""
lensing_shear_field.py — 剪切场生成与噪声注入
=============================================

来源种子: 128_bvec (位变换/状态编码),
          1109_marekgluza_Fidelity_witnesses_example (协方差矩阵演化)
科学角色: 从真实收敛场生成可观测剪切场 (含噪声),
         并计算源星系的椭率分布。

物理公式:
=========
1. 约化剪切 (Reduced shear):
   g = γ / (1 - κ)
   在弱透镜近似 (κ << 1) 下: g ≈ γ

2. 观测椭率:
   ε_obs = ε_intrinsic + γ/(1-κ) + noise
   其中 ε_intrinsic 是内禀椭率 (随机取向, σ_ε ≈ 0.3)

3. 形状噪声:
   σ_shape = σ_ε / √(n_gal) per pixel
   其中 n_gal 是每像素的平均星系数

4. B-mode 检验:
   真实引力透镜只产生 E-mode 剪切:
   γ_E = γ₁cos(2φ) + γ₂sin(2φ)  (非零)
   γ_B = -γ₁sin(2φ) + γ₂cos(2φ) (应为零)
   B-mode 的存在指示系统误差或非引力透镜效应

5. 剪切协方差:
   C_{αβ}(θ) = <γ_α(θ')γ_β(θ'+θ)>
   对于高斯随机场: C = P_κ(l) δ_{αβ} / (2π)
"""

import numpy as np
from lensing_config import GRID, COSMO
from lensing_finite_diff import gradient_4th, laplacian_4th


def compute_reduced_shear(gamma1, gamma2, kappa):
    """
    计算约化剪切 g = γ/(1-κ)

    在强透镜区域 (κ ≥ 1), 约化剪切发散,
    需要截断处理:
        g = γ / max(1-κ, κ_floor)

    Parameters
    ----------
    gamma1, gamma2 : ndarray
        剪切分量
    kappa : ndarray
        收敛场

    Returns
    -------
    g1, g2 : ndarray
        约化剪切分量
    """
    kappa_floor = 1.0e-3
    one_minus_kappa = np.maximum(1.0 - kappa, kappa_floor)
    g1 = gamma1 / one_minus_kappa
    g2 = gamma2 / one_minus_kappa
    return g1, g2


def generate_intrinsic_ellipticities(n_gal, rng, sigma_eps=0.27):
    """
    生成源星系内禀椭率

    内禀椭率分布通常建模为二维高斯:
        p(ε₁, ε₂) = (1/(2πσ²)) exp(-(ε₁²+ε₂²)/(2σ²))
    其中 σ ≈ 0.27 (观测值)

    等价地, |ε| 服从 Rayleigh 分布:
        p(|ε|) = (|ε|/σ²) exp(-|ε|²/(2σ²))

    Parameters
    ----------
    n_gal : int
        源星系数量
    rng : np.random.Generator
        随机数生成器
    sigma_eps : float
        椭率弥散 (per component)

    Returns
    -------
    eps1, eps2 : ndarray
        椭率分量
    """
    eps1 = rng.normal(0.0, sigma_eps, n_gal)
    eps2 = rng.normal(0.0, sigma_eps, n_gal)
    return eps1, eps2


def generate_shear_catalog(gamma1_true, gamma2_true, kappa_true, rng):
    """
    生成弱透镜剪切目录 (模拟观测)

    步骤:
    1. 在网格上放置 n_sources 个源星系 (均匀分布)
    2. 在每个源位置插值真实剪切 (γ₁,γ₂)
    3. 添加内禀椭率
    4. 计算观测椭率: ε_obs = ε_int + g

    观测椭率是收敛和剪切无偏估计量:
        <ε_obs> = <ε_int + g> = g (因为 <ε_int> = 0)

    Parameters
    ----------
    gamma1_true, gamma2_true : ndarray (Ny, Nx)
        真实剪切场
    kappa_true : ndarray (Ny, Nx)
        真实收敛场
    rng : np.random.Generator
        随机数生成器

    Returns
    -------
    catalog : dict
        包含以下字段:
        - 'x': 源 x 坐标 (arcmin)
        - 'y': 源 y 坐标 (arcmin)
        - 'eps1_obs': 观测椭率分量1
        - 'eps2_obs': 观测椭率分量2
        - 'eps1_int': 内禀椭率分量1
        - 'eps2_int': 内禀椭率分量2
        - 'g1_true': 真实约化剪切1
        - 'g2_true': 真实约化剪切2
    """
    n_src = GRID.n_sources
    N = GRID.N_grid
    field = GRID.field_size_arcmin

    # 源位置 (在网格上均匀分布 + 微扰)
    x_src = rng.uniform(-field/2, field/2, n_src)
    y_src = rng.uniform(-field/2, field/2, n_src)

    # 网格坐标
    x_grid = np.linspace(-field/2, field/2, N)
    y_grid = np.linspace(-field/2, field/2, N)

    # 约化剪切
    g1_true, g2_true = compute_reduced_shear(gamma1_true, gamma2_true, kappa_true)

    # 双线性插值真实剪切到源位置
    g1_interp = _bilinear_interp(g1_true, x_grid, y_grid, x_src, y_src)
    g2_interp = _bilinear_interp(g2_true, x_grid, y_grid, x_src, y_src)

    # 内禀椭率
    eps1_int, eps2_int = generate_intrinsic_ellipticities(n_src, rng)

    # 观测椭率 = 内禀 + 约化剪切
    eps1_obs = eps1_int + g1_interp
    eps2_obs = eps2_int + g2_interp

    catalog = {
        'x': x_src,
        'y': y_src,
        'eps1_obs': eps1_obs,
        'eps2_obs': eps2_obs,
        'eps1_int': eps1_int,
        'eps2_int': eps2_int,
        'g1_true': g1_interp,
        'g2_true': g2_interp,
    }
    return catalog


def catalog_to_grid_shear(catalog, N=None, field_size=None):
    """
    将剪切目录网格化 (binning)

    在每个像素内, 估计平均剪切:
        γ̂_α = (1/n_pix) Σ_i ε_{α,i}
    方差:
        σ²(γ̂_α) = σ²_ε / n_pix

    Parameters
    ----------
    catalog : dict
        剪切目录
    N : int
        网格分辨率
    field_size : float
        天区大小 (arcmin)

    Returns
    -------
    gamma1_grid, gamma2_grid : ndarray (N, N)
        网格化剪切场
    weight_grid : ndarray (N, N)
        权重 (每像素星系数)
    """
    if N is None:
        N = GRID.N_grid
    if field_size is None:
        field_size = GRID.field_size_arcmin

    x = catalog['x']
    y = catalog['y']
    e1 = catalog['eps1_obs']
    e2 = catalog['eps2_obs']

    # 像素索引
    ix = np.clip(((x + field_size/2) / field_size * N).astype(int), 0, N-1)
    iy = np.clip(((y + field_size/2) / field_size * N).astype(int), 0, N-1)

    # 累加
    gamma1_grid = np.zeros((N, N))
    gamma2_grid = np.zeros((N, N))
    weight_grid = np.zeros((N, N))

    np.add.at(gamma1_grid, (iy, ix), e1)
    np.add.at(gamma2_grid, (iy, ix), e2)
    np.add.at(weight_grid, (iy, ix), 1.0)

    # 平均
    mask = weight_grid > 0
    gamma1_grid[mask] /= weight_grid[mask]
    gamma2_grid[mask] /= weight_grid[mask]

    return gamma1_grid, gamma2_grid, weight_grid


def add_shape_noise_to_grid(gamma1_grid, gamma2_grid, weight_grid, rng):
    """
    向网格化剪切场添加形状噪声

    形状噪声方差:
        σ²_noise = σ²_ε / n_eff

    其中 n_eff = <n_gal per pixel> 是有效星系密度。

    Parameters
    ----------
    gamma1_grid, gamma2_grid : ndarray
        无噪声剪切场
    weight_grid : ndarray
        权重图
    rng : np.random.Generator
        随机数生成器

    Returns
    -------
    gamma1_noisy, gamma2_noisy : ndarray
        含噪声剪切场
    """
    sigma_eps = 0.27
    noise1 = np.zeros_like(gamma1_grid)
    noise2 = np.zeros_like(gamma2_grid)

    mask = weight_grid > 0
    sigma_per_pix = sigma_eps / np.sqrt(np.maximum(weight_grid, 1.0))
    noise1[mask] = rng.normal(0.0, 1.0, np.sum(mask)) * sigma_per_pix[mask]
    noise2[mask] = rng.normal(0.0, 1.0, np.sum(mask)) * sigma_per_pix[mask]

    return gamma1_grid + noise1, gamma2_grid + noise2


def compute_EB_decomposition(gamma1, gamma2, h):
    """
    E-mode / B-mode 分解

    在傅里叶空间中:
        γ̃(l) = γ̃₁(l) + iγ̃₂(l)
        γ̃(l) = [cos(2φ_l) + i sin(2φ_l)] × [Ẽ(l) + iB̃(l)]

    因此:
        Ẽ(l) = cos(2φ_l) γ̃₁ + sin(2φ_l) γ̃₂
        B̃(l) = -sin(2φ_l) γ̃₁ + cos(2φ_l) γ̃₂

    对于纯引力透镜剪切, B-mode 应为零。
    非零 B-mode 指示系统误差。

    Parameters
    ----------
    gamma1, gamma2 : ndarray
        剪切场
    h : float
        网格间距

    Returns
    -------
    E_power : float
        E-mode 功率
    B_power : float
        B-mode 功率
    EB_ratio : float
        B/E 比率 (应接近 0)
    """
    Ny, Nx = gamma1.shape
    ky = np.fft.fftfreq(Ny, d=h) * 2.0 * np.pi
    kx = np.fft.fftfreq(Nx, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    k_mag = np.sqrt(KX**2 + KY**2)
    phi_l = np.arctan2(KY, KX)

    cos2phi = np.cos(2.0 * phi_l)
    sin2phi = np.sin(2.0 * phi_l)

    g1_hat = np.fft.fft2(gamma1)
    g2_hat = np.fft.fft2(gamma2)

    E_hat = cos2phi * g1_hat + sin2phi * g2_hat
    B_hat = -sin2phi * g1_hat + cos2phi * g2_hat

    E_power = np.sum(np.abs(E_hat)**2) / (Nx * Ny)
    B_power = np.sum(np.abs(B_hat)**2) / (Nx * Ny)

    EB_ratio = B_power / max(E_power, 1.0e-30)

    return E_power, B_power, EB_ratio


def _bilinear_interp(field_2d, x_grid, y_grid, x_pts, y_pts):
    """
    双线性插值

    Parameters
    ----------
    field_2d : ndarray (Ny, Nx)
    x_grid, y_grid : ndarray
        1D 网格坐标
    x_pts, y_pts : ndarray
        插值点坐标

    Returns
    -------
    vals : ndarray
        插值结果
    """
    Nx = len(x_grid)
    Ny = len(y_grid)
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    x0 = x_grid[0]
    y0 = y_grid[0]

    # 归一化坐标
    fx = (x_pts - x0) / dx
    fy = (y_pts - y0) / dy

    # 整数和小数部分
    ix = np.floor(fx).astype(int)
    iy = np.floor(fy).astype(int)
    sx = fx - ix
    sy = fy - iy

    # 周期边界
    ix = ix % Nx
    iy = iy % Ny
    ix1 = (ix + 1) % Nx
    iy1 = (iy + 1) % Ny

    vals = (
        field_2d[iy, ix] * (1 - sx) * (1 - sy)
        + field_2d[iy, ix1] * sx * (1 - sy)
        + field_2d[iy1, ix] * (1 - sx) * sy
        + field_2d[iy1, ix1] * sx * sy
    )
    return vals


def compute_shear_correlation(gamma1, gamma2, h, n_bins=20):
    """
    剪切两点相关函数

    ξ₊(θ) = <γ_t(θ₁)γ_t(θ₁+θ)> + <γ_x γ_x>
           = <|γ|²>(θ)

    对于纯 E-mode:
    ξ₊(θ) = ∫ dl/(2π) l P_E(l) J₀(lθ)
    ξ₋(θ) = ∫ dl/(2π) l P_E(l) J₄(lθ)

    Parameters
    ----------
    gamma1, gamma2 : ndarray
        剪切场
    h : float
        网格间距
    n_bins : int
        角度 bin 数

    Returns
    -------
    theta_bins : ndarray
        角度 bin 中心
    xi_plus : ndarray
        ξ₊(θ)
    xi_minus : ndarray
        ξ₋(θ)
    """
    Ny, Nx = gamma1.shape
    ky = np.fft.fftfreq(Ny, d=h) * 2.0 * np.pi
    kx = np.fft.fftfreq(Nx, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    k_mag = np.sqrt(KX**2 + KY**2)
    phi_l = np.arctan2(KY, KX)

    g1_hat = np.fft.fft2(gamma1)
    g2_hat = np.fft.fft2(gamma2)

    # E-mode power spectrum
    cos2phi = np.cos(2.0 * phi_l)
    sin2phi = np.sin(2.0 * phi_l)
    E_hat = cos2phi * g1_hat + sin2phi * g2_hat
    B_hat = -sin2phi * g1_hat + cos2phi * g2_hat

    P_E = np.abs(E_hat)**2 / (Nx * Ny)
    P_B = np.abs(B_hat)**2 / (Nx * Ny)

    # Bin by |k|
    k_max = np.pi / h
    k_bins = np.linspace(0.01 * k_max, 0.9 * k_max, n_bins)
    dk = k_bins[1] - k_bins[0]

    xi_plus = np.zeros(n_bins)
    xi_minus = np.zeros(n_bins)

    for i, kb in enumerate(k_bins):
        mask = (k_mag >= kb - dk/2) & (k_mag < kb + dk/2)
        if np.sum(mask) > 0:
            xi_plus[i] = np.mean(P_E[mask])
            xi_minus[i] = np.mean(P_B[mask])

    theta_bins = k_bins
    return theta_bins, xi_plus, xi_minus
