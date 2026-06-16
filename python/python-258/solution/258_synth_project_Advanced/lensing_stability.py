"""
lensing_stability.py — von Neumann 稳定性分析与条件数诊断
=========================================================

来源种子: 1109_marekgluza_Fidelity_witnesses_example (保真度见证),
          782_msm_to_mm (稀疏矩阵分析/条件数)
科学角色: 对弱透镜质量重建的数值方案进行严格的稳定性分析,
         包括 von Neumann 放大因子、矩阵条件数、
         谱半径计算和 Monte Carlo 鲁棒性测试。

稳定性理论:
===========
1. von Neumann 稳定性分析:
   对离散 PDE ∂κ/∂t = L_h(κ), 代入傅里叶模式 κ_j = G^n e^{ijkh}:

   纯扩散 ∂κ/∂t = ν∇²κ:
   2阶: G(k) = 1 - (4νΔt/h²) sin²(kh/2)
   稳定性: |G(k)| ≤ 1  ∀k  ⟹  νΔt/h² ≤ 1/2

   4阶: G(k) = 1 - (νΔt/(3h²)) × [16sin²(kh/2) - sin²(kh)]
   稳定性: νΔt/h² ≤ 3/8 (略宽于2阶)

   双调和 ∂κ/∂t = -ν₂∇⁴κ:
   2阶: G(k) = 1 - ν₂Δt × k_eff⁴
   k_eff⁴ = (4/h²)² × (sin²(kx h/2) + sin²(ky h/2))²
   稳定性: ν₂Δt/h⁴ ≤ 1/8

2. 矩阵条件数:
   离散化系统 Aκ = b 的条件数:
   κ(A) = ||A|| × ||A⁻¹|| = σ_max/σ_min

   对于 Laplacian:
   κ(Δ_h) ≈ O(N²)  (病态!)

   对于正则化系统 (A†A + λI):
   κ(A†A + λI) = (σ_max² + λ)/(σ_min² + λ)
   当 λ >> σ_min² 时条件数改善

3. 放大因子谱半径:
   ρ(G) = max_k |G(k)|
   稳定性要求 ρ(G) ≤ 1 + O(Δt) (Lax-Richtmyer)

4. Monte Carlo 鲁棒性:
   对 N 次随机噪声实现, 计算:
   - 重建偏差: <κ_recon> - κ_true
   - 重建方差: Var(κ_recon)
   - SNR: max(κ_true) / std(κ_recon - κ_true)
"""

import numpy as np
from lensing_config import GRID, STAB, RECON
from lensing_finite_diff import modified_wavenumber_analysis


def von_neumann_diffusion(h, dt, nu, N_modes=256, order=2):
    """
    计算扩散方程的 von Neumann 放大因子

    对于 ∂u/∂t = ν∇²u, 离散后:
        u_j^{n+1} = G(k) × u_j^n

    2阶 Laplacian:
        G(k) = 1 - (4νΔt/h²) sin²(kh/2)

    4阶 Mehrstellen:
        G(k) = 1 - (νΔt/(3h²)) × [16sin²(kh/2) - sin²(kh)]

    稳定性条件:
        |G(k)| ≤ 1  ∀k ∈ [-π/h, π/h]

    最坏情况: k = π/h (Nyquist 模式)
    2阶: G(π/h) = 1 - 4νΔt/h²
    |G| ≤ 1 ⟹ 0 ≤ 4νΔt/h² ≤ 2 ⟹ νΔt/h² ≤ 1/2

    Parameters
    ----------
    h : float
        网格间距
    dt : float
        时间步长
    nu : float
        扩散系数 (对应 λ₁ 正则化)
    N_modes : int
        波数采样点数
    order : int
        差分阶数

    Returns
    -------
    k_array : ndarray
        波数数组
    G_array : ndarray
        放大因子 (复数, 扩散方程为实数)
    is_stable : bool
        是否满足稳定性条件
    spectral_radius : float
        谱半径 max|G(k)|
    """
    k_array = np.linspace(-np.pi / h, np.pi / h, N_modes)
    r = nu * dt / h**2

    if order == 2:
        G_array = 1.0 - 4.0 * r * np.sin(k_array * h / 2.0)**2
    elif order == 4:
        G_array = 1.0 - (r / 3.0) * (
            16.0 * np.sin(k_array * h / 2.0)**2
            - np.sin(k_array * h)**2
        )
    else:
        raise ValueError(f"Order {order} not supported")

    spectral_radius = np.max(np.abs(G_array))
    is_stable = spectral_radius <= 1.0 + 1.0e-12

    return k_array, G_array, is_stable, spectral_radius


def von_neumann_biharmonic(h, dt, nu2, N_modes=256, order=2):
    """
    计算双调和方程的 von Neumann 放大因子

    对于 ∂u/∂t = -ν₂∇⁴u:
        G(k) = 1 - ν₂Δt × k_eff⁴

    2阶 k_eff⁴:
        k_eff⁴ = ((4/h²)(sin²(kx h/2) + sin²(ky h/2)))²

    沿一维: k_eff⁴ = (4/h² × sin²(kh/2))² = (16/h⁴) sin⁴(kh/2)

    稳定性: ν₂Δt × 16/h⁴ ≤ 2 ⟹ ν₂Δt/h⁴ ≤ 1/8

    Parameters
    ----------
    h : float
        网格间距
    dt : float
        时间步长
    nu2 : float
        双调和系数 (对应 λ₂)
    N_modes : int
        波数采样点数
    order : int
        差分阶数

    Returns
    -------
    k_array : ndarray
        波数数组
    G_array : ndarray
        放大因子
    is_stable : bool
        稳定性
    spectral_radius : float
        谱半径
    """
    k_array = np.linspace(-np.pi / h, np.pi / h, N_modes)
    r2 = nu2 * dt / h**4

    if order == 2:
        keff4 = (4.0 / h**2 * np.sin(k_array * h / 2.0)**2)**2
        G_array = 1.0 - r2 * h**4 * keff4
    elif order == 4:
        keff2 = (1.0 / (3.0 * h**2)) * (
            16.0 * np.sin(k_array * h / 2.0)**2
            - np.sin(k_array * h)**2
        )
        keff4 = keff2**2
        G_array = 1.0 - nu2 * dt * keff4
    else:
        raise ValueError(f"Order {order} not supported")

    spectral_radius = np.max(np.abs(G_array))
    is_stable = spectral_radius <= 1.0 + 1.0e-12

    return k_array, G_array, is_stable, spectral_radius


def compute_system_condition_number(N, h, lam1, lam2, order=4):
    """
    计算离散重建系统的条件数

    系统矩阵 (Fourier 空间对角):
        M(k) = |A(k)|² + λ₁|k|² + λ₂|k|⁴

    其中 A(k) 是 KS 正向算子的符号:
        |A(k)|² = 1 (在连续极限下)

    因此:
        M(k) = 1 + λ₁k² + λ₂k⁴

    条件数:
        κ(M) = max_k M(k) / min_k M(k)

    对于正则化系统:
        min_k M(k) = M(0) = 1 (DC 被排除)
        max_k M(k) = M(k_Nyquist) = 1 + λ₁k_N² + λ₂k_N⁴

    Parameters
    ----------
    N : int
        网格大小
    h : float
        网格间距
    lam1 : float
        Tikhonov 系数
    lam2 : float
        Biharmonic 系数
    order : int
        差分阶数

    Returns
    -------
    cond_number : float
        条件数
    eigenvalues : ndarray
        特征值数组
    k_array : ndarray
        波数数组
    """
    k_array = np.fft.fftfreq(N, d=h) * 2.0 * np.pi

    # 修正波数
    if order == 2:
        k2_eff = (4.0 / h**2) * np.sin(k_array * h / 2.0)**2
    elif order == 4:
        k2_eff = (1.0 / (3.0 * h**2)) * (
            16.0 * np.sin(k_array * h / 2.0)**2
            - np.sin(k_array * h)**2
        )
    else:
        k2_eff = k_array**2

    # 2D: 取所有 kx, ky 组合
    KX, KY = np.meshgrid(k2_eff, k2_eff)
    k2_total = KX + KY

    # 系统矩阵特征值
    eigenvalues = 1.0 + lam1 * k2_total + lam2 * k2_total**2

    # 排除 DC 模式 (k=0)
    eigenvalues_no_dc = eigenvalues[1:, :].flatten()  # 去掉 DC 行
    eigenvalues_no_dc = eigenvalues_no_dc[eigenvalues_no_dc > 0]

    if len(eigenvalues_no_dc) == 0:
        cond_number = np.inf
    else:
        cond_number = np.max(eigenvalues_no_dc) / np.min(eigenvalues_no_dc)

    return cond_number, eigenvalues, k_array


def monte_carlo_stability_test(kappa_true_func, shear_forward_func,
                                h, n_trials=None):
    """
    Monte Carlo 稳定性测试

    对多次随机噪声实现进行重建, 评估:
    1. 偏差: bias(θ) = <κ_recon(θ)> - κ_true(θ)
    2. 方差: σ²(θ) = Var(κ_recon(θ))
    3. 信噪比: SNR = max(κ_true) / RMS(κ_recon - κ_true)
    4. 峰值位置偏移: Δθ_peak = |θ_peak,recon - θ_peak,true|

    Parameters
    ----------
    kappa_true_func : callable
        真实收敛场生成函数
    shear_forward_func : callable
        正向模型 (κ → γ)
    h : float
        网格间距
    n_trials : int
        试验次数

    Returns
    -------
    stats : dict
        统计量字典
    """
    if n_trials is None:
        n_trials = STAB.n_monte_carlo

    N = GRID.N_grid
    rng = np.random.default_rng(GRID.seed)

    # 生成真实场
    x = np.linspace(-GRID.field_size_arcmin/2, GRID.field_size_arcmin/2, N)
    y = np.linspace(-GRID.field_size_arcmin/2, GRID.field_size_arcmin/2, N)
    X, Y = np.meshgrid(x, y)
    kappa_true = kappa_true_func(X, Y)

    recon_stack = np.zeros((n_trials, N, N))

    for trial in range(n_trials):
        # 正向模型 + 噪声
        g1, g2 = shear_forward_func(kappa_true)
        noise1 = rng.normal(0.0, GRID.noise_level, (N, N))
        noise2 = rng.normal(0.0, GRID.noise_level, (N, N))
        g1_noisy = g1 + noise1
        g2_noisy = g2 + noise2

        # 简单 KS 重建
        from lensing_kaiser_squires import kaiser_squires
        kappa_recon, _ = kaiser_squires(g1_noisy, g2_noisy, h, apply_filter=True)
        recon_stack[trial] = kappa_recon

    # 统计
    mean_recon = np.mean(recon_stack, axis=0)
    var_recon = np.var(recon_stack, axis=0)
    bias = mean_recon - kappa_true
    rms_error = np.sqrt(np.mean((recon_stack - kappa_true[None, :, :])**2, axis=0))

    # 全局统计
    bias_global = np.sqrt(np.mean(bias**2))
    var_global = np.mean(var_recon)
    snr = np.max(np.abs(kappa_true)) / max(np.sqrt(np.mean(rms_error**2)), 1.0e-15)

    # 峰值偏移
    true_peak = np.unravel_index(np.argmax(kappa_true), kappa_true.shape)
    mean_peak = np.unravel_index(np.argmax(mean_recon), mean_recon.shape)
    peak_shift = np.sqrt(
        (true_peak[0] - mean_peak[0])**2 + (true_peak[1] - mean_peak[1])**2
    ) * h

    stats = {
        'bias_global': bias_global,
        'variance_global': var_global,
        'snr': snr,
        'peak_shift_pixels': peak_shift / h,
        'peak_shift_arcmin': peak_shift,
        'rms_error_mean': np.mean(rms_error),
        'n_trials': n_trials,
    }

    return stats


def noise_sweep_analysis(kappa_true_func, shear_forward_func, h,
                         noise_levels=None):
    """
    噪声水平扫描分析

    系统性地改变形状噪声水平, 评估重建质量退化:
    - SNR vs σ_noise
    - Bias vs σ_noise
    - 重建偏差的标度律: bias ∝ σ_noise^α

    预期标度:
    - KS 重建: SNR ∝ 1/σ_noise (线性)
    - PDE 重建: SNR ∝ 1/σ_noise^β (β < 1, 正则化抑制噪声)

    Parameters
    ----------
    kappa_true_func : callable
    shear_forward_func : callable
    h : float
    noise_levels : ndarray or None

    Returns
    -------
    results : dict
        各噪声水平下的重建统计
    """
    if noise_levels is None:
        noise_levels = np.linspace(0.005, STAB.max_noise_sigma, STAB.noise_sweep_levels)

    snr_values = []
    bias_values = []

    for sigma in noise_levels:
        GRID.noise_level = sigma
        stats = monte_carlo_stability_test(
            kappa_true_func, shear_forward_func, h, n_trials=10
        )
        snr_values.append(stats['snr'])
        bias_values.append(stats['bias_global'])

    results = {
        'noise_levels': noise_levels,
        'snr_values': np.array(snr_values),
        'bias_values': np.array(bias_values),
    }

    # 拟合标度律: SNR ∝ σ^α
    if len(noise_levels) > 2 and all(s > 0 for s in snr_values):
        log_sigma = np.log(noise_levels)
        log_snr = np.log(np.maximum(snr_values, 1.0e-15))
        # 线性拟合: log(SNR) = α log(σ) + C
        A = np.vstack([log_sigma, np.ones_like(log_sigma)]).T
        result = np.linalg.lstsq(A, log_snr, rcond=None)
        scaling_exponent = result[0][0] if result[0].size > 0 else np.nan
        results['snr_scaling_exponent'] = scaling_exponent

    return results
