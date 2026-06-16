"""
lensing_kaiser_squires.py — Kaiser-Squires 逆问题求解
====================================================

来源种子: 1050_ml-jku_LaM-SLidE (滑动/迭代重建协议),
          210_continuation (参数延拓方法)
科学角色: 实现经典 Kaiser & Squires (1993) 方法:
         从观测剪切场恢复收敛场。这是所有弱透镜质量
         重建方法的基线。

核心公式:
=========
KS 反演 (Fourier 空间):
    κ̃(l_x, l_y) = D(l_x, l_y) × γ̃(l_x, l_y)

    其中:
    D(l_x, l_y) = (l_x² - l_y² + 2i l_x l_y) / (l_x² + l_y²)
              = e^{2iφ_l}     (φ_l = atan2(l_y, l_x))

    γ̃ = γ̃₁ + iγ̃₂ (复剪切)

等价形式:
    κ̃ = [(l_x² - l_y²)/(l_x²+l_y²)] γ̃₁ + [2l_x l_y/(l_x²+l_y²)] γ̃₂

有限场校正:
    在实际观测中, 天区有限导致模式泄漏。
    窗函数 W(l) 导致:
    κ̃_obs(l) = κ̃(l) * W̃(l) / |W̃(l)|²

E-mode 投影:
    纯 E-mode 约束: κ 为实数场
    κ = Re[KF⁻¹(D × γ̃)]

B-mode 残差:
    κ_B = Im[KF⁻¹(D × γ̃)]
    理想情况下 κ_B = 0

质量片简并:
    κ → κ + κ_const 不改变剪切
    因此 KS 只能恢复 κ 的相对分布,
    绝对质量标定需要外部信息 (如 Σ_crit)。
"""

import numpy as np
from lensing_config import GRID


def kaiser_squires(gamma1_obs, gamma2_obs, h, apply_filter=True):
    """
    经典 Kaiser-Squires 反演

    在傅里叶空间:
        κ̃(l) = D(l) × γ̃(l)

    其中 D(l) = (l₁²-l₂²+2il₁l₂)/(l₁²+l₂²) 是 KS 核

    Parameters
    ----------
    gamma1_obs, gamma2_obs : ndarray (Ny, Nx)
        观测剪切场 (含噪声)
    h : float
        网格间距 (弧度)
    apply_filter : bool
        是否应用高斯平滑滤波器

    Returns
    -------
    kappa_E : ndarray
        E-mode 收敛场重建
    kappa_B : ndarray
        B-mode 残差 (应接近零)
    """
    Ny, Nx = gamma1_obs.shape
    ky = np.fft.fftfreq(Ny, d=h) * 2.0 * np.pi
    kx = np.fft.fftfreq(Nx, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0  # 避免除以零

    # KS 核: D = (kx² - ky² + 2i kx ky) / k²
    # 即 D = (kx + i ky)² / k² = e^{2iφ_l}
    D_real = (KX**2 - KY**2) / k2
    D_imag = 2.0 * KX * KY / k2
    D = D_real + 1j * D_imag
    D[0, 0] = 0.0  # DC 分量设为零

    # 复剪切
    gamma_hat = np.fft.fft2(gamma1_obs) + 1j * np.fft.fft2(gamma2_obs)

    # KS 反演
    kappa_hat = D * gamma_hat

    # 高斯滤波 (可选, 减少噪声放大)
    if apply_filter:
        # 滤波尺度: 4 像素 (抑制形状噪声)
        sigma_filter = 4.0 * h
        k2_smooth = k2 * sigma_filter**2 / 2.0
        G = np.exp(-k2_smooth)
        kappa_hat *= G

    kappa_hat[0, 0] = 0.0  # 零均值

    # E/B 分解
    # E-mode: Re(KF^{-1}(κ̃))
    # B-mode: Im(KF^{-1}(κ̃))
    kappa_full = np.fft.ifft2(kappa_hat)
    kappa_E = np.real(kappa_full)
    kappa_B = np.imag(kappa_full)

    return kappa_E, kappa_B


def kaiser_squires_with_masssheet(kappa_ks, gamma1, gamma2, h, kappa_ext=0.0):
    """
    带质量片变换的 KS 反演

    质量片变换 (Mass Sheet Transformation, MST):
        κ' = λκ + (1-λ)
        γ' = λγ
    其中 λ 是缩放参数。

    当 λ=1, κ' = κ (无变换)。
    当 λ≠1, 收敛场被重标定但剪切不变。

    这导致 "质量片简并": 从剪切观测无法确定
    收敛场的绝对归一化。

    Parameters
    ----------
    kappa_ks : ndarray
        KS 重建收敛场
    gamma1, gamma2 : ndarray
        剪切场
    h : float
        网格间距
    kappa_ext : float
        外部质量片 (external convergence)

    Returns
    -------
    kappa_transformed : ndarray
        MST 后的收敛场
    """
    kappa_transformed = kappa_ks + kappa_ext
    return kappa_transformed


def iterative_ks(gamma1_obs, gamma2_obs, h, n_iter=10, threshold=0.0):
    """
    迭代 Kaiser-Squires 方法 (GLIMPSE 类)

    基于 Lanusse et al. (2016) 的稀疏重建思路:
    1. 从 KS 重建初始 κ₀
    2. 计算残差剪切: δγ = γ_obs - γ(κ_n)
    3. 从残差重建 δκ = KS(δγ)
    4. 更新: κ_{n+1} = κ_n + δκ
    5. 应用稀疏先验 (软阈值): κ_{n+1} = S_λ(κ_{n+1})

    正向模型:
        γ(κ) = KS_forward(κ)
    即: 将 κ 变回 γ 进行一致性检查

    Parameters
    ----------
    gamma1_obs, gamma2_obs : ndarray
        观测剪切场
    h : float
        网格间距
    n_iter : int
        迭代次数
    threshold : float
        软阈值参数

    Returns
    -------
    kappa_iter : ndarray
        迭代重建的收敛场
    residual_history : list
        残差范数历史
    """
    Ny, Nx = gamma1_obs.shape
    ky = np.fft.fftfreq(Ny, d=h) * 2.0 * np.pi
    kx = np.fft.fftfreq(Nx, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)

    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0

    # KS 核
    D = (KX**2 - KY**2 + 2j * KX * KY) / k2
    D[0, 0] = 0.0

    # 正向 KS 核 (γ from κ)
    D_conj = np.conj(D)  # 正向核

    # 初始 KS 重建
    gamma_hat = np.fft.fft2(gamma1_obs) + 1j * np.fft.fft2(gamma2_obs)
    kappa_hat = D * gamma_hat
    kappa_hat[0, 0] = 0.0
    kappa_n = np.real(np.fft.ifft2(kappa_hat))

    residual_history = []

    for it in range(n_iter):
        # 正向模型: 从 κ_n 计算预测剪切
        kappa_n_hat = np.fft.fft2(kappa_n)
        gamma_pred_hat = D_conj * kappa_n_hat
        gamma_pred_1 = np.real(np.fft.ifft2(gamma_pred_hat))
        gamma_pred_2 = np.imag(np.fft.ifft2(gamma_pred_hat))

        # 残差
        res1 = gamma1_obs - gamma_pred_1
        res2 = gamma2_obs - gamma_pred_2
        res_norm = np.sqrt(np.mean(res1**2 + res2**2))
        residual_history.append(res_norm)

        # 从残差重建修正量
        res_hat = np.fft.fft2(res1) + 1j * np.fft.fft2(res2)
        delta_kappa_hat = D * res_hat
        delta_kappa_hat[0, 0] = 0.0
        delta_kappa = np.real(np.fft.ifft2(delta_kappa_hat))

        # 更新
        kappa_n = kappa_n + delta_kappa

        # 软阈值 (稀疏先验)
        if threshold > 0.0:
            kappa_n = _soft_threshold(kappa_n, threshold)

    return kappa_n, residual_history


def _soft_threshold(x, lam):
    """
    软阈值算子 (用于稀疏重建)

    S_λ(x) = sign(x) × max(|x| - λ, 0)

    在 GLIMPSE 中用于促进收敛场的稀疏性:
    假设 κ 在某个字典 (如 wavelet/starlet) 中是稀疏的。
    这里简化为在像素域中直接应用。

    Parameters
    ----------
    x : ndarray
        输入信号
    lam : float
        阈值参数

    Returns
    -------
    x_thresh : ndarray
        阈值后的信号
    """
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def ks_power_spectrum(kappa, h):
    """
    计算收敛场功率谱 P_κ(l)

    P_κ(l) = <|κ̃(l)|²> / A

    其中 A = N_x × N_y × h² 是天区面积。

    对于 ΛCDM, 理论功率谱:
    P_κ(l) = ∫ dz (H(z)/c) (W_κ(z))²/(χ²(z)) P_δ(l/χ(z), z)

    其中 W_κ(z) 是透镜权重函数:
    W_κ(z) = (3Ω_m H₀²)/(2c²) (1+z) χ(z) ∫_z^∞ dz' n(z') (1-z/z')

    Parameters
    ----------
    kappa : ndarray
        收敛场
    h : float
        网格间距

    Returns
    -------
    l_bins : ndarray
        多极矩 bin 中心
    P_l : ndarray
        功率谱
    """
    Ny, Nx = kappa.shape
    area = Nx * Ny * h**2

    kappa_hat = np.fft.fft2(kappa)
    power_2d = np.abs(kappa_hat)**2 / area

    ky = np.fft.fftfreq(Ny, d=h) * 2.0 * np.pi
    kx = np.fft.fftfreq(Nx, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k_mag = np.sqrt(KX**2 + KY**2)

    # Bin 化
    n_bins = 20
    l_max = np.pi / h
    l_bins = np.linspace(0.05 * l_max, 0.9 * l_max, n_bins)
    dl = l_bins[1] - l_bins[0]

    P_l = np.zeros(n_bins)
    for i, lb in enumerate(l_bins):
        mask = (k_mag >= lb - dl/2) & (k_mag < lb + dl/2)
        if np.sum(mask) > 0:
            P_l[i] = np.mean(power_2d[mask])

    return l_bins, P_l
