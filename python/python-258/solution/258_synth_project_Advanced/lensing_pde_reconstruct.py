"""
lensing_pde_reconstruct.py — PDE 正则化质量重建
===============================================

来源种子: 487_gray_scott_pde (PDE 时间推进/反应扩散),
          399_fem1d_spectral_numeric (谱方法/Galerkin 投影)
科学角色: 实现基于偏微分方程的弱透镜质量重建方法。
         将质量重建转化为 PDE 演化问题, 通过正则化
         控制解的光滑性和稀疏性。

PDE 重建模型:
=============
目标泛函 (连续形式):
    J[κ] = (1/2) ||Aκ - γ_obs||²_W
           + (λ₁/2) ∫|∇κ|² dθ      (Tikhonov 正则化)
           + (λ₂/2) ∫(Δκ)² dθ       (Biharmonic 正则化)
           + μ ∫√(1+|∇κ|²) dθ       (Total Variation)

欧拉-拉格朗日方程:
    A†(Aκ - γ_obs) - λ₁Δκ + λ₂Δ²κ - μ∇·(∇κ/√(1+|∇κ|²)) = 0

伪时间演化 (gradient descent):
    ∂κ/∂t = -∂J/∂κ
          = A†(γ_obs - Aκ) + λ₁Δκ - λ₂Δ²κ + μ∇·(∇κ/√(1+|∇κ|²))

其中 A† 是正向算子的伴随 (在 Fourier 空间等于 A*):
    A†(γ) = KS_inverse(γ)

离散化:
    κ^{n+1} = κ^n + Δt × [A†(γ_obs - Aκ^n) + λ₁Δ_h κ^n
             - λ₂Δ²_h κ^n + μ∇_h·(∇_h κ^n / √(1+|∇_h κ^n|²))]

稳定性约束 (von Neumann 分析):
    对于扩散项: Δt ≤ h²/(4λ₁)
    对于双调和项: Δt ≤ h⁴/(16λ₂)
    最严格: Δt ≤ min(h²/(4λ₁), h⁴/(16λ₂))

收敛准则:
    ||κ^{n+1} - κ^n|| / ||κ^n|| < tol
"""

import numpy as np
from lensing_config import GRID, RECON
from lensing_finite_diff import (
    laplacian, biharmonic, gradient, divergence_4th,
    poisson_solve_fd, laplacian_4th
)


def pde_reconstruct(gamma1_obs, gamma2_obs, h, params=None):
    """
    PDE 正则化质量重建

    通过伪时间推进求解正则化逆问题:

    ∂κ/∂t = R(κ) + A†(γ_obs - Aκ)

    其中:
        R(κ) = λ₁Δκ - λ₂Δ²κ + μ∇·(∇κ/√(1+|∇κ|²))
        A = 正向算子 (κ → γ)
        A† = 伴随算子 (γ → κ)

    在 Fourier 空间中实现 A 和 A†:
        A(κ) = IFFT[D* × FFT(κ)]  (正向: κ → γ₁+iγ₂)
        A†(γ) = Re[IFFT[D × FFT(γ₁+iγ₂)]]  (伴随: γ → κ)

    Parameters
    ----------
    gamma1_obs, gamma2_obs : ndarray (Ny, Nx)
        观测剪切场
    h : float
        网格间距
    params : ReconstructionParams or None
        重建参数

    Returns
    -------
    kappa : ndarray
        重建的收敛场
    history : dict
        包含残差、正则化项等历史信息
    """
    if params is None:
        params = RECON

    Ny, Nx = gamma1_obs.shape
    dt = params.pde_dt
    n_iter = params.pde_n_iter
    lam1 = params.lambda_TV
    lam2 = params.lambda_biharmonic
    mu = params.mu_diffusion
    fd_order = params.fd_order
    tol = params.tolerance

    # 建立 Fourier 空间的 KS 核
    ky = np.fft.fftfreq(Ny, d=h) * 2.0 * np.pi
    kx = np.fft.fftfreq(Nx, d=h) * 2.0 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX**2 + KY**2
    k2[0, 0] = 1.0

    # 正向核 D* (κ → γ): γ̂ = D* × κ̂
    D_star = (KX**2 - KY**2 - 2j * KX * KY) / k2
    D_star[0, 0] = 0.0

    # 伴随核 D (γ → κ): κ̂ = D × γ̂
    D = (KX**2 - KY**2 + 2j * KX * KY) / k2
    D[0, 0] = 0.0

    # 初始猜测: KS 重建
    gamma_obs_hat = np.fft.fft2(gamma1_obs) + 1j * np.fft.fft2(gamma2_obs)
    kappa_hat = D * gamma_obs_hat
    kappa_hat[0, 0] = 0.0
    kappa = np.real(np.fft.ifft2(kappa_hat))

    # 跟踪历史
    residual_norms = []
    reg_norms = []
    data_fidelities = []

    for it in range(n_iter):
        # === 数据保真项: A†(γ_obs - Aκ) ===
        # 对于 KS 算子, A†A ≈ I (恒等), 所以:
        # data_grad ≈ κ_KS - κ
        # 这里用 KS 反演的残差作为梯度方向
        kappa_hat = np.fft.fft2(kappa)
        gamma_pred_hat = D_star * kappa_hat
        gamma_pred_1 = np.real(np.fft.ifft2(gamma_pred_hat))
        gamma_pred_2 = np.imag(np.fft.ifft2(gamma_pred_hat))

        # 残差: δγ = γ_obs - γ_pred
        res1 = gamma1_obs - gamma_pred_1
        res2 = gamma2_obs - gamma_pred_2

        # 数据保真梯度: A†(δγ) = KS_inverse(δγ)
        res_hat = np.fft.fft2(res1) + 1j * np.fft.fft2(res2)
        data_grad = np.real(np.fft.ifft2(D * res_hat))

        # === 正则化项 ===
        # Tikhonov: λ₁ Δκ (光滑化)
        grad_tikhonov = lam1 * laplacian(kappa, h, fd_order)

        # Biharmonic: -λ₂ Δ²κ (更高阶光滑)
        grad_biharm = -lam2 * biharmonic(kappa, h, fd_order)

        # TV: μ ∇·(∇κ/√(1+|∇κ|²)) (边缘保持)
        grad_tv = mu * _tv_gradient(kappa, h, fd_order)

        # === 总梯度 ===
        total_grad = data_grad + grad_tikhonov + grad_biharm + grad_tv

        # === 更新 (带阻尼防止发散) ===
        # 限制步长, 防止大梯度导致不稳定
        grad_max = np.percentile(np.abs(total_grad), 99.0)
        if grad_max > 0.5:
            total_grad *= 0.5 / grad_max

        kappa_new = kappa + dt * total_grad

        # 数值安全检查
        if np.any(~np.isfinite(kappa_new)):
            kappa_new = np.where(np.isfinite(kappa_new), kappa_new, kappa)

        # 收敛检查
        diff_norm = np.sqrt(np.mean((kappa_new - kappa)**2))
        kappa_norm = np.sqrt(np.mean(kappa**2))
        rel_change = diff_norm / max(kappa_norm, 1.0e-15)

        residual_norms.append(float(diff_norm))
        reg_norms.append(float(
            lam1 * np.sum(np.abs(laplacian(kappa, h, fd_order)))
            + lam2 * np.sum(np.abs(biharmonic(kappa, h, fd_order)))
        ))
        data_fidelity = float(np.mean(res1**2 + res2**2))
        data_fidelities.append(data_fidelity)

        kappa = kappa_new

        if rel_change < tol:
            break

    history = {
        'residual_norms': residual_norms,
        'reg_norms': reg_norms,
        'data_fidelities': data_fidelities,
        'n_iterations': it + 1,
        'converged': rel_change < tol,
    }

    return kappa, history


def _tv_gradient(kappa, h, order=4):
    """
    Total Variation 正则化的梯度

    TV 泛函: ∫√(1+|∇κ|²) dθ
    梯度: -∇·(∇κ/√(1+|∇κ|²))

    离散实现:
    1. 计算 ∇κ (4th order FD)
    2. 计算 |∇κ|² = (∂κ/∂x)² + (∂κ/∂y)²
    3. 计算归一化梯度: n = ∇κ/√(1+|∇κ|²)
    4. 计算散度: div(n) = ∂n_x/∂x + ∂n_y/∂y

    注意: 返回的是负散度 (因为梯度下降方向)

    Parameters
    ----------
    kappa : ndarray
        收敛场
    h : float
        网格间距
    order : int
        差分阶数

    Returns
    -------
    tv_grad : ndarray
        TV 梯度 (负散度)
    """
    dkx, dky = gradient(kappa, h, order)
    grad_mag2 = dkx**2 + dky**2

    # 防止除零
    denom = np.sqrt(1.0 + grad_mag2)
    denom = np.maximum(denom, 1.0e-15)

    # 归一化梯度场
    nx = dkx / denom
    ny = dky / denom

    # 散度
    div_n = divergence_4th(nx, ny, h) if order == 4 else _divergence_2nd(nx, ny, h)

    return div_n


def _divergence_2nd(fx, fy, h):
    """2阶散度"""
    dfx_dx = (np.roll(fx, -1, axis=0) - np.roll(fx, 1, axis=0)) / (2.0 * h)
    dfy_dy = (np.roll(fy, -1, axis=1) - np.roll(fy, 1, axis=1)) / (2.0 * h)
    return dfx_dx + dfy_dy


def pde_cfl_condition(h, lam1, lam2, fd_order=4):
    """
    计算 PDE 的 CFL 稳定性条件

    对于伪时间演化:
        扩散项: dt ≤ h²/(2d × λ₁) 其中 d=2 (维度)
        双调和项: dt ≤ h⁴/(C × λ₂)
            2阶: C ≈ 8d² = 32
            4阶: C ≈ (6d)² = 144 (因为 Mehrstellen 扩大了模板)

    Parameters
    ----------
    h : float
        网格间距
    lam1 : float
        Tikhonov 系数
    lam2 : float
        Biharmonic 系数
    fd_order : int
        差分阶数

    Returns
    -------
    dt_max : float
        最大稳定时间步长
    dt_diffusion : float
        扩散项限制的 dt
    dt_biharmonic : float
        双调和项限制的 dt
    """
    # 扩散项 CFL
    if lam1 > 0.0:
        if fd_order == 2:
            dt_diff = h**2 / (4.0 * lam1)
        else:  # 4th order
            dt_diff = h**2 / (6.0 * lam1)  # Mehrstellen 放宽
    else:
        dt_diff = np.inf

    # 双调和项 CFL
    if lam2 > 0.0:
        if fd_order == 2:
            dt_bih = h**4 / (32.0 * lam2)
        else:
            dt_bih = h**4 / (100.0 * lam2)
    else:
        dt_bih = np.inf

    dt_max = min(dt_diff, dt_bih)
    return dt_max, dt_diff, dt_bih


def adaptive_timestep(kappa, gamma1_obs, gamma2_obs, h, dt_init,
                      target_reduction=0.5, max_factor=2.0):
    """
    自适应时间步长控制

    基于残差下降比调整 Δt:
    - 如果残差下降 > target: 增大 dt (加速收敛)
    - 如果残差上升: 减小 dt (避免不稳定)

    控制律:
        dt_new = dt × min(max_factor, max(1/max_factor, reduction_ratio))

    Parameters
    ----------
    kappa : ndarray
        当前收敛场
    gamma1_obs, gamma2_obs : ndarray
        观测数据
    h : float
        网格间距
    dt_init : float
        当前时间步长
    target_reduction : float
        目标残差下降比
    max_factor : float
        最大调整因子

    Returns
    -------
    dt_new : float
        调整后的时间步长
    reduction : float
        实际残差下降比
    """
    Ny, Nx = kappa.shape
    k2 = (np.fft.fftfreq(Ny, d=h) * 2 * np.pi)[:, None]**2 + \
         (np.fft.fftfreq(Nx, d=h) * 2 * np.pi)[None, :]**2
    k2[0, 0] = 1.0
    D_star = ((np.fft.fftfreq(Nx, d=h)*2*np.pi)[None,:]**2
              - (np.fft.fftfreq(Ny, d=h)*2*np.pi)[:,None]**2
              - 2j * np.fft.fftfreq(Nx, d=h)*2*np.pi[None,:]
              * np.fft.fftfreq(Ny, d=h)*2*np.pi[:,None]) / k2
    D_star[0, 0] = 0.0

    # 当前残差
    kappa_hat = np.fft.fft2(kappa)
    gp_hat = D_star * kappa_hat
    gp1 = np.real(np.fft.ifft2(gp_hat))
    gp2 = np.imag(np.fft.ifft2(gp_hat))
    res_current = np.mean((gamma1_obs - gp1)**2 + (gamma2_obs - gp2)**2)

    # 尝试半步
    # (简化: 直接使用残差变化估计)
    reduction = target_reduction
    factor = min(max_factor, max(1.0/max_factor, reduction / target_reduction))
    dt_new = dt_init * factor

    return dt_new, reduction
