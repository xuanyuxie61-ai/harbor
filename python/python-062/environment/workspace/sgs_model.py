"""
sgs_model.py
================================================================================
亚格子尺度（SGS）模型模块

在 LES 中，只有大尺度涡被直接解析，小尺度涡（< Δ，Δ 为过滤宽度）
需要通过亚格子模型来参数化。本模块实现两种经典模型：
1. Smagorinsky 模型（1963）
2. 动态 Smagorinsky 模型（Germano et al., 1991; Lilly, 1992）

核心物理公式
--------------------------------------------------------------------------------
亚格子应力张量：
    τ_{ij}^{sgs} = ū_iū_j - ū_i ū_j

Smagorinsky 模型将 SGS 应力与解析应变率关联：
    τ_{ij}^{sgs} - (1/3) δ_{ij} τ_{kk}^{sgs} = -2 ν_{sgs} S̄_{ij}

其中解析应变率：
    S̄_{ij} = 1/2 (∂ū_i/∂x_j + ∂ū_j/∂x_i)

SGS 涡粘性：
    ν_{sgs} = (C_s Δ)² |S̄|

|S̄| = √(2 S̄_{ij} S̄_{ij}) 为应变率模，C_s ≈ 0.1–0.2 为 Smagorinsky 常数，
Δ = (Δx Δy Δz)^{1/3} 为过滤宽度。

动态模型（Germano 恒等式）：
    L_{ij} = ū_i ū_j - ū_i ū_j = -2 C Δ² |S̄| S̄_{ij} + 2 C (2Δ)² |Ŝ| Ŝ_{ij}

通过最小二乘可得动态系数：
    C = ⟨L_{ij} M_{ij}⟩ / ⟨M_{kl} M_{kl}⟩

其中 M_{ij} = 2 Δ² (2² |Ŝ| Ŝ_{ij} - |S̄| S̄_{ij})
"""

import numpy as np


def compute_strain_rate_tensor(u, v, w, dx, dy, dz):
    """
    计算解析应变率张量 S_ij。

    参数
    ----------
    u, v, w : np.ndarray, shape (nx, ny, nz)
    dx, dy, dz : float

    返回
    -------
    S11, S22, S33, S12, S13, S23 : np.ndarray
    """
    def grad_central(f, axis, h):
        df = np.zeros_like(f)
        slc_p = [slice(None)] * 3
        slc_m = [slice(None)] * 3
        slc_p[axis] = slice(2, None)
        slc_m[axis] = slice(None, -2)
        df[tuple(slc_p)] = (f[tuple(slc_p)] - f[tuple(slc_m)]) / (2 * h)
        return df

    dudx = grad_central(u, 0, dx)
    dudy = grad_central(u, 1, dy)
    dudz = grad_central(u, 2, dz)

    dvdx = grad_central(v, 0, dx)
    dvdy = grad_central(v, 1, dy)
    dvdz = grad_central(v, 2, dz)

    dwdx = grad_central(w, 0, dx)
    dwdy = grad_central(w, 1, dy)
    dwdz = grad_central(w, 2, dz)

    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)

    return S11, S22, S33, S12, S13, S23


def smagorinsky_model(u, v, w, dx, dy, dz, Cs=0.16):
    """
    标准 Smagorinsky SGS 模型。

    参数
    ----------
    u, v, w : np.ndarray
    dx, dy, dz : float
    Cs : float
        Smagorinsky 常数

    返回
    -------
    nu_sgs : np.ndarray
        SGS 涡粘性场
    tau_sgs : dict
        SGS 应力分量
    """
    # === HOLE 1 BEGIN ===
    # 此处应实现标准 Smagorinsky SGS 模型的核心计算：
    # 1. 调用 compute_strain_rate_tensor 获取解析应变率张量 S_ij
    # 2. 计算应变率模 |S| = sqrt(2 S_ij S_ij)
    # 3. 计算过滤宽度 Delta = (dx*dy*dz)^{1/3}
    # 4. 计算 SGS 涡粘性 nu_sgs = (Cs * Delta)^2 * |S|
    # 5. 计算 SGS 应力 tau_sgs = -2 * nu_sgs * S_ij
    # 返回: nu_sgs (np.ndarray), tau_sgs (dict)
    raise NotImplementedError("HOLE 1: 请实现 Smagorinsky SGS 模型核心公式")
    # === HOLE 1 END ===


def dynamic_smagorinsky_model(u, v, w, dx, dy, dz, test_filter_width=2):
    """
    动态 Smagorinsky 模型（简化实现，基于 Germano 恒等式）。

    参数
    ----------
    u, v, w : np.ndarray
    dx, dy, dz : float
    test_filter_width : int
        测试过滤宽度倍数

    返回
    -------
    nu_sgs : np.ndarray
    C_dynamic : np.ndarray
        动态系数场
    """
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        # 回退：简单平均过滤
        def uniform_filter(arr, size, mode='nearest'):
            from scipy.ndimage import uniform_filter as uf
            return uf(arr, size=size, mode=mode)

    S11, S22, S33, S12, S13, S23 = compute_strain_rate_tensor(u, v, w, dx, dy, dz)

    S2 = 2.0 * (S11**2 + S22**2 + S33**2 + 2.0 * S12**2 + 2.0 * S13**2 + 2.0 * S23**2)
    S_mag = np.sqrt(np.clip(S2, 0.0, 1e12))

    Delta = (dx * dy * dz) ** (1.0 / 3.0)

    # 测试过滤（盒式过滤）
    w_test = max(test_filter_width, 2)

    def safe_filter(f):
        return uniform_filter(f, size=w_test, mode='nearest')

    # 测试过滤速度
    u_hat = safe_filter(u)
    v_hat = safe_filter(v)
    w_hat = safe_filter(w)

    # 测试过滤应变率
    Sh11, Sh22, Sh33, Sh12, Sh13, Sh23 = compute_strain_rate_tensor(
        u_hat, v_hat, w_hat, dx, dy, dz)
    Sh2 = 2.0 * (Sh11**2 + Sh22**2 + Sh33**2 + 2.0 * Sh12**2 + 2.0 * Sh13**2 + 2.0 * Sh23**2)
    Sh_mag = np.sqrt(np.clip(Sh2, 0.0, 1e12))

    # Germano 恒等式：L_ij = ū_i ū_j - û_i û_j
    L11 = safe_filter(u * u) - u_hat * u_hat
    L22 = safe_filter(v * v) - v_hat * v_hat
    L33 = safe_filter(w * w) - w_hat * w_hat
    L12 = safe_filter(u * v) - u_hat * v_hat
    L13 = safe_filter(u * w) - u_hat * w_hat
    L23 = safe_filter(v * w) - v_hat * w_hat

    # M_ij = 2 Δ² (α² |Ŝ| Ŝ_ij - |S̄| S̄_ij)  （这里 α = 2）
    alpha = 2.0
    scale = 2.0 * Delta**2
    M11 = scale * (alpha**2 * Sh_mag * Sh11 - S_mag * S11)
    M22 = scale * (alpha**2 * Sh_mag * Sh22 - S_mag * S22)
    M33 = scale * (alpha**2 * Sh_mag * Sh33 - S_mag * S33)
    M12 = scale * (alpha**2 * Sh_mag * Sh12 - S_mag * S12)
    M13 = scale * (alpha**2 * Sh_mag * Sh13 - S_mag * S13)
    M23 = scale * (alpha**2 * Sh_mag * Sh23 - S_mag * S23)

    # 最小二乘：C = <L_ij M_ij> / <M_kl M_kl>
    LM = L11 * M11 + L22 * M22 + L33 * M33 + 2.0 * (L12 * M12 + L13 * M13 + L23 * M23)
    MM = M11**2 + M22**2 + M33**2 + 2.0 * (M12**2 + M13**2 + M23**2)

    # 数值保护
    LM = np.clip(LM, -1e12, 1e12)
    MM = np.clip(MM, 1e-30, 1e12)

    # 平面平均（避免空间振荡）
    nz = u.shape[2]
    C_dynamic = np.zeros_like(u)
    for k in range(nz):
        lm_sum = np.sum(LM[:, :, k])
        mm_sum = np.sum(MM[:, :, k])
        if abs(mm_sum) > 1e-15:
            c_k = lm_sum / mm_sum
        else:
            c_k = 0.0
        C_dynamic[:, :, k] = c_k

    # 限制动态系数范围
    C_dynamic = np.clip(C_dynamic, -0.5, 0.5)

    # SGS 涡粘性
    nu_sgs = C_dynamic * Delta**2 * S_mag
    nu_sgs = np.clip(nu_sgs, 0.0, 5.0)

    return nu_sgs, C_dynamic
