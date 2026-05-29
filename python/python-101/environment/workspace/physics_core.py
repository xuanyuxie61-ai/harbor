"""
physics_core.py
==============
核心物理模型与公式库 —— 光子晶体带隙工程的电磁学基础

本模块严格遵循麦克斯韦方程组，提供二维/三维光子晶体的介电函数、
本构关系、能带结构计算所需的全部物理常数与解析公式。
"""

import numpy as np

# =============================================================================
# 物理常数 (SI 单位制)
# =============================================================================
C_0 = 2.99792458e8          # 真空光速 [m/s]
MU_0 = 4.0 * np.pi * 1.0e-7 # 真空磁导率 [H/m]
EPS_0 = 1.0 / (MU_0 * C_0 ** 2)  # 真空介电常数 [F/m]
ETA_0 = np.sqrt(MU_0 / EPS_0)    # 真空波阻抗 [Ohm]
H_BAR = 1.054571817e-34     # 约化普朗克常数 [J·s]
EV_TO_J = 1.602176634e-19   # 电子伏特转焦耳


# =============================================================================
# 介电材料模型
# =============================================================================

def eps_lorentz(omega, eps_inf, omega_p, gamma, omega_0):
    """
    Lorentz 振子模型 —— 描述色散介质的复介电常数
    
    公式:
        ε(ω) = ε_∞ + ω_p² / (ω_0² - ω² - iγω)
    
    其中:
        ε_∞    : 高频介电常数
        ω_p    : 等离子体频率 [rad/s]
        γ      : 阻尼系数 [rad/s]  
        ω_0    : 共振频率 [rad/s]
        ω      : 入射光角频率 [rad/s]
    
    Returns
    -------
    complex
        复介电常数 ε(ω)
    """
    if omega < 0:
        raise ValueError("角频率 ω 必须非负")
    if gamma < 0:
        raise ValueError("阻尼系数 γ 必须非负")
    return eps_inf + (omega_p ** 2) / (omega_0 ** 2 - omega ** 2 - 1j * gamma * omega)


def eps_drude(omega, eps_inf, omega_p, gamma):
    """
    Drude 自由电子气模型 —— 金属介电响应
    
    公式:
        ε(ω) = ε_∞ - ω_p² / (ω² + iγω)
    
    Parameters
    ----------
    omega : float
        角频率 [rad/s]
    eps_inf : float
        高频介电常数
    omega_p : float
        等离子体频率 [rad/s]
    gamma : float
        碰撞频率 [rad/s]
    
    Returns
    -------
    complex
        复介电常数
    """
    if omega <= 1e-12:
        return complex(-1e18, 0)  # 直流极限近似
    if gamma < 0:
        raise ValueError("阻尼系数 γ 必须非负")
    return eps_inf - (omega_p ** 2) / (omega ** 2 + 1j * gamma * omega)


def eps_sellmeier(wavelength, B_coeffs, C_coeffs):
    """
    Sellmeier 方程 —— 透明光学材料色散
    
    公式:
        n²(λ) = 1 + Σᵢ Bᵢλ² / (λ² - Cᵢ)
    
        ε(λ) = n²(λ)
    
    Parameters
    ----------
    wavelength : float
        波长 [μm], 必须 > 0
    B_coeffs : list of float
        Sellmeier B 系数
    C_coeffs : list of float
        Sellmeier C 系数 [μm²]
    
    Returns
    -------
    float
        介电常数 ε = n²
    """
    if wavelength <= 0:
        raise ValueError("波长必须为正")
    if len(B_coeffs) != len(C_coeffs):
        raise ValueError("B_coeffs 与 C_coeffs 长度必须一致")
    n2 = 1.0
    for B, C in zip(B_coeffs, C_coeffs):
        if wavelength ** 2 <= C:
            # 边界处理: 接近共振时进行正则化
            n2 += B * wavelength ** 2 / max(wavelength ** 2 - C, 1e-12)
        else:
            n2 += B * wavelength ** 2 / (wavelength ** 2 - C)
    return n2


# =============================================================================
# 光子晶体几何与晶格
# =============================================================================

def reciprocal_lattice_2d(a1, a2):
    """
    计算二维晶格的倒格矢
    
    原胞基矢 a₁, a₂，倒格矢 b₁, b₂ 满足:
        aᵢ · bⱼ = 2π δᵢⱼ
    
    公式:
        b₁ = 2π (a₂ × ẑ) / (a₁ · (a₂ × ẑ))
        b₂ = 2π (ẑ × a₁) / (a₁ · (a₂ × ẑ))
    
    Parameters
    ----------
    a1, a2 : ndarray, shape (2,)
        实空间晶格基矢 [m]
    
    Returns
    -------
    b1, b2 : ndarray, shape (2,)
        倒空间晶格基矢 [m⁻¹]
    """
    a1 = np.asarray(a1, dtype=float)
    a2 = np.asarray(a2, dtype=float)
    if a1.shape != (2,) or a2.shape != (2,):
        raise ValueError("a1, a2 必须为二维向量")
    
    area = a1[0] * a2[1] - a1[1] * a2[0]
    if abs(area) < 1e-18:
        raise ValueError("晶格基矢共线，无法构成二维晶格")
    
    b1 = 2.0 * np.pi * np.array([a2[1], -a2[0]]) / area
    b2 = 2.0 * np.pi * np.array([-a1[1], a1[0]]) / area
    return b1, b2


def brillouin_zone_path_2d(b1, b2, num_points, lattice_type='square'):
    """
    生成二维光子晶体布里渊区高对称点路径
    
    对于正方晶格:
        Γ = (0, 0)
        X = (π/a, 0) = b₁/2
        M = (π/a, π/a) = (b₁+b₂)/2
    
    对于三角晶格:
        Γ = (0, 0)
        K = (4π/3a, 0)
        M = (π/√3a, π/a)
    
    Parameters
    ----------
    b1, b2 : ndarray
        倒格矢
    num_points : int
        每段路径采样点数
    lattice_type : str
        'square' 或 'triangular'
    
    Returns
    -------
    k_points : ndarray, shape (N, 2)
        k 点路径
    labels : list of str
        高对称点标签
    """
    if num_points < 2:
        raise ValueError("采样点数至少为 2")
    
    if lattice_type == 'square':
        Gamma = np.array([0.0, 0.0])
        X = 0.5 * b1
        M = 0.5 * (b1 + b2)
        segments = [
            (Gamma, X, 'Γ→X'),
            (X, M, 'X→M'),
            (M, Gamma, 'M→Γ')
        ]
    elif lattice_type == 'triangular':
        Gamma = np.array([0.0, 0.0])
        # 六角晶格特殊高对称点
        K = (2.0 / 3.0) * b1 + (1.0 / 3.0) * b2
        M = 0.5 * b1
        segments = [
            (Gamma, K, 'Γ→K'),
            (K, M, 'K→M'),
            (M, Gamma, 'M→Γ')
        ]
    else:
        raise ValueError(f"不支持的晶格类型: {lattice_type}")
    
    k_points = []
    labels = []
    for start, end, label in segments:
        for t in np.linspace(0, 1, num_points):
            k_points.append((1 - t) * start + t * end)
        labels.append(label)
    
    return np.array(k_points), labels


# =============================================================================
# 电磁场本征方程相关公式
# =============================================================================

def helmholtz_operator_2d_te(psi, eps_r, kx, ky, dx, dy):
    """
    二维 TE 偏振麦克斯韦-赫姆霍兹方程的离散算子作用
    
    TE 模式 (电场垂直于传播平面):
        ∇ × (1/ε(r) ∇ × H) = (ω²/c²) H
    
    对于 z 方向磁场 H_z:
        -∂/∂x(1/ε ∂H_z/∂x) - ∂/∂y(1/ε ∂H_z/∂y) = (ω²/c²) H_z
    
    采用中心差分离散 (Yee 网格):
        ∇²_ε H_z ≈ [H_z(i+1,j) - 2H_z(i,j) + H_z(i-1,j)] / (ε_avg_x dx²)
                  + [H_z(i,j+1) - 2H_z(i,j) + H_z(i,j-1)] / (ε_avg_y dy²)
    
    Parameters
    ----------
    psi : ndarray, shape (nx, ny)
        磁场 H_z 分量
    eps_r : ndarray, shape (nx, ny)
        相对介电常数分布
    kx, ky : float
        布洛赫波矢分量 [m⁻¹]
    dx, dy : float
        网格间距 [m]
    
    Returns
    -------
    ndarray
        算子作用结果 L·ψ
    """
    nx, ny = psi.shape
    if eps_r.shape != (nx, ny):
        raise ValueError("eps_r 与 psi 形状必须一致")
    if dx <= 0 or dy <= 0:
        raise ValueError("网格间距必须为正")
    
    result = np.zeros_like(psi, dtype=complex)
    
    # 平均介电常数 (避免介电突变处的奇异性)
    eps_avg_x = np.zeros_like(eps_r)
    eps_avg_y = np.zeros_like(eps_r)
    
    eps_avg_x[1:-1, :] = 0.5 * (eps_r[1:-1, :] + eps_r[:-2, :])
    eps_avg_x[0, :] = eps_r[0, :]
    eps_avg_x[-1, :] = eps_r[-1, :]
    
    eps_avg_y[:, 1:-1] = 0.5 * (eps_r[:, 1:-1] + eps_r[:, :-2])
    eps_avg_y[:, 0] = eps_r[:, 0]
    eps_avg_y[:, -1] = eps_r[:, -1]
    
    # 中心差分 (含布洛赫周期性边界相位因子)
    for i in range(nx):
        ip1 = (i + 1) % nx
        im1 = (i - 1) % nx
        phase_x_p = np.exp(1j * kx * dx) if i == nx - 1 else 1.0
        phase_x_m = np.exp(-1j * kx * dx) if i == 0 else 1.0
        
        for j in range(ny):
            jp1 = (j + 1) % ny
            jm1 = (j - 1) % ny
            phase_y_p = np.exp(1j * ky * dy) if j == ny - 1 else 1.0
            phase_y_m = np.exp(-1j * ky * dy) if j == 0 else 1.0
            
            # x 方向二阶差分 (含周期性边界)
            d2x = (psi[ip1, j] * phase_x_p - 2 * psi[i, j] + psi[im1, j] * phase_x_m) / dx ** 2
            # y 方向二阶差分
            d2y = (psi[i, jp1] * phase_y_p - 2 * psi[i, j] + psi[i, jm1] * phase_y_m) / dy ** 2
            
            # 有效介电常数倒数加权 (处理介电不连续)
            inv_eps_x = 1.0 / max(eps_avg_x[i, j], 1e-12)
            inv_eps_y = 1.0 / max(eps_avg_y[i, j], 1e-12)
            
            result[i, j] = -(inv_eps_x * d2x + inv_eps_y * d2y)
    
    return result


def normalized_frequency(a, omega):
    """
    光子晶体的归一化频率 (无量纲带隙参数)
    
    公式:
        ω_norm = ωa / (2πc) = a / λ
    
    Parameters
    ----------
    a : float
        晶格常数 [m]
    omega : float
        角频率 [rad/s]
    
    Returns
    -------
    float
        归一化频率 ωa/(2πc)
    """
    if a <= 0:
        raise ValueError("晶格常数必须为正")
    if omega < 0:
        raise ValueError("角频率必须非负")
    return omega * a / (2.0 * np.pi * C_0)


def bandgap_ratio(omega_lower, omega_upper):
    """
    计算光子带隙的相对宽度
    
    公式:
        Δω/ω_mid = (ω_upper - ω_lower) / [(ω_upper + ω_lower)/2]
                 = 2(ω_upper - ω_lower) / (ω_upper + ω_lower)
    
    Parameters
    ----------
    omega_lower, omega_upper : float
        带隙下边界和上边界频率
    
    Returns
    -------
    float
        相对带隙宽度
    """
    if omega_lower <= 0 or omega_upper <= omega_lower:
        return 0.0
    return 2.0 * (omega_upper - omega_lower) / (omega_upper + omega_lower)


# =============================================================================
# 耦合模理论 (Coupled-Mode Theory, CMT)
# =============================================================================

def coupled_mode_equations(z, A, kappa, delta_beta, alpha=0.0):
    """
    一维耦合模方程 —— 描述布拉格光栅/光子晶体波导中的模式耦合
    
    对于前向模 A⁺ 和后向模 A⁻:
        dA⁺/dz =  iδβ A⁺ + iκ A⁻
        dA⁻/dz = -iδβ A⁻ + iκ* A⁺
    
    其中:
        δβ = β - β_B   (偏离布拉格条件)
        κ  = 耦合系数
        α  = 损耗系数
    
    Parameters
    ----------
    z : float
        传播距离 [m]
    A : ndarray, shape (2,)
        [A⁺, A⁻] 前向和后向模振幅
    kappa : complex
        耦合系数 [m⁻¹]
    delta_beta : float
        传播常数失谐 [m⁻¹]
    alpha : float
        损耗系数 [m⁻¹]
    
    Returns
    -------
    ndarray, shape (2,)
        [dA⁺/dz, dA⁻/dz]
    """
    A = np.asarray(A, dtype=complex)
    if A.shape != (2,):
        raise ValueError("A 必须为长度 2 的向量 [A+, A-]")
    
    dAp = 1j * delta_beta * A[0] + 1j * kappa * A[1] - alpha * A[0]
    dAm = -1j * np.conj(kappa) * A[0] - 1j * delta_beta * A[1] - alpha * A[1]
    return np.array([dAp, dAm], dtype=complex)


def bragg_reflectivity(kappa, L, delta_beta):
    """
    布拉格光栅反射率 —— 耦合模理论解析解
    
    基于标准耦合模方程:
        dA⁺/dz =  iδβ A⁺ + iκ A⁻
        dA⁻/dz = -iκ* A⁺ - iδβ A⁻
    
    边界条件: A⁺(0)=1, A⁻(L)=0
    
    解析解:
        带隙内 (|κ| > |δβ|):  S = √(|κ|² - δβ²)
            R = |κ|² sinh²(SL) / [|κ|² sinh²(SL) + S² cosh²(SL)]
        
        带隙外 (|κ| < |δβ|):  S' = √(δβ² - |κ|²)
            R = |κ|² sin²(S'L) / [|κ|² sin²(S'L) + S'² cos²(S'L)]
    
    Parameters
    ----------
    kappa : complex
        耦合系数 [m⁻¹]
    L : float
        光栅长度 [m]
    delta_beta : float
        传播常数失谐 [m⁻¹]
    
    Returns
    -------
    float
        功率反射率 R ∈ [0, 1]
    """
    # TODO: Implement the Bragg reflectivity based on Coupled-Mode Theory.
    # This function computes the analytical power reflectivity R for a Bragg grating.
    #
    # Key physics:
    #   - Inside bandgap (|κ| > |δβ|): evanescent-wave solution with hyperbolic functions
    #   - At bandgap edge (|κ| ≈ |δβ|): special limit
    #   - Outside bandgap (|κ| < |δβ|): oscillatory solution with trigonometric functions
    #
    # The standard CMT equations are:
    #   dA⁺/dz =  iδβ A⁺ + iκ A⁻
    #   dA⁻/dz = -iκ* A⁺ - iδβ A⁻
    #
    # Boundary conditions: A⁺(0)=1, A⁻(L)=0
    #
    # Return: power reflectivity R ∈ [0, 1]
    raise NotImplementedError("Hole 1: bragg_reflectivity needs to be implemented.")


# =============================================================================
# 品质因子与态密度
# =============================================================================

def cavity_q_factor(omega_res, delta_omega):
    """
    光子晶体微腔的品质因子
    
    公式:
        Q = ω_res / Δω = f_res / Δf
    
    Parameters
    ----------
    omega_res : float
        共振角频率 [rad/s]
    delta_omega : float
        半高全宽 (FWHM) [rad/s]
    
    Returns
    -------
    float
        品质因子 Q
    """
    if omega_res <= 0:
        raise ValueError("共振频率必须为正")
    if delta_omega <= 0:
        return float('inf')
    return omega_res / delta_omega


def local_density_of_states_3d(omega, n, V_eff):
    """
    三维自由空间局域态密度 (LDOS)
    
    公式:
        ρ(ω) = V_eff · ω²n³ / (π²c³)
    
    Parameters
    ----------
    omega : float
        角频率 [rad/s]
    n : float
        折射率
    V_eff : float
        有效模式体积 [m³]
    
    Returns
    -------
    float
        态密度 [s/m³]
    """
    if omega < 0 or n < 0 or V_eff < 0:
        raise ValueError("参数必须非负")
    return V_eff * omega ** 2 * n ** 3 / (np.pi ** 2 * C_0 ** 3)
