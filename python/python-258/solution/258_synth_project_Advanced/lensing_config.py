"""
lensing_config.py — 全局配置与物理常数
====================================

来源种子: 487_gray_scott_pde (参数配置系统)
科学角色: 定义弱引力透镜质量重建中所有宇宙学常数、网格参数、
         正则化权重和数值精度阈值。

宇宙学参数基于 Planck 2018 TT,TE,EE+lowE+lensing 基线:
    Ω_m = 0.3153    (物质密度参数)
    Ω_Λ = 0.6847    (暗能量密度参数)
    h     = 0.6736   (约化 Hubble 常数 H₀/(100 km/s/Mpc))
    σ_8   = 0.811    (密度涨落振幅)
    n_s   = 0.9649   (标量谱指数)
"""

import numpy as np


# ======================================================================
# 宇宙学基本参数 (Planck 2018 baseline)
# ======================================================================
class CosmoParams:
    """
    宇宙学参数容器

    关键物理关系:
        H(z) = H₀ √(Ω_m(1+z)³ + Ω_k(1+z)² + Ω_Λ)
        其中 Ω_k = 1 - Ω_m - Ω_Λ (空间曲率)

    临界密度:
        ρ_crit(z) = 3H(z)² / (8πG)

    角直径距离:
        D_A(z₁,z₂) = c/(1+z₂) ∫_{z₁}^{z₂} dz'/H(z')   (平坦宇宙)
    """
    def __init__(self):
        self.Omega_m = 0.3153          # 物质密度参数
        self.Omega_L = 0.6847          # 暗能量密度参数 (宇宙常数)
        self.Omega_k = 0.0             # 空间曲率 (平坦宇宙)
        self.h = 0.6736                # 约化 Hubble 常数
        self.H0 = 67.36                # km/s/Mpc
        self.c_km_s = 299792.458       # 光速 km/s
        self.sigma8 = 0.811            # 密度涨落振幅 (8 h⁻¹ Mpc)
        self.ns = 0.9649               # 标量谱指数
        self.G_newton = 6.674e-11      # 万有引力常数 m³/(kg·s²)
        self.M_sun = 1.989e30          # 太阳质量 kg
        self.Mpc_m = 3.0857e22         # 1 Mpc = 米
        self.arcsec_rad = np.pi / 648000.0  # 角秒 → 弧度


# ======================================================================
# 数值网格参数
# ======================================================================
class GridParams:
    """
    计算网格参数

    在弱透镜质量重建中，天区被离散为 N_grid × N_grid 的正交网格。
    每个网格单元的物理尺寸由 field_size_arcmin 和 N_grid 决定。

    关键约束:
        - Nyquist 频率: l_max = π / Δθ (Δθ = 网格间距)
        - 最大探测尺度: L = N_grid × Δθ
        - 有限差分截断误差: O(Δθ^4) for 4th order stencil
    """
    def __init__(self):
        self.N_grid = 50               # 网格分辨率 (50×50)
        self.field_size_arcmin = 60.0  # 天区大小 (角分)
        self.n_sources = 8000          # 背景源星系数量
        self.noise_level = 0.03        # 形状噪声 σ_ε per component
        self.seed = 258                # 随机种子 (可复现)

        # 导出量
        self.pixel_scale = self.field_size_arcmin / self.N_grid  # arcmin/pixel
        self.pixel_scale_rad = self.pixel_scale * np.pi / 10800.0  # rad/pixel
        self.l_max = np.pi / self.pixel_scale_rad  # Nyquist multipole
        # 内部计算统一使用 arcmin 单位以避免量纲混淆
        self.h_internal = self.pixel_scale  # arcmin/pixel (内部 FD 使用)


# ======================================================================
# 质量重建算法参数
# ======================================================================
class ReconstructionParams:
    """
    质量重建算法参数

    PDE 正则化重建的目标泛函:
        min_κ  J(κ) = ||γ - Aκ||²_W + λ₁||∇κ||² + λ₂||∇²κ||² + μ∫√(1+|∇κ|²)dθ

    其中:
        A = Kaiser-Squares 正向算子 (shear → convergence)
        W = 噪声权重矩阵
        λ₁ = TV 正则化强度 (一阶梯度)
        λ₂ = Biharmonic 正则化强度 (二阶梯度)
        μ = 表面扩散正则化强度

    GLIMPSE 参数:
        Φ = 字典 (curvelet/wavelet basis)
        λ = 稀疏惩罚系数
        n_iter = 迭代次数
    """
    def __init__(self):
        # PDE 正则化
        self.lambda_TV = 1.0e-3        # TV (Total Variation) 正则化系数
        self.lambda_biharmonic = 1.0e-5  # Biharmonic 正则化系数
        self.mu_diffusion = 5.0e-4     # 扩散正则化系数
        self.pde_dt = 0.01             # PDE 伪时间步长
        self.pde_n_iter = 80           # PDE 迭代步数

        # GLIMPSE 稀疏重建
        self.glimpse_lambda = 0.1      # 稀疏惩罚系数
        self.glimpse_n_iter = 30       # GLIMPSE 迭代次数
        self.glimpse_threshold_mode = 'soft'  # 软/硬阈值

        # 高阶有限差分
        self.fd_order = 4              # 有限差分阶数 (2 或 4)
        self.boundary_mode = 'periodic'  # 边界条件

        # 迭代收敛
        self.tolerance = 1.0e-8        # 收敛容差
        self.max_cg_iter = 200         # 共轭梯度最大迭代

        # 连续性方法
        self.continuation_steps = 15   # 同伦步数
        self.s_min = 0.0               # 同伦参数起点 (纯 KS)
        self.s_max = 1.0               # 同伦参数终点 (完全 PDE)


# ======================================================================
# 稳定性分析参数
# ======================================================================
class StabilityParams:
    """
    稳定性分析参数

    von Neumann 稳定性条件:
        对于扩散方程 ∂κ/∂t = ν∇²κ, 显式格式稳定要求:
        CFL: Δt ≤ Δθ² / (4ν)

    对于双调和方程 ∂κ/∂t = -ν₂∇⁴κ:
        Δt ≤ Δθ⁴ / (16ν₂)  (二维)

    放大因子:
        G(k) = 1 - 4νΔt/Δθ² (sin²(kx Δθ/2) + sin²(ky Δθ/2))
        稳定性要求 |G(k)| ≤ 1 ∀k
    """
    def __init__(self):
        self.n_monte_carlo = 30        # Monte Carlo 稳定性测试次数
        self.noise_sweep_levels = 8    # 噪声水平扫描点数
        self.max_noise_sigma = 0.10    # 最大形状噪声
        self.condition_number_threshold = 1.0e6  # 病态阈值
        self.eigenvalue_check = True   # 是否计算谱半径


# ======================================================================
# NFW 质量模型参数
# ======================================================================
class NFWParams:
    """
    NFW (Navarro-Frenk-White) 暗物质晕参数

    NFW 密度分布:
        ρ(r) = ρ_s / ((r/r_s)(1 + r/r_s)²)

    其中:
        r_s = 特征半径 (scale radius)
        ρ_s = 特征密度
        c = r_vir/r_s (浓度参数)

    投影收敛:
        κ(x) = 2ρ_s r_s Σ_crit⁻¹ f(x)
        f(x) = { (1/(x²-1))[1 - (1/√(1-x²)) arccosh(1/x)]  if x < 1
               { 1/3                                          if x = 1
               { (1/(x²-1))[1 - (1/√(x²-1)) arccos(1/x)]    if x > 1
        x = θ/θ_s, θ_s = r_s/D_L
    """
    def __init__(self):
        self.n_halos = 5               # 暗物质晕数量
        self.r_s_range = (0.5, 3.0)    # 特征半径范围 (arcmin)
        self.kappa_0_range = (0.05, 0.25)  # 中心收敛范围
        self.x_center_range = (-20.0, 20.0)  # 中心位置范围 (arcmin)
        self.y_center_range = (-20.0, 20.0)


# ======================================================================
# 全局单例
# ======================================================================
COSMO = CosmoParams()
GRID = GridParams()
RECON = ReconstructionParams()
STAB = StabilityParams()
NFW = NFWParams()
