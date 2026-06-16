"""
detector_response.py — 探测器响应函数与 Navier-Stokes 启发的平滑核
=================================================================
本模块模拟探测器效应对顶夸克 ne变质量谱的修正。
使用从不可压缩 Navier-Stokes 方程导出的扩散型平滑核
来参数化探测器分辨率函数。

核心物理:
  探测器分辨率函数 R(M_obs | M_true) 描述
  从真实 ne变质量到观测质量的映射:
    dN_obs/dM_obs = ∫ R(M_obs|M_true) × dN_true/dM_true dM_true

  对于量能器, R 通常为高斯型:
    R(M_obs|M_true) = (1/(√(2π)σ)) exp(-(M_obs-M_true)²/(2σ²))
  其中 σ/M = a/√E ⊕ b ⊕ c/E (随机项 ⊕ 常数项 ⊕ 噪声项)

  非高斯尾部 (non-closure):
    R_tail = ε × (1/(√(2π)σ_t)) exp(-(M_obs-M_true)²/(2σ_t²))
    其中 σ_t = κσ, κ ~ 2-3

  Navier-Stokes 启发的扩散核:
    将探测器响应类比为扩散过程:
      ∂R/∂t = ν ∇²R
    其中 ν 是"探测器粘度", t 是"扩散时间"
    解为高斯核: R(r,t) = (4πνt)^{-d/2} exp(-r²/(4νt))

  映射种子项目:
    - 142_cavity_flow_movie: Navier-Stokes 求解器
    - 237_cuda_loop: 并行核计算架构
"""

import numpy as np
from topmass_constants import PI


# ============================================================
# 探测器参数 (ATLAS/CMS 典型值)
# ============================================================
JET_ENERGY_RESOLUTION_STOCHASTIC = 0.50  # a (随机项, √GeV)
JET_ENERGY_RESOLUTION_CONSTANT = 0.03  # b (常数项, 3%)
JET_ENERGY_RESOLUTION_NOISE = 1.0  # c (噪声项, GeV)
LEPTON_MOMENTUM_RESOLUTION = 0.02  # 2% for electrons/muons
MET_RESOLUTION = 0.5  # 缺失横能量分辨率

# b-tagging 参数
B_TAG_EFFICIENCY = 0.77  # b-tag 效率
LIGHT_JET_MISTAG_RATE = 0.01  # 轻喷注误标率
C_JET_MISTAG_RATE = 0.20  # c-jet 误标率


def jet_energy_resolution(E_jet):
    """
    喷注能量分辨率 σ_E/E 的参数化。

    标准三参数公式:
      σ_E/E = √((a/√E)² + b² + (c/E)²)
            = √(a²/E + b² + c²/E²)

    对于 ATLAS Run 2 典型值:
      a = 50% √GeV, b = 3%, c = 1.0 GeV

    参数:
        E_jet: 喷注能量 (GeV)

    返回:
        σ_E (GeV)
    """
    if E_jet <= 0.0:
        return 1e-6

    a = JET_ENERGY_RESOLUTION_STOCHASTIC
    b = JET_ENERGY_RESOLUTION_CONSTANT
    c = JET_ENERGY_RESOLUTION_NOISE

    sigma_rel = np.sqrt(a**2 / E_jet + b**2 + c**2 / E_jet**2)
    return sigma_rel * E_jet


def detector_resolution_function(M_true, channel='dilepton'):
    """
    计算 ne变质量分辨率 σ(M_tt̄)。

    对于半轻道 (tt̄ → bW⁺b̄W⁻ → blνbqq̄'):
      σ(M) ≈ √(σ_j1² + σ_j2² + σ_b1² + σ_b2² + σ_lep²) × c_factor

    对于双轻道 (tt̄ → blνblν):
      由于两个中微子, 无法直接重建 M_tt̄,
      需使用 M_T2 或 ne变质量方法。

    简化参数化:
      σ(M)/M = √(4(σ_j/E_j)² + (σ_b/E_b)² + (2σ_lep/E_lep)²)

    参数:
        M_true: 真实 ne变质量 (GeV)
        channel: 'dilepton', 'semileptonic', 'allhadronic'

    返回:
        σ_M (GeV)
    """
    if M_true <= 0.0:
        return 1e-6

    # 典型喷注能量 (~ m_top/2)
    E_jet = M_true / 4.0
    sigma_j = jet_energy_resolution(E_jet)

    # b 喷注分辨率 (略差)
    sigma_b = 1.2 * jet_energy_resolution(E_jet)

    # 轻子分辨率
    E_lep = M_true / 6.0
    sigma_lep = LEPTON_MOMENTUM_RESOLUTION * E_lep

    if channel == 'allhadronic':
        sigma_M = np.sqrt(4.0 * sigma_j**2 + 2.0 * sigma_b**2)
    elif channel == 'semileptonic':
        sigma_M = np.sqrt(2.0 * sigma_j**2 + 2.0 * sigma_b**2 + sigma_lep**2)
    else:  # dilepton
        sigma_M = np.sqrt(2.0 * sigma_b**2 + 2.0 * sigma_lep**2) * 2.0

    return max(sigma_M, 0.1)


def gaussian_response(M_obs, M_true, sigma):
    """
    高斯探测器响应函数。

    R(M_obs | M_true) = (1/(√(2π)σ)) exp(-(M_obs-M_true)²/(2σ²))

    包含非高斯尾部修正:
      R_total = (1-ε) R_gauss(M, σ) + ε R_gauss(M, κσ)

    参数:
        M_obs: 观测质量 (GeV)
        M_true: 真实质量 (GeV)
        sigma: 分辨率 σ (GeV)

    返回:
        R(M_obs | M_true)
    """
    epsilon = 0.05  # 尾部比例
    kappa = 2.5  # 尾部展宽因子

    dm = M_obs - M_true

    # 核心高斯
    if sigma > 0:
        core = np.exp(-0.5 * (dm / sigma)**2) / (np.sqrt(2.0 * PI) * sigma)
    else:
        core = 1e30 if abs(dm) < 1e-10 else 0.0

    # 尾部高斯
    sigma_tail = kappa * sigma
    if sigma_tail > 0:
        tail = np.exp(-0.5 * (dm / sigma_tail)**2) / (np.sqrt(2.0 * PI) * sigma_tail)
    else:
        tail = 0.0

    return (1.0 - epsilon) * core + epsilon * tail


def navier_stokes_diffusion_kernel(M_obs, M_true, t_diff, nu=1.0):
    """
    Navier-Stokes 启发的探测器响应扩散核。

    类比不可压缩流体的 vorticity 扩散:
      ∂ω/∂t = ν ∇²ω (vorticity transport equation)

    在一维 (质量轴):
      ∂R/∂t = ν ∂²R/∂M²

    基本解 (Green's function):
      R(M, t) = (4πνt)^{-1/2} exp(-M²/(4νt))

    "扩散时间" t 参数化探测器分辨率:
      t = σ² / (2ν)

    映射种子项目:
      - 142_cavity_flow_movie: 扩散项隐式求解

    参数:
        M_obs: 观测质量
        M_true: 真实质量
        t_diff: 扩散时间 (与分辨率相关)
        nu: 有效粘度

    返回:
        R(M_obs | M_true)
    """
    dm = M_obs - M_true
    D = 2.0 * nu * t_diff  # 有效扩散系数

    if D <= 0:
        return 1e30 if abs(dm) < 1e-10 else 0.0

    return np.exp(-dm**2 / (2.0 * D)) / np.sqrt(2.0 * PI * D)


def convolve_with_detector(theoretical_masses, theoretical_xs,
                           observed_masses, channel='semileptonic'):
    """
    将理论截面与探测器响应函数卷积。

    (dσ/dM)_obs(M_obs) = ∫ R(M_obs|M) × (dσ/dM)(M) dM

    使用梯形积分规则实现卷积。

    映射种子项目:
      - 945_quad_trapezoid: 梯形积分规则

    参数:
        theoretical_masses: 理论质量网格 (GeV)
        theoretical_xs: 理论截面值 (pb/GeV)
        observed_masses: 观测质量网格 (GeV)
        channel: 探测器道

    返回:
        observed_xs: 观测截面 (pb/GeV)
    """
    observed_xs = np.zeros(len(observed_masses))

    for i, M_obs in enumerate(observed_masses):
        sigma = detector_resolution_function(M_obs, channel)

        # 积分核 (仅计算有效范围 ±5σ)
        mask = np.abs(theoretical_masses - M_obs) < 5.0 * sigma
        if not np.any(mask):
            continue

        M_valid = theoretical_masses[mask]
        xs_valid = theoretical_xs[mask]

        # 响应函数值
        R_values = np.array([gaussian_response(M_obs, M, sigma)
                             for M in M_valid])

        # 梯形积分
        integrand = R_values * xs_valid
        observed_xs[i] = np.trapz(integrand, M_valid)

    return observed_xs


def btag_weight(n_jets, n_btags, n_true_b=2):
    """
    b-tagging 权重 (基于二项分布)。

    P(n_btags | n_true_b, n_jets) =
      C(n_true_b, k) ε^k (1-ε)^{n_true_b-k} ×
      C(n_jets-n_true_b, n_btags-k) f^{n_btags-k} (1-f)^{n_jets-n_true_b-n_btags+k}

    参数:
        n_jets: 总喷注数
        n_btags: b-tagged 喷注数
        n_true_b: 真实 b 喷注数

    返回:
        权重 (似然贡献)
    """
    from scipy.special import comb

    eps = B_TAG_EFFICIENCY
    f = LIGHT_JET_MISTAG_RATE

    weight = 0.0
    for k in range(min(n_btags, n_true_b) + 1):
        if n_btags - k > n_jets - n_true_b:
            continue

        prob_b = comb(n_true_b, k) * eps**k * (1.0 - eps)**(n_true_b - k)
        prob_light = (comb(n_jets - n_true_b, n_btags - k) *
                      f**(n_btags - k) * (1.0 - f)**(n_jets - n_true_b - n_btags + k))
        weight += prob_b * prob_light

    return max(weight, 1e-30)


def parallel_kernel_response(mass_grid, center_mass, sigma, n_threads=4):
    """
    并行化高斯核响应计算 (模拟 CUDA 线程架构)。

    将质量网格分割为 n_threads 个块, 每块独立计算。
    全局索引: K = tx + Bx×ty + Bx×By×tz + ... (CUDA 3D 索引)

    映射种子项目:
      - 237_cuda_loop: CUDA 线程索引计算

    参数:
        mass_grid: 质量网格
        center_mass: 中心质量
        sigma: 分辨率
        n_threads: 线程数

    返回:
        响应值数组
    """
    N = len(mass_grid)
    result = np.zeros(N)

    # 模拟 CUDA grid-stride loop
    threads_per_block = max(1, N // n_threads)

    for block_id in range(n_threads):
        for thread_id in range(threads_per_block):
            # CUDA 全局索引
            global_idx = thread_id + block_id * threads_per_block
            if global_idx >= N:
                break

            M = mass_grid[global_idx]
            dm = M - center_mass
            if sigma > 0:
                result[global_idx] = np.exp(-0.5 * (dm / sigma)**2) / (np.sqrt(2.0 * PI) * sigma)
            else:
                result[global_idx] = 0.0

    return result
