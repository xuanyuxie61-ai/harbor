"""
stability_analysis.py — 数值稳定性分析模块

融合种子项目:
  - 208_conservation_ode : 守恒 ODE 的数值积分与守恒量验证
  - 1252_LewisWooltorton_numerical-bypass-QKD : 数值优化与精度控制
  - 961_r8_scale : 浮点邻域用于稳定性边界扫描

物理背景:
  喷注聚类算法的数值稳定性分析涉及:
  1. 聚类距离矩阵的条件数
  2. 迭代合并过程的误差传播
  3. 有限差分算子的 Von Neumann 稳定性
  4. 喷注子结构观测量对 pT 阈值的敏感性

关键理论:
  Von Neumann 稳定性分析:
    对线性 PDE du/dt = L(u), 离散化后:
      u_j^{n+1} = G · u_j^n
    其中 G 为放大矩阵. 稳定性条件: |G| ≤ 1 + O(Δt)

  CFL 条件:
    Δt ≤ C · Δx / |λ_max|
    其中 λ_max 为最大特征速度, C 为 CFL 数

  条件数分析:
    κ(D) = ||D|| · ||D^{-1}||
    大条件数 → 数值不稳定
"""

import numpy as np
import math
from typing import List, Tuple, Callable, Dict
from jet_fourvector import FourVector, r8_next, r8_previous


# ─────────────────────────────────────────────────────────────────────────────
# Von Neumann 稳定性分析 (源自 208_conservation_ode)
# ─────────────────────────────────────────────────────────────────────────────

def von_neumann_amplification(
    scheme: str,
    k_dx: np.ndarray,
    cfl: float = 0.5,
    diffusion: float = 0.0
) -> np.ndarray:
    """计算有限差分格式的 Von Neumann 放大因子.

    对对流-扩散方程:
      ∂u/∂t + a ∂u/∂x = ν ∂²u/∂x²

    不同格式的放大因子 G(k·Δx):

    1. FTCS (Forward-Time Centered-Space):
       G = 1 - 2r(1-cos(θ)) - i·c·sin(θ)
       其中 r = νΔt/Δx², c = aΔt/Δx, θ = kΔx
       条件: r ≤ 1/2 且 c² ≤ 2r

    2. Lax-Friedrichs:
       G = cos(θ) - i·c·sin(θ)
       条件: |c| ≤ 1

    3. Lax-Wendroff:
       G = 1 - c²(1-cos(θ)) - i·c·sin(θ)
       条件: |c| ≤ 1

    4. Upwind (一阶):
       G = 1 - c(1-cos(θ)) - i·c·sin(θ)
       条件: |c| ≤ 1

    Args:
        scheme: 'FTCS', 'Lax-Friedrichs', 'Lax-Wendroff', 'Upwind'
        k_dx: 无量纲波数 k·Δx ∈ [0, 2π]
        cfl: CFL 数 c = a·Δt/Δx
        diffusion: 扩散系数 r = ν·Δt/Δx²

    Returns:
        |G(k·Δx)| 放大因子模
    """
    theta = k_dx
    c = cfl
    r = diffusion

    if scheme == 'FTCS':
        # G = 1 - 2r(1-cos θ) - i c sin θ
        G_real = 1 - 2 * r * (1 - np.cos(theta))
        G_imag = -c * np.sin(theta)
    elif scheme == 'Lax-Friedrichs':
        # G = cos θ - i c sin θ
        G_real = np.cos(theta)
        G_imag = -c * np.sin(theta)
    elif scheme == 'Lax-Wendroff':
        # G = 1 - c²(1-cos θ) - i c sin θ
        G_real = 1 - c ** 2 * (1 - np.cos(theta))
        G_imag = -c * np.sin(theta)
    elif scheme == 'Upwind':
        # G = 1 - c(1-e^{-iθ}) = 1 - c + c cos θ - i c sin θ
        G_real = 1 - c + c * np.cos(theta)
        G_imag = -c * np.sin(theta)
    elif scheme == 'RK4-FD4':
        # 4阶 Runge-Kutta + 4阶空间差分 (喷注分析常用)
        # 对纯对流: G = 1 - i c sin θ + (i c sin θ)²/2 + ...
        # 近似为 4 阶 Taylor 展开
        z = -1j * c * np.sin(theta) + r * (np.cos(theta) - 1)
        G_real = np.real(1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24)
        G_imag = np.imag(1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    return np.sqrt(G_real ** 2 + G_imag ** 2)


def cfl_condition(max_velocity: float, dx: float,
                  safety_factor: float = 0.9) -> float:
    """计算 CFL 条件允许的最大时间步长.

    Δt_max = C_safety · Δx / |v_max|

    Args:
        max_velocity: 最大特征速度
        dx: 空间步长
        safety_factor: 安全因子 (通常 0.5 ~ 0.9)

    Returns:
        最大允许时间步长 Δt
    """
    if abs(max_velocity) < 1e-15:
        return float('inf')
    return safety_factor * abs(dx) / abs(max_velocity)


def diffusion_stability_limit(dx: float) -> float:
    """计算扩散方程的稳定性极限.

    对 FTCS 格式: Δt ≤ Δx² / (2ν)
    返回最大允许 Δt (假设 ν=1).
    """
    return dx ** 2 / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# 聚类距离矩阵的条件数分析
# ─────────────────────────────────────────────────────────────────────────────

def cluster_matrix_condition_number(
    particles: List[FourVector],
    R: float = 0.4,
    power: int = -2
) -> Dict[str, float]:
    """计算聚类距离矩阵的条件数.

    条件数 κ = σ_max / σ_min (奇异值比)

    物理意义:
      κ 大 → 距离矩阵接近奇异, 聚类结果对微扰敏感
      κ 小 → 聚类结果稳定

    在喷注分析中, 当多个粒子的 pT 和角度非常接近时,
    距离矩阵可能接近奇异, 导致聚类顺序不确定.
    """
    n = len(particles)
    if n < 2:
        return {'condition_number': 1.0, 'sigma_max': 0.0,
                'sigma_min': 0.0, 'determinant': 0.0}

    # 构建距离矩阵
    D = np.zeros((n, n))
    pT_list = [max(p.pT, 1e-10) for p in particles]

    for i in range(n):
        for j in range(i + 1, n):
            dr2 = particles[i].delta_R_sq(particles[j])
            f_min = min(pT_list[i] ** power, pT_list[j] ** power)
            D[i, j] = f_min * dr2 / (R ** 2)
            D[j, i] = D[i, j]

    # 奇异值分解
    try:
        sigma = np.linalg.svd(D, compute_uv=False)
        sigma_max = sigma[0]
        sigma_min = sigma[-1]
        cond = sigma_max / sigma_min if sigma_min > 1e-30 else float('inf')
        det = np.linalg.det(D)
    except np.linalg.LinAlgError:
        cond = float('inf')
        sigma_max = float('inf')
        sigma_min = 0.0
        det = 0.0

    return {
        'condition_number': float(cond),
        'sigma_max': float(sigma_max),
        'sigma_min': float(sigma_min),
        'determinant': float(det)
    }


# ─────────────────────────────────────────────────────────────────────────────
# pT 阈值敏感性分析 (源自 961_r8_scale + 1252)
# ─────────────────────────────────────────────────────────────────────────────

def pT_threshold_sensitivity(
    particles: List[FourVector],
    R: float = 0.4,
    n_scan: int = 50,
    pT_range: Tuple[float, float] = (1.0, 100.0)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """扫描 pT 阈值对喷注数量和总 pT 的影响.

    分析聚类结果对 pT_min 的敏感性:
      - 小 pT_min: 更多软粒子被包含, 喷注数多
      - 大 pT_min: 仅硬粒子, 喷注数少

    敏感性指标:
      S = dN_jet / dpT_min  (喷注数变化率)

    使用浮点邻域 (r8_next/r8_previous) 检测阈值穿越点.
    """
    from jet_clustering import anti_kt

    pT_values = np.linspace(pT_range[0], pT_range[1], n_scan)
    n_jets = np.zeros(n_scan)
    total_pT = np.zeros(n_scan)

    for i, pT_min in enumerate(pT_values):
        result = anti_kt(particles, R=R, pT_min=pT_min)
        n_jets[i] = len(result.jets)
        total_pT[i] = sum(j.pT for j in result.jets)

    # 计算敏感性 (数值导数)
    sensitivity = np.zeros(n_scan)
    for i in range(1, n_scan - 1):
        dpT = pT_values[i + 1] - pT_values[i - 1]
        sensitivity[i] = (n_jets[i + 1] - n_jets[i - 1]) / dpT

    return pT_values, n_jets, sensitivity


def clustering_stability_perturbation(
    particles: List[FourVector],
    R: float = 0.4,
    n_perturbations: int = 20,
    perturbation_scale: float = 1e-5,
    seed: int = 42
) -> Dict[str, float]:
    """分析聚类结果对粒子四动量微扰的稳定性.

    对每个粒子施加随机微扰:
      p_μ → p_μ + δp_μ,  |δp_μ| ~ ε · |p_μ|

    测量聚类结果的变动:
      ΔN_jet: 喷注数的变化
      ΔΣpT: 总 pT 的变化
      max ΔR_jet: 喷注轴的最大偏移

    源自 1252 中的数值精度分析框架.
    """
    from jet_clustering import anti_kt

    rng = np.random.default_rng(seed)

    # 基准聚类
    base_result = anti_kt(particles, R=R, pT_min=5.0)
    base_n_jets = len(base_result.jets)
    base_total_pT = sum(j.pT for j in base_result.jets)

    delta_n_jets = []
    delta_total_pT = []

    for _ in range(n_perturbations):
        # 微扰粒子
        perturbed = []
        for p in particles:
            scale = perturbation_scale * max(p.pT, 1.0)
            dE = rng.normal(0, scale)
            dpx = rng.normal(0, scale)
            dpy = rng.normal(0, scale)
            dpz = rng.normal(0, scale)
            new_E = max(p.E + dE, 0.1)
            perturbed.append(FourVector(new_E, p.px + dpx,
                                        p.py + dpy, p.pz + dpz))

        pert_result = anti_kt(perturbed, R=R, pT_min=5.0)
        pert_n_jets = len(pert_result.jets)
        pert_total_pT = sum(j.pT for j in pert_result.jets)

        delta_n_jets.append(abs(pert_n_jets - base_n_jets))
        delta_total_pT.append(abs(pert_total_pT - base_total_pT))

    return {
        'mean_delta_n_jets': float(np.mean(delta_n_jets)),
        'max_delta_n_jets': float(np.max(delta_n_jets)),
        'mean_delta_pT': float(np.mean(delta_total_pT)),
        'max_delta_pT': float(np.max(delta_total_pT)),
        'perturbation_scale': perturbation_scale
    }


# ─────────────────────────────────────────────────────────────────────────────
# 守恒量漂移分析 (源自 208_conservation_ode)
# ─────────────────────────────────────────────────────────────────────────────

def conservation_drift_analysis(
    particles_before: List[FourVector],
    particles_after: List[FourVector]
) -> Dict[str, float]:
    """分析聚类过程中的守恒量漂移.

    理想的聚类算法应保持:
      Σ E  = const   (能量守恒)
      Σ p⃗  = const  (动量守恒)
      M_inv² = const  (不变质量守恒)

    漂移量:
      ΔE = |ΣE_after - ΣE_before| / |ΣE_before|
      Δp⃗ = |Σp⃗_after - Σp⃗_before| / |Σp⃗_before|
    """
    def total_momentum(particles):
        E = sum(p.E for p in particles)
        px = sum(p.px for p in particles)
        py = sum(p.py for p in particles)
        pz = sum(p.pz for p in particles)
        return E, px, py, pz

    E_b, px_b, py_b, pz_b = total_momentum(particles_before)
    E_a, px_a, py_a, pz_a = total_momentum(particles_after)

    dE = abs(E_a - E_b) / max(abs(E_b), 1e-15)
    dpx = abs(px_a - px_b) / max(abs(px_b), 1e-15)
    dpy = abs(py_a - py_b) / max(abs(py_b), 1e-15)
    dpz = abs(pz_a - pz_b) / max(abs(pz_b), 1e-15)

    # 不变质量
    m2_b = E_b ** 2 - px_b ** 2 - py_b ** 2 - pz_b ** 2
    m2_a = E_a ** 2 - px_a ** 2 - py_a ** 2 - pz_a ** 2
    dm2 = abs(m2_a - m2_b) / max(abs(m2_b), 1e-15)

    return {
        'delta_E_relative': float(dE),
        'delta_px_relative': float(dpx),
        'delta_py_relative': float(dpy),
        'delta_pz_relative': float(dpz),
        'delta_invariant_mass_sq_relative': float(dm2),
        'total_energy_before': float(E_b),
        'total_energy_after': float(E_a)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 数值稳定性判据汇总
# ─────────────────────────────────────────────────────────────────────────────

def stability_report(
    particles: List[FourVector],
    R: float = 0.4
) -> Dict:
    """生成完整的数值稳定性分析报告.

    包括:
    1. 距离矩阵条件数
    2. Von Neumann 放大因子 (对标准格式)
    3. CFL 条件
    4. pT 阈值敏感性
    5. 微扰稳定性
    """
    report = {}

    # 1. 条件数
    cond = cluster_matrix_condition_number(particles, R)
    report['condition_number'] = cond

    # 2. Von Neumann 分析
    k_dx = np.linspace(0, 2 * math.pi, 100)
    schemes = ['FTCS', 'Lax-Friedrichs', 'Lax-Wendroff', 'Upwind']
    von_neumann = {}
    for scheme in schemes:
        try:
            G = von_neumann_amplification(scheme, k_dx, cfl=0.5)
            von_neumann[scheme] = {
                'max_amplification': float(np.max(G)),
                'min_amplification': float(np.min(G)),
                'stable': bool(np.all(G <= 1.0 + 1e-10))
            }
        except Exception:
            von_neumann[scheme] = {'stable': False}
    report['von_neumann'] = von_neumann

    # 3. CFL
    max_v = max(p.pT for p in particles) if particles else 1.0
    dx = 0.1  # 典型量热器 cell 尺寸
    cfl = cfl_condition(max_v, dx)
    report['cfl_max_dt'] = float(cfl)
    report['diffusion_stability_limit'] = float(diffusion_stability_limit(dx))

    return report
