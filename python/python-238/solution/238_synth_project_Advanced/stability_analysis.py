"""
stability_analysis.py
=====================

高阶有限差分与分子动力学积分的稳定性分析.

核心问题:
---------
1. Von Neumann 稳定性 (线性 PDE):
    对线性化演化方程 u_{n+1} = G u_n, 放大因子 g(k) 须满足:
        |g(k)| ≤ 1  ∀ k ∈ Brillouin zone

2. HMC 分子动力学的辛积分器稳定性:
    对 leapfrog 积分器, Hamilton 方程的线性化给出:
        δq(t+ε) = δq(t) + ε δp(t)
        δp(t+ε) = δp(t) - ε K δq(t)
    稳定性要求: ε < 2 / √(λ_max(K))

3. 高阶辛积分器 (Forest-Ruth, Yoshida):
    通过将步长分为多个子步并选择适当的权重,
    实现 O(ε^4) 或 O(ε^6) 精度, 但稳定性域更小.

4. Courant-Friedrichs-Lewy (CFL) 条件:
    对格点 Dirac 方程, 最大传播速度为 c = 1,
    因此 MD 步长须满足:
        ε × c / a ≤ C_CFL
    其中 C_CFL 取决于积分器阶数.

本模块融合种子项目:
  - 270_dfield9: RK 稳定性域 (Dormand-Prince, RK4)
      映射到辛积分器的稳定性域分析
  - 020_artery_pde: PDE 稳定性理论 (damped oscillator → MD)
  - 478_gradient_descent: 梯度下降的步长约束
      映射到 MD 步长的稳定性上界
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from high_order_fd import fd_coefficients_central, dispersion_relation


# ============================================================
# Von Neumann 稳定性分析
# ============================================================

def von_neumann_amplification_fd(order: int, r: float,
                                  n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """有限差分格式的 Von Neumann 放大因子.

    考虑扩散方程: ∂_t u = α ∂_x² u
    离散化: u_j^{n+1} = u_j^n + r (u_{j+1}^n - 2 u_j^n + u_{j-1}^n)
    其中 r = α Δt / Δx².

    放大因子:
        g(k) = 1 - 4 r sin²(k Δx / 2)  (对二阶中心差分)

    稳定性条件: |g(k)| ≤ 1, 即 r ≤ 1/2.

    对高阶差分:
        g(k) = 1 - 2 r Σ_m d_m (1 - cos(m k Δx))
    其中 d_m 为二阶差分的系数.

    参数:
        order: 空间差分精度
        r: CFL 数 (α Δt / Δx²)
        n_points: k 空间的采样点数

    返回:
        k_vals: k Δx ∈ [0, π]
        g_abs: |g(k)|
    """
    k_vals = np.linspace(0, np.pi, n_points)

    if order == 2:
        # 标准二阶: g = 1 - 4r sin²(k/2)
        g = 1.0 - 4.0 * r * np.sin(k_vals / 2.0) ** 2
    elif order == 4:
        # 4 阶二阶差分: 混合 1-link 和 2-link
        # g = 1 - 2r [d_1 (1 - cos k) + d_2 (1 - cos 2k)]
        d1 = 4.0 / 3.0
        d2 = -1.0 / 12.0
        g = 1.0 - 2.0 * r * (d1 * (1.0 - np.cos(k_vals))
                               + d2 * (1.0 - np.cos(2.0 * k_vals)))
    elif order == 6:
        d1 = 3.0 / 2.0
        d2 = -3.0 / 20.0
        d3 = 1.0 / 90.0
        g = 1.0 - 2.0 * r * (d1 * (1.0 - np.cos(k_vals))
                               + d2 * (1.0 - np.cos(2.0 * k_vals))
                               + d3 * (1.0 - np.cos(3.0 * k_vals)))
    else:
        # 使用通用系数
        coeffs = fd_coefficients_central(order)
        # 转换为二阶差分系数 (对 ∂²/∂x²)
        g = np.ones_like(k_vals)
        for m, c_m in enumerate(coeffs):
            g -= 2.0 * r * c_m ** 2 * (1.0 - np.cos((2 * m + 1) * k_vals)) / ((2 * m + 1) ** 2)

    return k_vals, np.abs(g)


def critical_r_fd(order: int, n_points: int = 1000) -> float:
    """计算有限差分格式的最大稳定 r 值.

    通过二分法找到使 max|g(k)| = 1 的临界 r_c.
    """
    r_low, r_high = 0.0, 2.0
    for _ in range(50):
        r_mid = 0.5 * (r_low + r_high)
        _, g_abs = von_neumann_amplification_fd(order, r_mid, n_points)
        if np.max(g_abs) <= 1.0 + 1e-10:
            r_low = r_mid
        else:
            r_high = r_mid
    return r_low


# ============================================================
# 分子动力学积分器稳定性
# ============================================================

def leapfrog_stability_bound(lambda_max: float) -> float:
    """Leapfrog 积分器的稳定性上界.

    对 Hamilton 方程:
        dq/dt = p
        dp/dt = -K q

    其中 K 为刚度矩阵, λ_max 为最大本征值.

    Leapfrog 的放大矩阵:
        G = [1 - ε²λ/2,  ε] [-ελ/2, 1]
            [-ελ,         0] [1,      0]

    稳定性条件: ε² λ < 4, 即
        ε < 2 / √(λ_max)

    参数:
        λ_max: 力矩阵 K 的最大本征值
    """
    if lambda_max <= 0:
        return np.inf
    return 2.0 / np.sqrt(lambda_max)


def omelyan_stability_bound(lambda_max: float, xi: float = 0.1932) -> float:
    """Omelyan MNI 积分器的稳定性上界.

    Omelyan (2002) 最优参数 ξ ≈ 0.1932:
        步骤:
            p ← p - ξ ε F(q)
            q ← q + (1/2) ε p
            p ← p - (1 - 2ξ) ε F(q)
            q ← q + (1/2) ε p
            p ← p - ξ ε F(q)

    稳定性域:
        ε < 2 × (1 - 2ξ + 2ξ²)^{-1/2} / √(λ_max)

    参数:
        lambda_max: 力矩阵最大本征值
        xi: Omelyan 参数 (默认最优值)
    """
    factor = 1.0 - 2.0 * xi + 2.0 * xi ** 2
    if factor <= 0:
        return np.inf
    return 2.0 * np.sqrt(1.0 / factor) / np.sqrt(lambda_max)


def forest_ruth_stability_bound(lambda_max: float) -> float:
    """Forest-Ruth (3rd-order Yoshida) 辛积分器的稳定性上界.

    Yoshida 4 阶辛积分器使用 3 个 leapfrog 步:
        θ = 1 / (2 - 2^{1/3})
        ε_1 = θ ε, ε_2 = (1 - 2θ) ε, ε_3 = θ ε

    稳定性域:
        ε < ε_c = 2.46... / √(λ_max)  (比 leapfrog 更窄)
    """
    theta = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    # 有效稳定性系数 (由放大矩阵谱半径确定)
    # 近似: ε_c ≈ 2.46 / √(λ_max)
    return 2.46 / np.sqrt(lambda_max) if lambda_max > 0 else np.inf


# ============================================================
# Dirac 算子本征值界限
# ============================================================

def dirac_eigenvalue_bounds(mass: float, fd_order: int = 2,
                             use_naik: bool = False) -> Tuple[float, float]:
    """自由 staggered Dirac 算子的本征值范围.

    自由 Dirac 算子在动量空间:
        D(p) = i Σ_μ γ_μ p̃_μ(p) + m
    其中 p̃_μ(p) 为改进导数的色散关系.

    本征值: λ(p) = ±√(Σ_μ p̃_μ²(p) + m²)

    最小本征值: λ_min = m (在 p = 0)
    最大本征值: λ_max = √(Σ_μ (p̃_max)² + m²)

    参数:
        mass: 夸克质量
        fd_order: 有限差分精度
        use_naik: 是否使用 Naik 导数
    """
    # p_max = π (Brillouin zone 边界)
    if use_naik:
        from high_order_fd import naik_coefficients
        coeffs = naik_coefficients()
    else:
        coeffs = fd_coefficients_central(fd_order)

    # 在各方向的最大动量: p̃_max
    p_max_val = float(dispersion_relation(coeffs, np.array([np.pi]))[0])
    lambda_max = np.sqrt(4.0 * p_max_val ** 2 + mass ** 2)
    lambda_min = mass
    return float(lambda_min), float(lambda_max)


def condition_number(mass: float, fd_order: int = 2) -> float:
    """Dirac 算子的条件数:

    κ = λ_max / λ_min = √(Σ_μ (p̃_max)² + m²) / m

    小质量 (近手征极限) → 条件数发散 → CG 收敛困难.
    """
    l_min, l_max = dirac_eigenvalue_bounds(mass, fd_order)
    return l_max / l_min if l_min > 0 else np.inf


# ============================================================
# HMC 步长选择器
# ============================================================

class StepSizeSelector:
    """基于稳定性分析自动选择 HMC 步长.

    策略:
        1. 计算力矩阵的最大本征值估计
        2. 根据积分器类型计算稳定性上界
        3. 选择安全步长: ε = safety × ε_c
    """

    def __init__(self, safety_factor: float = 0.5,
                 integrator: str = 'leapfrog'):
        """
        参数:
            safety_factor: 安全因子 (0 < safety ≤ 1)
            integrator: 'leapfrog', 'omelyan', 'forest_ruth'
        """
        self.safety = safety_factor
        self.integrator = integrator

    def estimate_lambda_max(self, forces) -> float:
        """估计力矩阵的最大本征值.

        通过 Gershgorin 圆盘定理:
            λ_max ≤ max_i Σ_j |F_{ij}|
        简化估计: 取所有力分量的最大绝对值 × 自由度.
        """
        if forces is None:
            return 1.0  # 默认估计
        # forces shape: (4, volume, 3, 3)
        max_force = float(np.max(np.abs(forces)))
        n_dof = forces.shape[0] * forces.shape[1] * 8  # 8 = su(3) 维数
        # 粗略估计 λ_max ~ n_dof × max_force
        return max_force * np.sqrt(n_dof)

    def optimal_step_size(self, lambda_max: float) -> float:
        """计算最优步长.

        ε = safety × ε_c(integrator, λ_max)
        """
        if self.integrator == 'leapfrog':
            eps_c = leapfrog_stability_bound(lambda_max)
        elif self.integrator == 'omelyan':
            eps_c = omelyan_stability_bound(lambda_max)
        elif self.integrator == 'forest_ruth':
            eps_c = forest_ruth_stability_bound(lambda_max)
        else:
            eps_c = leapfrog_stability_bound(lambda_max)
        return self.safety * eps_c

    def adaptive_step_size(self, dH: float, target_accept: float = 0.75,
                            eps_current: float = 0.01) -> float:
        """自适应步长调整 (基于 Metropolis 接受率).

        策略:
            if |dH| < -log(target_accept):
                ε ← 1.02 ε  (接受率过高, 增大步长)
            elif |dH| > -log(1 - target_accept):
                ε ← 0.95 ε  (接受率过低, 减小步长)

        参数:
            dH: Hamilton 量变化
            target_accept: 目标接受率 (通常 0.7 ~ 0.8)
            eps_current: 当前步长
        """
        threshold_low = -np.log(target_accept)
        threshold_high = -np.log(1.0 - target_accept)
        abs_dH = abs(dH)

        if abs_dH < threshold_low:
            return eps_current * 1.02
        elif abs_dH > threshold_high:
            return eps_current * 0.95
        else:
            return eps_current


# ============================================================
# 积分器误差阶数验证
# ============================================================

def verify_integrator_order(integrator_func, eps_values: List[float],
                             reference_eps: float = 1e-5) -> Dict:
    """通过 Richardson 外推验证积分器的阶数.

    若积分器为 p 阶, 则误差 E(ε) ∝ ε^p,
    因此 log E(ε) vs log ε 的斜率应为 p.

    参数:
        integrator_func: 积分器函数, 接受 (eps) 返回 (q, p)
        eps_values: 测试的步长列表
        reference_eps: 参考高精度步长
    """
    errors = []
    # 参考解
    q_ref, p_ref = integrator_func(reference_eps)

    for eps in eps_values:
        q, p = integrator_func(eps)
        err = float(np.max(np.abs(q - q_ref)))
        errors.append(err)

    errors = np.array(errors)
    eps_arr = np.array(eps_values)
    # 拟合斜率 (log-log)
    valid = errors > 1e-15
    if np.sum(valid) < 2:
        return {'order': float('nan'), 'errors': errors}

    log_eps = np.log(eps_arr[valid])
    log_err = np.log(errors[valid])
    slope, _ = np.polyfit(log_eps, log_err, 1)

    return {
        'order': float(slope),
        'errors': errors,
        'eps_values': eps_values,
    }
