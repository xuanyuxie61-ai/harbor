"""
monte_carlo.py — Monte Carlo 相空间积分
=========================================
种子项目映射:
  501_hand_area (MC面积) → 相空间体积的MC计算
  1000_dasayan05 (Brownian运动) → 随机游动模拟

物理: 使用Monte Carlo方法计算多体相空间积分.
"""
import numpy as np
import math


def mc_integrate_2body(s, m1, m2, n_samples=10000, seed=42):
    """
    二体相空间 Monte Carlo 积分.
    Φ₂(s; m1, m2) = π/(2s) * sqrt(λ(s, m1², m2²))

    λ(a,b,c) = a²+b²+c²-2ab-2ac-2bc (Källén函数)
    """
    if s < (m1 + m2)**2:
        return 0.0, 0.0

    lam = s**2 + m1**4 + m2**4 - 2*s*m1**2 - 2*s*m2**2 - 2*m1**2*m2**2
    if lam < 0:
        return 0.0, 0.0

    # 解析结果
    phi2_exact = math.pi / (2.0 * s) * math.sqrt(lam)
    return phi2_exact, 0.0  # 无MC误差


def mc_integrate_3body(s, m1, m2, m3, n_samples=50000, seed=42):
    """
    三体相空间 Monte Carlo 积分.
    种子项目 501_hand_area 映射: 在Dalitz图中采样.
    """
    if s < (m1 + m2 + m3)**2:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    sqrt_s = math.sqrt(s)

    # Dalitz图边界
    E1_max = (s + m1**2 - (m2 + m3)**2) / (2 * sqrt_s)
    E2_max = (s + m2**2 - (m1 + m3)**2) / (2 * sqrt_s)
    E1_min, E2_min = m1, m2

    vol_box = (E1_max - E1_min) * (E2_max - E2_min)

    hits = 0
    weights = []

    for _ in range(n_samples):
        E1 = rng.uniform(E1_min, E1_max)
        E2 = rng.uniform(E2_min, E2_max)
        E3 = sqrt_s - E1 - E2

        if E3 < m3:
            continue

        p1 = math.sqrt(max(E1**2 - m1**2, 0))
        p2 = math.sqrt(max(E2**2 - m2**2, 0))
        p3 = math.sqrt(max(E3**2 - m3**2, 0))

        # cosθ12 范围
        m12_sq = s + m1**2 + m2**2 - 2*sqrt_s*(E1+E2) + 2*E1*E2
        if m12_sq < (m1+m2)**2:
            continue

        # 简化: 假设全相空间
        hits += 1
        weights.append(1.0)

    if hits == 0:
        return 0.0, 0.0

    vol = vol_box * hits / n_samples
    err = vol / math.sqrt(max(hits, 1))
    return vol, err


def mc_cross_section(sqrt_s, M_min, M_max, n_events=10000, seed=42):
    """
    蒙特卡洛截面积分.
    σ = ∫ dM² dcosθ |M|² * (相空间因子)
    """
    rng = np.random.RandomState(seed)

    total_weight = 0.0
    total_weight_sq = 0.0

    for _ in range(n_events):
        M = rng.uniform(M_min, M_max)
        cos_theta = rng.uniform(-1, 1)

        # 简化矩阵元: |M|² ∝ 1/((M²-mZ²)² + mZ²ΓZ²)
        mZ, GammaZ = 91.2, 2.5
        amp_sq = 1.0 / ((M**2 - mZ**2)**2 + mZ**2 * GammaZ**2)

        # 角分布: 1 + cos²θ
        angular = 1.0 + cos_theta**2

        w = amp_sq * angular
        total_weight += w
        total_weight_sq += w**2

    mean_w = total_weight / n_events
    var_w = total_weight_sq / n_events - mean_w**2

    # 积分体积
    vol = (M_max**2 - M_min**2) * 2.0  # dM² * dcosθ
    sigma = vol * mean_w
    sigma_err = vol * math.sqrt(max(var_w, 0) / n_events)

    return sigma, sigma_err


def importance_sampling(func, proposal_sampler, proposal_pdf, n_samples=10000, seed=42):
    """
    重要性采样:
      I = ∫ f(x) dx ≈ (1/N) Σ f(x_i) / q(x_i)
    其中 x_i ~ q(x).
    """
    rng = np.random.RandomState(seed)
    total = 0.0
    total_sq = 0.0

    for _ in range(n_samples):
        x = proposal_sampler(rng)
        q = proposal_pdf(x)
        if q > 1e-30:
            w = func(x) / q
            total += w
            total_sq += w**2

    mean = total / n_samples
    var = total_sq / n_samples - mean**2
    return mean, math.sqrt(max(var, 0) / n_samples)
