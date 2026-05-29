r"""
monte_carlo_generator.py
蒙特卡洛事件产生模块

本模块实现暗物质直接探测实验的完整蒙特卡洛模拟链：
1. 随机事件采样（参考 clock_solitaire_simulation 的随机状态机思想）
2. 背景事件模拟（参考 craps_simulation 的概率分支模型）
3. WIMP 散射事件生成
4. 探测器响应模拟（能量分辨率、阈值效应）

核心算法：

A. WIMP 事件生成（拒绝采样）：
    1. 从微分率 dR/dE 生成能量样本
    2. 对每个能量，随机抽取相互作用位置（均匀分布）
    3. 随机抽取时间（考虑年度调制）
    4. 应用探测器效率与能量分辨率

B. 背景事件生成：
    模拟多种背景来源：
    - 环境 γ 射线（指数谱）
    - 中子散射（均匀分布）
    - 表面 β 衰变（高斯峰）
    每种背景有其独立的产生率和能谱。

C. 随机数引擎：
    采用 Park-Miller LCG（参考 968_r85 中的 r8_uniform_01），
    保证跨平台可复现性。

物理公式：
    能量分辨率（高斯展宽）：
        \sigma_E = \sqrt{\sigma_{\rm elec}^2 + \sigma_{\rm noise}^2}
        \sigma_{\rm elec} = \sqrt{F \cdot N_e} \cdot \varepsilon_e

    探测效率（阈值 + 饱和）：
        \epsilon(E) = \frac{1}{2} \left[ 1 + \operatorname{erf}\left(
            \frac{E - E_{\rm th}}{\sqrt{2} \sigma_E}
        \right) \right] \times
        \left[ 1 - \operatorname{erf}\left(
            \frac{E - E_{\rm sat}}{\sqrt{2} \sigma_E}
        \right) \right]

参考文献：
- Press, W. H., et al. (2007). Numerical Recipes, 3rd ed.
- Lewin, J. D., & Smith, P. F. (1996). Astroparticle Physics, 6, 87.
"""

import numpy as np
from typing import List, Tuple, Dict
from utils import r8_uniform_01, erf_approx
from wimp_physics import differential_rate, annual_modulated_rate, total_events_in_range


# ============================================================================
# 可复现随机数生成器（Park-Miller LCG）
# ============================================================================

class ReproducibleRNG:
    """
    基于 Park-Miller LCG 的可复现伪随机数生成器。
    """

    def __init__(self, seed: int = 123456789):
        self.seed = int(seed) % 2147483647
        if self.seed == 0:
            self.seed = 1

    def uniform(self) -> float:
        """生成 (0, 1) 均匀分布随机数。"""
        r, self.seed = r8_uniform_01(self.seed)
        return r

    def randn(self) -> float:
        """使用 Box-Muller 变换生成标准正态分布随机数。"""
        u1 = self.uniform()
        u2 = self.uniform()
        while u1 <= 1.0e-15:
            u1 = self.uniform()
        return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

    def choice(self, weights: np.ndarray) -> int:
        """按权重随机选择索引。"""
        weights = np.asarray(weights, dtype=float)
        if np.sum(weights) <= 0.0:
            return 0
        cdf = np.cumsum(weights / np.sum(weights))
        u = self.uniform()
        return int(np.searchsorted(cdf, u))

    def exponential(self, scale: float) -> float:
        """指数分布随机数。"""
        u = self.uniform()
        while u <= 1.0e-15:
            u = self.uniform()
        return -scale * np.log(u)


# ============================================================================
# 探测器响应模型
# ============================================================================

def detection_efficiency(
    er_kev: float,
    threshold_kev: float = 0.5,
    saturation_kev: float = 100.0,
    sigma_e_kev: float = 0.1,
) -> float:
    """
    计算探测效率（S 曲线 + 饱和截止）。

    公式：
        \epsilon(E) = \epsilon_{\rm th}(E) \cdot \epsilon_{\rm sat}(E)

    其中：
        \epsilon_{\rm th}(E) = \frac{1}{2} \left[ 1 + \operatorname{erf}
            \left( \frac{E - E_{\rm th}}{\sqrt{2} \sigma_E} \right) \right]
        \epsilon_{\rm sat}(E) = \frac{1}{2} \left[ 1 + \operatorname{erf}
            \left( \frac{E_{\rm sat} - E}{\sqrt{2} \sigma_E} \right) \right]
    """
    if er_kev <= 0.0:
        return 0.0
    arg_th = (er_kev - threshold_kev) / (np.sqrt(2.0) * sigma_e_kev)
    arg_sat = (saturation_kev - er_kev) / (np.sqrt(2.0) * sigma_e_kev)
    eps_th = 0.5 * (1.0 + erf_approx(arg_th))
    eps_sat = 0.5 * (1.0 + erf_approx(arg_sat))
    return eps_th * eps_sat


def apply_energy_resolution(
    er_true_kev: float,
    fano_factor: float,
    epsilon_eV: float,
    rng: ReproducibleRNG,
) -> float:
    """
    对真实能量施加高斯展宽（Fano 噪声）。

    公式：
        \sigma_E = \frac{\sqrt{F \cdot N_e} \cdot \varepsilon}{1000}
        N_e = E_R / \varepsilon  （简化为理想情况）
        E_{\rm obs} = E_{\rm true} + \mathcal{N}(0, \sigma_E^2)
    """
    if er_true_kev <= 0.0:
        return 0.0
    N_e = er_true_kev * 1000.0 / epsilon_eV
    sigma_e_kev = np.sqrt(fano_factor * N_e) * epsilon_eV / 1000.0
    if sigma_e_kev < 1.0e-6:
        return er_true_kev
    noise = rng.randn() * sigma_e_kev
    return max(er_true_kev + noise, 0.0)


# ============================================================================
# WIMP 事件产生
# ============================================================================

def generate_wimp_events(
    n_events_target: int,
    m_chi_gev: float,
    sigma_pb: float,
    a_mass: float,
    target_mass_kg: float,
    exposure_days: float,
    e_min_kev: float,
    e_max_kev: float,
    detector_radius_m: float,
    detector_thickness_m: float,
    rng: ReproducibleRNG,
    apply_modulation: bool = True,
) -> List[Dict]:
    """
    使用拒绝采样生成 WIMP 散射事件列表。

    算法：
        1. 在能量区间 [e_min, e_max] 上计算最大微分率 R_max
        2. 均匀随机采样能量 E ~ U(e_min, e_max)
        3. 均匀随机采样 u ~ U(0, R_max)
        4. 若 u < dR/dE(E)，接受该能量
        5. 随机采样空间位置和时间
        6. 应用探测效率与能量分辨率

    参数：
        n_events_target: 目标事件数
        m_chi_gev: WIMP 质量
        sigma_pb: 截面
        a_mass: 靶核质量数
        target_mass_kg: 探测器质量
        exposure_days: 曝光时间
        e_min_kev, e_max_kev: 能量范围
        detector_radius_m: 探测器半径
        detector_thickness_m: 探测器厚度
        rng: 随机数生成器
        apply_modulation: 是否应用年度调制

    返回：
        events: 事件字典列表，每项包含 'type', 'energy_true', 'energy_obs',
                'x', 'y', 'z', 'time_day'
    """
    events = []
    if n_events_target <= 0:
        return events

    # 预计算最大微分率（网格扫描）
    n_scan = 200
    e_scan = np.linspace(e_min_kev, e_max_kev, n_scan)
    rates = np.array([
        differential_rate(e, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)
        for e in e_scan
    ])
    max_rate = float(np.max(rates)) * 1.2  # 留余量
    if max_rate <= 0.0:
        return events

    max_attempts = n_events_target * 10000
    attempts = 0

    while len(events) < n_events_target and attempts < max_attempts:
        attempts += 1
        # 均匀采样能量
        e_trial = e_min_kev + (e_max_kev - e_min_kev) * rng.uniform()
        # 计算微分率
        rate = differential_rate(e_trial, m_chi_gev, sigma_pb, a_mass, target_mass_kg, exposure_days)
        # 拒绝采样
        if rate > max_rate:
            max_rate = rate * 1.2
            continue
        u = rng.uniform() * max_rate
        if u > rate:
            continue

        # 年度调制时间采样
        if apply_modulation:
            # 根据调制因子加权采样时间
            t_trial = 365.25 * rng.uniform()
            mod_factor = 1.0 + 0.05 * np.cos(2.0 * np.pi * (t_trial - 152.0) / 365.25)
            if rng.uniform() > mod_factor / 1.06:  # 1.06 ≈ 1 + 0.05（最大调制因子）
                continue
        else:
            t_trial = 365.25 * rng.uniform()

        # 空间位置（均匀圆柱体）
        r_pos = detector_radius_m * np.sqrt(rng.uniform())
        theta = 2.0 * np.pi * rng.uniform()
        x = r_pos * np.cos(theta)
        y = r_pos * np.sin(theta)
        z = detector_thickness_m * rng.uniform()

        # 探测效率
        eps = detection_efficiency(e_trial)
        if rng.uniform() > eps:
            continue

        # 能量分辨率展宽
        e_obs = apply_energy_resolution(e_trial, fano_factor=0.15, epsilon_eV=3.0, rng=rng)
        if e_obs < e_min_kev:
            continue

        events.append({
            "type": "WIMP",
            "energy_true": float(e_trial),
            "energy_obs": float(e_obs),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "time_day": float(t_trial),
        })

    return events


# ============================================================================
# 背景事件产生（多源背景模型）
# ============================================================================

def generate_background_events(
    n_events_target: int,
    e_min_kev: float,
    e_max_kev: float,
    detector_radius_m: float,
    detector_thickness_m: float,
    rng: ReproducibleRNG,
    gamma_rate_per_day: float = 5.0,
    neutron_rate_per_day: float = 0.5,
    beta_rate_per_day: float = 2.0,
) -> List[Dict]:
    """
    生成多种背景来源的模拟事件。

    背景模型：
        1. γ 射线：指数谱 dN/dE ∝ exp(-E/E_0)，E_0 = 10 keV
        2. 中子散射：平坦能谱 + 特征峰（~30 keV）
        3. 表面 β：高斯峰 E ≈ 10 keV，σ ≈ 2 keV

    参数：
        n_events_target: 目标事件数
        e_min_kev, e_max_kev: 能量范围
        detector_radius_m, detector_thickness_m: 几何参数
        rng: 随机数生成器
        gamma_rate_per_day: γ 产生率 [events/day]
        neutron_rate_per_day: 中子产生率
        beta_rate_per_day: β 产生率

    返回：
        events: 事件字典列表
    """
    events = []
    if n_events_target <= 0:
        return events

    # 背景类型权重
    rates = np.array([gamma_rate_per_day, neutron_rate_per_day, beta_rate_per_day])
    labels = ["gamma", "neutron", "beta"]

    max_attempts = n_events_target * 10000
    attempts = 0

    while len(events) < n_events_target and attempts < max_attempts:
        attempts += 1
        bg_type = labels[rng.choice(rates)]

        if bg_type == "gamma":
            # 指数谱，拒绝采样
            E0 = 10.0
            e_trial = e_min_kev + (e_max_kev - e_min_kev) * rng.uniform()
            prob = np.exp(-e_trial / E0)
            if rng.uniform() > prob:
                continue
        elif bg_type == "neutron":
            # 80% 平坦 + 20% 30 keV 高斯峰
            if rng.uniform() < 0.8:
                e_trial = e_min_kev + (e_max_kev - e_min_kev) * rng.uniform()
            else:
                e_trial = 30.0 + 5.0 * rng.randn()
                if e_trial < e_min_kev or e_trial > e_max_kev:
                    continue
        else:  # beta
            e_trial = 10.0 + 2.0 * rng.randn()
            if e_trial < e_min_kev or e_trial > e_max_kev:
                continue

        # 探测效率
        eps = detection_efficiency(e_trial)
        if rng.uniform() > eps:
            continue

        # 能量分辨率
        e_obs = apply_energy_resolution(e_trial, fano_factor=0.15, epsilon_eV=3.0, rng=rng)
        if e_obs < e_min_kev:
            continue

        # 空间位置
        r_pos = detector_radius_m * np.sqrt(rng.uniform())
        theta = 2.0 * np.pi * rng.uniform()
        x = r_pos * np.cos(theta)
        y = r_pos * np.sin(theta)
        z = detector_thickness_m * rng.uniform()
        t_trial = 365.25 * rng.uniform()

        events.append({
            "type": bg_type,
            "energy_true": float(e_trial),
            "energy_obs": float(e_obs),
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "time_day": float(t_trial),
        })

    return events


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    rng = ReproducibleRNG(seed=42)

    # 测试探测效率
    assert detection_efficiency(0.1) < detection_efficiency(10.0)
    assert 0.0 <= detection_efficiency(50.0) <= 1.0

    # 测试能量分辨率
    e_obs = apply_energy_resolution(10.0, 0.15, 3.0, rng)
    assert e_obs >= 0.0

    # 测试 WIMP 事件生成
    events = generate_wimp_events(
        50, m_chi_gev=50.0, sigma_pb=1.0, a_mass=73.0,
        target_mass_kg=10.0, exposure_days=365.0,
        e_min_kev=0.5, e_max_kev=50.0,
        detector_radius_m=0.05, detector_thickness_m=0.02,
        rng=rng,
    )
    assert len(events) > 0, "WIMP 事件生成失败"
    for ev in events:
        assert ev["type"] == "WIMP"
        assert e_min_kev <= ev["energy_obs"] <= e_max_kev * 2.0

    # 测试背景事件
    bg = generate_background_events(
        50, 0.5, 50.0, 0.05, 0.02, rng,
    )
    assert len(bg) > 0

    print("monte_carlo_generator.py: 所有自测通过")
