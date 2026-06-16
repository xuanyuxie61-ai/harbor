#!/usr/bin/env python3
"""
symmetry_decomposition.py
=========================
ICF 内爆对称性分解模块。

物理背景:
  ICF 内爆的理想状态是完美球对称 (1D 径向内爆)。
  实际中, 由于驱动不对称、靶丸缺陷、激光功率失衡等因素,
  内爆会产生非对称变形。

对称性度量:
  将内爆界面 (热斑边界) 的形状展开为球谐函数:
    R(θ, φ) = R₀ [1 + Σ_{l,m} a_{lm} Y_{lm}(θ, φ)]
  在轴对称近似下, 仅需 Legendre 多项式:
    R(θ) = R₀ [1 + Σ_l a_l P_l(cos θ)]

  关键模式:
    l=0: 平均半径 (内爆压缩度)
    l=1: 偶极 (质心偏移, 可通过参考系选择消除)
    l=2: 椭球不对称 (最关键的对称性模式)
    l=4: 四极不对称 (激光束排列引起)
    l=6: 六极不对称

  对称性要求 (NIF 标准):
    |a_l/a_0| < 1%  for l ≥ 2
    |v_impulse asymmetry| < 2%

Growth 因子:
  Rayleigh-Taylor 不稳定性增长:
    a_l(t) = a_l(0) × exp(∫ γ_{RT}(t) dt)
  其中 γ_{RT} = √(A k a_{imp} - (l(l+1)σ/(ρR³)))
    A = (ρ_h - ρ_l)/(ρ_h + ρ_l) 为 Atwood 数
    k = l/R 为波数
    σ 为表面张力
"""

import numpy as np
from scipy.special import legendre


class LegendreDecomposition:
    """
    Legendre 多项式分解。
    将 ICF 内爆界面的角分布分解为 P_l 模式。
    """

    @staticmethod
    def decompose(R_theta, theta, max_mode=8):
        """
        对 R(θ) 进行 Legendre 分解。

        a_l = (2l+1)/2 ∫₀^π R(θ) P_l(cos θ) sin θ dθ

        参数:
            R_theta: R(θ) 的值, shape=(N_theta,)
            theta: 极角, shape=(N_theta,), ∈ [0, π]
            max_mode: 最大展开阶数
        返回:
            modes: dict {l: a_l}
        """
        R = np.asarray(R_theta, dtype=np.float64)
        mu = np.cos(theta)
        sin_theta = np.sin(theta)
        dmu = np.gradient(mu)

        modes = {}
        for l in range(max_mode + 1):
            # P_l(cos θ)
            P_l = legendre(l)(mu)
            # 积分: a_l = (2l+1)/2 ∫ R(θ) P_l(cos θ) sin θ dθ
            integrand = R * P_l * sin_theta
            a_l = (2 * l + 1) / 2.0 * np.sum(integrand * np.abs(dmu))
            modes[l] = a_l

        return modes

    @staticmethod
    def reconstruct(modes, theta):
        """
        由 Legendre 系数重构 R(θ)。
        R(θ) = Σ_l a_l P_l(cos θ)
        """
        mu = np.cos(theta)
        R = np.zeros_like(mu)
        for l, a_l in modes.items():
            R += a_l * legendre(l)(mu)
        return R

    @staticmethod
    def symmetry_metrics(modes):
        """
        计算对称性度量指标。

        返回:
            dict: {
                'R0': 平均半径 (a_0),
                'P2_asymmetry': |a_2/a_0|,
                'P4_asymmetry': |a_4/a_0|,
                'total_asymmetry': sqrt(Σ_{l≥2} a_l²) / a_0,
                'mode_spectrum': {l: |a_l/a_0|}
            }
        """
        a0 = abs(modes.get(0, 1.0))
        if a0 < 1e-30:
            a0 = 1.0

        metrics = {
            'R0': modes.get(0, 0.0),
            'a0': a0,
            'P2_asymmetry': abs(modes.get(2, 0.0)) / a0,
            'P4_asymmetry': abs(modes.get(4, 0.0)) / a0,
            'P6_asymmetry': abs(modes.get(6, 0.0)) / a0,
        }

        # 总不对称度
        total_sq = sum(a_l ** 2 for l, a_l in modes.items() if l >= 2)
        metrics['total_asymmetry'] = np.sqrt(total_sq) / a0

        # 各模式相对幅度
        metrics['mode_spectrum'] = {l: abs(a_l) / a0 for l, a_l in modes.items()}

        return metrics


class RTGrowthModel:
    """
    Rayleigh-Taylor 不稳定性增长模型。

    RT 不稳定性: 当重流体在轻流体上方, 或界面被减速时,
    小扰动会指数增长。

    线性增长阶段:
      a_l(t) = a_l(0) × exp(γ t)
    其中增长率为 (Bellan 2012):
      γ = √(A k g_eff - (l(l+1) σ)/(ρ R³))

    A = Atwood 数 = (ρ_h - ρ_l) / (ρ_h + ρ_l)
    k = l / R = 波数
    g_eff = 有效加速度 (减速时为负)
    σ = 表面张力

    饱和振幅:
      a_l^{sat} ≈ C / k = C R / l  (C ≈ 1~3)
    """

    @staticmethod
    def growth_rate(l, R, A, g_eff, sigma=0.0, rho_h=1.0):
        """
        计算 RT 增长率 γ。

        参数:
            l: 模式阶数
            R: 界面半径 [cm]
            A: Atwood 数
            g_eff: 有效加速度 [cm/s²] (减速时为负值)
            sigma: 表面张力 [dyn/cm]
            rho_h: 重流体密度 [g/cm³]
        返回:
            gamma: 增长率 [1/s]
        """
        if R < 1e-30 or l < 1:
            return 0.0

        k = l / R  # 波数

        # 驱动项: A k g_eff
        # 注意: g_eff 为减速时 (指向轻流体), RT 不稳定
        driving = A * k * abs(g_eff)

        # 稳定项: 表面张力
        stabilizing = l * (l + 1) * sigma / (rho_h * R ** 3) if sigma > 0 else 0.0

        gamma_sq = driving - stabilizing
        if gamma_sq > 0:
            return np.sqrt(gamma_sq)
        else:
            return 0.0  # 稳定

    @staticmethod
    def growth_factor(l, R_history, A_history, g_history, dt_history, sigma=0.0, rho_h=1.0):
        """
        计算随时间累积的 RT 增长因子。
        G_l = exp(∫ γ_l(t) dt)

        参数:
            R_history: 半径历史 [cm]
            A_history: Atwood 数历史
            g_history: 加速度历史 [cm/s²]
            dt_history: 时间步长历史 [s]
        返回:
            G_l: 总增长因子
        """
        integral = 0.0
        for i in range(len(R_history)):
            gamma = RTGrowthModel.growth_rate(
                l, R_history[i], A_history[i], g_history[i], sigma, rho_h
            )
            integral += gamma * dt_history[i]
        return np.exp(integral)

    @staticmethod
    def mode_amplitude_evolution(a0_l, l, R_history, A_history, g_history, dt_history,
                                  sigma=0.0, rho_h=1.0):
        """
        计算扰动振幅的时间演化。
        a_l(t) = a_l(0) × G_l(t)

        当 a_l 达到饱和振幅时停止增长:
          a_l^{sat} ≈ R / l × C_sat
        """
        C_sat = 2.0  # 饱和系数
        a_l = a0_l
        a_history = [a_l]

        for i in range(len(R_history)):
            gamma = RTGrowthModel.growth_rate(
                l, R_history[i], A_history[i], g_history[i], sigma, rho_h
            )
            a_l *= np.exp(gamma * dt_history[i])

            # 饱和限制
            a_sat = C_sat * R_history[i] / max(l, 1)
            a_l = min(a_l, a_sat)
            a_history.append(a_l)

        return np.array(a_history)


class ImplosionSymmetryAnalyzer:
    """
    ICF 内爆对称性综合分析器。
    """

    @staticmethod
    def analyze_implosion_symmetry(R_theta_data, theta_pts, time_stamps=None):
        """
        对内爆界面进行完整的对称性分析。

        参数:
            R_theta_data: shape=(N_time, N_theta) 或 (N_theta,)
            theta_pts: 极角数组
            time_stamps: 时间戳 (可选)
        返回:
            dict: 完整分析结果
        """
        R_data = np.atleast_2d(R_theta_data)
        N_time = R_data.shape[0]
        N_theta = R_data.shape[1] if R_data.ndim > 1 else len(R_data)

        if R_data.ndim == 1:
            R_data = R_data.reshape(1, -1)
            N_time = 1

        ld = LegendreDecomposition()
        results = {
            'modes_history': [],
            'metrics_history': [],
            'time': time_stamps if time_stamps is not None else list(range(N_time))
        }

        for t_idx in range(N_time):
            modes = ld.decompose(R_data[t_idx], theta_pts, max_mode=8)
            metrics = ld.symmetry_metrics(modes)
            results['modes_history'].append(modes)
            results['metrics_history'].append(metrics)

        return results

    @staticmethod
    def assess_symmetry_quality(metrics, threshold=0.01):
        """
        评估内爆对称性质量。
        NIF 标准: |a_l/a_0| < 1% for l ≥ 2

        返回:
            dict: {pass: bool, details: str, violations: list}
        """
        last = metrics[-1] if isinstance(metrics, list) else metrics
        violations = []

        for l in [2, 4, 6]:
            key = f'P{l}_asymmetry'
            val = last.get(key, 0.0)
            if val > threshold:
                violations.append(f"P_{l} = {val:.4f} > {threshold}")

        passed = len(violations) == 0
        details = "PASS" if passed else f"FAIL: {len(violations)} violations"
        return {
            'pass': passed,
            'details': details,
            'violations': violations,
            'P2': last.get('P2_asymmetry', 0.0),
            'P4': last.get('P4_asymmetry', 0.0),
            'P6': last.get('P6_asymmetry', 0.0),
            'total': last.get('total_asymmetry', 0.0)
        }
