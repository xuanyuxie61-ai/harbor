"""
event_generator.py — 对撞事件生成器
=====================================
种子项目映射:
  1000_dasayan05 (随机过程/混沌) → 部分子簇射的随机游动模拟
  501_hand_area (Monte Carlo面积) → 事件Acceptance计算

物理: 模拟 pp→Z'/γ*→ℓ⁺ℓ⁻ (Drell-Yan过程)
"""
import numpy as np
from special_functions import breit_wigner, alpha_s_running


class PartonDistribution:
    """简化部分子分布函数 (PDF)."""

    def __init__(self, Q=91.2):
        self.Q = Q
        self.alpha_s = alpha_s_running(Q)

    def xf(self, flavor, x):
        """xf(x, Q) 简化参数化."""
        if x <= 0 or x >= 1:
            return 0.0
        if flavor in ('u', 'uv'):
            return 3.0 * x**0.5 * (1 - x)**3
        elif flavor in ('d', 'dv'):
            return 1.5 * x**0.5 * (1 - x)**4
        elif flavor == 'gluon':
            return 5.0 * x**0.3 * (1 - x)**5
        elif flavor in ('s', 'c', 'b'):
            return 0.5 * x**0.3 * (1 - x)**6
        else:
            return 0.1 * x**0.5 * (1 - x)**5

    def parton_luminosity(self, M, flavor='u'):
        """部分子光度: dL/dM² = Σ ∫ dx1 dx2 f1(x1)f2(x2) δ(x1x2s - M²)."""
        tau = M**2 / (13000.0**2)  # √s = 13 TeV
        if tau >= 1:
            return 0.0
        x = math.sqrt(tau)
        f1 = self.xf(flavor, x)
        f2 = self.xf(flavor, x)
        return f1 * f2 / (2.0 * x * 13000.0**2) if x > 0 else 0.0


import math


class EventGenerator:
    """Drell-Yan 事件生成器."""

    def __init__(self, sqrt_s=13000.0, mZ=91.2, GammaZ=2.5, seed=42):
        self.sqrt_s = sqrt_s
        self.mZ = mZ
        self.GammaZ = GammaZ
        self.rng = np.random.RandomState(seed)
        self.pdf = PartonDistribution()

    def generate_drell_yan(self, n_events=100, M_min=60, M_max=120,
                           bsm_mass=0, bsm_width=0, bsm_coupling=0):
        """
        生成 Drell-Yan 事件.
        包含 SM (Z/γ*) 和可选的 BSM (Z') 贡献.

        截面: σ ∝ |A_SM + A_BSM|²
          A_SM ∝ 1/(M² - mZ² + i*mZ*ΓZ)
          A_BSM ∝ g'/(M² - MZ'² + i*MZ'*ΓZ')
        """
        events = []
        attempts = 0
        max_attempts = n_events * 100

        while len(events) < n_events and attempts < max_attempts:
            attempts += 1

            # 采样不变质量 M (重要性采样: Breit-Wigner)
            M = self.rng.uniform(M_min, M_max)

            # SM 振幅平方
            sm_amp_sq = 1.0 / ((M**2 - self.mZ**2)**2 + self.mZ**2 * self.GammaZ**2)

            # BSM 振幅平方 (如果有)
            bsm_amp_sq = 0.0
            interference = 0.0
            if bsm_mass > 0 and bsm_coupling > 0:
                bsm_amp_sq = bsm_coupling**2 / (
                    (M**2 - bsm_mass**2)**2 + bsm_mass**2 * bsm_width**2)
                # 干涉项
                re_sm = (M**2 - self.mZ**2) / (
                    (M**2 - self.mZ**2)**2 + self.mZ**2 * self.GammaZ**2)
                im_sm = self.mZ * self.GammaZ / (
                    (M**2 - self.mZ**2)**2 + self.mZ**2 * self.GammaZ**2)
                re_bsm = bsm_coupling * (M**2 - bsm_mass**2) / (
                    (M**2 - bsm_mass**2)**2 + bsm_mass**2 * bsm_width**2)
                im_bsm = bsm_coupling * bsm_mass * bsm_width / (
                    (M**2 - bsm_mass**2)**2 + bsm_mass**2 * bsm_width**2)
                interference = 2.0 * (re_sm * re_bsm + im_sm * im_bsm)

            sigma = sm_amp_sq + bsm_amp_sq + interference
            sigma_max = 1.0 / (self.mZ**2 * self.GammaZ**2) * 2.0

            # 接受-拒绝
            if self.rng.uniform() < sigma / sigma_max:
                # 生成运动学变量
                y = self.rng.uniform(-2.5, 2.5)  # 快度
                cos_theta = self.rng.uniform(-1, 1)  # 散射角
                pT = M / 2.0 * math.sqrt(1 - cos_theta**2)

                event = {
                    'M': M,
                    'rapidity': y,
                    'cos_theta': cos_theta,
                    'pT': pT,
                    'weight': sigma,
                    'is_bsm': bsm_amp_sq > sm_amp_sq,
                }
                events.append(event)

        return events


def invariant_mass(p1, p2):
    """计算两粒子的不变质量 M² = (p1+p2)²."""
    E = p1[0] + p2[0]
    px = p1[1] + p2[1]
    py = p1[2] + p2[2]
    pz = p1[3] + p2[3]
    M2 = E**2 - px**2 - py**2 - pz**2
    return math.sqrt(max(M2, 0))


def transverse_mass(pT, MET, delta_phi):
    """横向质量: M_T² = 2*pT*MET*(1 - cos(Δφ))."""
    return math.sqrt(max(2.0 * pT * MET * (1.0 - math.cos(delta_phi)), 0))
