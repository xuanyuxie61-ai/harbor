"""
detector_sim.py — 探测器模拟
==============================
种子项目映射:
  1289_BABILong (长上下文/噪声注入) → 探测器噪声与效率建模

物理: 模拟探测器效应 (能量分辨率、效率、噪声) 对事件重建的影响.
"""
import numpy as np
import math


class DetectorGeometry:
    """简化探测器几何."""

    def __init__(self, eta_max=2.5, n_layers=10):
        self.eta_max = eta_max
        self.n_layers = n_layers

    def acceptance(self, eta):
        """几何接受度: |η| < eta_max."""
        return abs(eta) < self.eta_max

    def efficiency(self, pT, eta):
        """
        重建效率参数化:
          ε(pT, η) = ε_0 * f(pT) * g(η)
          f(pT) = 1 - exp(-pT/pT_0)
          g(η) = 1 / (1 + (η/η_0)^4)
        """
        pT_0 = 5.0  # GeV
        eta_0 = 2.0
        eps_0 = 0.95
        f_pT = 1.0 - math.exp(-pT / pT_0) if pT > 0 else 0.0
        g_eta = 1.0 / (1.0 + (eta / eta_0)**4)
        return eps_0 * f_pT * g_eta


class EnergyResolution:
    """能量分辨率模型."""

    def __init__(self, stochastic=0.10, constant=0.01, noise=0.5):
        """
        σ_E/E = sqrt( (S/sqrt(E))² + N² + C² )
        S: 随机项 (stochastic), N: 噪声项, C: 常数项
        单位: E in GeV, σ_E in GeV
        """
        self.S = stochastic
        self.C = constant
        self.N = noise

    def sigma_E(self, E):
        """能量分辨率 σ_E (GeV)."""
        if E <= 0:
            return 0.0
        rel = math.sqrt((self.S / math.sqrt(E))**2 + self.N**2 + self.C**2)
        return rel * E

    def smear(self, E_true, rng=None):
        """对真实能量进行高斯展宽."""
        if rng is None:
            rng = np.random.RandomState()
        sigma = self.sigma_E(E_true)
        return max(rng.normal(E_true, sigma), 0.0)


class DetectorSimulation:
    """完整探测器模拟流水线."""

    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)
        self.geom = DetectorGeometry()
        self.resolution = EnergyResolution()

    def process_event(self, event):
        """处理单个事件."""
        reco = event.copy()

        # 快度 → η
        y = event.get('rapidity', 0)
        eta = y  # 简化: 对无质量粒子 η ≈ y

        # 接受度检查
        if not self.geom.acceptance(eta):
            reco['reconstructed'] = False
            return reco

        # 效率
        pT = event.get('pT', 30)
        eff = self.geom.efficiency(pT, eta)
        if self.rng.uniform() > eff:
            reco['reconstructed'] = False
            return reco

        # 能量展宽
        M_true = event.get('M', 91.2)
        M_reco = self.resolution.smear(M_true, self.rng)

        # pT展宽
        pT_reco = self.resolution.smear(pT, self.rng)

        # 缺失横能量 (MET)
        MET = self.rng.exponential(10.0)  # 简化: 指数分布本底

        reco.update({
            'M': M_reco,
            'pT_meas': pT_reco,
            'MET': MET,
            'reconstructed': True,
            'triggered': pT_reco > 20.0,
        })

        return reco

    def process_events(self, events):
        """批量处理事件."""
        return [self.process_event(e) for e in events]


def compute_data_quality(event):
    """计算数据质量标志."""
    flags = {
        'good_event': False,
        'high_quality': False,
    }
    if not event.get('reconstructed', False):
        return flags

    M = event.get('M', 0)
    pT = event.get('pT_meas', 0)
    MET = event.get('MET', 0)

    flags['good_event'] = M > 0 and pT > 10 and MET < 100
    flags['high_quality'] = flags['good_event'] and pT > 30 and MET < 30

    return flags
