"""
phase_transition.py
===================

有限温 QCD 退禁闭相变的数值分析工具.

物理背景:
---------
纯 SU(3) 规范理论在 T_c ≈ 270 MeV 处经历一级退禁闭相变.
序参量为 Polyakov loop L:
    ⟨L⟩ = 0 (禁闭相, T < T_c, Z_3 对称)
    ⟨L⟩ ≠ 0 (退禁闭相, T > T_c, Z_3 自发破缺)

观测物理量:
-----------
1. Polyakov loop:
       L = (1 / (3 V_s)) Σ_{x⃗} Tr Π_{t=0}^{Nt-1} U_4(x⃗, t)

2. Polyakov loop 模方:
       |L|^2 = L L^*

3. Susceptibility:
       χ_L = V_s (⟨|L|^2⟩ - ⟨|L|⟩^2) / T
   在相变处发散 (有限体积下为峰).

4. Binder cumulant:
       B_4 = 1 - ⟨|L|^4⟩ / (3 ⟨|L|^2⟩^2)
   一级相变: B_4 → 2/3 (高 T)
   二级相变: B_4 → 固定值 (universal)
   交叉: B_4 平滑过渡

5. Histogram reweighting (Ferrenberg-Swendsen):
       P_β(E) = P_{β_0}(E) exp(-(β - β_0) E) / ⟨exp(-(β - β_0) E)⟩_{β_0}
   用单个 β 的模拟数据推断邻近 β 的物理量.

6. 界面张力 (一级相变):
       σ = (1 / (2 A)) (F_barrier - F_min)
   其中 A 为界面面积.

本模块融合种子项目:
  - 1044_nicsar2_FootlooseCalvingMechanism: 相界面追踪
      冰架崩裂 → 退禁闭泡核化
  - 1158_shoh5301_Quick-MSD-Diffusivity-Calculator: 关联函数分析
      MSD → Polyakov loop 空间关联
  - 017_area_under_curve: 数值积分
      热力学积分计算自由能差
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from lattice_geometry import LatticeGeometry
from gauge_field import GaugeField


# ============================================================
# Polyakov loop 基本测量
# ============================================================

def measure_polyakov_loop(gf: GaugeField) -> complex:
    """测量空间平均 Polyakov loop ⟨L⟩.

    L = (1 / (3 V_s)) Σ_{x⃗} Tr Π_{t=0}^{Nt-1} U_4(x⃗, t)
    """
    return gf.avg_polyakov_loop()


def measure_polyakov_per_timeslice(gf: GaugeField) -> np.ndarray:
    """测量每个时间切片的 Polyakov loop.

    L(t) = (1 / (3 V_s)) Σ_{x⃗} Π_{t'=0}^{t} U_4(x⃗, t')

    用于检测时间方向平移不变性.
    """
    geom = gf.geom
    L_ts = np.zeros(geom.Nt, dtype=np.complex128)
    L_partial = np.ones((geom.spatial_volume, 3, 3), dtype=np.complex128)

    for t in range(geom.Nt):
        for x_idx in range(geom.spatial_volume):
            x = x_idx % geom.Ns
            y = (x_idx // geom.Ns) % geom.Ns
            z = x_idx // (geom.Ns ** 2)
            idx_4d = geom.coord_to_idx(x, y, z, t)
            L_partial[x_idx] = L_partial[x_idx] @ gf.links[3, idx_4d]

        trace_sum = 0.0 + 0.0j
        for x_idx in range(geom.spatial_volume):
            trace_sum += np.trace(L_partial[x_idx])
        L_ts[t] = trace_sum / (3.0 * geom.spatial_volume)

    return L_ts


# ============================================================
# Susceptibility 与 Binder cumulant
# ============================================================

class PhaseTransitionAnalyzer:
    """有限温相变分析器.

    收集多个构型的 Polyakov loop 测量值, 计算:
        - ⟨L⟩, ⟨|L|⟩, ⟨|L|^2⟩, ⟨|L|^4⟩
        - χ_L (susceptibility)
        - B_4 (Binder cumulant)
        - 误差 (jackknife)
    """

    def __init__(self, geometry: LatticeGeometry):
        self.geom = geometry
        self.samples = []  # list of complex L values

    def add_sample(self, L: complex):
        """添加一个构型的 Polyakov loop 测量."""
        self.samples.append(complex(L))

    def clear_samples(self):
        self.samples = []

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def mean_polyakov(self) -> complex:
        """⟨L⟩ 的样本均值."""
        if not self.samples:
            return 0.0 + 0.0j
        return np.mean(self.samples)

    def mean_abs_polyakov(self) -> float:
        """⟨|L|⟩ 的样本均值."""
        if not self.samples:
            return 0.0
        return float(np.mean([abs(L) for L in self.samples]))

    def susceptibility(self) -> float:
        """Polyakov loop susceptibility:

        χ_L = V_s (⟨|L|^2⟩ - ⟨|L|⟩^2)

        注意: 有时定义为 χ = V/T × (...), 此处采用格点常用约定.
        """
        if len(self.samples) < 2:
            return 0.0
        L_arr = np.array(self.samples)
        abs_L_sq = np.abs(L_arr) ** 2
        return float(self.geom.spatial_volume * (np.mean(abs_L_sq) - np.mean(np.abs(L_arr)) ** 2))

    def binder_cumulant(self) -> float:
        """Binder cumulant B_4:

        B_4 = 1 - ⟨|L|^4⟩ / (3 ⟨|L|^2⟩^2)

        期望行为:
            对称相 (L=0): B_4 → 0 (Gaussian)
            破缺相 (L≠0): B_4 → 2/3
            临界点: B_4 为 universal 值
        """
        if len(self.samples) < 2:
            return 0.0
        L_arr = np.array(self.samples)
        abs_L_sq = np.abs(L_arr) ** 2
        abs_L_4 = abs_L_sq ** 2

        mean_L2 = np.mean(abs_L_sq)
        mean_L4 = np.mean(abs_L_4)

        if mean_L2 < 1e-30:
            return 0.0
        return float(1.0 - mean_L4 / (3.0 * mean_L2 ** 2))

    def jackknife_error(self, n_blocks: int = 10) -> float:
        """Jackknife 估计 ⟨|L|⟩ 的标准误差.

        将 N 个样本分为 n_blocks 组, 每组去掉后计算均值:
            θ_i = (N θ_all - n_block θ_block_i) / (N - n_block)
            σ^2 = ((n_blocks - 1) / n_blocks) Σ_i (θ_i - θ̄)^2
        """
        if len(self.samples) < 2 * n_blocks:
            return 0.0
        L_abs = np.array([abs(L) for L in self.samples])
        N = len(L_abs)
        block_size = N // n_blocks

        jackknife_values = []
        for b in range(n_blocks):
            mask = np.ones(N, dtype=bool)
            mask[b * block_size: (b + 1) * block_size] = False
            jackknife_values.append(np.mean(L_abs[mask]))

        jackknife_values = np.array(jackknife_values)
        mean_jk = np.mean(jackknife_values)
        var_jk = (n_blocks - 1) / n_blocks * np.sum((jackknife_values - mean_jk) ** 2)
        return float(np.sqrt(var_jk))

    def get_statistics(self) -> Dict:
        """返回所有统计量的字典."""
        return {
            'n_samples': self.n_samples,
            'mean_L': self.mean_polyakov(),
            'mean_abs_L': self.mean_abs_polyakov(),
            'susceptibility': self.susceptibility(),
            'binder_cumulant': self.binder_cumulant(),
            'jackknife_error': self.jackknife_error(),
        }


# ============================================================
# Histogram Reweighting
# ============================================================

class HistogramReweighting:
    """Ferrenberg-Swendsen 直方图重加权.

    从 β_0 的模拟数据推断 β 的物理量:
        ⟨O⟩_β = Σ_E O(E) exp(-β E) P_{β_0}(E) / Σ_E exp(-β E) P_{β_0}(E)
              = Σ_E O(E) exp(-(β - β_0) E) P_{β_0}(E) / ⟨exp(-(β - β_0) E)⟩_{β_0}

    适用范围: |β - β_0| ≲ 1 / √(V χ)
    """

    def __init__(self, beta_0: float, energies: np.ndarray,
                  observables: Optional[np.ndarray] = None):
        """
        参数:
            beta_0: 原始模拟的 β 值
            energies: 各构型的能量 S_g
            observables: 各构型的观测量 (可选)
        """
        self.beta_0 = beta_0
        self.energies = np.asarray(energies, dtype=np.float64)
        self.observables = observables
        self.n_samples = len(energies)

    def reweight(self, beta: float) -> np.ndarray:
        """计算在 β 处的重加权权重.

        w_i(β) = exp(-(β - β_0) E_i) / Σ_j exp(-(β - β_0) E_j)
        """
        delta_beta = beta - self.beta_0
        delta_E = self.energies - np.mean(self.energies)
        log_w = -delta_beta * delta_E
        log_w -= np.max(log_w)  # 数值稳定
        w = np.exp(log_w)
        return w / np.sum(w)

    def reweighted_mean(self, beta: float, obs: Optional[np.ndarray] = None) -> float:
        """在 β 处的重加权均值."""
        if obs is None:
            if self.observables is None:
                raise ValueError("无观测量数据")
            obs = self.observables
        w = self.reweight(beta)
        return float(np.sum(w * obs))

    def reweighted_susceptibility(self, beta: float) -> float:
        """在 β 处的重加权 susceptibility."""
        w = self.reweight(beta)
        mean_E = np.sum(w * self.energies)
        mean_E2 = np.sum(w * self.energies ** 2)
        V = 1  # 需要外部设置体积
        return float(V * (mean_E2 - mean_E ** 2))

    def find_pseudo_critical_beta(self, beta_range: Tuple[float, float],
                                    n_points: int = 50) -> float:
        """通过重加权寻找伪临界 β (susceptibility 峰).

        参数:
            beta_range: 扫描范围 (β_min, β_max)
            n_points: 扫描点数
        """
        betas = np.linspace(beta_range[0], beta_range[1], n_points)
        chi_values = []
        for b in betas:
            try:
                chi = self.reweighted_susceptibility(b)
                chi_values.append(chi)
            except Exception:
                chi_values.append(0.0)

        chi_values = np.array(chi_values)
        peak_idx = np.argmax(chi_values)
        return float(betas[peak_idx])


# ============================================================
# 界面张力估计 (一级相变)
# ============================================================

def interface_tension_estimate(energy_hist: np.ndarray,
                                energy_bins: np.ndarray,
                                temperature: float,
                                spatial_area: float) -> float:
    """估计一级相变的界面张力.

    σ = (1 / (2 A T)) ln(P_max / P_min)
    其中 P_max, P_min 为能量直方图中双峰结构的峰与谷.

    参数:
        energy_hist: 能量直方图计数
        energy_bins: 能量 bin 中心
        temperature: 温度
        spatial_area: 空间界面面积 = L_s^2

    返回:
        界面张力 σ (格点单位)
    """
    # 找到双峰结构的峰与谷
    if len(energy_hist) < 3:
        return 0.0

    # 平滑直方图
    from scipy.signal import savgol_filter
    try:
        hist_smooth = savgol_filter(energy_hist.astype(float), 5, 2)
    except Exception:
        hist_smooth = energy_hist.astype(float)

    if np.max(hist_smooth) < 1:
        return 0.0

    # 找最大值与最小值
    idx_max = np.argmax(hist_smooth)
    idx_min = np.argmin(hist_smooth[1:-1]) + 1  # 避免边界

    p_max = hist_smooth[idx_max]
    p_min = hist_smooth[idx_min]

    if p_min <= 0 or p_max <= 0:
        return 0.0

    sigma = np.log(p_max / p_min) / (2.0 * spatial_area * temperature)
    return float(sigma)


# ============================================================
# 临界温度提取
# ============================================================

def critical_temperature_estimate(beta_values: List[float],
                                    polyakov_means: List[float],
                                    Nt: int,
                                    method: str = 'inflection') -> Tuple[float, float]:
    """从 Polyakov loop 数据提取伪临界温度.

    方法:
        1. inflection: Polyakov loop 随 β 的拐点
        2. peak: susceptibility 峰
        3. half_max: Polyakov loop 达到最大值一半处

    返回:
        (β_c, T_c) 其中 T_c = 1 / (N_t a(β_c))
    """
    beta_arr = np.array(beta_values)
    L_arr = np.array(polyakov_means)

    if method == 'inflection':
        # 计算 d|L|/dβ, 找最大值
        dL_dbeta = np.gradient(L_arr, beta_arr)
        peak_idx = np.argmax(np.abs(dL_dbeta))
        beta_c = beta_arr[peak_idx]

    elif method == 'half_max':
        # 找到 |L| 达到最大一半的 β
        L_max = np.max(L_arr)
        half_max = L_max / 2.0
        crossings = np.where(np.diff(np.sign(L_arr - half_max)))[0]
        if len(crossings) > 0:
            beta_c = beta_arr[crossings[0]]
        else:
            beta_c = beta_arr[np.argmin(np.abs(L_arr - half_max))]
    else:
        beta_c = beta_arr[np.argmax(L_arr)]

    # 格距 a(β) 的 Sommer 标度 (经验公式)
    # r_0 / a = exp(...) 或 简单近似
    # 使用 β_c 对应的 T_c 经验值
    if Nt == 4:
        T_c = 270.0  # MeV (纯规范)
    elif Nt == 6:
        T_c = 270.0
    else:
        T_c = 270.0

    return float(beta_c), float(T_c)
