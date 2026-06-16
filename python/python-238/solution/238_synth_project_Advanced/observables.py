"""
observables.py
==============

格点 QCD 物理观测量的测量与统计分析.

观测量:
-------
1. 平均 plaquette:
       P = (1 / (6V)) Σ_{x,μ<ν} (1/3) Re Tr U_μν(x)
   与耦合常数的关系 (微扰展开):
       P = 1 - (g^2/6) C_F - ... (弱耦合)

2. Polyakov loop 关联函数 (静态夸克势):
       G(r) = ⟨L(x) L^†(x+r)⟩ - ⟨L⟩^2
   在大距离:
       G(r) ∝ exp(-m_D r) / r  (Debye 屏蔽)
   或:
       G(r) ∝ exp(-σ r / T)    (弦张力)

3. 拓扑荷 Q:
       Q = (1 / (32 π^2)) Σ_x ε_{μνρσ} Tr F_μν F_ρσ
   期望值: ⟨Q⟩ = 0
   涨落: ⟨Q^2⟩ = V χ_t
   其中 χ_t 为拓扑 susceptibility.

4. 自关联时间 τ_int:
       C(t) = ⟨O(n) O(n+t)⟩ - ⟨O⟩^2
       τ_int = (1/2) + Σ_{t=1}^∞ C(t) / C(0)
   有效样本数: N_eff = N / (2 τ_int)

5. 拓扑荷扩散 (MSD 类似物, 融合 1158_shoh5301):
       ⟨(Q(t) - Q(0))^2⟩ = 2 D_t t
   其中 D_t 为拓扑荷的 "扩散系数" (Monte Carlo 时间单位).

本模块融合种子项目:
  - 1158_shoh5301_Quick-MSD-Diffusivity-Calculator:
      MSD → 拓扑荷扩散系数
  - 977_r8col: 排序与去重
      用于观测值序列的排序统计
  - 017_area_under_curve: 数值积分
      自关联函数的积分 → τ_int
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from lattice_geometry import LatticeGeometry
from gauge_field import GaugeField
from gauge_actions import GaugeAction


# ============================================================
# Plaquette 与基本观测量
# ============================================================

def measure_plaquette(gf: GaugeField) -> float:
    """平均归一化 plaquette."""
    return gf.avg_plaquette()


def measure_plaquette_per_direction(gf: GaugeField) -> Dict:
    """按方向对分解 plaquette.

    返回:
        {'spatial': P_s, 'temporal': P_t, 'total': P}
    其中:
        P_s = 平均空间 plaquette (μ,ν ∈ {0,1,2})
        P_t = 平均时空 plaquette (μ ∈ {0,1,2}, ν = 3)
    """
    geom = gf.geom
    P_s_sum = 0.0
    P_s_count = 0
    P_t_sum = 0.0
    P_t_count = 0

    for idx in range(geom.volume):
        for mu in range(4):
            for nu in range(mu + 1, 4):
                plaq = gf.plaquette(idx, mu, nu)
                tr = np.trace(plaq).real / 3.0
                if mu < 3 and nu < 3:
                    P_s_sum += tr
                    P_s_count += 1
                else:
                    P_t_sum += tr
                    P_t_count += 1

    P_s = P_s_sum / P_s_count if P_s_count > 0 else 0.0
    P_t = P_t_sum / P_t_count if P_t_count > 0 else 0.0
    return {
        'spatial': P_s,
        'temporal': P_t,
        'total': (P_s * P_s_count + P_t * P_t_count) / (P_s_count + P_t_count),
    }


# ============================================================
# Polyakov loop 关联函数 (MSD 类似物)
# ============================================================

def polyakov_loop_correlator_vs_r(gf: GaugeField,
                                    max_r_sq: int = 10) -> Dict:
    """计算 Polyakov loop 关联函数作为距离的函数.

    C(r) = ⟨L(x) L^†(x+r)⟩

    对每个可达的距离平方 r^2, 计算空间平均.

    类比种子项目 1158: MSD(t) → C(r),
    其中 r 为空间距离而非 Monte Carlo 时间.

    返回:
        {r_sq: {'correlator': C(r), 'count': N_pairs}}
    """
    geom = gf.geom

    # 计算所有空间位置的 Polyakov loop
    L_spatial = []
    for x_idx in range(geom.spatial_volume):
        L = gf.polyakov_loop_spatial(x_idx)
        L_spatial.append(np.trace(L) / 3.0)
    L_spatial = np.array(L_spatial)

    results = {}
    for x_idx in range(geom.spatial_volume):
        x = x_idx % geom.Ns
        y = (x_idx // geom.Ns) % geom.Ns
        z = x_idx // (geom.Ns ** 2)
        x_4d = geom.coord_to_idx(x, y, z, 0)

        for y_idx in range(x_idx + 1, geom.spatial_volume):
            yx = y_idx % geom.Ns
            yy = (y_idx // geom.Ns) % geom.Ns
            yz = y_idx // (geom.Ns ** 2)
            y_4d = geom.coord_to_idx(yx, yy, yz, 0)

            r_sq = int(round(geom.distance_sq(x_4d, y_4d)))
            if r_sq > max_r_sq or r_sq == 0:
                continue

            corr = (L_spatial[x_idx] * np.conj(L_spatial[y_idx])).real
            if r_sq not in results:
                results[r_sq] = {'sum': 0.0, 'count': 0}
            results[r_sq]['sum'] += corr
            results[r_sq]['count'] += 1

    # 平均
    for r_sq in results:
        if results[r_sq]['count'] > 0:
            results[r_sq]['correlator'] = results[r_sq]['sum'] / results[r_sq]['count']
        else:
            results[r_sq]['correlator'] = 0.0

    return results


# ============================================================
# 自关联分析
# ============================================================

def autocorrelation_function(series: np.ndarray,
                               max_lag: Optional[int] = None) -> np.ndarray:
    """计算时间序列的自关联函数.

    C(t) = (1/(N-t)) Σ_{n=0}^{N-t-1} (O(n) - ⟨O⟩)(O(n+t) - ⟨O⟩)
    C(0) = var(O)

    返回:
        C(t) / C(0) for t = 0, 1, ..., max_lag
    """
    N = len(series)
    if max_lag is None:
        max_lag = min(N // 4, 100)

    mean = np.mean(series)
    centered = series - mean
    var = np.var(series)

    if var < 1e-30:
        return np.zeros(max_lag + 1)

    acf = np.zeros(max_lag + 1)
    for t in range(max_lag + 1):
        if t >= N:
            break
        acf[t] = np.sum(centered[:N-t] * centered[t:]) / ((N - t) * var)
    return acf


def integrated_autocorrelation_time(series: np.ndarray,
                                       max_lag: Optional[int] = None,
                                       window: int = 10) -> float:
    """计算积分自关联时间 τ_int.

    τ_int = 0.5 + Σ_{t=1}^{W} C(t)/C(0)

    其中 W 为窗口 (通常选为 C(t) 首次变负处或固定值).

    参数:
        series: 时间序列
        max_lag: 最大 lag
        window: 求和窗口宽度

    返回:
        τ_int (单位: 构型数)
    """
    acf = autocorrelation_function(series, max_lag)
    if len(acf) < 2:
        return 0.5

    # 求和至窗口或首次变负
    tau = 0.5
    for t in range(1, min(window, len(acf))):
        if acf[t] < 0:
            break
        tau += acf[t]
    return float(tau)


def effective_sample_size(series: np.ndarray) -> float:
    """有效样本数:

    N_eff = N / (2 τ_int)

    反映自关联对统计误差的放大.
    """
    N = len(series)
    tau = integrated_autocorrelation_time(series)
    return N / (2.0 * tau) if tau > 0 else float(N)


# ============================================================
# 拓扑荷扩散 (MSD 类似)
# ============================================================

def topological_charge_diffusion(Q_series: np.ndarray) -> Dict:
    """计算拓扑荷的均方位移 (MSD).

    MSD(Δt) = ⟨(Q(t + Δt) - Q(t))^2⟩_t

    扩散系数:
        D_Q = lim_{Δt→∞} MSD(Δt) / (2 Δt)

    类比 1158_shoh5301 中的 MSD → diffusivity 拟合.
    """
    N = len(Q_series)
    max_dt = min(N // 2, 50)
    msd = np.zeros(max_dt + 1)
    counts = np.zeros(max_dt + 1)

    for dt in range(1, max_dt + 1):
        for t in range(N - dt):
            msd[dt] += (Q_series[t + dt] - Q_series[t]) ** 2
            counts[dt] += 1
        if counts[dt] > 0:
            msd[dt] /= counts[dt]

    # 线性拟合求扩散系数
    if max_dt >= 5:
        dt_vals = np.arange(1, max_dt + 1, dtype=float)
        msd_vals = msd[1:]
        # 线性回归: MSD = 2 D t
        slope, _ = np.polyfit(dt_vals, msd_vals, 1)
        D_Q = slope / 2.0
    else:
        D_Q = 0.0

    return {
        'msd': msd,
        'dt_values': np.arange(max_dt + 1),
        'diffusion_coefficient': float(D_Q),
    }


# ============================================================
# 排序与分位数 (融合 977_r8col)
# ============================================================

def sorted_statistics(series: np.ndarray) -> Dict:
    """对观测值序列进行排序统计.

    融合 977_r8col 的列排序与唯一值识别思想.

    返回:
        均值, 中位数, 标准差, 分位数 (25%, 75%), 唯一值数
    """
    arr = np.asarray(series)
    sorted_arr = np.sort(arr)
    unique_vals = np.unique(sorted_arr)

    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr)),
        'q25': float(np.percentile(arr, 25)),
        'q75': float(np.percentile(arr, 75)),
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'n_unique': len(unique_vals),
        'n_samples': len(arr),
    }


# ============================================================
# Bootstrap 误差分析
# ============================================================

def bootstrap_error(series: np.ndarray,
                     statistic_func,
                     n_bootstrap: int = 500,
                     seed: int = 42) -> Tuple[float, float]:
    """Bootstrap 估计统计量的标准误差.

    从原始数据中重采样 n_bootstrap 次, 计算统计量分布.

    参数:
        series: 原始数据
        statistic_func: 统计量函数 (接受数组返回标量)
        n_bootstrap: bootstrap 样本数
        seed: 随机种子

    返回:
        (统计量估计值, bootstrap 标准误差)
    """
    rng = np.random.default_rng(seed)
    N = len(series)

    theta_original = statistic_func(series)
    theta_bootstrap = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.integers(0, N, size=N)
        theta_bootstrap[b] = statistic_func(series[indices])

    se = float(np.std(theta_bootstrap))
    return float(theta_original), se


# ============================================================
# 热力学积分 (融合 017_area_under_curve)
# ============================================================

def thermodynamic_integration(beta_values: np.ndarray,
                                plaquette_values: np.ndarray) -> float:
    """通过热力学积分计算自由能差.

    关系:
        d(ln Z)/dβ = ⟨S_plaq / β⟩ = 6V ⟨P⟩

    因此:
        ln Z(β_2) - ln Z(β_1) = 6V ∫_{β_1}^{β_2} ⟨P(β)⟩ dβ

    使用梯形积分计算.

    参数:
        beta_values: β 值数组 (升序)
        plaquette_values: 对应的平均 plaquette

    返回:
        Δ(ln Z) / (6V)
    """
    # 梯形积分
    delta_ln_Z = float(np.trapz(plaquette_values, beta_values))
    return delta_ln_Z
