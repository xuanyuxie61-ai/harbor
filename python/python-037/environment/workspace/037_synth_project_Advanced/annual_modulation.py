r"""
annual_modulation.py
年度调制信号分析模块

本模块实现暗物质直接探测实验中的年度调制效应分析。
地球绕太阳公转叠加太阳系在银河系中的运动导致
WIMP 相对地球的速度呈年度周期性变化，
从而在探测器散射率中引入 ~5% 的调制信号。

核心内容：
1. 年度调制曲线建模（参考 lissajous 的参数化振荡思想）
2. 多通道调制分析（不同能量区间）
3. 调制显著性检验（χ² 拟合）
4. 相位重建与振幅提取

物理公式：

A. 地球速度矢量（Galactic 坐标系）：
    \vec{v}_e(t) = \vec{v}_0 + \vec{v}_{\odot} + \vec{v}_{\rm orb}(t)

    其中：
        v_0 = 220 km/s      (本地标准静止系圆速度)
        v_\odot ≈ 16 km/s   (太阳 peculiar velocity)
        v_{\rm orb}(t) = 30 km/s \cdot \cos[2\pi(t - t_0)/T]

B. 调制散射率：
    S(t, E) = S_0(E) + S_m(E) \cos\left[ \omega (t - t_0) \right]
    \omega = 2\pi / T, \quad T = 365.25 \text{ days}

C. 调制振幅（Lewin & Smith 近似）：
    \frac{S_m}{S_0} \approx \frac{2 v_e v_{\rm orb}}{v_0^2}
                              \approx 0.03 \text{ – } 0.07

D. Lissajous 型调制信号（二维参数化）：
    将调制信号表示为参数化曲线：
        X(t) = S_0 + S_m \cos(\omega t)
        Y(t) = S_0 + S_m \sin(\omega t)
    形成闭合椭圆轨迹，用于多通道联合显著性分析。

参考文献：
- Drukier, A. K., Freese, K., & Spergel, D. N. (1986). Phys. Rev. D 33, 3495.
- Freese, K., Frieman, J. A., & Gould, A. (1988). Phys. Rev. D 37, 3388.
- Bernabei, R., et al. (2000). Phys. Lett. B 480, 23.
"""

import numpy as np
from typing import Tuple, List, Dict


# ============================================================================
# 年度调制模型
# ============================================================================

def modulation_curve(
    t_days: np.ndarray,
    s0: float,
    sm: float,
    t0: float = 152.0,
    period: float = 365.25,
) -> np.ndarray:
    """
    计算标准余弦型调制曲线。

    公式：
        S(t) = S_0 + S_m \cos\left[ \frac{2\pi (t - t_0)}{T} \right]

    参数：
        t_days: 时间数组 [天]
        s0: 平均计数率
        sm: 调制振幅
        t0: 最大速度对应时刻（约 6 月 2 日 ≈ 152 天）
        period: 周期 [天]

    返回：
        S: 与 t_days 同形数组
    """
    return s0 + sm * np.cos(2.0 * np.pi * (t_days - t0) / period)


def modulation_curve_lissajous(
    t_days: np.ndarray,
    s0: float,
    sm: float,
    t0: float = 152.0,
    period: float = 365.25,
    phase_shift: float = np.pi / 2.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    生成 Lissajous 型调制参数曲线（X-Y 平面）。

    参数化方程：
        X(t) = S_0 + S_m \cos\left[ \omega (t - t_0) \right]
        Y(t) = S_0 + S_m \cos\left[ \omega (t - t_0) + \delta \right]

    当 δ = π/2 时，轨迹为圆；δ ≠ π/2 时为椭圆。
    此曲线用于可视化多探测器联合调制相位关系。

    参数：
        t_days: 时间数组
        s0: 基准值
        sm: 振幅
        t0: 相位参考时刻
        period: 周期
        phase_shift: Y 通道相位偏移 [rad]

    返回：
        (X, Y): 参数曲线坐标
    """
    omega = 2.0 * np.pi / period
    phase = omega * (t_days - t0)
    X = s0 + sm * np.cos(phase)
    Y = s0 + sm * np.cos(phase + phase_shift)
    return X, Y


# ============================================================================
# 时间分箱与速率计算
# ============================================================================

def bin_events_by_time(
    events: List[Dict],
    n_bins: int = 12,
    period: float = 365.25,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将事件按时间分箱，计算每 bin 的计数与误差。

    参数：
        events: 事件列表（含 'time_day' 字段）
        n_bins: 时间分箱数（通常 12，即月度 bin）
        period: 周期长度 [天]

    返回：
        bin_centers: (n_bins,) bin 中心时刻
        counts: (n_bins,) 每 bin 计数
        errors: (n_bins,) Poisson 误差 sqrt(N)
    """
    counts = np.zeros(n_bins)
    bin_edges = np.linspace(0.0, period, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    for ev in events:
        t = ev.get("time_day", 0.0) % period
        idx = int(np.clip(np.floor(t / period * n_bins), 0, n_bins - 1))
        counts[idx] += 1.0

    errors = np.sqrt(np.where(counts > 0.0, counts, 1.0))
    return bin_centers, counts, errors


def fit_modulation_amplitude(
    t_bins: np.ndarray,
    counts: np.ndarray,
    errors: np.ndarray,
    period: float = 365.25,
) -> Tuple[float, float, float, float]:
    """
    用最小二乘法拟合调制曲线 S(t) = S_0 + S_m cos[ω(t - t_0)]。

    线性化方法：
        令 θ_i = 2π(t_i - t_0)/T，展开为：
        S(t) = S_0 + A \cos θ + B \sin θ
        其中 S_m = √(A² + B²)，相位 φ = arctan2(B, A)

    最小二乘矩阵形式：
        [ 1, cos θ_1, sin θ_1 ]   [ S_0 ]   [ c_1 ]
        [ 1, cos θ_2, sin θ_2 ] · [  A  ] = [ c_2 ]
        [ 1, cos θ_3, sin θ_3 ]   [  B  ]   [ c_3 ]
                 ...

    参数：
        t_bins: 时间 bin 中心 [天]
        counts: 观测计数
        errors: 计数误差
        period: 周期

    返回：
        s0: 平均计数率
        sm: 调制振幅
        phase: 相位 [rad]
        chi2: 拟合 χ²
    """
    t_bins = np.asarray(t_bins)
    counts = np.asarray(counts)
    errors = np.asarray(errors)

    # 默认相位参考 t0 = 152 天
    t0 = 152.0
    theta = 2.0 * np.pi * (t_bins - t0) / period

    # 构造设计矩阵，按误差加权
    w = 1.0 / np.where(errors > 0.0, errors, 1.0)
    A_mat = np.column_stack([np.ones_like(t_bins), np.cos(theta), np.sin(theta)])
    Aw = A_mat * w[:, None]
    cw = counts * w

    # 最小二乘求解
    coeffs, residuals, rank, s = np.linalg.lstsq(Aw, cw, rcond=None)
    s0_fit = coeffs[0]
    A_fit = coeffs[1]
    B_fit = coeffs[2]

    sm_fit = np.sqrt(A_fit ** 2 + B_fit ** 2)
    phase_fit = np.arctan2(B_fit, A_fit)

    # χ²
    model = s0_fit + A_fit * np.cos(theta) + B_fit * np.sin(theta)
    chi2 = np.sum(((counts - model) / errors) ** 2)

    return float(s0_fit), float(sm_fit), float(phase_fit), float(chi2)


def modulation_significance(
    s0: float,
    sm: float,
    total_counts: float,
    n_bins: int = 12,
) -> float:
    """
    估计调制信号的统计显著性（σ 水平）。

    公式：
        \sigma = \frac{S_m}{\sqrt{2/N_{\rm total}}} \cdot \sqrt{\frac{n_{\rm bins}}{2}}

    简化为：
        \sigma \approx \frac{S_m \cdot N_{\rm total}}{S_0 \cdot \sqrt{N_{\rm total}}}
                 = \frac{S_m}{S_0} \sqrt{N_{\rm total}}

    参数：
        s0: 平均计数率
        sm: 调制振幅
        total_counts: 总事件数
        n_bins: 时间分箱数

    返回：
        sigma: 显著性水平 [σ]
    """
    if s0 <= 0.0 or total_counts <= 0.0:
        return 0.0
    modulation_fraction = sm / s0
    sigma = modulation_fraction * np.sqrt(total_counts)
    return float(sigma)


# ============================================================================
# 多能量区间调制分析
# ============================================================================

def analyze_modulation_by_energy_bins(
    events: List[Dict],
    energy_edges: np.ndarray,
    n_time_bins: int = 12,
) -> List[Dict]:
    """
    对不同能量区间分别进行调制分析。

    参数：
        events: 事件列表
        energy_edges: 能量分箱边界 [keV]
        n_time_bins: 时间分箱数

    返回：
        results: 每个能量区间的分析结果字典列表
    """
    results = []
    n_ebins = len(energy_edges) - 1

    for i in range(n_ebins):
        e_low = energy_edges[i]
        e_high = energy_edges[i + 1]
        selected = [ev for ev in events if e_low <= ev.get("energy_obs", 0.0) < e_high]

        if len(selected) < n_time_bins * 2:
            results.append({
                "energy_low": e_low,
                "energy_high": e_high,
                "n_events": len(selected),
                "s0": None,
                "sm": None,
                "significance": 0.0,
            })
            continue

        t_bins, counts, errors = bin_events_by_time(selected, n_time_bins)
        s0, sm, phase, chi2 = fit_modulation_amplitude(t_bins, counts, errors)
        sig = modulation_significance(s0, sm, float(len(selected)), n_time_bins)

        results.append({
            "energy_low": e_low,
            "energy_high": e_high,
            "n_events": len(selected),
            "s0": s0,
            "sm": sm,
            "phase": phase,
            "chi2": chi2,
            "significance": sig,
        })

    return results


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    # 测试调制曲线
    t = np.linspace(0.0, 365.25, 100)
    s = modulation_curve(t, s0=100.0, sm=5.0)
    assert abs(np.mean(s) - 100.0) < 0.1, "调制曲线平均值异常"
    assert abs(np.max(s) - 105.0) < 0.1, "调制曲线最大值异常"
    assert abs(np.min(s) - 95.0) < 0.1, "调制曲线最小值异常"

    # 测试 Lissajous 曲线
    X, Y = modulation_curve_lissajous(t, 100.0, 5.0)
    # 当相位差为 π/2 时应形成圆（闭合曲线）
    assert abs(X[0] - X[-1]) < 1e-10, "Lissajous 曲线未闭合"

    # 测试拟合
    np.random.seed(0)
    true_s0, true_sm = 100.0, 5.0
    t_bins = np.linspace(15.0, 350.0, 12)
    counts = modulation_curve(t_bins, true_s0, true_sm) + np.random.normal(0.0, 3.0, size=12)
    errors = np.sqrt(np.where(counts > 0, counts, 1.0))
    s0_fit, sm_fit, phase_fit, chi2 = fit_modulation_amplitude(t_bins, counts, errors)
    assert abs(s0_fit - true_s0) / true_s0 < 0.1, "S0 拟合偏差过大"
    assert abs(sm_fit - true_sm) / true_sm < 0.3, "Sm 拟合偏差过大"

    # 测试显著性
    sig = modulation_significance(100.0, 5.0, 10000.0)
    assert sig > 0.0, "显著性应为正"

    print("annual_modulation.py: 所有自测通过")
