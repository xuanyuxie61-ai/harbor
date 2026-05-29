"""
磁场分析与极性反转检测 (field_analysis.py)
===========================================
基于种子项目 1192_svd_sphere 的 SVD 模态分析思想与 116_box_plot 的
数据分类思想，为地核发电机模拟提供：
  - 磁场球谐系数的 SVD 主成分分析
  - 偶极子倾角与强度时间演化
  - 极性反转自动检测
  - 能量谱与功率谱分析
  - 反转统计（反转频率、持续时间）
"""

import numpy as np
from typing import List, Tuple, Dict


# ---------------------------------------------------------------------------
# 1. 磁场 SVD 模态分析
#    将多时刻的球谐系数矩阵进行 SVD，提取主导模态。
# ---------------------------------------------------------------------------
def svd_field_modes(coeffs_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对球谐系数时间序列进行 SVD 分解。

    参数:
      coeffs_matrix : shape (n_times, n_modes)，每行为一个时刻的所有球谐系数

    返回:
      U : 时间模态 (n_times, n_times)
      S : 奇异值
      Vt: 空间模态 (n_modes, n_modes)

    物理意义:
      最大奇异值对应的时间-空间模态对代表磁场的主导演化模式
      （如稳态偶极子、振荡四极子等）。
    """
    coeffs_matrix = np.asarray(coeffs_matrix, dtype=float)
    # 去除时间均值（异常场分析）
    mean = np.mean(coeffs_matrix, axis=0)
    Anom = coeffs_matrix - mean[np.newaxis, :]
    U, S, Vt = np.linalg.svd(Anom, full_matrices=False)
    return U, S, Vt


# ---------------------------------------------------------------------------
# 2. 偶极子参数提取
# ---------------------------------------------------------------------------
def extract_dipole_parameters(coeffs: Dict[Tuple[int, int], complex]) -> Dict[str, float]:
    """
    从球谐系数字典中提取偶极子参数。

    采用国际地磁参考场 (IGRF) 系数约定：
      g_l^m = Re(coeff[l,m])
      h_l^m = Im(coeff[l,m])

    返回:
      g10, g11, h11        : 偶极子高斯系数
      inclination          : 磁倾角 (rad)
      declination          : 磁偏角 (rad)
      dipole_moment_norm   : 归一化偶极矩
      dipole_tilt          : 偶极子倾角（相对于自转轴）
    """
    # TODO(Hole_3): 从球谐系数字典中提取地磁偶极子参数。
    # 输入: coeffs 是 dict[(l,m)] -> complex，采用 IGRF 系数约定。
    # 需计算:
    #   g10, g11, h11         : 高斯系数 (Re/Im 提取)
    #   inclination (rad)     : 磁倾角，公式 tan(I) = 2*g10 / sqrt(g11^2+h11^2)
    #   declination (rad)     : 磁偏角，公式 tan(D) = h11 / g11
    #   dipole_moment_norm    : 归一化偶极矩 sqrt(g10^2+g11^2+h11^2)
    #   dipole_tilt (rad)     : 相对于自转轴的倾角
    # 注意: 这些参数用于后续极性反转检测，倾角符号决定极性。
    raise NotImplementedError("Hole_3: 偶极子参数提取待实现")

    return {
        "g10": 0.0,
        "g11": 0.0,
        "h11": 0.0,
        "inclination": 0.0,
        "declination": 0.0,
        "dipole_moment_norm": 0.0,
        "dipole_tilt": 0.0,
    }


# ---------------------------------------------------------------------------
# 3. 极性反转检测
#    根据地磁偶极子轴向分量 g10 的符号翻转检测反转事件。
# ---------------------------------------------------------------------------
def detect_reversals(g10_series: np.ndarray, time_series: np.ndarray,
                     threshold_ratio: float = 0.3) -> List[Dict[str, float]]:
    """
    自动检测地磁场极性反转事件。

    算法:
      1. 计算 g10 时间序列的滑动平均以降噪
      2. 识别 g10 穿越零点的时刻
      3. 要求反转前后 |g10| 超过阈值（避免噪声穿越）
      4. 记录反转开始、结束、持续时间

    参数:
      g10_series     : 偶极子轴向系数时间序列
      time_series    : 对应时间（秒）
      threshold_ratio: 触发反转所需的振幅阈值（相对于最大振幅的比例）

    返回:
      反转事件列表，每个事件为 dict:
        { 'time_start', 'time_end', 'duration', 'polarity_before', 'polarity_after' }
    """
    g10 = np.asarray(g10_series, dtype=float)
    t = np.asarray(time_series, dtype=float)
    n = len(g10)
    if n < 3:
        return []

    # 滑动平均降噪（窗口大小 3）
    g10_smooth = g10.copy()
    for i in range(1, n - 1):
        g10_smooth[i] = (g10[i - 1] + g10[i] + g10[i + 1]) / 3.0

    max_amp = np.max(np.abs(g10_smooth))
    threshold = threshold_ratio * max_amp
    if threshold < 1e-30:
        return []

    reversals = []
    in_reversal = False
    reversal_start = 0.0
    polarity_before = 0

    for i in range(1, n):
        prev = g10_smooth[i - 1]
        curr = g10_smooth[i]
        # 检测符号翻转
        if prev * curr < 0:
            if not in_reversal:
                # 检查穿越前振幅是否足够大
                if abs(prev) >= threshold:
                    in_reversal = True
                    reversal_start = t[i - 1]
                    polarity_before = 1 if prev > 0 else -1
            else:
                # 反转结束
                if abs(curr) >= threshold:
                    reversal_end = t[i]
                    duration = reversal_end - reversal_start
                    polarity_after = 1 if curr > 0 else -1
                    reversals.append({
                        "time_start": reversal_start,
                        "time_end": reversal_end,
                        "duration": duration,
                        "polarity_before": polarity_before,
                        "polarity_after": polarity_after,
                    })
                    in_reversal = False

    return reversals


# ---------------------------------------------------------------------------
# 4. 能量谱分析
# ---------------------------------------------------------------------------
def magnetic_energy_spectrum(coeffs_time: List[Dict[Tuple[int, int], complex]],
                              l_max: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算磁场能量谱的时间平均。

    能量谱定义:
      E_l = sum_{m=-l}^{l} (l+1) * |g_l^m|^2

    返回:
      l_values : 球谐阶数数组
      E_mean   : 时间平均能量谱
    """
    n_times = len(coeffs_time)
    if n_times == 0:
        return np.array([]), np.array([])

    E_all = np.zeros((n_times, l_max + 1), dtype=float)
    for it, coeffs in enumerate(coeffs_time):
        for l in range(l_max + 1):
            energy = 0.0
            for m in range(-l, l + 1):
                c = coeffs.get((l, m), 0.0)
                energy += abs(c) ** 2
            E_all[it, l] = (l + 1.0) * energy

    E_mean = np.mean(E_all, axis=0)
    return np.arange(l_max + 1), E_mean


# ---------------------------------------------------------------------------
# 5. 反转统计量
# ---------------------------------------------------------------------------
def reversal_statistics(reversals: List[Dict[str, float]],
                        total_time: float) -> Dict[str, float]:
    """
    计算极性反转的统计特征。

    返回:
      reversal_rate        : 反转频率 (次/百万年)
      mean_duration        : 平均反转持续时间 (秒)
      std_duration         : 持续时间标准差
      total_reversal_time  : 处于反转状态的总时间比例
    """
    if not reversals:
        return {
            "reversal_rate": 0.0,
            "mean_duration": 0.0,
            "std_duration": 0.0,
            "total_reversal_time_ratio": 0.0,
        }

    durations = np.array([ev["duration"] for ev in reversals], dtype=float)
    total_rev_time = np.sum(durations)
    myr = 1e6 * 365.25 * 24 * 3600.0

    return {
        "reversal_rate": len(reversals) / (total_time / myr),
        "mean_duration": float(np.mean(durations)),
        "std_duration": float(np.std(durations)),
        "total_reversal_time_ratio": total_rev_time / total_time,
    }


# ---------------------------------------------------------------------------
# 6. 综合报告生成
# ---------------------------------------------------------------------------
def generate_field_report(coeffs_history: List[Dict[Tuple[int, int], complex]],
                          time_history: np.ndarray,
                          l_max: int) -> Dict[str, any]:
    """
    生成地磁场的综合分析报告。
    """
    g10_series = np.array([extract_dipole_parameters(c)["g10"] for c in coeffs_history])
    reversals = detect_reversals(g10_series, time_history)
    stats = reversal_statistics(reversals, time_history[-1] - time_history[0])
    l_vals, E_mean = magnetic_energy_spectrum(coeffs_history, l_max)

    # 最新时刻的偶极子参数
    latest_dipole = extract_dipole_parameters(coeffs_history[-1])

    return {
        "dipole_latest": latest_dipole,
        "reversals": reversals,
        "statistics": stats,
        "energy_spectrum_l": l_vals,
        "energy_spectrum_E": E_mean,
    }


# ---------------------------------------------------------------------------
# 自测试
# ---------------------------------------------------------------------------
def _self_test():
    # 构造模拟数据：g10 振荡并反转
    t = np.linspace(0.0, 1e6 * 365.25 * 24 * 3600, 1000)
    g10 = np.sin(2.0 * np.pi * t / (2.5e5 * 365.25 * 24 * 3600))
    revs = detect_reversals(g10, t)
    assert len(revs) >= 3  # 约 4 次反转

    # 偶极子提取
    coeffs = {(1, 0): 1.0 + 0.0j, (1, 1): 0.5 + 0.3j, (2, 0): 0.1 + 0.0j}
    dp = extract_dipole_parameters(coeffs)
    assert abs(dp["g10"] - 1.0) < 1e-10
    assert dp["dipole_moment_norm"] > 0.0

    # SVD
    mat = np.random.randn(20, 5)
    U, S, Vt = svd_field_modes(mat)
    assert len(S) <= 5

    print("field_analysis: self-test passed.")


if __name__ == "__main__":
    _self_test()
