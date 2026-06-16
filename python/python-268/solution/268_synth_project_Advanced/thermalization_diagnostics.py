"""
thermalization_diagnostics.py - DQMC 热化诊断与非平衡分析
============================================================

科学背景 (Scientific Background):
    DQMC 的可靠性取决于马尔可夫链是否达到热平衡.
    热化诊断包括:
        1. 时间序列分析: 可观测量是否稳定
        2. 自相关时间: 统计独立样本的间隔
        3. 遍历性检查: 不同初值是否收敛到同一分布
        4. 非平衡检测: 系统是否仍在弛豫

    本模块融合种子项目 1014_GlacierWeilin_AAR_disequilibrium 的思想:
        原项目分析冰川积累区比率 (AAR) 的"非平衡" (disequilibrium).
        类比: DQMC 中能量/双占据的时间序列类似 AAR 的时间序列,
        偏离稳态即为 "非平衡".

    关键量:
        AAR (Accumulation Area Ratio): 冰川正向质量平衡的面积占比
        DQMC 类比: "平衡比率" = 处于稳态范围内的 MC 步占比

核心公式 (Key Formulas):
    自相关函数:
        C(t) = ⟨O(s) O(s+t)⟩ - ⟨O⟩²
    归一化:
        ρ(t) = C(t) / C(0)

    积分自相关时间:
        τ_int = 1/2 + Σ_{t=1}^{∞} ρ(t)

    有效样本数:
        N_eff = N_total / (2 τ_int)

    Gelman-Rubin 统计量 (多链诊断):
        R̂ = √(V̂ / W)
        V̂ = (n-1)/n W + (m+1)/(mn) B
        W = 链内方差, B = 链间方差
        收敛条件: R̂ < 1.1
"""

import numpy as np
from typing import Tuple, Dict, List, Optional


# ==========================================================================
#  自相关分析
# ==========================================================================

def autocorrelation_function(series: np.ndarray, max_lag: int = None
                             ) -> np.ndarray:
    """
    计算时间序列的归一化自相关函数.

    ρ(t) = [⟨O(s) O(s+t)⟩ - ⟨O⟩²] / [⟨O²⟩ - ⟨O⟩²]

    使用 FFT 加速:
        C(t) = IFFT[ |FFT[O - ⟨O⟩]|² ] / N
    """
    N = len(series)
    if max_lag is None:
        max_lag = min(N // 4, 200)

    mean = np.mean(series)
    x = series - mean
    var = np.var(series)

    if var < 1e-30:
        return np.zeros(max_lag + 1)

    # FFT 加速自相关
    n_fft = 2 ** int(np.ceil(np.log2(2 * N - 1)))
    x_fft = np.fft.fft(x, n=n_fft)
    acf_full = np.fft.ifft(x_fft * np.conj(x_fft)).real
    acf_full = acf_full[:N] / (np.arange(N, 0, -1) * var)

    return acf_full[:max_lag + 1]


def integrated_autocorrelation_time(series: np.ndarray,
                                    window_factor: float = 10.0
                                    ) -> Dict[str, float]:
    """
    计算积分自相关时间 τ_int.

    τ_int = 1/2 + Σ_{t=1}^{M} ρ(t)

    窗口 M 的选择 (自洽窗口):
        M = C * τ_int,  其中 C ≈ 10

    物理意义:
        每 2τ_int 步才有一个统计独立的样本.
        N_eff = N_total / (2 τ_int)
    """
    N = len(series)
    acf = autocorrelation_function(series, max_lag=N // 4)

    # 自洽窗口
    tau_est = 0.5
    for _ in range(5):
        window = int(window_factor * tau_est)
        window = max(1, min(window, len(acf) - 1))
        tau_est = 0.5 + np.sum(acf[1:window + 1])

    tau_int = max(tau_est, 0.5)
    n_eff = N / (2.0 * tau_int)

    return {
        'tau_int': float(tau_int),
        'n_eff': float(n_eff),
        'n_total': N,
        'efficiency': float(n_eff / N) if N > 0 else 0.0,
    }


# ==========================================================================
#  非平衡检测 (融合 AAR disequilibrium 思想)
# ==========================================================================

def compute_aar_disequilibrium(series: np.ndarray,
                               window_size: int = 20,
                               threshold_std: float = 2.0
                               ) -> Dict[str, float]:
    """
    计算"积累区比率非平衡度" (AAR Disequilibrium).

    融合种子项目 1014_GlacierWeilin_AAR_disequilibrium.

    冰川学中的 AAR:
        AAR = A_accumulation / A_total
        平衡态 AAR ≈ 0.5-0.8 (取决于气候)

    DQMC 类比:
        将时间序列分窗口, 计算每个窗口的"平衡指标":
        AAR(t) = #{s ∈ window: |O(s) - ⟨O⟩| < threshold × σ} / window_size

        disequilibrium = 1 - AAR (偏离平衡的程度)

    返回值:
        'aar': 平均积累区比率
        'disequilibrium': 非平衡度 (0=完全平衡, 1=完全不平衡)
        'trend_slope': 线性趋势斜率 (非零 → 仍在弛豫)
    """
    N = len(series)
    if N < window_size * 2:
        return {
            'aar': 0.5,
            'disequilibrium': 0.5,
            'trend_slope': 0.0,
            'is_equilibrated': False,
        }

    mean = np.mean(series)
    std = np.std(series)
    if std < 1e-15:
        return {
            'aar': 1.0,
            'disequilibrium': 0.0,
            'trend_slope': 0.0,
            'is_equilibrated': True,
        }

    # 滑动窗口 AAR
    aar_values = []
    for start in range(0, N - window_size + 1, max(window_size // 2, 1)):
        window = series[start:start + window_size]
        in_balance = np.sum(np.abs(window - mean) < threshold_std * std)
        aar_values.append(in_balance / window_size)

    aar_mean = np.mean(aar_values) if aar_values else 0.5
    disequilibrium = 1.0 - aar_mean

    # 线性趋势 (最小二乘)
    t = np.arange(N, dtype=float)
    t_mean = np.mean(t)
    o_mean = np.mean(series)
    slope = np.sum((t - t_mean) * (series - o_mean)) / max(np.sum((t - t_mean) ** 2), 1e-30)
    # 归一化斜率
    normalized_slope = slope / max(std, 1e-15)

    return {
        'aar': float(aar_mean),
        'disequilibrium': float(disequilibrium),
        'trend_slope': float(normalized_slope),
        'is_equilibrated': disequilibrium < 0.2 and abs(normalized_slope) < 0.01,
    }


# ==========================================================================
#  Gelman-Rubin 多链诊断
# ==========================================================================

def gelman_rubin_statistic(chains: List[np.ndarray]) -> Dict[str, float]:
    """
    Gelman-Rubin R̂ 统计量 (多链收敛诊断).

    融合多链思想: 运行 m 条独立 MC 链, 比较链内/链间方差.

    计算:
        W = (1/m) Σ_j s_j²     (平均链内方差)
        B = n/(m-1) Σ_j (x̄_j - x̄)²  (链间方差)
        V̂ = (n-1)/n W + (m+1)/(mn) B
        R̂ = √(V̂ / W)

    收敛条件: R̂ < 1.1 (严格), R̂ < 1.01 (精确)
    """
    m = len(chains)
    if m < 2:
        return {'R_hat': 1.0, 'converged': True, 'n_chains': 1}

    n = min(len(c) for c in chains)
    chain_means = np.array([np.mean(c[:n]) for c in chains])
    chain_vars = np.array([np.var(c[:n], ddof=1) for c in chains])

    overall_mean = np.mean(chain_means)
    W = np.mean(chain_vars)
    B = n * np.var(chain_means, ddof=1)

    if W < 1e-30:
        return {'R_hat': 1.0, 'converged': True, 'W': W, 'B': B}

    V_hat = (n - 1) / n * W + (m + 1) / (m * n) * B
    R_hat = np.sqrt(V_hat / W)

    return {
        'R_hat': float(R_hat),
        'converged': R_hat < 1.1,
        'W': float(W),
        'B': float(B),
        'n_chains': m,
        'chain_length': n,
    }


# ==========================================================================
#  阻塞分析 (Blocking analysis)
# ==========================================================================

def blocking_analysis(series: np.ndarray,
                      max_block_size: int = None
                      ) -> Dict[str, np.ndarray]:
    """
    阻塞分析: 估计统计误差随块大小的变化.

    将 N 个数据点分为 N/b 个大小为 b 的块, 计算块平均.
    块平均的方差:
        σ²_b = Var[块平均] ≈ σ² (2τ_int) / b  (for b >> τ_int)

    当 b >> τ_int 时, σ_b 趋于平台 → 真实统计误差.
    """
    N = len(series)
    if max_block_size is None:
        max_block_size = N // 4

    block_sizes = []
    errors = []

    b = 1
    while b <= max_block_size:
        n_blocks = N // b
        if n_blocks < 2:
            break
        block_means = np.array([
            np.mean(series[i * b:(i + 1) * b])
            for i in range(n_blocks)
        ])
        se = np.std(block_means, ddof=1) / np.sqrt(n_blocks)
        block_sizes.append(b)
        errors.append(se)
        # 指数增长块大小
        if b < 10:
            b += 1
        else:
            b = int(b * 1.5)

    return {
        'block_sizes': np.array(block_sizes),
        'standard_errors': np.array(errors),
    }


# ==========================================================================
#  综合热化诊断报告
# ==========================================================================

def comprehensive_thermalization_check(measurements: Dict[str, np.ndarray]
                                       ) -> Dict[str, Dict]:
    """
    对 DQMC 测量结果进行综合热化诊断.

    检查:
        1. 各可观测量的自相关时间
        2. AAR 非平衡度
        3. 趋势检验

    参数:
        measurements: DQMC 输出字典 (包含 'energy', 'double_occ', 'sign' 等)
    """
    report = {}
    for obs_name in ['energy', 'double_occ', 'sign']:
        key = obs_name + '_series' if obs_name + '_series' in measurements else None
        if key is None:
            key = obs_name
        if key not in measurements:
            continue
        series = measurements[key]
        if len(series) < 10:
            continue

        autocorr_info = integrated_autocorrelation_time(series)
        aar_info = compute_aar_disequilibrium(series)
        blocking_info = blocking_analysis(series)

        report[obs_name] = {
            'autocorrelation': autocorr_info,
            'aar_disequilibrium': aar_info,
            'blocking': {
                'max_block_size': int(blocking_info['block_sizes'][-1]) if len(blocking_info['block_sizes']) > 0 else 0,
                'final_error': float(blocking_info['standard_errors'][-1]) if len(blocking_info['standard_errors']) > 0 else 0.0,
            },
            'is_thermalized': (
                aar_info['is_equilibrated'] and
                autocorr_info['n_eff'] > 10
            ),
        }
    return report
