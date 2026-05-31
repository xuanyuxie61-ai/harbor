# -*- coding: utf-8 -*-
"""
power_spectrum.py
CMB 角功率谱 C_l 计算与声学峰定位

核心物理：
    温度各向异性功率谱：
        C_l^{TT} = 4π ∫ (dk/k) P_R(k) [T_l^T(k)]^2
    极化功率谱 C_l^{EE}, C_l^{TE} 类似。

    声学峰位置 l_p 满足 dC_l/dl = 0。
    使用二分法（bisection）在导数变号区间内精确寻找峰位。
    峰间距比 R = l_3 / l_2 对宇宙几何敏感，是暗能量的关键探针。

融合种子项目 094_bisection（二分法根搜索）。
"""

import numpy as np
from typing import List, Tuple
from utils import ensure_positive


# ---------------------------------------------------------------------------
# 原初功率谱
# ---------------------------------------------------------------------------
def primordial_power_spectrum(k: float, A_s: float = 2.1e-9,
                               n_s: float = 0.965, k_pivot: float = 0.05) -> float:
    """
    标量原初功率谱：
        P_R(k) = A_s (k / k_pivot)^{n_s - 1}
    """
    if k <= 0.0:
        return 0.0
    return A_s * (k / k_pivot) ** (n_s - 1.0)


# ---------------------------------------------------------------------------
# 角功率谱计算
# ---------------------------------------------------------------------------
def compute_Cl_spectrum(l_values: np.ndarray,
                        transfer_func: callable,
                        k_min: float = 1e-4,
                        k_max: float = 1.0,
                        n_k: int = 200) -> np.ndarray:
    """
    对给定的 l 数组计算 C_l 功率谱。
    使用对数均匀 k 网格上的梯形积分。
    """
    from los_integration import los_integral_power_spectrum
    Cl = np.zeros(len(l_values))
    for idx, l in enumerate(l_values):
        def Tl(k: float) -> float:
            return transfer_func(l, k)
        Cl[idx] = los_integral_power_spectrum(
            l, Tl, primordial_power_spectrum, k_min, k_max, n_quad=min(n_k, 64)
        )
    return Cl


# ---------------------------------------------------------------------------
# 二分法寻找声学峰
# ---------------------------------------------------------------------------
def bisection_root(f: callable, a: float, b: float,
                   tol: float = 1e-6, max_iter: int = 100) -> float:
    """
    经典二分法求根：寻找 f(x)=0 在 [a,b] 内的根，要求 f(a)*f(b) < 0。
    迭代公式：c = (a+b)/2，保留异号子区间。
    收敛速率线性，保证 |x_n - x*| ≤ (b-a)/2^n。
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError(f"二分法要求端点异号，当前 f(a)={fa}, f(b)={fb}")
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if abs(fc) < tol or (b - a) < 2.0 * tol:
            return c
        if fa * fc < 0.0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
    return 0.5 * (a + b)


def find_acoustic_peaks(l_values: np.ndarray, Cl: np.ndarray,
                        n_peaks: int = 3) -> List[Tuple[int, float]]:
    """
    使用数值微分 + 二分法定位 C_l 的前 n_peaks 个声学峰。
    返回 [(l_peak, C_l_peak), ...]。
    """
    # 数值微分 dC_l/dl（中心差分）
    dCdl = np.zeros(len(Cl))
    dCdl[0] = (Cl[1] - Cl[0]) / (l_values[1] - l_values[0])
    dCdl[-1] = (Cl[-1] - Cl[-2]) / (l_values[-1] - l_values[-2])
    for i in range(1, len(Cl) - 1):
        dCdl[i] = (Cl[i + 1] - Cl[i - 1]) / (l_values[i + 1] - l_values[i - 1])

    peaks = []
    sign_changes = []
    for i in range(len(dCdl) - 1):
        if dCdl[i] > 0 and dCdl[i + 1] < 0:
            sign_changes.append((l_values[i], l_values[i + 1]))

    for a, b in sign_changes[:n_peaks]:
        try:
            def dCdl_interp(l: float) -> float:
                # 线性插值求导数
                if l <= l_values[0] or l >= l_values[-1]:
                    return 0.0
                idx = int(np.searchsorted(l_values, l)) - 1
                idx = max(0, min(idx, len(l_values) - 2))
                frac = (l - l_values[idx]) / (l_values[idx + 1] - l_values[idx])
                return dCdl[idx] + frac * (dCdl[idx + 1] - dCdl[idx])

            l_peak = bisection_root(dCdl_interp, a, b, tol=0.5)
            # 插值 C_l 峰值
            idx = int(np.searchsorted(l_values, l_peak)) - 1
            idx = max(0, min(idx, len(l_values) - 2))
            frac = (l_peak - l_values[idx]) / (l_values[idx + 1] - l_values[idx])
            C_peak = Cl[idx] + frac * (Cl[idx + 1] - Cl[idx])
            peaks.append((int(round(l_peak)), C_peak))
        except ValueError:
            continue

    return peaks


def compute_peak_spacing_ratio(peaks: List[Tuple[int, float]]) -> float:
    """
    计算声学峰间距比 R = l_{n+1} / l_n。
    对平坦 ΛCDM，理论预测 R ≈ 1.5–1.6（第三峰/第二峰）。
    """
    if len(peaks) < 2:
        return 0.0
    l_positions = [p[0] for p in peaks]
    ratios = []
    for i in range(1, len(l_positions)):
        if l_positions[i - 1] > 0:
            ratios.append(l_positions[i] / l_positions[i - 1])
    return float(np.mean(ratios)) if ratios else 0.0
