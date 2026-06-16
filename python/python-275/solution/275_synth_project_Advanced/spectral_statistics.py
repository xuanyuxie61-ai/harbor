"""
spectral_statistics.py
======================
非厄米谱的统计性质分析.
融合种子项目:
  - 024_asa005: 正态分布函数 (统计检验)
  - 1377_usa_box_plot: 分位数与统计摘要 (无可视化)
  - 284_digital_dice: 蒙特卡洛采样与估计

物理背景:
  随机矩阵理论 (RMT) 对非厄米系统的谱统计给出普适预言:
    - 厄米 (GUE): Wigner-Dyson 分布, 能级排斥 p(s) ~ s*exp(-πs²/4).
    - 非厄米 (Ginibre): 本征值在复平面均匀, 间距分布不同.
    - PT 对称 (过渡区): 从 Poisson 到 Wigner-Dyson 的 crossover.
  统计量:
    - 最近邻间距分布 P(s).
    - 谱刚性 Σ²(L).
    - 比率 r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, List
import nonhermitian_hamiltonian as nh


# ---------------------------------------------------------------------------
# 正态分布函数 (源自 024_asa005 alnorm)
# ---------------------------------------------------------------------------
def normal_cdf(z: float) -> float:
    """标准正态分布累积函数 Φ(z).

    采用有理逼近 (Abramowitz & Stegun 26.2.17):
      Φ(z) = 1 - φ(z)(b1*t + b2*t² + b3*t³ + b4*t⁴ + b5*t⁵)
    其中 t = 1/(1 + 0.2316419*|z|), φ(z) = exp(-z²/2)/√(2π).

    精度: |误差| < 7.5e-8.
    """
    if z == 0.0:
        return 0.5
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    b = np.array([0.319381530, -0.356563782, 1.781477937,
                  -1.821255978, 1.330274429])
    poly = 0.0
    t_power = t
    for bi in b:
        poly += bi * t_power
        t_power *= t
    cdf = 1.0 - phi * poly
    if z < 0:
        cdf = 1.0 - cdf
    return float(cdf)


def normal_inverse_cdf(p: float) -> float:
    """正态分布逆函数 (分位点).

    Rational approximation (Beasley-Springer-Moro).
    """
    if p <= 0 or p >= 1:
        raise ValueError("p must be in (0, 1)")
    if p == 0.5:
        return 0.0
    if p < 0.5:
        t = np.sqrt(-2 * np.log(p))
    else:
        t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    z = t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)
    if p < 0.5:
        z = -z
    return float(z)


# ---------------------------------------------------------------------------
# 谱统计量
# ---------------------------------------------------------------------------
def level_spacing_distribution(E: np.ndarray,
                               n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """计算最近邻间距分布 P(s).

    步骤:
      1. 排序本征值 (取实部).
      2. 计算间距 s_n = E_{n+1} - E_n.
      3.  unfolding: s_n → s_n / <s>, 使平均间距为 1.
      4. 统计直方图.

    对 GUE: P(s) = (π/2) s exp(-π s² / 4).
    对 Poisson: P(s) = exp(-s).

    Returns
    -------
    s_centers, P_s : 直方图中心与概率密度.
    """
    E_real = np.sort(E.real)
    spacings = np.diff(E_real)
    spacings = spacings[spacings > 1e-12]
    if len(spacings) < 3:
        return np.array([0.0]), np.array([1.0])
    mean_s = np.mean(spacings)
    if mean_s < 1e-15:
        return np.array([0.0]), np.array([1.0])
    s_unfolded = spacings / mean_s
    bins = np.linspace(0, max(3.0, np.percentile(s_unfolded, 99)), n_bins + 1)
    hist, _ = np.histogram(s_unfolded, bins=bins, density=True)
    s_centers = 0.5 * (bins[:-1] + bins[1:])
    return s_centers, hist


def level_spacing_ratio(E: np.ndarray) -> Tuple[float, float]:
    """计算相邻间距比率 r 及其平均值 <r>.

    r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1})
    对 GOE: <r> ≈ 0.5307
    对 GUE: <r> ≈ 0.6027
    对 Poisson: <r> = 2 ln 2 - 1 ≈ 0.3863

    Returns
    -------
    r_mean, r_std : 平均比率与标准差.
    """
    E_real = np.sort(E.real)
    spacings = np.diff(E_real)
    spacings = spacings[spacings > 1e-12]
    if len(spacings) < 2:
        return 0.0, 0.0
    r_vals = np.zeros(len(spacings) - 1)
    for i in range(len(spacings) - 1):
        s1, s2 = spacings[i], spacings[i + 1]
        r_vals[i] = min(s1, s2) / max(s1, s2)
    return float(np.mean(r_vals)), float(np.std(r_vals))


def spectral_rigidity(E: np.ndarray, L_max: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """谱刚性 Σ²(L): 区间 [E, E+L] 中本征值数的涨落.

    Σ²(L) = <(N(L) - L)²>
    对 GUE: Σ²(L) ~ (2/π²) ln(L) (对大 L).
    对 Poisson: Σ²(L) = L.

    Returns
    -------
    L_vals, Sigma2 : L 与刚性.
    """
    E_real = np.sort(E.real)
    N_total = len(E_real)
    if N_total < 5:
        return np.array([1.0]), np.array([0.0])
    mean_density = N_total / (E_real[-1] - E_real[0] + 1e-15)
    L_vals = np.arange(1, min(L_max + 1, N_total // 2))
    Sigma2 = np.zeros(len(L_vals))
    for idx, L in enumerate(L_vals):
        n_windows = max(1, N_total - L)
        fluctuations = []
        for start in range(0, N_total - L, max(1, L // 2)):
            n_in_window = np.sum((E_real >= E_real[start]) &
                                 (E_real < E_real[start] + L / mean_density))
            expected = L
            fluctuations.append((n_in_window - expected) ** 2)
        Sigma2[idx] = np.mean(fluctuations) if fluctuations else 0.0
    return L_vals.astype(float), Sigma2


# ---------------------------------------------------------------------------
# 蒙特卡洛无序平均 (源自 284_digital_dice)
# ---------------------------------------------------------------------------
def disordered_average(H_builder: Callable, n_samples: int,
                       disorder_strength: float,
                       seed: int = 42) -> Dict:
    """对非厄米系统做蒙特卡洛无序平均.

    在位能添加随机非厄米扰动: ε_i ~ Uniform(-W, W) + i*γ*randn.

    Returns
    -------
    info : dict with 'mean_spectrum', 'std_spectrum', 'samples'.
    """
    rng = np.random.default_rng(seed)
    H_base = H_builder()
    N = H_base.shape[0]
    all_E = []
    for _ in range(n_samples):
        disorder_real = rng.uniform(-disorder_strength, disorder_strength, N)
        disorder_imag = 1j * disorder_strength * rng.standard_normal(N)
        H_dis = H_base + np.diag(disorder_real + disorder_imag)
        E = np.linalg.eigvals(H_dis)
        all_E.append(E)
    all_E = np.array(all_E)
    return {
        "mean_spectrum": np.mean(all_E, axis=0),
        "std_spectrum": np.std(all_E, axis=0),
        "n_samples": n_samples,
        "disorder_strength": disorder_strength,
        "all_eigenvalues": all_E,
    }


# ---------------------------------------------------------------------------
# 统计摘要 (源自 1377_usa_box_plot 的分位数思想, 无绘图)
# ---------------------------------------------------------------------------
def spectral_summary(E: np.ndarray) -> Dict:
    """谱的分位数统计摘要.

    计算实部/虚部的五数概要 (min, Q1, median, Q3, max).
    """
    E_real = np.sort(E.real)
    E_imag = np.sort(E.imag)
    return {
        "real_min": float(E_real[0]),
        "real_Q1": float(np.percentile(E_real, 25)),
        "real_median": float(np.median(E_real)),
        "real_Q3": float(np.percentile(E_real, 75)),
        "real_max": float(E_real[-1]),
        "imag_min": float(E_imag[0]),
        "imag_Q1": float(np.percentile(E_imag, 25)),
        "imag_median": float(np.median(E_imag)),
        "imag_Q3": float(np.percentile(E_imag, 75)),
        "imag_max": float(E_imag[-1]),
        "n_levels": len(E),
    }
