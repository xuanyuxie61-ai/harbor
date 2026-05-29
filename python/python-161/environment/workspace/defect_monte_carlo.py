"""
defect_monte_carlo.py
基于种子项目 1006_random_data (uniform/random sampling in various domains)
和 780_mortality (mortality PDF/CDF statistics)
改造为钙钛矿太阳能电池中材料缺陷随机分布建模与载流子寿命统计模块。

在钙钛矿薄膜中，点缺陷（空位、间隙原子）的空间分布服从泊松过程，
其密度受合成条件影响具有空间非均匀性。缺陷作为 Shockley-Read-Hall (SRH)
复合中心，直接决定载流子寿命 τ。

核心公式：
  1. 缺陷在三角形区域 T 内的均匀采样（Turk's rule）：
       给定随机数 r1, r2 ~ U(0,1)
       a = 1 - sqrt(r2), b = (1-r1)*sqrt(r2), c = r1*sqrt(r2)
       P = a*v1 + b*v2 + c*v3
  2. 正态分布采样（Box-Muller）：
       Z = sqrt(-2 ln U1) * cos(2π U2)
  3. 缺陷密度服从对数正态分布：
       ln(N_t) ~ N(μ_N, σ_N^2)
  4. 载流子寿命（SRH，单能级陷阱）：
       τ_n = 1 / (N_t * σ_n * v_th)
       τ_p = 1 / (N_t * σ_p * v_th)
     其中 σ 为俘获截面，v_th = sqrt(3 k_B T / m*) 为热运动速度。
  5. 复合率：
       R_SRH = (n p - n_i^2) / (τ_p (n + n1) + τ_n (p + p1))
     n1 = N_c exp((E_t - E_c)/kT), p1 = N_v exp((E_v - E_t)/kT)

借用 780_mortality 的 PDF/CDF 思想，将载流子“死亡”（复合）建模为
年龄（寿命）相关的概率过程。
"""

import numpy as np
from typing import Tuple


# 物理常数
k_B = 1.380649e-23  # J/K
m_e = 9.10938356e-31  # kg
q = 1.602176634e-19  # C


def uniform_in_triangle(
    n: int, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray
) -> np.ndarray:
    """
    在三角形内均匀采样 n 个点（Turk's rule #1）。
    对应原项目 uniform_in_triangle。
    """
    v1, v2, v3 = np.asarray(v1, dtype=float), np.asarray(v2, dtype=float), np.asarray(v3, dtype=float)
    pts = np.zeros((n, len(v1)))
    rng = np.random.default_rng(123)
    r = rng.random((n, 2))
    for i in range(n):
        a = 1.0 - np.sqrt(r[i, 1])
        b = (1.0 - r[i, 0]) * np.sqrt(r[i, 1])
        c = r[i, 0] * np.sqrt(r[i, 1])
        pts[i] = a * v1 + b * v2 + c * v3
    return pts


def normal_square(n: int, d: int) -> np.ndarray:
    """
    生成 d 维标准正态分布的 n 个样本点。
    对应原项目 normal_square。
    """
    rng = np.random.default_rng(456)
    return rng.standard_normal((n, d))


def sample_defect_positions(
    n_defects: int,
    domain: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """
    在三角形区域集合内均匀采样缺陷位置。

    Parameters
    ----------
    n_defects : int
    domain : (v1_array, v2_array, v3_array)
        每个数组形状为 (n_tri, 2)，表示 n_tri 个三角形的顶点

    Returns
    -------
    positions : (n_defects, 2) array
    """
    v1, v2, v3 = domain
    n_tri = v1.shape[0]
    if n_tri == 0:
        return np.zeros((n_defects, 2))

    # 按三角形面积比例分配缺陷数
    areas = np.zeros(n_tri)
    for i in range(n_tri):
        a = v2[i] - v1[i]
        b = v3[i] - v1[i]
        areas[i] = 0.5 * abs(a[0] * b[1] - a[1] * b[0])
    total_area = areas.sum()
    if total_area <= 0:
        counts = np.full(n_tri, n_defects // n_tri)
    else:
        probs = areas / total_area
        counts = np.random.multinomial(n_defects, probs)

    positions = []
    for i in range(n_tri):
        if counts[i] > 0:
            pts = uniform_in_triangle(counts[i], v1[i], v2[i], v3[i])
            positions.append(pts)
    if positions:
        return np.vstack(positions)
    return np.zeros((0, 2))


def defect_density_lognormal(
    n_samples: int, mu_log: float = 24.0, sigma_log: float = 1.0
) -> np.ndarray:
    """
    缺陷密度的对数正态采样 [cm^{-3}]。
    典型钙钛矿缺陷密度：10^15–10^17 cm^{-3}。
    ln(N_t) ~ N(ln(1e16), 1.0^2) 对应 mu_log ≈ 36.8（以自然对数计，1e16 ≈ 36.8? 不对，ln(1e16)=36.8?)
    等等：ln(1e16) = 16 * ln(10) ≈ 36.84。但这里我们使用更方便的尺度。
    实际上使用 log10 更直观：log10(N_t) ~ N(16, 0.5^2)。
    下面用自然对数实现：mu_log = 16 * ln(10)。
    """
    mu_ln = mu_log * np.log(10.0)
    sigma_ln = sigma_log * np.log(10.0)
    z = normal_square(n_samples, 1).flatten()
    logN = mu_ln + sigma_ln * z
    N_t = np.exp(logN)
    return np.clip(N_t, 1e12, 1e19)


def carrier_lifetime_from_defects(
    N_t: np.ndarray,
    T: float = 300.0,
    sigma_n: float = 1e-16,  # cm^2 (典型深能级陷阱截面较小)
    sigma_p: float = 1e-16,  # cm^2
    m_eff: float = 0.15 * 9.10938356e-31,  # kg，有效质量
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从缺陷密度计算载流子寿命。

    τ = 1 / (N_t * σ * v_th)
    v_th = sqrt(3 k_B T / m_eff)
    """
    if T <= 0 or m_eff <= 0:
        raise ValueError("温度和有效质量必须为正")
    v_th = np.sqrt(3.0 * k_B * T / m_eff)  # m/s
    v_th_cm = v_th * 100.0  # cm/s
    tau_n = 1.0 / (N_t * sigma_n * v_th_cm)
    tau_p = 1.0 / (N_t * sigma_p * v_th_cm)
    return tau_n, tau_p


class MortalityStyleLifetimeModel:
    """
    借用 780_mortality 的 PDF/CDF 框架，将载流子复合建模为“寿命”过程。
    """

    def __init__(self, tau_values: np.ndarray):
        """
        Parameters
        ----------
        tau_values : array
            离散化的载流子寿命样本 [s]
        """
        self.tau = np.asarray(tau_values)
        self.tau = self.tau[self.tau > 0]
        if len(self.tau) == 0:
            self.tau = np.array([1e-9])

    def pdf(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算寿命的 PDF（概率密度函数）。
        返回 (tau_bins, pdf_values)
        """
        log_tau = np.log10(self.tau)
        bins = np.linspace(log_tau.min(), log_tau.max(), max(20, int(np.sqrt(len(self.tau)))))
        counts, edges = np.histogram(log_tau, bins=bins)
        pdf = counts / (counts.sum() * np.diff(edges))
        centers = 0.5 * (edges[:-1] + edges[1:])
        return 10.0 ** centers, pdf

    def cdf(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算寿命的 CDF（累积分布函数）。
        """
        log_tau = np.log10(self.tau)
        sorted_log = np.sort(log_tau)
        cdf = np.arange(1, len(sorted_log) + 1) / len(sorted_log)
        return 10.0 ** sorted_log, cdf

    def expected_lifetime(self) -> float:
        """平均寿命。"""
        return float(np.mean(self.tau))

    def survival_probability(self, t: float) -> float:
        """
        载流子在时间 t 后仍存活的概率（类比生存函数）。
        S(t) = exp(-t / τ_mean)
        """
        tau_mean = self.expected_lifetime()
        if tau_mean <= 0:
            return 0.0
        return float(np.exp(-t / tau_mean))


def srh_recombination_rate(
    n: float, p: float, n_i: float,
    tau_n: float, tau_p: float,
    E_t: float = 0.0,  # 陷阱能级相对本征费米能级 [eV]
    T: float = 300.0,
    N_c: float = 1e19, N_v: float = 1e19,
) -> float:
    """
    Shockley-Read-Hall 复合率 [cm^{-3}·s^{-1}]。

    R = (n p - n_i^2) / (τ_p (n + n1) + τ_n (p + p1))
    n1 = N_c exp((E_t - E_c)/kT)  (简化：假设 E_t 相对于本征能级)
    """
    kT_eV = 8.617333e-5 * T  # eV
    if kT_eV <= 0:
        raise ValueError("温度必须为正")
    n1 = N_c * np.exp(-E_t / kT_eV)
    p1 = N_v * np.exp(E_t / kT_eV)
    denom = tau_p * (n + n1) + tau_n * (p + p1)
    if denom <= 0:
        return 0.0
    return (n * p - n_i * n_i) / denom


if __name__ == "__main__":
    N_t = defect_density_lognormal(1000)
    tau_n, tau_p = carrier_lifetime_from_defects(N_t)
    model = MortalityStyleLifetimeModel(tau_n)
    t_bins, pdf_vals = model.pdf()
    print(f"平均电子寿命: {model.expected_lifetime():.3e} s")
    print(f"100 ns 存活概率: {model.survival_probability(1e-7):.4f}")
    R = srh_recombination_rate(1e15, 1e15, 1e10, tau_n.mean(), tau_p.mean())
    print(f"SRH 复合率: {R:.3e} cm^-3 s^-1")
