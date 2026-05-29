"""
power_spectrum.py
=================
功率谱与统计量计算模块

计算宇宙密度场的功率谱 P(k)、相关函数 ξ(r)、以及通过多维数值积分
计算高阶统计量（融入 nintlib 的 Monte Carlo 多维积分核心思想）。

核心物理公式
------------
功率谱定义:
    P(k) = V ⟨|δ_k|²⟩

    其中 δ_k 为密度场的傅里叶分量，V 为模拟体积，
    尖括号表示对同一 |k| 壳层的所有模式取平均。

相关函数（Wiener-Khinchin 定理）:
    ξ(r) = (1/2π²) ∫_0^∞ k² P(k) (sin(kr) / (kr)) dk

    或等价地，通过 FFT:
    ξ(r) = FFT⁻¹[|δ_k|²]

质量方差（top-hat 窗）:
    σ²(R) = (1/2π²) ∫_0^∞ k² P(k) W²(kR) dk

    W(kR) = 3 [sin(kR) - kR cos(kR)] / (kR)³

Monte Carlo 多维积分（融入 nintlib / monte_carlo_nd）:
    对于 d 维积分:
        I = ∫_Ω f(x) dx
    采用均匀随机采样估计:
        I ≈ V_Ω / N Σ_{i=1}^N f(x_i)

    其中 x_i ~ U(Ω)，V_Ω 为积分区域体积。
    误差估计: σ_I = V_Ω / √N · std(f(x_i))

离散化功率谱估计（binning）:
    将 k 空间按 |k| 分 bin，每 bin 内取平均:
        P(k_i) = (V / N_modes) Σ_{k ∈ bin_i} |δ_k|²
"""

import numpy as np
from typing import Tuple
from statistics import tophat_window


class PowerSpectrumEstimator:
    """
    从模拟密度场估计功率谱。
    """

    def __init__(self, N: int, L: float):
        """
        Parameters
        ----------
        N : int
            每维网格数
        L : float
            盒子边长
        """
        self.N = N
        self.L = L
        self.V = L ** 3
        self.dk = 2.0 * np.pi / L

    def estimate(self, delta: np.ndarray, n_bins: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        从密度对比度场估计功率谱 P(k)。

        Parameters
        ----------
        delta : np.ndarray, shape (N, N, N)
            密度对比度场
        n_bins : int, optional
            k 的 bin 数，默认 N//2

        Returns
        -------
        k_bins : np.ndarray
            每个 bin 的中心 k 值
        Pk : np.ndarray
            功率谱估计
        N_modes : np.ndarray
            每个 bin 的模式数
        """
        if n_bins is None:
            n_bins = self.N // 2
        delta_k = np.fft.fftn(delta) / (self.N ** 3)
        power = np.abs(delta_k) ** 2 * self.V

        # 构造 k 网格
        k_vec = 2.0 * np.pi * np.fft.fftfreq(self.N, d=self.L / self.N)
        kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
        k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

        k_min = self.dk
        k_max = np.pi * self.N / self.L
        k_edges = np.linspace(k_min, k_max, n_bins + 1)
        k_bins = 0.5 * (k_edges[:-1] + k_edges[1:])
        Pk = np.zeros(n_bins)
        N_modes = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = (k_mag >= k_edges[i]) & (k_mag < k_edges[i + 1])
            N_modes[i] = mask.sum()
            if N_modes[i] > 0:
                Pk[i] = power[mask].mean()
            else:
                Pk[i] = 0.0

        return k_bins, Pk, N_modes

    def compute_correlation_function(
        self, delta: np.ndarray, n_bins: int = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        通过 FFT 计算相关函数 ξ(r)。

        算法:
            ξ(r) = FFT⁻¹[|δ_k|²]
            然后对 |r| 分 bin 平均。
        """
        if n_bins is None:
            n_bins = self.N // 2
        delta_k = np.fft.fftn(delta) / (self.N ** 3)
        xi_grid = np.fft.ifftn(np.abs(delta_k) ** 2).real * (self.N ** 3) / self.V

        # r 网格
        r_vec = np.fft.fftfreq(self.N, d=1.0 / self.N) * (self.L / self.N)
        rx, ry, rz = np.meshgrid(r_vec, r_vec, r_vec, indexing="ij")
        r_mag = np.sqrt(rx ** 2 + ry ** 2 + rz ** 2)

        r_max = self.L / 2.0
        r_edges = np.linspace(0.0, r_max, n_bins + 1)
        r_bins = 0.5 * (r_edges[:-1] + r_edges[1:])
        xi = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            mask = (r_mag >= r_edges[i]) & (r_mag < r_edges[i + 1])
            counts[i] = mask.sum()
            if counts[i] > 0:
                xi[i] = xi_grid[mask].mean()
            else:
                xi[i] = 0.0

        return r_bins, xi


def monte_carlo_nd_integral(
    func: callable,
    dim: int,
    a: np.ndarray,
    b: np.ndarray,
    n_samples: int,
    rng: np.random.Generator = None,
) -> Tuple[float, float]:
    """
    多维 Monte Carlo 积分（融入 nintlib / monte_carlo_nd 核心算法）。

    Parameters
    ----------
    func : callable
        被积函数 f(x)，x 为长度为 dim 的向量
    dim : int
        维度
    a, b : np.ndarray
        积分下限与上限
    n_samples : int
        采样点数
    rng : np.random.Generator

    Returns
    -------
    result : float
        积分估计值
    error : float
        标准误差估计
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    a = np.asarray(a)
    b = np.asarray(b)
    volume = np.prod(b - a)
    samples = rng.random((n_samples, dim)) * (b - a) + a
    values = np.array([func(x) for x in samples])
    mean_val = values.mean()
    std_val = values.std(ddof=1)
    result = volume * mean_val
    error = volume * std_val / np.sqrt(n_samples)
    return result, error


def compute_sigma_r(
    k_arr: np.ndarray,
    P_arr: np.ndarray,
    R: float,
    n_int: int = 2000,
) -> float:
    """
    通过数值积分计算 σ(R)。

    σ²(R) = (1/2π²) ∫_0^∞ k² P(k) W²(kR) dk
    """
    if R <= 0.0:
        raise ValueError("R 必须为正")
    integrand = k_arr ** 2 * P_arr * tophat_window(k_arr * R) ** 2
    sigma2 = np.trapezoid(integrand, k_arr) / (2.0 * np.pi ** 2)
    return np.sqrt(sigma2)


def press_schechter_mass_function(
    M: np.ndarray,
    sigma_M: np.ndarray,
    rho_mean: float,
    delta_c: float = 1.686,
) -> np.ndarray:
    """
    Press-Schechter 质量函数:
        n(M) = √(2/π) (ρ̄/M²) (δ_c/σ) |d ln σ / d ln M| exp(-δ_c² / (2σ²))

    Parameters
    ----------
    M : np.ndarray
        质量数组
    sigma_M : np.ndarray
        对应质量上的 σ(M)
    rho_mean : float
        平均物质密度
    delta_c : float
        线性临界过密度

    Returns
    -------
    n_M : np.ndarray
        单位体积单位质量内的暗物质晕数量
    """
    M = np.asarray(M)
    sigma = np.asarray(sigma_M)
    if np.any(sigma <= 0.0):
        raise ValueError("σ(M) 必须为正")
    # 数值微分计算 d ln σ / d ln M
    lnM = np.log(M)
    lns = np.log(sigma)
    dln_sigma_dlnM = np.gradient(lns, lnM)

    nu = delta_c / sigma
    prefactor = np.sqrt(2.0 / np.pi) * (rho_mean / M ** 2) * nu * np.abs(dln_sigma_dlnM)
    n_M = prefactor * np.exp(-0.5 * nu ** 2)
    return n_M


def spherical_overdensity_criterion(
    delta_grid: np.ndarray,
    threshold: float,
    L: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    球形过密度判据下的粗粒化密度峰值识别。

    对每个网格点，计算以其为中心的球形区域（半径 R）内的平均过密度，
    若超过 threshold 则标记为候选晕中心。

    公式:
        δ̄(R, x) = ∫ W(y-x, R) δ(y) d³y
    """
    N = delta_grid.shape[0]
    # 简单最近邻球形平均（效率考虑，用 top-hat 窗的 FFT 卷积）
    delta_k = np.fft.fftn(delta_grid)
    k_vec = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    # 默认 R = L / N * 3
    R = L / N * 3.0
    W = tophat_window(k_mag * R)
    delta_smooth_k = delta_k * W
    delta_smooth = np.fft.ifftn(delta_smooth_k).real
    peaks = delta_smooth > threshold
    # 返回峰值坐标与平滑密度
    coords = np.argwhere(peaks)
    # 转换为物理坐标
    phys_coords = coords * (L / N)
    return phys_coords, delta_smooth[peaks]


if __name__ == "__main__":
    # 自检
    N = 32
    L = 100.0
    est = PowerSpectrumEstimator(N, L)
    delta = np.random.randn(N, N, N) * 0.1
    k_bins, Pk, Nm = est.estimate(delta)
    print(f"功率谱估计: k_min={k_bins[0]:.4f}, P(k_min)={Pk[0]:.4e}")

    # Monte Carlo 积分测试
    def test_func(x):
        return np.prod(x ** 2)

    val, err = monte_carlo_nd_integral(test_func, 3, [0, 0, 0], [1, 1, 1], 100000)
    print(f"Monte Carlo ∫ x²y²z² dxdydz = {val:.5f} ± {err:.5f} (理论=1/27≈0.03704)")

    # Press-Schechter 测试
    M = np.logspace(10, 15, 50)
    sigma = 2.0 * (M / 1e14) ** (-0.2)
    nM = press_schechter_mass_function(M, sigma, 2.7e11, delta_c=1.686)
    print(f"Press-Schechter n(M=1e14) = {np.interp(1e14, M, nM):.4e}")
