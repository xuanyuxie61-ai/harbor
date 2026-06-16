"""
initial_conditions.py  —  宇宙学初始条件生成 (有色噪声 + 素数采样)
===============================================================

科学来源种子:
  - 201_colored_noise / f_alpha.m, r8vec_sftf.m, r8vec_sftb.m
    直接使用其 f_alpha 有色噪声生成思路:
        功率谱 P(k) ∝ k^(-α)
    通过傅里叶合成生成空间序列。r8vec_sftf/sftb 为正/逆正弦
    傅里叶变换,对应 FFT 的实值版本。
  - 1290_Dohoon1_Prediction-of-Halitosis (PCR microbiome data)
    借鉴其伪随机模采样与分组策略,映射为"原初宇宙随机模"生成:
    在 Fourier 空间对每个 k 模独立采样高斯随机相位,模的幅度
    由 P(k) 决定。
  - 915_prime_plot / prime_plot.m
    使用素数序列构造准蒙特卡洛 (QMC) 采样点,用于相位生成
    与功率谱验证的 k-模抽样。

物理背景:
  宇宙密度扰动 δ(x) 在初始时刻 (z ≈ 49) 为高斯随机场,
  其功率谱由 inflation 预言:
      P(k) = A_s (k/k_*)^{n_s - 1} T²(k)
  其中 A_s 为振幅, n_s ≈ 0.9667 为谱指数, T(k) 为转移函数。
  本代码使用 Eisenstein-Hu 无重子近似:
      T(k) = L(q) / (L(q) + C q²)
      L(q) = ln(2e + 1.8 q)
      C = 14.7 + 133/(169 + 0.3 q^{-1/4})  # 近似
      q = k / (Ω_m h² Mpc⁻¹)
  生成流程:
      1. 在 3D k-网格上计算 P(k)
      2. 对每个 k,采样 δ(k) = √P(k) · (高斯随机相位)
      3. IFFT 得到 δ(x)
      4. 使用 Zel'dovich 近似计算位移场 ψ = -∇Φ/(a² H f)
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Tuple

from cosmo_config import CosmoParams, z_to_a, hubble_factor


# ---------------------------------------------------------------------------- #
#                          素数序列工具 (seed 915)
# ---------------------------------------------------------------------------- #
def prime_sieve(limit: int) -> NDArray:
    """
    Eratosthenes 筛法返回 ≤ limit 的所有素数。
    对应 prime_plot.m 的素数检测逻辑,用于 QMC 采样。
    """
    if limit < 2:
        return np.array([], dtype=int)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0:2] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if is_prime[p]:
            is_prime[p * p: limit + 1: p] = False
    return np.nonzero(is_prime)[0].astype(int)


def prime_qmc_phases(n_modes: int, seed: int = 42) -> NDArray:
    """
    使用素数倒数构造准蒙特卡洛相位:
        φ_j = 2π · frac(j · p_j / q_j)
    其中 p_j, q_j 为相邻素数对,frac 为小数部分。
    此方法确保相位在 [0, 2π] 上低偏差分布。
    """
    primes = prime_sieve(10 * n_modes + 100)[: 2 * n_modes]
    rng = np.random.default_rng(seed)
    shifts = rng.uniform(0, 1, size=n_modes)
    phases = np.zeros(n_modes)
    for j in range(n_modes):
        p = primes[2 * j]
        q = primes[2 * j + 1]
        phases[j] = 2.0 * np.pi * ((j * p / q + shifts[j]) % 1.0)
    return phases


# ---------------------------------------------------------------------------- #
#                  E-H 无重子转移函数 (Eisenstein & Hu 1998)
# ---------------------------------------------------------------------------- #
def eisenstein_hu_transfer(k: NDArray, p: CosmoParams) -> NDArray:
    """
    无重子近似下的转移函数 T(k):
        q = k / (Ω_m h² Mpc⁻¹)
        L = ln(2e + 1.8 q)
        C = 14.7 + 133 / (169 + 0.3 q^{-1/4})   (q>0)
        T = L / (L + C q²)
    对 k=0 模返回 T(0)=1。
    """
    k = np.asarray(k, dtype=float)
    T = np.ones_like(k)
    nonzero = k > 1e-10
    q = k[nonzero] / (p.omega_m * p.hubble ** 2)
    Lq = np.log(2.0 * np.e + 1.8 * q)
    Cq = 14.7 + 133.0 / (169.0 + 0.3 * q ** (-0.25))
    T[nonzero] = Lq / (Lq + Cq * q * q)
    return T


def primordial_power_spectrum(k: NDArray, p: CosmoParams) -> NDArray:
    """
    物质功率谱:
        P(k) = A_s · (k/k_*)^{n_s-1} · T²(k) · (2π²/k³) · ...
    简化为 dimensionless Δ²(k) ∝ k^{n_s+3} T²(k) 用于生成。
    A_s 由 σ_8 归一化确定。
    """
    k = np.asarray(k, dtype=float)
    T = eisenstein_hu_transfer(k, p)
    # Dimensionless: Δ²(k) = k³ P(k) / (2π²)
    # P(k) ∝ k^{n_s} T²(k)
    P = np.zeros_like(k)
    nz = k > 1e-10
    P[nz] = (k[nz] ** p.spectral_index) * (T[nz] ** 2)
    return P


# ---------------------------------------------------------------------------- #
#                     3D 高斯随机场生成
# ---------------------------------------------------------------------------- #
def generate_delta_k(grid_shape: Tuple[int, int, int], box_length: float,
                     p: CosmoParams, seed: int = 42) -> NDArray:
    """
    在 3D 傅里叶空间生成密度扰动 δ(k):
        δ(k) = √P(k) · √V · (a + ib) / √2
    其中 a,b 为标准高斯, V = L³ 为盒子体积。
    实空间 δ(x) 需满足 Hermitian 对称: δ(-k) = δ*(k)。

    Parameters
    ----------
    grid_shape : (Nx, Ny, Nz)  必须相等
    box_length : float
    p : CosmoParams
    seed : int  随机种子

    Returns
    -------
    delta_k : (Nx, Ny, Nz//2+1) complex array  (rfft 存储)
    """
    nx, ny, nz = grid_shape
    if nx != ny or ny != nz:
        raise ValueError("仅支持立方网格")
    rng = np.random.default_rng(seed)
    # 波数:
    kfreq_x = np.fft.fftfreq(nx, d=box_length / nx) * (2.0 * np.pi)
    kfreq_y = np.fft.fftfreq(ny, d=box_length / ny) * (2.0 * np.pi)
    kfreq_z = np.fft.rfftfreq(nz, d=box_length / nz) * (2.0 * np.pi)
    kx, ky, kz = np.meshgrid(kfreq_x, kfreq_y, kfreq_z, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    k_mag = np.sqrt(k2)
    Pk = primordial_power_spectrum(k_mag, p)
    # 振幅:
    volume = box_length ** 3
    amp = np.sqrt(Pk * volume)
    # 高斯随机 (实部+虚部):
    gauss_real = rng.standard_normal(amp.shape)
    gauss_imag = rng.standard_normal(amp.shape)
    delta_k = amp * (gauss_real + 1j * gauss_imag) / np.sqrt(2.0)
    # k=0 模为零 (周期域平均扰动为零):
    delta_k[0, 0, 0] = 0.0
    return delta_k


def generate_density_field(grid_shape: Tuple[int, int, int],
                           box_length: float, p: CosmoParams,
                           seed: int = 42) -> NDArray:
    """
    生成实空间密度扰动 δ(x) = IFFT[δ(k)]。
    返回已减均值、无量纲的密度对比。
    """
    delta_k = generate_delta_k(grid_shape, box_length, p, seed)
    delta_x = np.fft.irfftn(delta_k, s=grid_shape)
    # 归一化至 σ_8 (在 8 Mpc/h 球窗上):
    delta_x = _normalize_to_sigma8(delta_x, box_length, p)
    return delta_x


def _normalize_to_sigma8(delta: NDArray, box_length: float,
                         p: CosmoParams) -> NDArray:
    """
    将密度场归一化至 σ_8:
        σ_8² = ∫ d³k/(2π)³ P(k) W²(kR)
    其中 W(x) = 3 (sin x - x cos x) / x³ 为 top-hat 窗函数。
    简化实现: 用实测 RMS 与理论 RMS 之比缩放。
    """
    # 实测当前 RMS:
    sigma_meas = float(np.std(delta))
    if sigma_meas < 1e-30:
        return delta
    # 理论 σ_8 (简化: 取 P(k) 在 k=0.125 h/Mpc 处的值):
    k_pivot = 0.125  # 2π/8 / (2π) = 1/8
    Pk_pivot = primordial_power_spectrum(np.array([k_pivot]), p)[0]
    sigma_theory = np.sqrt(Pk_pivot) * p.sigma_8 / max(np.sqrt(Pk_pivot), 1e-10)
    # 直接缩放至 sigma_8:
    scale = p.sigma_8 / max(sigma_meas, 1e-30)
    return delta * scale


# ---------------------------------------------------------------------------- #
#                    Zel'dovich 近似位移场
# ---------------------------------------------------------------------------- #
def zeldovich_displacement(delta_k: NDArray, grid_shape: Tuple[int, int, int],
                           box_length: float, p: CosmoParams,
                           z_init: float) -> NDArray:
    """
    Zel'dovich 近似: 粒子位移 ψ 与密度场的关系:
        ∇·ψ = -δ / D1(a)
    在 Fourier 空间:
        ψ(k) = i k δ(k) / (|k|² D1(a))
    其中 D1(a) 为线性增长因子。
    返回 ψ 作为 (3, Nx, Ny, Nz) 数组 (位移场)。
    """
    nx, ny, nz = grid_shape
    kfreq_x = np.fft.fftfreq(nx, d=box_length / nx) * (2.0 * np.pi)
    kfreq_y = np.fft.fftfreq(ny, d=box_length / ny) * (2.0 * np.pi)
    kfreq_z = np.fft.rfftfreq(nz, d=box_length / nz) * (2.0 * np.pi)
    kx, ky, kz = np.meshgrid(kfreq_x, kfreq_y, kfreq_z, indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    a_init = z_to_a(z_init)
    # 简化 D1 ≈ a_init (EdS 近似,对 Ω_m=1 精确; 对 ΛCDM 有修正):
    D1 = a_init * _growth_factor_approx(a_init, p)
    psi_k = np.zeros((3,) + delta_k.shape, dtype=complex)
    nz_mask = k2 > 1e-12
    # 对每个分量 α ∈ {x,y,z}:
    for alpha, kk in enumerate((kx, ky, kz)):
        psi_k[alpha, nz_mask] = (1j * kk[nz_mask] * delta_k[nz_mask]
                                  / (k2[nz_mask] * D1))
    # 转回实空间:
    psi = np.zeros((3, nx, ny, nz))
    for alpha in range(3):
        psi[alpha] = np.fft.irfftn(psi_k[alpha], s=grid_shape)
    return psi


def _growth_factor_approx(a: float, p: CosmoParams) -> float:
    """
    线性增长因子近似 (Carroll et al. 1992):
        D1(a) ≈ a · [ 1 + (Ω_Λ/Ω_m)^{1/4} · a^{3/2} ]^{-2/5}
                × 5 Ω_m / 2
    归一化为 D1(a=1) = 1 的近似。
    """
    if a <= 0:
        return 0.0
    x = (p.omega_l / p.omega_m) ** (1.0 / 3.0) * a
    D = 5.0 * p.omega_m / 2.0 * a / (
        1.0 + x ** (3.0 / 2.0) + (1.0 + p.omega_l / p.omega_m) ** (1.0 / 5.0) * a
    )
    # 归一化到 a=1:
    x1 = (p.omega_l / p.omega_m) ** (1.0 / 3.0)
    D1_now = 5.0 * p.omega_m / 2.0 / (
        1.0 + x1 ** (3.0 / 2.0) + (1.0 + p.omega_l / p.omega_m) ** (1.0 / 5.0)
    )
    return D / D1_now


# ---------------------------------------------------------------------------- #
#                    粒子采样 (基于 QMC 素数)
# ---------------------------------------------------------------------------- #
def sample_particle_positions(n_particles: int, box_length: float,
                              seed: int = 42) -> NDArray:
    """
    在盒子 [0, L]³ 内均匀采样 n_particles 个粒子位置。
    使用素数序列构造 Halton 低偏差序列 (QMC):
        x_j = L · frac(j / p_1)
        y_j = L · frac(j / p_2)
        z_j = L · frac(j / p_3)
    其中 p_1=2, p_2=3, p_3=5 为前 3 个素数。
    """
    primes = [2, 3, 5]
    pos = np.zeros((n_particles, 3))
    for j in range(1, n_particles + 1):
        for d, pr in enumerate(primes):
            # Halton 序列:
            f = 1.0
            r = 0.0
            jj = j
            while jj > 0:
                f /= pr
                r += f * (jj % pr)
                jj //= pr
            pos[j - 1, d] = r * box_length
    return pos
