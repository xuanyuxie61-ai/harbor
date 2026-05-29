"""
statistics.py
=============
统计工具与随机采样模块

融入 asa005（标准正态分布累积密度函数 alnorm）、
fair_dice_simulation（离散概率分布采样）、
casino_simulation（随机过程与长时间统计演化）的核心算法，
为宇宙学模拟提供概率统计支撑。

核心公式
--------
标准正态分布累积分布函数 Φ(x):
    Φ(x) = ∫_{-∞}^{x} (1/√(2π)) exp(-t²/2) dt

采用 Hill (1973) AS 66 算法的有理函数近似:
    当 |z| ≤ 1.28 时，使用连分式展开
    当 |z| > 1.28 时，使用指数衰减尾部近似

高斯随机场的功率谱与方差关系:
    σ²(R) = ∫₀^∞ (k²/2π²) P(k) W²(kR) dk
    其中 W(kR) = 3 [sin(kR) - kR cos(kR)] / (kR)³ 为 top-hat 窗函数

离散分布采样（逆变换法）:
    给定概率质量函数 p(x_i)，构造累积分布:
        F_i = Σ_{j≤i} p(x_j)
    生成均匀随机数 u ~ U(0,1)，返回满足 F_{i-1} < u ≤ F_i 的 x_i
"""

import numpy as np
from typing import Tuple, List


def alnorm(x: float, upper: bool = False) -> float:
    """
    计算标准正态分布累积密度函数 Φ(x)（融入 asa005 / alnorm 核心算法）。

    Parameters
    ----------
    x : float
        积分上限/下限
    upper : bool
        若为 True，计算 ∫_x^∞ φ(t) dt；否则计算 ∫_{-∞}^x φ(t) dt

    Returns
    -------
    float
        累积概率值，范围 [0, 1]
    """
    # Hill (1973) AS 66 系数
    a1 = 5.75885480458
    a2 = 2.62433121679
    a3 = 5.92885724438
    b1 = -29.8213557807
    b2 = 48.6959930692
    c1 = -0.000000038052
    c2 = 0.000398064794
    c3 = -0.151679116635
    c4 = 4.8385912808
    c5 = 0.742380924027
    c6 = 3.99019417011
    con = 1.28
    d1 = 1.00000615302
    d2 = 1.98615381364
    d3 = 5.29330324926
    d4 = -15.1508972451
    d5 = 30.789933034
    ltone = 7.0
    p = 0.39894228044
    q = 0.39990348504
    r = 0.398942280385
    utzero = 18.66

    up = upper
    z = float(x)

    if z < 0.0:
        up = not up
        z = -z

    if ltone < z and (not up or utzero < z):
        return 0.0 if up else 1.0

    y = 0.5 * z * z

    if z <= con:
        value = 0.5 - z * (
            p
            - q
            * y
            / (y + a1 + b1 / (y + a2 + b2 / (y + a3)))
        )
    else:
        value = (
            r
            * np.exp(-y)
            / (
                z
                + c1
                + d1
                / (
                    z
                    + c2
                    + d2
                    / (
                        z
                        + c3
                        + d3
                        / (z + c4 + d4 / (z + c5 + d5 / (z + c6)))
                    )
                )
            )
        )

    if not up:
        value = 1.0 - value

    return value


def alnorm_array(x: np.ndarray, upper: bool = False) -> np.ndarray:
    """
    向量化版本的 alnorm。
    """
    vec = np.vectorize(lambda xi: alnorm(xi, upper))
    return vec(x)


def sample_discrete_cdf(n: int, pmf: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    根据离散概率质量函数进行逆变换采样（融入 fair_dice_simulation 核心思想）。

    Parameters
    ----------
    n : int
        采样次数
    pmf : np.ndarray
        概率质量函数，非负且和为 1
    rng : np.random.Generator, optional
        随机数生成器

    Returns
    -------
    samples : np.ndarray
        采样结果，为索引值 0, 1, ..., len(pmf)-1
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    pmf = np.asarray(pmf, dtype=float)
    if np.any(pmf < -1e-15):
        raise ValueError("概率质量函数包含负值")
    s = pmf.sum()
    if abs(s - 1.0) > 1e-6:
        if s == 0.0:
            raise ValueError("概率质量函数全为零")
        pmf = pmf / s
    cdf = np.cumsum(pmf)
    u = rng.random(n)
    samples = np.searchsorted(cdf, u)
    return samples


def gaussian_random_field_1d(
    N: int,
    L: float,
    power_spectrum: callable,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    生成一维高斯随机场（用于测试初始条件）。

    算法:
        1. 在傅里叶空间生成复高斯随机数 δ_k ~ N(0, P(k))
        2. 实数场要求 δ_{-k} = δ_k^*
        3. 逆 FFT 得到实空间场 δ(x)

    公式:
        ⟨δ_k δ_{k'}^*⟩ = (2π/L) P(k) δ_{k,k'}
        δ(x) = Σ_k δ_k exp(i k x)
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    Pk = power_spectrum(np.abs(k))
    # 实数场的 Hermitian 对称性
    real_part = rng.standard_normal(N)
    imag_part = rng.standard_normal(N)
    delta_k = np.zeros(N, dtype=complex)
    delta_k[0] = real_part[0] * np.sqrt(Pk[0])
    for i in range(1, N // 2):
        amp = np.sqrt(0.5 * Pk[i])
        delta_k[i] = (real_part[i] + 1j * imag_part[i]) * amp
        delta_k[N - i] = delta_k[i].conjugate()
    if N % 2 == 0:
        i = N // 2
        delta_k[i] = real_part[i] * np.sqrt(Pk[i])
    delta_x = np.fft.ifft(delta_k).real * N
    return delta_x


def tophat_window(kR: np.ndarray) -> np.ndarray:
    """
    三维 top-hat 窗函数的傅里叶变换:
        W(kR) = 3 [sin(kR) - kR cos(kR)] / (kR)³

    当 kR → 0 时，W → 1。
    """
    kr = np.asarray(kR, dtype=float)
    out = np.ones_like(kr)
    mask = kr > 1e-6
    kr_m = kr[mask]
    out[mask] = 3.0 * (np.sin(kr_m) - kr_m * np.cos(kr_m)) / (kr_m ** 3)
    return out


def variance_from_power_spectrum(
    R: float, k_arr: np.ndarray, P_arr: np.ndarray
) -> float:
    """
    通过功率谱计算尺度 R 上的质量涨落方差:
        σ²(R) = (1/2π²) ∫_0^∞ k² P(k) W²(kR) dk

    采用 Simpson 数值积分。
    """
    if R <= 0.0:
        raise ValueError("R 必须为正")
    integrand = k_arr ** 2 * P_arr * tophat_window(k_arr * R) ** 2
    # 确保 k_arr 等距或采用梯形法则
    sigma2 = np.trapezoid(integrand, k_arr) / (2.0 * np.pi ** 2)
    return sigma2


def casino_random_walk(
    initial_stakes: float,
    n_steps: int,
    win_factor: float = 1.2,
    loss_factor: float = 0.83,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, int, int]:
    """
    多plicative 随机行走模型（融入 casino_simulation 核心思想）。

    在宇宙学中可类比为:
        - 密度扰动的随机倍增（非线性 regime 的粗粒化描述）
        - 或 Monte Carlo 积分中的重要性采样权重演化

    演化规则:
        每一步以 1/2 概率 stakes *= win_factor，以 1/2 概率 stakes *= loss_factor

    长期期望:
        E[stakes] = stakes₀ · [(win_factor + loss_factor)/2]^n

    Parameters
    ----------
    initial_stakes : float
        初始值
    n_steps : int
        步数
    win_factor, loss_factor : float
        赢/输时的乘子
    rng : np.random.Generator
        随机数生成器

    Returns
    -------
    trajectory : np.ndarray
        演化轨迹
    n_wins, n_losses : int
        赢/输次数
    """
    if initial_stakes <= 0.0:
        raise ValueError("initial_stakes 必须为正")
    if n_steps < 0:
        raise ValueError("n_steps 不能为负")
    if rng is None:
        rng = np.random.default_rng(seed=42)
    trajectory = np.zeros(n_steps + 1)
    trajectory[0] = initial_stakes
    n_wins = 0
    n_losses = 0
    for i in range(1, n_steps + 1):
        if rng.random() < 0.5:
            trajectory[i] = trajectory[i - 1] * win_factor
            n_wins += 1
        else:
            trajectory[i] = trajectory[i - 1] * loss_factor
            n_losses += 1
    return trajectory, n_wins, n_losses


if __name__ == "__main__":
    # 自检
    print("alnorm(0) =", alnorm(0.0))
    print("alnorm(1.96) =", alnorm(1.96))
    print("alnorm(-1.96) =", alnorm(-1.96))
    pmf = np.array([1, 2, 3, 4, 5, 6]) / 21.0
    samples = sample_discrete_cdf(10000, pmf)
    print("离散采样均值 ≈", samples.mean(), "(理论=3.333)")
    traj, w, l = casino_random_walk(1.0, 100)
    print(f"Casino 随机行走最终值: {traj[-1]:.4f}, wins={w}, losses={l}")
