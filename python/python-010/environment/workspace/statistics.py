
import numpy as np
from typing import Tuple, List


def alnorm(x: float, upper: bool = False) -> float:

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
    vec = np.vectorize(lambda xi: alnorm(xi, upper))
    return vec(x)


def sample_discrete_cdf(n: int, pmf: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
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
    if rng is None:
        rng = np.random.default_rng(seed=42)
    k = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    Pk = power_spectrum(np.abs(k))

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
    kr = np.asarray(kR, dtype=float)
    out = np.ones_like(kr)
    mask = kr > 1e-6
    kr_m = kr[mask]
    out[mask] = 3.0 * (np.sin(kr_m) - kr_m * np.cos(kr_m)) / (kr_m ** 3)
    return out


def variance_from_power_spectrum(
    R: float, k_arr: np.ndarray, P_arr: np.ndarray
) -> float:
    if R <= 0.0:
        raise ValueError("R 必须为正")
    integrand = k_arr ** 2 * P_arr * tophat_window(k_arr * R) ** 2

    sigma2 = np.trapezoid(integrand, k_arr) / (2.0 * np.pi ** 2)
    return sigma2


def casino_random_walk(
    initial_stakes: float,
    n_steps: int,
    win_factor: float = 1.2,
    loss_factor: float = 0.83,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, int, int]:
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

    print("alnorm(0) =", alnorm(0.0))
    print("alnorm(1.96) =", alnorm(1.96))
    print("alnorm(-1.96) =", alnorm(-1.96))
    pmf = np.array([1, 2, 3, 4, 5, 6]) / 21.0
    samples = sample_discrete_cdf(10000, pmf)
    print("离散采样均值 ≈", samples.mean(), "(理论=3.333)")
    traj, w, l = casino_random_walk(1.0, 100)
    print(f"Casino 随机行走最终值: {traj[-1]:.4f}, wins={w}, losses={l}")
