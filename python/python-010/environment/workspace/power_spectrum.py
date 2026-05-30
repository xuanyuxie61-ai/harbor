
import numpy as np
from typing import Tuple
from statistics import tophat_window


class PowerSpectrumEstimator:

    def __init__(self, N: int, L: float):
        self.N = N
        self.L = L
        self.V = L ** 3
        self.dk = 2.0 * np.pi / L

    def estimate(self, delta: np.ndarray, n_bins: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_bins is None:
            n_bins = self.N // 2
        delta_k = np.fft.fftn(delta) / (self.N ** 3)
        power = np.abs(delta_k) ** 2 * self.V


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
        if n_bins is None:
            n_bins = self.N // 2
        delta_k = np.fft.fftn(delta) / (self.N ** 3)
        xi_grid = np.fft.ifftn(np.abs(delta_k) ** 2).real * (self.N ** 3) / self.V


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
    M = np.asarray(M)
    sigma = np.asarray(sigma_M)
    if np.any(sigma <= 0.0):
        raise ValueError("σ(M) 必须为正")

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
    N = delta_grid.shape[0]

    delta_k = np.fft.fftn(delta_grid)
    k_vec = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
    k_mag = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)

    R = L / N * 3.0
    W = tophat_window(k_mag * R)
    delta_smooth_k = delta_k * W
    delta_smooth = np.fft.ifftn(delta_smooth_k).real
    peaks = delta_smooth > threshold

    coords = np.argwhere(peaks)

    phys_coords = coords * (L / N)
    return phys_coords, delta_smooth[peaks]


if __name__ == "__main__":

    N = 32
    L = 100.0
    est = PowerSpectrumEstimator(N, L)
    delta = np.random.randn(N, N, N) * 0.1
    k_bins, Pk, Nm = est.estimate(delta)
    print(f"功率谱估计: k_min={k_bins[0]:.4f}, P(k_min)={Pk[0]:.4e}")


    def test_func(x):
        return np.prod(x ** 2)

    val, err = monte_carlo_nd_integral(test_func, 3, [0, 0, 0], [1, 1, 1], 100000)
    print(f"Monte Carlo ∫ x²y²z² dxdydz = {val:.5f} ± {err:.5f} (理论=1/27≈0.03704)")


    M = np.logspace(10, 15, 50)
    sigma = 2.0 * (M / 1e14) ** (-0.2)
    nM = press_schechter_mass_function(M, sigma, 2.7e11, delta_c=1.686)
    print(f"Press-Schechter n(M=1e14) = {np.interp(1e14, M, nM):.4e}")
