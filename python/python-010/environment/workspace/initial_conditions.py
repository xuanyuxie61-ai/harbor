
import numpy as np
from typing import Tuple
from statistics import variance_from_power_spectrum, tophat_window


class TransferFunction:

    def __init__(self, cosmology):
        self.Omega_m = cosmology.Omega_m
        self.h = cosmology.h
        self.T_cmb = cosmology.T_cmb

    def __call__(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        out = np.ones_like(k)
        mask = k > 0.0
        k_m = k[mask]
        q = k_m / (self.Omega_m * self.h ** 2) * (self.T_cmb / 2.7) ** 2

        q = np.clip(q, 1e-10, None)
        ln_term = np.log(1.0 + 2.34 * q) / (2.34 * q)
        poly_term = (
            1.0
            + 3.89 * q
            + (16.1 * q) ** 2
            + (5.46 * q) ** 3
            + (6.71 * q) ** 4
        ) ** (-0.25)
        out[mask] = ln_term * poly_term
        return out


class PowerSpectrum:

    def __init__(self, cosmology, transfer_fn: TransferFunction = None):
        self.cosmo = cosmology
        self.transfer = transfer_fn or TransferFunction(cosmology)

        self.A_s = self._normalize_amplitude()

    def _normalize_amplitude(self) -> float:
        R8 = 8.0
        k_arr = np.logspace(-4, 2, 2000)
        T_arr = self.transfer(k_arr)
        P_unnorm = k_arr ** self.cosmo.ns * T_arr ** 2
        sigma2_unnorm = variance_from_power_spectrum(R8, k_arr, P_unnorm)
        if sigma2_unnorm <= 0.0:
            sigma2_unnorm = 1e-30
        A = self.cosmo.sigma8 ** 2 / sigma2_unnorm
        return A

    def __call__(self, k: np.ndarray) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        T = self.transfer(k)
        return self.A_s * (k ** self.cosmo.ns) * (T ** 2)


def latin_edge_sample(dim_num: int, point_num: int, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(seed=42)
    if point_num == 1:
        return np.full((dim_num, 1), 0.5)
    x = np.zeros((dim_num, point_num))
    for i in range(dim_num):
        perm = rng.permutation(point_num)
        x[i, :] = perm / (point_num - 1.0)
    return x


def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        raise ValueError("n 必须为正")

    diag = np.zeros(n)
    offdiag = np.sqrt(0.5 * np.arange(1, n))

    J = np.diag(diag) + np.diag(offdiag, 1) + np.diag(offdiag, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    nodes = eigenvalues

    weights = np.sqrt(np.pi) * eigenvectors[0, :] ** 2
    return nodes, weights


def generate_zeldovich_displacement(
    N: int,
    L: float,
    power_spectrum: PowerSpectrum,
    D_growth: float = 1.0,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng(seed=42)


    k_vec = 2.0 * np.pi * np.fft.fftfreq(N, d=L / N)
    kx, ky, kz = np.meshgrid(k_vec, k_vec, k_vec, indexing="ij")
    k2 = kx ** 2 + ky ** 2 + kz ** 2
    k_mag = np.sqrt(k2)


    Pk = power_spectrum(k_mag.flatten()).reshape(k_mag.shape)

    V = L ** 3
    amplitude = np.sqrt(Pk / V)


    real_part = rng.standard_normal((N, N, N))
    imag_part = rng.standard_normal((N, N, N))
    delta_k = (real_part + 1j * imag_part) * amplitude


    for i in range(N):
        for j in range(N):
            for kk in range(N):
                ii = (N - i) % N
                jj = (N - j) % N
                kk2 = (N - kk) % N
                if (ii, jj, kk2) > (i, j, kk):
                    delta_k[ii, jj, kk2] = delta_k[i, j, kk].conjugate()
    delta_k[0, 0, 0] = delta_k[0, 0, 0].real


    delta = np.fft.ifftn(delta_k).real * (N ** 3)


    Psi_k = np.zeros_like(delta_k)
    mask = k2 > 0.0
    Psi_k[mask] = delta_k[mask] / k2[mask]


    Sx_k = -1j * kx * Psi_k
    Sy_k = -1j * ky * Psi_k
    Sz_k = -1j * kz * Psi_k

    Sx = np.fft.ifftn(Sx_k).real * (N ** 3)
    Sy = np.fft.ifftn(Sy_k).real * (N ** 3)
    Sz = np.fft.ifftn(Sz_k).real * (N ** 3)


    qx = np.linspace(0.0, L, N, endpoint=False)
    qgrid = np.meshgrid(qx, qx, qx, indexing="ij")


    x_pos = qgrid[0] + D_growth * Sx
    y_pos = qgrid[1] + D_growth * Sy
    z_pos = qgrid[2] + D_growth * Sz


    x_pos = x_pos % L
    y_pos = y_pos % L
    z_pos = z_pos % L

    pos = np.stack([x_pos.ravel(), y_pos.ravel(), z_pos.ravel()], axis=1)


    f_growth = 1.0
    a_scale = 1.0

    H_a = 100.0
    vel_factor = a_scale ** 2 * f_growth * H_a * D_growth
    vel = vel_factor * np.stack([Sx.ravel(), Sy.ravel(), Sz.ravel()], axis=1)

    return pos, vel, delta


def particle_mass_from_cosmology(N: int, L: float, cosmology) -> float:
    n_part = N ** 3
    return cosmology.Omega_m * cosmology.rho_crit_0 * (L ** 3) / n_part


if __name__ == "__main__":
    from cosmology import Cosmology

    cosmo = Cosmology()
    ps = PowerSpectrum(cosmo)
    k_test = np.logspace(-3, 1, 100)
    P_test = ps(k_test)
    print(f"P(k=0.1) = {np.interp(0.1, k_test, P_test):.4e}")


    latin = latin_edge_sample(3, 8)
    print("Latin edge 采样 shape:", latin.shape)


    nodes, weights = gauss_hermite_nodes_weights(8)

    integral = np.sum(weights)
    print(f"Gauss-Hermite ∫ exp(-x²) dx = {integral:.8f} (理论 √π = {np.sqrt(np.pi):.8f})")


    pos, vel, delta = generate_zeldovich_displacement(16, 100.0, ps, D_growth=1.0)
    print(f"初始位置 shape: {pos.shape}, 速度 shape: {vel.shape}")
    print(f"密度场 std: {delta.std():.4f}")
