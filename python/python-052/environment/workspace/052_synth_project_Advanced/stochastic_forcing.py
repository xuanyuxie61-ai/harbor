
import numpy as np
from numerics_core import cholesky_decompose, hermite_polynomial_prob
from typing import Tuple, Optional






def gaussian_covariance_matrix(coords: np.ndarray, sigma: float, Lc: float,
                                nugget: float = 1e-10) -> np.ndarray:
    n = coords.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        dx = coords[i, 0] - coords[:, 0]
        dy = coords[i, 1] - coords[:, 1]
        r2 = dx ** 2 + dy ** 2
        C[i, :] = sigma ** 2 * np.exp(-r2 / (2.0 * Lc ** 2))
    C += nugget * np.eye(n)

    C = 0.5 * (C + C.T)
    return C


def exponential_covariance_matrix(coords: np.ndarray, sigma: float, Lc: float,
                                  nugget: float = 1e-10) -> np.ndarray:
    n = coords.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        dx = coords[i, 0] - coords[:, 0]
        dy = coords[i, 1] - coords[:, 1]
        r = np.sqrt(dx ** 2 + dy ** 2)
        C[i, :] = sigma ** 2 * np.exp(-r / Lc)
    C += nugget * np.eye(n)
    C = 0.5 * (C + C.T)
    return C


def matern_covariance_matrix(coords: np.ndarray, sigma: float, Lc: float,
                             nu: float = 1.5, nugget: float = 1e-10) -> np.ndarray:
    from scipy.special import kv, gamma
    n = coords.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        dx = coords[i, 0] - coords[:, 0]
        dy = coords[i, 1] - coords[:, 1]
        r = np.sqrt(dx ** 2 + dy ** 2)

        r_safe = np.where(r < 1e-10, 1e-10, r)
        scale = np.sqrt(2.0 * nu) * r_safe / Lc
        C[i, :] = sigma ** 2 * (2.0 ** (1.0 - nu) / gamma(nu)) * (scale ** nu) * kv(nu, scale)

    np.fill_diagonal(C, sigma ** 2)
    C += nugget * np.eye(n)
    C = 0.5 * (C + C.T)
    return C






def generate_gaussian_random_field(C: np.ndarray, rng: Optional[np.random.Generator] = None,
                                   n_samples: int = 1) -> np.ndarray:
    n = C.shape[0]
    if rng is None:
        rng = np.random.default_rng(42)

    try:
        U = cholesky_decompose(C, tol=1e-12)
    except ValueError:

        C2 = C + 1e-8 * np.eye(n)
        U = cholesky_decompose(C2, tol=1e-12)

    z = rng.standard_normal((n, n_samples))
    f = U.T @ z
    return f


def generate_spectral_forcing(Nx: int, Ny: int, Lx: float, Ly: float,
                              forcing_amplitude: float = 1.0,
                              k_inject_min: float = 2.0, k_inject_max: float = 6.0,
                              rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng(42)

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, Lx / Nx)[:Nx // 2 + 1]
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, Ly / Ny)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX ** 2 + KY ** 2)

    mask = (K >= k_inject_min) & (K <= k_inject_max)
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(Ny, Nx // 2 + 1))
    amplitude = forcing_amplitude * mask.astype(float)

    F_h = amplitude * np.exp(1j * phase)

    F_h[0, 0] = 0.0
    if Ny % 2 == 0:
        F_h[Ny // 2, :] = 0.0
    if Nx % 2 == 0:
        F_h[:, Nx // 2] = 0.0

    return F_h






class OrnsteinUhlenbeckProcess:

    def __init__(self, theta: float, sigma: float, X0: float = 0.0,
                 dt: float = 0.01, shape: Tuple[int, ...] = ()):
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.X = np.full(shape, X0, dtype=float)
        self.dt = float(dt)
        self.shape = shape

    def step(self, rng: Optional[np.random.Generator] = None):
        if rng is None:
            rng = np.random.default_rng()
        dW = rng.standard_normal(self.shape) * np.sqrt(self.dt)
        self.X = self.X - self.theta * self.X * self.dt + self.sigma * dW
        return self.X

    def steady_state_std(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.theta)






def forcing_hermite_expansion(x: np.ndarray, y: np.ndarray,
                              coeffs: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    X, Y = np.meshgrid(x, y)
    nx, ny = len(x), len(y)
    N = int(np.sqrt(len(coeffs)))
    if N * N != len(coeffs):
        N = int(np.ceil(np.sqrt(len(coeffs))))
        coeffs = np.pad(coeffs, (0, N * N - len(coeffs)))

    result = np.zeros((ny, nx))
    Hx = hermite_polynomial_prob(N - 1, X.flatten() / sigma)
    Hy = hermite_polynomial_prob(N - 1, Y.flatten() / sigma)
    env = np.exp(-0.5 * (X.flatten() ** 2 + Y.flatten() ** 2) / sigma ** 2)

    idx = 0
    for m in range(N):
        for n in range(N):
            if idx < len(coeffs):
                result.flat += coeffs[idx] * Hx[m, :] * Hy[n, :] * env
                idx += 1

    return result


if __name__ == "__main__":

    coords = np.random.default_rng(42).random((20, 2))
    C = gaussian_covariance_matrix(coords, sigma=1.0, Lc=0.3)
    f = generate_gaussian_random_field(C, n_samples=3)
    print("Random field shape:", f.shape)
    print("Empirical covariance vs target:", np.max(np.abs(np.cov(f) - C)))


    ou = OrnsteinUhlenbeckProcess(theta=1.0, sigma=0.5, dt=0.01, shape=(1000,))
    for _ in range(1000):
        ou.step()
    print("OU steady std:", ou.steady_state_std(), "actual:", np.std(ou.X))
