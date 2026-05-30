
import numpy as np
from scipy.stats import norm
from typing import Tuple






def truncated_normal_ab_sample(mu: float, sigma: float, a: float, b: float, size: int = 1) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("sigma > 0 required")
    if a >= b:
        raise ValueError("a < b required")
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    alpha_cdf = norm.cdf(alpha)
    beta_cdf = norm.cdf(beta)
    if alpha_cdf >= beta_cdf:
        return np.full(size, mu)
    u = np.random.rand(size)
    xi_cdf = alpha_cdf + u * (beta_cdf - alpha_cdf)

    xi_cdf = np.clip(xi_cdf, 1e-10, 1 - 1e-10)
    xi = norm.ppf(xi_cdf)
    return mu + sigma * xi


def truncated_normal_a_sample(mu: float, sigma: float, a: float, size: int = 1) -> np.ndarray:
    return truncated_normal_ab_sample(mu, sigma, a, mu + 10.0 * sigma, size)


def truncated_normal_b_sample(mu: float, sigma: float, b: float, size: int = 1) -> np.ndarray:
    return truncated_normal_ab_sample(mu, sigma, mu - 10.0 * sigma, b, size)


def generate_anderson_disorder(nsites: int, W: float, mu: float = 0.0, sigma: float = 1.0) -> np.ndarray:
    if W <= 0:
        raise ValueError("W > 0 required")
    return truncated_normal_ab_sample(mu, sigma, -W / 2.0, W / 2.0, size=nsites)






def square_surface_sample(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("n >= 1 required")
    p = np.random.rand(n, 2)

    i = np.random.randint(0, 2, size=n)
    s = np.random.randint(0, 2, size=n)

    for idx in range(n):
        p[idx, i[idx]] = float(s[idx])
    return p


def boundary_site_indices(nx: int, ny: int) -> np.ndarray:
    indices = []
    for iy in range(ny):
        for ix in range(nx):
            if ix == 0 or ix == nx - 1 or iy == 0 or iy == ny - 1:
                indices.append(ix + iy * nx)
    return np.array(indices, dtype=int)






def thermal_spin_configuration(nsites: int, beta: float, J: float = 1.0) -> np.ndarray:
    if nsites < 1:
        raise ValueError("nsites >= 1")
    if beta < 0:
        raise ValueError("beta >= 0")

    spins = np.random.randn(nsites, 3)

    norms = np.sqrt(np.sum(spins ** 2, axis=1))
    norms = np.where(norms > 0, norms, 1.0)
    spins = spins / norms[:, np.newaxis]

    if beta * J > 1.0:
        bias = np.tanh(beta * J)
        spins[:, 2] = spins[:, 2] * (1.0 - bias) + bias
        norms = np.sqrt(np.sum(spins ** 2, axis=1))
        norms = np.where(norms > 0, norms, 1.0)
        spins = spins / norms[:, np.newaxis]
    return spins


def random_phase_vector(dim: int) -> np.ndarray:
    if dim < 1:
        raise ValueError("dim >= 1")
    theta = np.random.uniform(0, 2.0 * np.pi, size=dim)
    return np.exp(1j * theta)


def disordered_hubbard_parameters(nsites: int, W: float, U_base: float, U_var: float) -> Tuple[np.ndarray, np.ndarray]:
    epsilon = generate_anderson_disorder(nsites, W, mu=0.0, sigma=W / 3.0)
    delta_U = truncated_normal_ab_sample(0.0, U_var, -U_base * 0.5, U_base * 0.5, size=nsites)
    U = U_base + delta_U
    U = np.clip(U, 0.1, None)
    return epsilon, U


if __name__ == "__main__":
    eps, U = disordered_hubbard_parameters(10, W=2.0, U_base=4.0, U_var=0.5)
    print(f"Disorder mean eps={np.mean(eps):.3f}, U mean={np.mean(U):.3f}")
    sp = thermal_spin_configuration(8, beta=1.0)
    print(f"Spin norms: {np.mean(np.sqrt(np.sum(sp**2, axis=1))):.6f}")
