
import numpy as np
from typing import Tuple, Optional


def wishart_variate(
    d: np.ndarray, n: int, np_dim: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    if n < 1 or n > np_dim:
        raise ValueError(f"n must be in [1, {np_dim}]")

    nnp = np_dim * (np_dim + 1) // 2
    if len(d) != nnp:
        raise ValueError("d length must be np*(np+1)/2")


    sb = rng.standard_normal(nnp)


    sa = np.zeros(nnp)
    k = 0
    for i in range(1, np_dim + 1):

        df = n - i + 1
        if df <= 0:
            sa[k] = 0.0
        else:
            sa[k] = np.sqrt(rng.chisquare(df))
        k += 1

        for j in range(i + 1, np_dim + 1):
            sa[k] = sb[k]
            k += 1



    return sa


def generate_correlated_covariance(
    n_dim: int, n_samples: int = 1, alpha: float = 2.0, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    n = max(n_dim, int(alpha * n_dim))
    covs = np.zeros((n_samples, n_dim, n_dim))

    for s in range(n_samples):

        base = rng.standard_normal((n_dim, n))
        cov = base @ base.T / n
        covs[s] = cov

    return covs


def exponential_correlation_kernel(
    x1: np.ndarray, x2: np.ndarray, correlation_length: float
) -> float:
    r = np.linalg.norm(x1 - x2)
    return np.exp(-r / correlation_length)


def squared_exponential_kernel(
    x1: np.ndarray, x2: np.ndarray, correlation_length: float
) -> float:
    r2 = np.sum((x1 - x2) ** 2)
    return np.exp(-r2 / (2.0 * correlation_length ** 2))


def karhunen_loeve_expansion(
    nodes: np.ndarray,
    correlation_length: float,
    n_modes: int,
    kernel_type: str = "squared_exponential",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    n_nodes = nodes.shape[0]


    K = np.zeros((n_nodes, n_nodes))
    for i in range(n_nodes):
        for j in range(n_nodes):
            if kernel_type == "exponential":
                K[i, j] = exponential_correlation_kernel(
                    nodes[i], nodes[j], correlation_length
                )
            else:
                K[i, j] = squared_exponential_kernel(
                    nodes[i], nodes[j], correlation_length
                )


    K += np.eye(n_nodes) * 1e-12


    eigenvalues, eigenvectors = np.linalg.eigh(K)


    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]


    n_modes = min(n_modes, n_nodes)
    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]


    coefficients = rng.standard_normal(n_modes)

    return eigenvalues, eigenvectors, coefficients


def generate_random_field(
    nodes: np.ndarray,
    mean: float,
    std: float,
    correlation_length: float,
    n_modes: int = 20,
    kernel_type: str = "squared_exponential",
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    eigenvalues, eigenvectors, coefficients = karhunen_loeve_expansion(
        nodes, correlation_length, n_modes, kernel_type, rng
    )

    field = mean * np.ones(len(nodes))
    for i in range(n_modes):
        field += std * np.sqrt(eigenvalues[i]) * coefficients[i] * eigenvectors[:, i]

    return field


def brownian_motion(
    n_steps: int, dt: float, n_paths: int = 1, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    dW = rng.standard_normal((n_paths, n_steps)) * np.sqrt(dt)
    W = np.zeros((n_paths, n_steps + 1))
    W[:, 1:] = np.cumsum(dW, axis=1)
    return W


def ornstein_uhlenbeck_process(
    n_steps: int, dt: float, theta: float, mu: float, sigma: float,
    n_paths: int = 1, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    X = np.zeros((n_paths, n_steps + 1))
    X[:, 0] = mu

    for k in range(n_steps):
        dW = rng.standard_normal(n_paths) * np.sqrt(dt)
        X[:, k + 1] = X[:, k] + theta * (mu - X[:, k]) * dt + sigma * dW

    return X


def lognormal_random_field(
    nodes: np.ndarray,
    median: float,
    cov: float,
    correlation_length: float,
    n_modes: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()

    sigma_ln = np.sqrt(np.log(1.0 + cov ** 2))
    mu_ln = np.log(median) - 0.5 * sigma_ln ** 2

    gauss_field = generate_random_field(
        nodes, mu_ln, sigma_ln, correlation_length, n_modes, "squared_exponential", rng
    )

    return np.exp(gauss_field)
