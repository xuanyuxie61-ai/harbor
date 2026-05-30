
import numpy as np
from typing import Callable, Tuple


def gauss_hermite_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n <= 0:
        return np.array([]), np.array([])
    

    if n == 1:
        x = np.array([0.0])
        w = np.array([np.sqrt(np.pi)])
        return x, w
    
    J = np.zeros((n, n))
    for i in range(n - 1):
        J[i, i + 1] = np.sqrt((i + 1) / 2.0)
        J[i + 1, i] = J[i, i + 1]
    
    eigenvalues, eigenvectors = np.linalg.eigh(J)
    x = eigenvalues
    w = np.sqrt(np.pi) * eigenvectors[0, :] ** 2
    
    return x, w


def generalized_hermite_integral(expon: int, alpha: float) -> float:
    from scipy.special import gamma as scipy_gamma
    
    if expon % 2 == 1:
        return 0.0
    
    a = alpha + expon
    if a <= -1.0:
        return -np.inf
    
    value = scipy_gamma((a + 1.0) / 2.0)
    return value


def dos_histogram(energies: np.ndarray, e_min: float, e_max: float,
                   n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    if len(energies) == 0:
        return np.zeros(n_bins), np.zeros(n_bins)
    
    hist, bin_edges = np.histogram(energies, bins=n_bins, range=(e_min, e_max))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_width = (e_max - e_min) / n_bins
    

    dos = hist / (len(energies) * bin_width)
    
    return bin_centers, dos


def dos_gaussian_broadening(energies: np.ndarray,
                             e_grid: np.ndarray,
                             sigma: float = 0.05) -> np.ndarray:
    N = len(energies)
    if N == 0:
        return np.zeros_like(e_grid)
    
    prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    dos = np.zeros_like(e_grid)
    
    for e_val in energies:
        dos += prefactor * np.exp(-0.5 * ((e_grid - e_val) / sigma) ** 2)
    
    dos /= N
    return dos


def dos_weyl_semimetal_analytic(e: np.ndarray, v_f: float = 1.0,
                                 hbar: float = 1.0) -> np.ndarray:
    e = np.asarray(e)
    dos = np.zeros_like(e)
    

    nonzero = np.abs(e) > 1e-14
    dos[nonzero] = e[nonzero] ** 2 / (2.0 * np.pi ** 2 * (hbar * v_f) ** 3)
    
    return dos


def integrate_dos_with_hermite(energy_func: Callable[[np.ndarray], np.ndarray],
                                k_bounds: np.ndarray,
                                e_ref: float,
                                n_hermite: int = 20,
                                n_k_samples: int = 1000) -> float:

    x_gh, w_gh = gauss_hermite_nodes_weights(n_hermite)
    

    k_samples = np.zeros((n_k_samples, 3))
    for d in range(3):
        k_samples[:, d] = np.random.uniform(k_bounds[d, 0], k_bounds[d, 1], n_k_samples)
    
    energies = energy_func(k_samples)
    

    sigma = 0.05 * (np.max(energies) - np.min(energies))
    if sigma < 1e-10:
        sigma = 0.01
    



    dos_val = np.sum(np.exp(-((energies - e_ref) / sigma) ** 2))
    dos_val /= (n_k_samples * sigma * np.sqrt(np.pi))
    

    vol = np.prod(k_bounds[:, 1] - k_bounds[:, 0])
    dos_val *= vol
    
    return dos_val


def test_hermite_exactness(alpha: float, max_degree: int = 10,
                            n_points: int = 10) -> np.ndarray:
    x, w = gauss_hermite_nodes_weights(n_points)
    
    errors = np.zeros(max_degree + 1)
    for degree in range(max_degree + 1):

        exact = generalized_hermite_integral(degree, alpha)
        

        if exact == 0.0:
            quad = np.sum(w * (x ** degree))
            errors[degree] = abs(quad)
        else:
            quad = np.sum(w * (x ** degree))
            errors[degree] = abs((quad - exact) / exact)
    
    return errors
