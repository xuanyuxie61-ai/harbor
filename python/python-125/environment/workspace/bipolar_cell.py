
import numpy as np
from typing import Tuple, Callable






def legendre_polynomial_value(n: int, x: np.ndarray) -> np.ndarray:
    m = x.shape[0]
    V = np.zeros((m, n + 1), dtype=np.float64)
    
    V[:, 0] = 1.0
    if n >= 1:
        V[:, 1] = x
    
    for k in range(2, n + 1):
        V[:, k] = ((2.0 * k - 1.0) * x * V[:, k - 1] - (k - 1.0) * V[:, k - 2]) / k
    
    return V


def legendre_polynomial_zeros(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])
    if n == 1:
        return np.array([0.0])
    

    J = np.zeros((n, n), dtype=np.float64)
    for i in range(1, n):
        b = np.sqrt(i * i / (4.0 * i * i - 1.0))
        J[i - 1, i] = b
        J[i, i - 1] = b
    

    eigenvalues = np.linalg.eigvalsh(J)
    return np.sort(eigenvalues)


def gauss_legendre_quadrature(n: int) -> Tuple[np.ndarray, np.ndarray]:
    x = legendre_polynomial_zeros(n)
    

    V = legendre_polynomial_value(n, x)
    P_n = V[:, n]
    P_n_minus_1 = V[:, n - 1]
    


    denom = 1.0 - x ** 2
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    Pn_prime = n * (P_n_minus_1 - x * P_n) / denom
    

    w = 2.0 / ((1.0 - x ** 2) * Pn_prime ** 2 + 1e-14)
    
    return x, w






def dog_receptive_field(
    x: np.ndarray, y: np.ndarray,
    A_c: float, sigma_c: float,
    A_s: float, sigma_s: float
) -> np.ndarray:
    r2 = x ** 2 + y ** 2
    center = A_c * np.exp(-r2 / (2.0 * sigma_c ** 2))
    surround = A_s * np.exp(-r2 / (2.0 * sigma_s ** 2))
    return center - surround


def compute_bipolar_response_convolution(
    stimulus: np.ndarray,
    rf_params: dict,
    grid_spacing: float
) -> float:
    ny, nx = stimulus.shape
    

    x_range = (nx - 1) * grid_spacing / 2.0
    y_range = (ny - 1) * grid_spacing / 2.0
    

    n_quad = min(16, nx, ny)
    x_nodes, x_weights = gauss_legendre_quadrature(n_quad)
    

    x_phys = x_range * x_nodes
    y_phys = y_range * x_nodes
    

    A_c = rf_params.get('A_c', 1.0)
    sigma_c = rf_params.get('sigma_c', 10.0)
    A_s = rf_params.get('A_s', 0.5)
    sigma_s = rf_params.get('sigma_s', 30.0)
    
    response = 0.0
    scale_x = x_range
    scale_y = y_range
    
    for i in range(n_quad):
        for j in range(n_quad):

            xi = (x_phys[i] + x_range) / (2.0 * x_range) * (nx - 1)
            yj = (y_phys[j] + y_range) / (2.0 * y_range) * (ny - 1)
            
            ix = int(np.clip(round(xi), 0, nx - 1))
            iy = int(np.clip(round(yj), 0, ny - 1))
            

            r2 = x_phys[i] ** 2 + y_phys[j] ** 2
            rf_val = A_c * np.exp(-r2 / (2.0 * sigma_c ** 2)) - A_s * np.exp(-r2 / (2.0 * sigma_s ** 2))
            
            response += scale_x * scale_y * x_weights[i] * x_weights[j] * rf_val * stimulus[iy, ix]
    
    return float(response)






def decompose_rf_with_legendre_basis(
    spatial_profile: Callable[[float], float],
    max_degree: int,
    n_quad: int = 64
) -> np.ndarray:
    x_nodes, weights = gauss_legendre_quadrature(n_quad)
    

    V = legendre_polynomial_value(max_degree, x_nodes)
    

    f_vals = np.array([spatial_profile(xi) for xi in x_nodes], dtype=np.float64)
    
    coeffs = np.zeros(max_degree + 1, dtype=np.float64)
    for k in range(max_degree + 1):
        integrand = f_vals * V[:, k]
        integral = np.sum(weights * integrand)
        coeffs[k] = (2.0 * k + 1.0) / 2.0 * integral
    
    return coeffs


def reconstruct_rf_from_legendre_coeffs(
    coeffs: np.ndarray,
    x: np.ndarray
) -> np.ndarray:
    n = len(coeffs) - 1
    V = legendre_polynomial_value(n, x)
    return V @ coeffs
