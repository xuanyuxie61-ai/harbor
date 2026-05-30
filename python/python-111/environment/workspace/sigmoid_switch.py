
import numpy as np
from typing import Tuple
from math import comb


def sigmoid_coef(n: int) -> np.ndarray:
    if n < 0:
        raise ValueError("n must be non-negative")
    coeffs = np.zeros(n + 1)
    for k in range(1, n + 2):
        c_nk = 0.0
        for j in range(k + 1):
            c_nk += ((-1) ** j) * ((j + 1) ** n) * comb(k, j)
        coeffs[k - 1] = c_nk
    return coeffs


def sigmoid_value(x: np.ndarray, n_derivative: int = 0) -> np.ndarray:
    if n_derivative < 0:
        raise ValueError("n_derivative must be non-negative")
    
    sigma = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
    if n_derivative == 0:
        return sigma
    
    coeffs = sigmoid_coef(n_derivative)
    values = np.zeros_like(x)
    for j, c in enumerate(coeffs):
        values += c * (sigma ** (j + 1))
    return values


def smooth_cutoff_function(r: np.ndarray, r_cut: float, width: float = 0.5) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be positive")
    x = (r_cut - r) / width
    return sigmoid_value(x, 0)


def smooth_cutoff_derivative(r: np.ndarray, r_cut: float, width: float = 0.5,
                              order: int = 1) -> np.ndarray:
    if width <= 0:
        raise ValueError("width must be positive")
    if order < 1:
        raise ValueError("order must be at least 1")
    
    x = (r_cut - r) / width
    sig_n = sigmoid_value(x, order)
    deriv = ((-1.0 / width) ** order) * sig_n
    return deriv


def dielectric_switch_function(r: np.ndarray, r_in: float, r_out: float,
                                eps_in: float = 4.0, eps_out: float = 80.0) -> np.ndarray:
    if r_out <= r_in:
        raise ValueError("r_out must be greater than r_in")
    width = 0.5 * (r_out - r_in)
    r_mid = 0.5 * (r_in + r_out)
    S = smooth_cutoff_function(r, r_mid, width)
    eps = eps_in + (eps_out - eps_in) * S
    return eps


def force_switching(r: np.ndarray, r_on: float, r_off: float) -> Tuple[np.ndarray, np.ndarray]:
    if r_off <= r_on:
        raise ValueError("r_off must be greater than r_on")
    
    width = 0.5 * (r_off - r_on)
    r_mid = 0.5 * (r_on + r_off)
    
    S = smooth_cutoff_function(r, r_mid, width)
    dS = smooth_cutoff_derivative(r, r_mid, width, order=1)
    return S, dS
