
import numpy as np
from typing import Tuple


def physicist_hermite_polynomials(x: np.ndarray, max_degree: int) -> np.ndarray:
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    x = np.asarray(x, dtype=float)
    m = x.size
    p = np.zeros((m, max_degree + 1))
    p[:, 0] = 1.0
    if max_degree == 0:
        return p
    p[:, 1] = 2.0 * x
    for j in range(2, max_degree + 1):
        p[:, j] = 2.0 * x * p[:, j - 1] - 2.0 * (j - 1) * p[:, j - 2]
    return p


def probabilist_hermite_polynomials(x: np.ndarray, max_degree: int) -> np.ndarray:
    if max_degree < 0:
        raise ValueError("max_degree must be non-negative")
    x = np.asarray(x, dtype=float)
    m = x.size
    p = np.zeros((m, max_degree + 1))
    p[:, 0] = 1.0
    if max_degree == 0:
        return p
    p[:, 1] = x
    for j in range(2, max_degree + 1):
        p[:, j] = x * p[:, j - 1] - (j - 1) * p[:, j - 2]
    return p


def normalized_probabilist_hermite(x: np.ndarray, max_degree: int) -> np.ndarray:
    p = probabilist_hermite_polynomials(x, max_degree)

    from math import factorial, sqrt, pi
    norms = np.array([1.0 / sqrt(sqrt(2.0 * pi) * factorial(n)) for n in range(max_degree + 1)])
    return p * norms[np.newaxis, :]


def hermite_function_basis(x: np.ndarray, max_degree: int) -> np.ndarray:
    p = physicist_hermite_polynomials(x, max_degree)
    from math import factorial, sqrt, pi
    m = x.size
    f = np.zeros((m, max_degree + 1))
    f[:, 0] = np.exp(-0.5 * x ** 2) / sqrt(sqrt(pi))
    if max_degree == 0:
        return f
    f[:, 1] = 2.0 * x * np.exp(-0.5 * x ** 2) / sqrt(2.0 * sqrt(pi))
    for j in range(2, max_degree + 1):
        f[:, j] = (np.sqrt(2.0) * x * f[:, j - 1] - np.sqrt(j - 1) * f[:, j - 2]) / np.sqrt(j)
    return f


def tunneling_amplitude_1d(x: float, gamma: float, n_basis: int = 8) -> float:
    if gamma < 0:
        raise ValueError("gamma must be non-negative")

    hvals = physicist_hermite_polynomials(np.array([gamma]), n_basis)[0, :]


    import math
    amp = 0.0
    for k in range(n_basis + 1):
        amp += hvals[k] * ((-1.0) ** k) / float(math.factorial(max(k, 1)))

    amp = float(np.clip(amp, -10.0, 10.0))
    return amp


def transverse_field_hamiltonian_dense(n_spins: int, gamma: float) -> np.ndarray:
    if n_spins > 12:
        raise ValueError("Dense H_D construction only for n_spins <= 12")
    dim = 2 ** n_spins
    H = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        for i in range(n_spins):
            flipped = state ^ (1 << i)
            H[state, flipped] -= gamma
    return H


def basis_change_matrix(n_spins: int, max_local_dim: int = 4) -> np.ndarray:
    if max_local_dim < 1:
        raise ValueError("max_local_dim must be >= 1")

    pts = np.cos(np.pi * (2 * np.arange(max_local_dim) + 1) / (2 * max_local_dim)) * 3.0
    psi = hermite_function_basis(pts, max_local_dim - 1)

    return psi


class TunnelingKernel:

    def __init__(self, beta: float, gamma: float, n_slices: int, n_basis: int = 8):
        if beta <= 0 or gamma < 0 or n_slices <= 0:
            raise ValueError("Invalid physical parameters")
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.n_slices = int(n_slices)
        self.n_basis = n_basis
        self.dtau = beta / n_slices

    def kinetic_matrix_element(self, s_left: int, s_right: int) -> float:
        a = self.dtau * self.gamma
        if s_left == s_right:
            return np.cosh(a)
        else:
            return np.sinh(a)

    def path_weight_ratio(self, config_old: np.ndarray, config_new: np.ndarray,
                          slice_idx: int) -> float:
        if config_old.shape != config_new.shape:
            raise ValueError("config shapes must match")
        if not np.all(np.isin(config_old, [-1, 1])):
            raise ValueError("configs must be +/-1")

        s_old = int(config_old[slice_idx])
        s_new = int(config_new[slice_idx])
        if s_old == s_new:
            return 1.0

        n = config_old.size
        prev_idx = (slice_idx - 1) % n
        next_idx = (slice_idx + 1) % n

        a = self.dtau * self.gamma

        ratio = (
            (np.cosh(a) if config_old[prev_idx] == s_new else np.sinh(a)) *
            (np.cosh(a) if s_new == config_old[next_idx] else np.sinh(a))
        ) / (
            (np.cosh(a) if config_old[prev_idx] == s_old else np.sinh(a)) *
            (np.cosh(a) if s_old == config_old[next_idx] else np.sinh(a))
        )
        return float(ratio)
