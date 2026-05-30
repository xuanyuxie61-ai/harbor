
import numpy as np
from typing import Tuple, List






def pca_vectors(A: np.ndarray, numvecs: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("pca_vectors: A must be 2D")
    m, n = A.shape
    if numvecs > n:
        numvecs = n
    if numvecs < 1:
        raise ValueError("pca_vectors: numvecs must be positive")


    Psi = np.mean(A, axis=1)
    A_centered = A - Psi[:, np.newaxis]


    L = A_centered.T @ A_centered
    eigvals, eigvecs = np.linalg.eigh(L)

    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]


    eigvals = eigvals / max(n - 1, 1)
    Vectors = A_centered @ eigvecs


    for j in range(n):
        norm = np.linalg.norm(Vectors[:, j])
        if norm > 1.0e-12:
            Vectors[:, j] = Vectors[:, j] / norm
        else:
            Vectors[:, j] = 0.0
            eigvals[j] = 0.0


    num_good = int(np.sum(eigvals > 1.0e-8))
    if num_good < numvecs:
        numvecs = num_good
    Vectors = Vectors[:, :numvecs]
    return Vectors, eigvals, Psi


def pca_transform(A: np.ndarray, V: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    if A.ndim == 1:
        return V.T @ (A - Psi)
    return V.T @ (A - Psi[:, np.newaxis])


def pca_reconstruct(z: np.ndarray, V: np.ndarray, Psi: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        return V @ z + Psi
    return V @ z + Psi[:, np.newaxis]






def legendre_polynomial_1d(n: int, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if n < 0:
        raise ValueError("legendre_polynomial_1d: n must be non-negative")
    if n == 0:
        return np.ones_like(x)
    if n == 1:
        return x.copy()
    P_prev2 = np.ones_like(x)
    P_prev1 = x.copy()
    for k in range(1, n):
        P_curr = ((2 * k + 1) * x * P_prev1 - k * P_prev2) / (k + 1)
        P_prev2 = P_prev1
        P_prev1 = P_curr
    return P_prev1


def legendre_product_polynomial(m: int, degrees: np.ndarray, X: np.ndarray) -> np.ndarray:
    degrees = np.asarray(degrees, dtype=int)
    X = np.asarray(X, dtype=float)
    if degrees.ndim != 1 or len(degrees) != m:
        raise ValueError("legendre_product_polynomial: degrees shape mismatch")
    if X.ndim == 1:
        X = X.reshape(m, 1)
    n_pts = X.shape[1]
    v = np.ones(n_pts)
    for i in range(m):
        vi = legendre_polynomial_1d(degrees[i], X[i, :])
        v = v * vi
    return v


def build_legendre_basis(m: int, max_degree: int, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(m, 1)

    indices = _generate_multi_indices(m, max_degree)
    K = len(indices)
    N = X.shape[1]
    B = np.zeros((K, N))
    for k, alpha in enumerate(indices):
        B[k, :] = legendre_product_polynomial(m, alpha, X)
    return B


def _generate_multi_indices(m: int, max_degree: int) -> List[np.ndarray]:
    result = []
    def backtrack(pos: int, current: List[int], remaining: int):
        if pos == m - 1:
            current.append(remaining)
            result.append(np.array(current, dtype=int))
            current.pop()
            return
        for val in range(remaining + 1):
            current.append(val)
            backtrack(pos + 1, current, remaining - val)
            current.pop()
    for total in range(max_degree + 1):
        backtrack(0, [], total)
    return result






def bessel_spectral_filter(freqs: np.ndarray, n: float, k: int,
                           kind: int = 1, bandwidth: float = 1.0) -> np.ndarray:
    from special_functions import bessel_zero
    response = np.ones_like(freqs)
    for j in range(1, k + 1):
        zj = bessel_zero(n, j, kind)
        response = response * np.exp(-(freqs - zj) ** 2 / (2.0 * bandwidth ** 2))
    return response
