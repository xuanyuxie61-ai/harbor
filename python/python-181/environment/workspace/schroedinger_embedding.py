
import numpy as np
from typing import Tuple, Optional
from linear_algebra_core import jacobi_eigenvalue, solve_cholesky, cholesky_factor


def nonlinear_schroedinger_parameters() -> dict:
    return {
        'alpha': 1.0,
        'c1': 1.0,
        'c2': 0.5,
        'delta': 0.1,
        'gamma': -0.5,
        't0': 0.0,
        'xmin': -5.0,
        'xmax': 5.0
    }


def build_effective_potential(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:



    raise NotImplementedError("Hole 1: build_effective_potential 待实现")


def schroedinger_deriv(u: np.ndarray, L: np.ndarray, V: np.ndarray,
                        gamma: float = -0.5) -> np.ndarray:
    nonlin = gamma * np.abs(u) ** 2 * u
    dudt = -0.5 * (L @ u) - V * u - nonlin
    return dudt


def finite_difference_evolve(u0: np.ndarray, L: np.ndarray, V: np.ndarray,
                              dt: float = 0.01, n_steps: int = 1000,
                              gamma: float = -0.5) -> np.ndarray:
    N = len(u0)
    u = u0.copy()

    A = np.eye(N) + 0.5 * dt * L
    B = np.eye(N) - 0.5 * dt * L
    for step in range(n_steps):

        nonlin = gamma * np.abs(u) ** 2 * u
        rhs = B @ u - dt * (V * u + nonlin)

        u_new = np.linalg.solve(A, rhs)

        norm = np.linalg.norm(u_new)
        if norm > 1e-15:
            u_new = u_new / norm

        if step > 0 and step % 20 == 0:
            residual = np.linalg.norm(u_new - u_old)
            if residual < 1e-7:
                u = u_new
                break
        u_old = u_new.copy()
        u = u_new
    return u


def schroedinger_spectral_embedding(data: np.ndarray, L: np.ndarray,
                                     n_components: int = 3,
                                     sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    N = len(data)
    V = build_effective_potential(data, sigma)

    D = np.diag(np.sum(np.exp(-np.linalg.norm(data[:, None] - data[None, :], axis=2) ** 2 / (2.0 * sigma ** 2)), axis=1))

    D_reg = D + 1e-6 * np.eye(N)

    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D_reg)))
    L_sym = D_inv_sqrt @ L @ D_inv_sqrt

    eigvals, eigvecs = np.linalg.eigh(L_sym)


    idx = np.argsort(eigvals)

    selected = idx[1:n_components + 1]
    eigenvalues = eigvals[selected]

    embedding = D_inv_sqrt @ eigvecs[:, selected]
    return embedding, eigenvalues


def nonlinear_spectral_coordinates(data: np.ndarray, n_components: int = 3,
                                    sigma: float = 1.0, gamma: float = -0.5,
                                    n_iterations: int = 50) -> np.ndarray:
    N, D = data.shape
    from neighbor_graph import build_knn_graph, graph_laplacian
    edges, weights = build_knn_graph(data, k=min(10, N - 1))
    L = graph_laplacian(edges, weights, N, normalize=True)
    V = build_effective_potential(data, sigma)
    coordinates = []
    for comp in range(n_components):

        np.random.seed(42 + comp)
        u0 = np.random.randn(N)
        u0 = u0 / np.linalg.norm(u0)
        for prev in coordinates:
            u0 = u0 - np.dot(u0, prev) * prev
            u0 = u0 / (np.linalg.norm(u0) + 1e-15)

        u = finite_difference_evolve(u0, L, V, dt=0.02, n_steps=80, gamma=gamma)

        for prev in coordinates:
            u = u - np.dot(u, prev) * prev
            u = u / (np.linalg.norm(u) + 1e-15)
        coordinates.append(u)
    return np.array(coordinates).T


def schroedinger_energy(u: np.ndarray, L: np.ndarray, V: np.ndarray,
                         gamma: float = -0.5) -> float:



    raise NotImplementedError("Hole 2: schroedinger_energy 待实现")
