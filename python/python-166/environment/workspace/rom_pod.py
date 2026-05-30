
import numpy as np
from typing import Tuple, Optional


def compute_svd_basis(snapshot_matrix: np.ndarray,
                      basis_num: int,
                      subtract_mean: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if snapshot_matrix.ndim != 2:
        raise ValueError("snapshot_matrix must be 2D")

    M, N = snapshot_matrix.shape
    basis_num = min(basis_num, M, N)

    mean_vector = np.zeros(M)
    A = snapshot_matrix.copy()

    if subtract_mean:
        mean_vector = np.mean(A, axis=1)
        A = A - mean_vector.reshape(-1, 1)


    try:
        U, S, Vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:

        U, S, Vt = randomized_svd(A, basis_num)

    basis = U[:, :basis_num]
    singular_values = S[:basis_num]

    return basis, singular_values, mean_vector


def randomized_svd(A: np.ndarray, k: int, p: int = 5, q: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    m, n = A.shape
    k = min(k, min(m, n))


    np.random.seed(42)
    Omega = np.random.randn(n, k + p)


    Y = A @ Omega
    for _ in range(q):
        Y = A @ (A.T @ Y)

    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A

    U_tilde, S, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ U_tilde

    return U[:, :k], S[:k], Vt[:k, :]


def project_onto_basis(field: np.ndarray, basis: np.ndarray,
                       mean_vector: Optional[np.ndarray] = None) -> np.ndarray:
    if mean_vector is not None:
        field = field - mean_vector
    coeffs = basis.T @ field
    return coeffs


def reconstruct_from_basis(coefficients: np.ndarray, basis: np.ndarray,
                           mean_vector: Optional[np.ndarray] = None) -> np.ndarray:
    field = basis @ coefficients
    if mean_vector is not None:
        field = field + mean_vector
    return field


def pod_galerkin_rom(M_mass: np.ndarray, K_stiff: np.ndarray, F_force: np.ndarray,
                     basis: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    M_rom = basis.T @ M_mass @ basis
    K_rom = basis.T @ K_stiff @ basis
    F_rom = basis.T @ F_force
    return M_rom, K_rom, F_rom


def generate_snapshots_soft_robot(L: float, Ns: int, n_snapshots: int,
                                  material_params: dict) -> np.ndarray:
    from cosserat_core import forward_kinematics_cosserat

    n_nodes = Ns + 1
    M = n_nodes * 3

    snapshots = np.zeros((M, n_snapshots))
    rng = np.random.RandomState(42)

    for i in range(n_snapshots):




        raise NotImplementedError("Hole 3: 实现快照生成循环")

    return snapshots


def energy_fraction(singular_values: np.ndarray) -> np.ndarray:
    total = np.sum(singular_values ** 2)
    if total < 1e-14:
        return np.ones(len(singular_values))
    cumsum = np.cumsum(singular_values ** 2)
    return cumsum / total


def optimal_basis_size(singular_values: np.ndarray,
                       threshold: float = 0.99) -> int:
    energy = energy_fraction(singular_values)
    size = np.searchsorted(energy, threshold) + 1
    return min(size, len(singular_values))
