
import numpy as np
from typing import Tuple, List


def hexity_rotate(hex_state: np.ndarray, k: int = 1) -> np.ndarray:
    rotated = np.roll(hex_state, k)
    return rotated


def hexity_reflect(hex_state: np.ndarray) -> np.ndarray:
    reflected = hex_state.copy()

    reflected = np.flip(reflected)
    return reflected


def dihedral_group_d6_action(state: np.ndarray, operation: str) -> np.ndarray:
    if operation.startswith('r'):
        k = int(operation[1])
        return hexity_rotate(state, k)
    elif operation.startswith('s'):
        k = int(operation[1])
        s = hexity_reflect(state)
        return hexity_rotate(s, k)
    else:
        return state.copy()


def orbit_under_d6(state: np.ndarray) -> List[np.ndarray]:
    orbit = []
    seen = set()
    for op in ['r0', 'r1', 'r2', 'r3', 'r4', 'r5',
               's0', 's1', 's2', 's3', 's4', 's5']:
        new_state = dihedral_group_d6_action(state, op)
        key = tuple(new_state.tolist())
        if key not in seen:
            seen.add(key)
            orbit.append(new_state)
    return orbit


def symmetry_order(state: np.ndarray) -> int:
    stabilizer_size = 0
    for op in ['r0', 'r1', 'r2', 'r3', 'r4', 'r5',
               's0', 's1', 's2', 's3', 's4', 's5']:
        new_state = dihedral_group_d6_action(state, op)
        if np.allclose(new_state, state):
            stabilizer_size += 1
    return stabilizer_size


def lights_out_matrix(mrow: int = 5, ncol: int = 5) -> np.ndarray:
    n = mrow * ncol
    A = np.zeros((n, n), dtype=int)
    def index(i, j):
        if i < 0 or i >= mrow or j < 0 or j >= ncol:
            return -1
        return i * ncol + j
    for i in range(mrow):
        for j in range(ncol):
            c = index(i, j)
            neighbors = [index(i, j), index(i - 1, j), index(i + 1, j),
                         index(i, j - 1), index(i, j + 1)]
            for nbr in neighbors:
                if nbr >= 0:
                    A[nbr, c] = 1
    return A


def lights_out_solve(initial: np.ndarray, mrow: int = 5, ncol: int = 5) -> np.ndarray:
    A = lights_out_matrix(mrow, ncol)
    n = mrow * ncol
    b = initial.copy() % 2

    aug = np.hstack([A.astype(int), b.reshape(-1, 1)])

    for col in range(n):

        pivot = -1
        for row in range(col, n):
            if aug[row, col] == 1:
                pivot = row
                break
        if pivot == -1:
            continue

        aug[[col, pivot]] = aug[[pivot, col]]

        for row in range(n):
            if row != col and aug[row, col] == 1:
                aug[row] = (aug[row] + aug[col]) % 2

    p = aug[:, n].copy()
    return p % 2


def betti_number_estimate(edges: np.ndarray, n_vertices: int) -> int:

    W = np.zeros((n_vertices, n_vertices), dtype=np.float64)
    for (i, j) in edges:
        W[i, j] = 1.0
        W[j, i] = 1.0
    D = np.diag(np.sum(W, axis=1))
    L = D - W

    eigvals = np.linalg.eigvalsh(L)

    threshold = 1e-10
    beta_0 = int(np.sum(eigvals < threshold))
    return max(beta_0, 1)


def persistence_homology_filtration(data: np.ndarray, radii: np.ndarray) -> dict:
    N = len(data)
    persistence = {}
    for r in radii:
        edges = []
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(data[i] - data[j]) <= r:
                    edges.append([i, j])
        edges = np.array(edges, dtype=int) if len(edges) > 0 else np.zeros((0, 2), dtype=int)
        beta_0 = betti_number_estimate(edges, N)
        persistence[r] = beta_0
    return persistence


def discrete_topological_features(data: np.ndarray, n_bins: int = 5) -> np.ndarray:
    from linear_algebra_core import jacobi_eigenvalue

    cov = np.cov(data.T)
    _, vecs = jacobi_eigenvalue(cov)
    proj = data @ vecs[:, :2]

    xmin, xmax = np.min(proj[:, 0]), np.max(proj[:, 0])
    ymin, ymax = np.min(proj[:, 1]), np.max(proj[:, 1])
    grid = np.zeros((n_bins, n_bins), dtype=int)
    for p in proj:
        ix = int((p[0] - xmin) / (xmax - xmin + 1e-10) * n_bins)
        iy = int((p[1] - ymin) / (ymax - ymin + 1e-10) * n_bins)
        ix = min(ix, n_bins - 1)
        iy = min(iy, n_bins - 1)
        grid[ix, iy] = 1

    return grid.ravel()
