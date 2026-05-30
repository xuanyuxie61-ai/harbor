
import numpy as np
from typing import Tuple, List


def berry_curvature_discrete(
    k_tri: np.ndarray,
    u_states: np.ndarray,
) -> float:
    U_link = np.zeros(3, dtype=complex)
    for e in range(3):
        i = e
        j = (e + 1) % 3
        overlap = np.vdot(u_states[i], u_states[j])
        if abs(overlap) < 1e-14:
            overlap = 1e-14
        U_link[e] = overlap / abs(overlap)

    product = U_link[0] * U_link[1] * U_link[2]
    flux = np.imag(np.log(product))
    return float(flux)


def chern_number_from_mesh(
    kpoints: np.ndarray,
    band_energies: np.ndarray,
    band_vectors: np.ndarray,
    band_index: int,
) -> float:
    from scipy.spatial import Delaunay

    try:
        tri = Delaunay(kpoints)
    except Exception:
        return 0.0

    total_flux = 0.0
    for simplex in tri.simplices:
        k_tri = kpoints[simplex]
        u_states = band_vectors[simplex, band_index, :]
        flux = berry_curvature_discrete(k_tri, u_states)
        total_flux += flux

    chern = total_flux / (2.0 * np.pi)
    return float(chern)


def parity_eigenvalues_at_trim(
    H_trim: np.ndarray,
    inversion_center: np.ndarray,
) -> np.ndarray:
    N = H_trim.shape[0]


    P = np.zeros((N, N))
    for i in range(N):
        j = N - 1 - i
        P[i, j] = 1.0




    energies, vectors = np.linalg.eigh(H_trim)
    parities = np.zeros(N)
    for n in range(N):
        psi = vectors[:, n]
        val = np.real(np.vdot(psi, P @ psi))

        parities[n] = 1.0 if val > 0 else -1.0

    return parities


def z2_index_from_parity(
    trim_hamiltonians: List[np.ndarray],
    trim_names: List[str],
    n_occupied: int,
) -> int:
    if len(trim_hamiltonians) != 4:
        raise ValueError("Exactly four TRIM points are required for 2D Z2.")

    delta_product = 1
    for i, H in enumerate(trim_hamiltonians):
        parities = parity_eigenvalues_at_trim(H, np.zeros(2))


        occ_parities = parities[:n_occupied]
        delta_i = int(np.prod(occ_parities))
        delta_product *= delta_i

    z2 = 0 if delta_product == 1 else 1
    return z2


def mod2_matrix_rank(
    M: np.ndarray,
    tol: float = 1e-10,
) -> int:
    M = np.asarray(M, dtype=float).copy()
    m, n = M.shape
    rank = 0
    for col in range(n):

        pivot = -1
        for row in range(rank, m):
            if abs(M[row, col]) > tol:
                pivot = row
                break
        if pivot == -1:
            continue

        M[[rank, pivot]] = M[[pivot, rank]]

        for row in range(m):
            if row != rank and abs(M[row, col]) > tol:

                M[row] = np.abs(M[row] - M[rank])
                M[row] = (M[row] > tol).astype(float)
        rank += 1
        if rank >= m:
            break
    return rank


def lights_out_matrix_moire(
    nx: int,
    ny: int,
) -> np.ndarray:
    N = nx * ny
    L = np.zeros((N, N), dtype=int)
    for ix in range(nx):
        for iy in range(ny):
            idx = ix * ny + iy
            L[idx, idx] = 1
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                jx = ix + dx
                jy = iy + dy
                if 0 <= jx < nx and 0 <= jy < ny:
                    jdx = jx * ny + jy
                    L[idx, jdx] = 1
    return L


def moire_wilson_loop(
    k_path: np.ndarray,
    H_builder_at_k: callable,
    band_index: int,
) -> complex:
    M = k_path.shape[0]
    if M < 2:
        return 1.0 + 0.0j

    W = 1.0 + 0.0j
    for j in range(M):
        k1 = k_path[j]
        k2 = k_path[(j + 1) % M]
        _, vecs1 = H_builder_at_k(k1)
        _, vecs2 = H_builder_at_k(k2)
        u1 = vecs1[:, band_index]
        u2 = vecs2[:, band_index]
        overlap = np.vdot(u1, u2)
        W *= overlap

    return W
