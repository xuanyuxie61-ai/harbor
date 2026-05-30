
import numpy as np


def dyadic_green_tensor(r_vec, k, eps0=8.854187817e-12):
    r = np.linalg.norm(r_vec)
    if r < 1e-18:

        return np.zeros((3, 3), dtype=complex)

    kr = k * r
    if abs(kr) < 1e-12:

        prefactor = 1.0 / (4.0 * np.pi * eps0 * (r ** 3))
        rr = np.outer(r_vec, r_vec) / (r ** 2)
        G = prefactor * (3.0 * rr - np.eye(3))
        return G

    prefactor = (k ** 2) / eps0 * np.exp(1j * kr) / (4.0 * np.pi * r)
    rr = np.outer(r_vec, r_vec) / (r ** 2)

    term1 = (1.0 + 1j / kr - 1.0 / (kr ** 2)) * np.eye(3)
    term2 = (1.0 + 3.0j / kr - 3.0 / (kr ** 2)) * rr
    G = prefactor * (term1 - term2)
    return G


def build_coupling_matrix(positions, polarizabilities, omega, eps_medium=1.0):
    c = 2.99792458e8
    N = positions.shape[0]
    if N == 0:
        raise ValueError("At least one particle required.")
    if positions.shape[1] != 3:
        raise ValueError("Positions must be N×3.")

    k_medium = omega * np.sqrt(eps_medium) / c
    A = np.zeros((3 * N, 3 * N), dtype=complex)

    for j in range(N):

        alpha_j = polarizabilities[j]
        if abs(alpha_j) < 1e-40:
            alpha_j = 1e-40
        inv_alpha = 1.0 / alpha_j
        A[3 * j:3 * j + 3, 3 * j:3 * j + 3] = inv_alpha * np.eye(3)

        for k_ in range(j + 1, N):
            r_vec = positions[j] - positions[k_]
            Gjk = dyadic_green_tensor(r_vec, k_medium)
            A[3 * j:3 * j + 3, 3 * k_:3 * k_ + 3] = -Gjk
            A[3 * k_:3 * k_ + 3, 3 * j:3 * j + 3] = -Gjk.T

    return A


def incident_plane_wave(positions, E0, kvec, pol):
    N = positions.shape[0]
    b = np.zeros(3 * N, dtype=complex)
    for j in range(N):
        phase = np.exp(1j * np.dot(kvec, positions[j]))
        b[3 * j:3 * j + 3] = E0 * pol * phase
    return b


def solve_dipole_moments(A, b, tol=1e-12, max_iter=5000):
    N = b.size
    if N <= 60:
        return np.linalg.solve(A, b)

    try:
        from scipy.sparse.linalg import bicgstab
        from scipy.sparse import csr_matrix
        p, info = bicgstab(csr_matrix(A), b, tol=tol, maxiter=max_iter)
        if info != 0:
            p = np.linalg.solve(A, b)
    except Exception:
        p = np.linalg.solve(A, b)
    return p


def build_coupling_graph(positions, omega, eps_medium=1.0, threshold=1.0e30):
    c = 2.99792458e8
    N = positions.shape[0]
    k = omega * np.sqrt(eps_medium) / c
    adjacency = [[] for _ in range(N)]
    arc_list = []

    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            r_vec = positions[i] - positions[j]
            G = dyadic_green_tensor(r_vec, k)
            strength = np.linalg.norm(G)
            if strength > threshold:
                adjacency[i].append(j)
                arc_list.append((i, j))

    return adjacency, arc_list


def polarizability_clausius_mossotti(eps_particle, eps_medium, volume):



    raise NotImplementedError("Hole 2: Clausius-Mossotti polarizability formula is missing.")
