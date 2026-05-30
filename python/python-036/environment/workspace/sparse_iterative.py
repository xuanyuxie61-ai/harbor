
import numpy as np
from constants import DELTA_M2_21, DELTA_M2_31, DELTA_M2_31_IH


def sparse_from_index_data(index_data):
    n = len(index_data)
    row = []
    col = []

    for i in range(n):
        neighbors = index_data[i]
        for j in neighbors:
            row.append(i)
            col.append(j)

    row = np.array(row, dtype=np.int64)
    col = np.array(col, dtype=np.int64)
    data = np.ones(len(row), dtype=np.float64)

    try:
        from scipy.sparse import coo_matrix
        A = coo_matrix((data, (row, col)), shape=(n, n))
        return A
    except ImportError:

        A = np.zeros((n, n), dtype=np.float64)
        A[row, col] = data
        return A


def power_iteration(A, n_iterations=100, tol=1e-10, seed=None):
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        raise ValueError("Empty matrix")

    rng = np.random.default_rng(seed)
    v = rng.random(n)
    v = v / np.linalg.norm(v)

    eigenvalue = 0.0

    for iteration in range(n_iterations):
        Av = A @ v
        norm = np.linalg.norm(Av)
        if norm < 1e-15:
            break
        v_new = Av / norm


        eigenvalue_new = np.dot(v_new, A @ v_new) / np.dot(v_new, v_new)

        if abs(eigenvalue_new - eigenvalue) < tol and np.linalg.norm(v_new - v) < tol:
            return float(eigenvalue_new), v_new, True

        eigenvalue = eigenvalue_new
        v = v_new

    return float(eigenvalue), v, False


def pagerank_style_matrix(H, damping=0.85):
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0]

    if damping <= 0 or damping >= 1:
        raise ValueError("damping must be in (0, 1)")


    A = np.abs(H)


    col_sums = np.sum(A, axis=0)
    for j in range(n):
        if col_sums[j] > 0:
            A[:, j] = A[:, j] / col_sums[j]
        else:
            A[:, j] = 1.0 / n


    P = damping * A + (1.0 - damping) / n * np.ones((n, n))

    return P


def find_dominant_oscillation_mode(H, n_iterations=200, tol=1e-12):
    H = np.asarray(H, dtype=np.complex128)


    eigenvalues, eigenvectors = np.linalg.eigh(H)


    idx_max = np.argmax(np.abs(eigenvalues))
    ev = eigenvalues[idx_max]
    state = eigenvectors[:, idx_max]


    state = state / np.linalg.norm(state)

    return {
        'energy': float(ev),
        'state': state,
        'frequency': float(abs(ev))
    }


def iterative_hierarchy_solver(
        energy_gev, baseline_km,
        n_iterations=50, tol=1e-10,
        theta12=None, theta23=None, theta13=None,
        delta_cp=None, delta_m2_21=None
):
    from pmns_matrix import build_pmns_matrix
    from neutrino_hamiltonian import build_vacuum_hamiltonian

    U = build_pmns_matrix(theta12, theta23, theta13, delta_cp)


    M2_NH = np.diag([0.0, DELTA_M2_21 if delta_m2_21 is None else delta_m2_21, DELTA_M2_31])
    H_NH = (1.0 / (2.0 * energy_gev * 1e9)) * (U @ M2_NH @ U.conj().T)


    M2_IH = np.diag([0.0, DELTA_M2_21 if delta_m2_21 is None else delta_m2_21, DELTA_M2_31_IH])
    H_IH = (1.0 / (2.0 * energy_gev * 1e9)) * (U @ M2_IH @ U.conj().T)

    L_ev_inv = baseline_km * 5.067730889e9

    psi0 = np.array([1.0, 0.0, 0.0], dtype=np.complex128)


    ev_NH, evec_NH = np.linalg.eigh(H_NH)
    D_NH = np.diag(np.exp(-1j * ev_NH * L_ev_inv))
    U_prop_NH = evec_NH @ D_NH @ evec_NH.conj().T
    psi_NH = U_prop_NH @ psi0
    P_NH = abs(psi_NH[0]) ** 2


    ev_IH, evec_IH = np.linalg.eigh(H_IH)
    D_IH = np.diag(np.exp(-1j * ev_IH * L_ev_inv))
    U_prop_IH = evec_IH @ D_IH @ evec_IH.conj().T
    psi_IH = U_prop_IH @ psi0
    P_IH = abs(psi_IH[0]) ** 2

    delta_P = abs(P_NH - P_IH)

    return {
        'P_ee_NH': float(P_NH),
        'P_ee_IH': float(P_IH),
        'delta_P': float(delta_P),
        'discrimination_power': float(delta_P / max(P_NH, P_IH, 1e-10))
    }
