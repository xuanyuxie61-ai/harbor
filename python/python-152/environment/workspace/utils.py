import numpy as np
from scipy.special import erf, erfc


def kronecker_delta(i: int, j: int) -> int:
    return 1 if i == j else 0


def pauli_operators():
    I = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return I, X, Y, Z


def pauli_commutator(P, Q):
    return P @ Q - Q @ P


def pauli_anticommutator(P, Q):
    return P @ Q + Q @ P


def trace_inner_product(A: np.ndarray, B: np.ndarray) -> complex:
    d = A.shape[0]
    return np.trace(A.conj().T @ B) / d


def von_neumann_entropy(rho: np.ndarray) -> float:
    eps = 1e-14
    w = np.linalg.eigvalsh(rho)
    w = w[w > eps]
    return -np.sum(w * np.log2(w))


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    sr = sqrtm_psd(rho)
    M = sr @ sigma @ sr
    w = np.linalg.eigvalsh(M)
    w = np.maximum(w, 0)
    return (np.sum(np.sqrt(w))) ** 2


def sqrtm_psd(A: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(A)
    w = np.maximum(w, 0)
    return v @ np.diag(np.sqrt(w)) @ v.conj().T


def tensor_pauli(indices, n_qubits):
    I, X, Y, Z = pauli_operators()
    pmap = [I, X, Y, Z]
    result = np.eye(1, dtype=complex)
    for idx in indices:
        result = np.kron(result, pmap[idx])
    return result


def depolarizing_channel(p: float, n_qubits: int = 1) -> np.ndarray:
    d = 2 ** n_qubits
    N = d * d
    chi = np.eye(N, dtype=complex)
    chi *= (1.0 - p)

    chi += p * np.eye(N, dtype=complex) / N
    return chi


def bit_flip_channel(p: float) -> np.ndarray:
    return np.array([
        [1.0 - p, 0, 0, p],
        [0, 1.0 - p, p, 0],
        [0, p, 1.0 - p, 0],
        [p, 0, 0, 1.0 - p]
    ], dtype=complex)


def phase_flip_channel(p: float) -> np.ndarray:
    return np.array([
        [1.0, 0, 0, 0],
        [0, 1.0 - 2 * p, 0, 0],
        [0, 0, 1.0 - 2 * p, 0],
        [0, 0, 0, 1.0]
    ], dtype=complex)


def chop_complex(z: complex, tol: float = 1e-14) -> complex:
    re = 0.0 if abs(z.real) < tol else z.real
    im = 0.0 if abs(z.imag) < tol else z.imag
    return re + 1j * im


def chop_array(arr: np.ndarray, tol: float = 1e-14) -> np.ndarray:
    out = arr.copy()
    if np.iscomplexobj(out):
        re = out.real
        im = out.imag
        re[np.abs(re) < tol] = 0.0
        im[np.abs(im) < tol] = 0.0
        mask = (np.abs(re) < tol) & (np.abs(im) < tol)
        re[mask] = 0.0
        im[mask] = 0.0
        out = re + 1j * im
    else:
        out[np.abs(out) < tol] = 0.0
    return out


def symplectic_inner_product(a: np.ndarray, b: np.ndarray) -> int:
    n = a.shape[0] // 2
    x_a, z_a = a[:n], a[n:]
    x_b, z_b = b[:n], b[n:]
    return int((np.dot(x_a, z_b) + np.dot(z_a, x_b)) % 2)


def hamming_weight(v: np.ndarray) -> int:
    return int(np.sum(v % 2))


def binary_gaussian_elimination(M: np.ndarray) -> tuple:




    raise NotImplementedError("Hole 3: binary_gaussian_elimination to be implemented.")


def stabilizer_centralizer(S: np.ndarray) -> np.ndarray:
    m, n2 = S.shape
    n = n2 // 2


    Omega = np.block([
        [np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
        [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]
    ])
    symp = (Omega @ S.T) % 2

    rref, rank, pivots = binary_gaussian_elimination(symp.T)
    free_cols = [c for c in range(n2) if c not in pivots]
    basis = []
    for fc in free_cols:
        vec = np.zeros(n2, dtype=int)
        vec[fc] = 1
        for i, p in enumerate(pivots):
            vec[p] = rref[i, fc]

        ok = True
        for s in S:
            if symplectic_inner_product(vec, s) % 2 != 0:
                ok = False
                break
        if ok:
            basis.append(vec % 2)
    if not basis:
        return np.zeros((0, n2), dtype=int)
    return np.array(basis, dtype=int)
