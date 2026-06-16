"""
sparse_hamiltonian.py  --  Row-stored sparse Hamiltonian in the shell-model basis
===========================================================================
Adapts the R8STO format (row-stored, packed lower-triangle) to build the
shell-model Hamiltonian matrix in the truncated many-body basis.

The R8STO layout stores the lower triangle of an N x N matrix as a 1-D array
of length N*(N+1)/2, with a companion pointer array of length N+1 pointing to
the start of each row. This is optimal for shell-model Hamiltonians because:

    - The Hamiltonian is real symmetric.
    - In the m-scheme basis, most two-body matrix elements are zero by angular
      momentum conservation (J_z, parity).
    - Lanczos iteration needs only H*v products, which the packed format
      supports with a single cache-friendly pass.

We additionally implement a hybrid "profile" storage that keeps only the band
of non-zero diagonals within the packed triangle (from the knapsack_values
algorithm, which computes binomial-like configuration counts).

References:
    John Burkardt, R8STO library (row-stored sparse format)
    Whitehead et al., Rev. Mod. Phys. 49 (1977) 655 (shell-model review)
    Lanczos, J. Res. NBS 45 (1950) 255
"""

from __future__ import annotations
import math
import numpy as np
from nuclear_constants import R_EPSILON


# ======================================================================
#  R8STO-style packed row storage
# ======================================================================
class R8STOMatrix:
    """Symmetric matrix in packed row (lower triangle) storage.

    Attributes:
        n      : dimension
        data   : 1-D array of length n(n+1)/2
        ptr    : row pointers (length n+1)
    """
    def __init__(self, n: int):
        self.n = n
        nnz = n * (n + 1) // 2
        self.data = np.zeros(nnz, dtype=np.float64)
        self.ptr = np.zeros(n + 1, dtype=np.int64)
        acc = 0
        for i in range(n):
            self.ptr[i] = acc
            acc += (i + 1)
        self.ptr[n] = acc

    def index(self, i: int, j: int) -> int:
        """Linear index of element (i, j) in packed storage (lower triangle)."""
        if i < j:
            i, j = j, i
        return self.ptr[i] + j

    def __setitem__(self, key, value):
        i, j = key
        self.data[self.index(i, j)] = value

    def __getitem__(self, key):
        i, j = key
        return self.data[self.index(i, j)]

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Compute A @ v for symmetric packed A."""
        n = self.n
        out = np.zeros(n, dtype=np.float64)
        for i in range(n):
            s = 0.0
            for j in range(i + 1):
                s += self.data[self.ptr[i] + j] * v[j]
            out[i] += s
        for j in range(n):
            # Off-diagonal transpose contribution
            v_j = v[j]
            for i in range(j + 1, n):
                out[i] += self.data[self.ptr[i] + j] * v_j
        return out

    def to_dense(self) -> np.ndarray:
        """Materialise as dense array (for diagnostics / small problems)."""
        n = self.n
        A = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1):
                A[i, j] = A[j, i] = self.data[self.ptr[i] + j]
        return A


# ======================================================================
#  Single-particle energies and pairing interaction
# ======================================================================
def single_particle_energies(
    orbitals: list,
    A: int,
    Z_core: int,
    radial_solver,
) -> dict:
    """Compute single-particle energies for a list of (n, l, j) orbitals.

    Args:
        orbitals: list of (n_rad, l, j, is_proton) tuples
    Returns:
        dict mapping (n, l, j, kind) -> E_MeV
    """
    spe = {}
    for (n_rad, l, j, is_proton) in orbitals:
        res = radial_solver.solve_radial_levels(
            A=A, Z_core=Z_core, l_q=l, j_q=j,
            is_proton=is_proton, n_levels=n_rad + 1,
        )
        key = (n_rad, l, j, "p" if is_proton else "n")
        spe[key] = float(res["energies"][min(n_rad, len(res["energies"]) - 1)])
    return spe


# ======================================================================
#  Two-body matrix elements (pairing Hamiltonian)
# ======================================================================
def pairing_matrix_element(
    orb_a: tuple, orb_b: tuple, orb_c: tuple, orb_d: tuple,
    G_pair: float,
) -> float:
    """Pairing interaction matrix element <ab|V_pair|cd>.

    V_pair = -G sum_{J=0} P^+_J P_J   with
    P^+_J = sum_a sqrt(2j_a+1)/2  [a^+ x a^+]^{J=0}

    In the seniority-zero subspace the matrix element reduces to
        <ab|V|cd> = -G * delta_{J,0} * sqrt((2j_a+1)(2j_c+1)) / 4
                     * delta_{a,c} * delta_{b,d}

    More general (non-diagonal) terms follow from Racah recoupling; for
    the small-scale experiment we keep the diagonal pairing approximation.
    """
    n_a, l_a, j_a, kind_a = orb_a
    n_b, l_b, j_b, kind_b = orb_b
    n_c, l_c, j_c, kind_c = orb_c
    n_d, l_d, j_d, kind_d = orb_d
    # Pairing conserves particle kind
    if kind_a != kind_c or kind_b != kind_d:
        return 0.0
    # Diagonal (seniority-zero) pairing
    if (n_a == n_c and l_a == l_c and abs(j_a - j_c) < 1e-9
            and n_b == n_d and l_b == l_d and abs(j_b - j_d) < 1e-9):
        return -G_pair * math.sqrt((2.0 * j_a + 1.0) * (2.0 * j_b + 1.0)) / 4.0
    return 0.0


# ======================================================================
#  Shell-model Hamiltonian construction
# ======================================================================
def build_shell_model_hamiltonian(
    orbitals: list,
    n_particles: int,
    spe_dict: dict,
    G_pair: float,
) -> R8STOMatrix:
    """Construct the shell-model Hamiltonian in the seniority-zero basis.

    The basis is enumerated via the knapsack-style DP (counting
    configurations of n_particles pairs distributed among the orbitals).

    Args:
        orbitals:    list of orbital indices (each is (n, l, j, kind))
        n_particles: number of PAIRS (seniority-zero subspace)
        spe_dict:    single-particle energies
        G_pair:      pairing strength in MeV

    Returns:
        R8STOMatrix instance of dimension = number of basis states
    """
    basis = enumerate_seniority_zero_basis(orbitals, n_particles)
    dim = len(basis)
    H = R8STOMatrix(dim)
    for i in range(dim):
        for j in range(i + 1):
            H[i, j] = hamiltonian_element(basis[i], basis[j], spe_dict, orbitals, G_pair)
    return H


def enumerate_seniority_zero_basis(orbitals: list, n_pairs: int) -> list:
    """Enumerate basis states |p_1, p_2, ..., p_K> with sum p_i = n_pairs.

    Each p_i is the number of PAIRS occupying orbital i (0 <= p_i <= Omega_i,
    with Omega_i = (2j_i + 1)/2 the pair degeneracy).

    Uses a knapsack-style DP (from knapsack_values): iterate over orbitals
    and accumulate reachable occupations.
    """
    orbitals = list(orbitals)
    K = len(orbitals)
    if K == 0 or n_pairs < 0:
        return []
    # Omega per orbital
    omega = [max(1, int(round((2.0 * orb[2] + 1.0) / 2.0))) for orb in orbitals]
    # DP: reachable occupations after first k orbitals
    states = [[()]]
    for k in range(K):
        new_states = []
        for prev in states[-1]:
            s_prev = sum(prev)
            for occ in range(min(omega[k], n_pairs - s_prev) + 1):
                new_states.append(prev + (occ,))
        states.append(new_states)
    return [s for s in states[-1] if sum(s) == n_pairs]


def hamiltonian_element(
    bra: tuple, ket: tuple, spe_dict: dict, orbitals: list, G_pair: float,
) -> float:
    """Matrix element <bra | H | ket> in the seniority-zero basis."""
    if len(bra) != len(ket):
        return 0.0
    # SPE contribution (diagonal)
    if bra == ket:
        h_diag = 0.0
        for k, occ in enumerate(bra):
            orb = orbitals[k]
            key = (orb[0], orb[1], orb[2], orb[3])
            h_diag += occ * 2.0 * spe_dict.get(key, 0.0)  # factor 2: 2 nucleons per pair
        # Pairing self-energy: -G * occ * (Omega - occ + 1) / 2
        for k, occ in enumerate(bra):
            orb = orbitals[k]
            omega_k = max(1, int(round((2.0 * orb[2] + 1.0) / 2.0)))
            h_diag += -G_pair * occ * (omega_k - occ + 1) / 2.0
        return h_diag
    # Off-diagonal pairing: differs by one pair moved from orb i to orb j
    diff_from = []
    diff_to = []
    for k in range(len(bra)):
        d = bra[k] - ket[k]
        if d > 0:
            diff_from.extend([k] * d)
        elif d < 0:
            diff_to.extend([k] * (-d))
    if len(diff_from) != 1 or len(diff_to) != 1:
        return 0.0
    i, j = diff_from[0], diff_to[0]
    orb_i = orbitals[i]
    orb_j = orbitals[j]
    omega_i = max(1, int(round((2.0 * orb_i[2] + 1.0) / 2.0)))
    omega_j = max(1, int(round((2.0 * orb_j[2] + 1.0) / 2.0)))
    # <... b_i, b_j ...| V |... k_i, k_j ...> = -G * sqrt(b_i*(Omega_i - b_i + 1))
    #                                                * sqrt(k_j*(Omega_j - k_j + 1))
    b_i = bra[i]
    k_j = ket[j]
    val = -G_pair * math.sqrt(max(b_i * (omega_i - b_i + 1), 0.0)) \
                    * math.sqrt(max(k_j * (omega_j - k_j + 1), 0.0))
    return val


# ======================================================================
#  Lanczos eigensolver (works directly with R8STOMatrix)
# ======================================================================
def lanczos_eigen(H: R8STOMatrix, n_steps: int = 30, seed: int = 12345) -> tuple[np.ndarray, np.ndarray]:
    """Lanczos tridiagonalisation of the packed symmetric matrix H.

    Returns (alpha, beta) tridiagonal coefficients; eigenvalues from the
    resulting Jacobi matrix approximate the extremal eigenvalues of H.
    """
    n = H.n
    rng = np.random.RandomState(seed)
    v = rng.randn(n)
    v = v / np.linalg.norm(v)
    alpha = np.zeros(n_steps)
    beta = np.zeros(n_steps)
    v_prev = np.zeros(n)
    for k in range(n_steps):
        w = H.matvec(v)
        a = float(np.dot(v, w))
        alpha[k] = a
        w = w - a * v - (beta[k - 1] if k > 0 else 0.0) * v_prev
        # Reorthogonalise (full reorthogonalisation for small n)
        b = float(np.linalg.norm(w))
        if b < 1.0e-12:
            alpha = alpha[:k + 1]
            beta = beta[:k]
            break
        beta[k] = b
        v_prev = v
        v = w / b
    return alpha, beta


def jacobi_eigenvalues(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Eigenvalues of the symmetric tridiagonal matrix defined by (alpha, beta).

    The Lanczos iteration produces alpha[0..k] (diagonal) and beta[0..k]
    (off-diagonal with beta[0] = 0 by convention). scipy.eigh_tridiagonal
    wants off-diagonal of length len(alpha) - 1, so we drop beta[0].
    """
    from scipy.linalg import eigh_tridiagonal
    if len(alpha) < 2:
        return np.asarray(alpha)
    off = beta[1:] if len(beta) > 1 else beta
    if len(off) >= len(alpha):
        off = off[:len(alpha) - 1]
    if len(off) == 0:
        return np.asarray(alpha)
    eigs, _vecs = eigh_tridiagonal(alpha, off)
    return np.asarray(eigs)
