
import numpy as np
from scipy.linalg import eigh
from typing import Tuple, Optional


def fermion_hilbert_dimension(nsites: int) -> int:
    if nsites < 0 or nsites > 6:

        raise ValueError("nsites 必须满足 0 <= nsites <= 6")
    return 4 ** nsites


def _occupation_state(nsites: int, state_idx: int, sigma: int) -> int:
    if not (0 <= state_idx < nsites):
        raise IndexError("state_idx 越界")
    bit_pos = 2 * state_idx + sigma
    return (state_idx >> bit_pos) & 1


def _apply_hopping(state: int, i: int, j: int, sigma: int, nsites: int) -> Tuple[int, float]:
    bit_j = 2 * j + sigma
    bit_i = 2 * i + sigma

    if not ((state >> bit_j) & 1):
        return -1, 0.0

    if (state >> bit_i) & 1:
        return -1, 0.0

    new_state = state ^ (1 << bit_j)
    new_state = new_state ^ (1 << bit_i)

    low, high = sorted((bit_i, bit_j))
    sign = (-1) ** bin((state >> low) & ((1 << (high - low)) - 1)).count("1")
    return new_state, float(sign)


def build_hubbard_hamiltonian(nsites: int, neighbors: list, t: float, U: float, mu: float = 0.0) -> np.ndarray:
    if nsites < 0 or nsites > 6:
        raise ValueError("nsites 必须在 [0, 6] 范围内")
    dim = 4 ** nsites
    H = np.zeros((dim, dim), dtype=np.float64)
    for state in range(dim):

        n_up_total = 0
        n_dn_total = 0
        for i in range(nsites):
            n_up = (state >> (2 * i)) & 1
            n_dn = (state >> (2 * i + 1)) & 1
            n_up_total += n_up
            n_dn_total += n_dn
            H[state, state] += U * n_up * n_dn
        H[state, state] -= mu * (n_up_total + n_dn_total)

        for i in range(nsites):
            for j in neighbors[i]:
                if j <= i:
                    continue
                for sigma in [0, 1]:
                    new_state, sign = _apply_hopping(state, i, j, sigma, nsites)
                    if sign != 0.0:
                        H[state, new_state] -= t * sign

    H = 0.5 * (H + H.T.conj())
    return H


def exact_diagonalization(H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if H.shape[0] != H.shape[1]:
        raise ValueError("H 必须是方阵")
    evals, evecs = eigh(H)
    return evals, evecs


def thermal_average(evals: np.ndarray, evecs: np.ndarray, operator: np.ndarray, beta: float) -> float:
    if beta < 0:
        raise ValueError("beta 必须 >= 0")
    if len(evals) == 0:
        return 0.0

    E0 = np.min(evals)
    weights = np.exp(-beta * (evals - E0))
    Z = np.sum(weights)
    if Z == 0:
        return 0.0

    O_diag = np.einsum('ni,ij,nj->n', evecs, operator, evecs)
    return np.sum(weights * O_diag) / Z


def double_occupancy_operator(nsites: int) -> np.ndarray:
    dim = 4 ** nsites
    D = np.zeros((dim, dim), dtype=np.float64)
    for state in range(dim):
        d = 0
        for i in range(nsites):
            n_up = (state >> (2 * i)) & 1
            n_dn = (state >> (2 * i + 1)) & 1
            d += n_up * n_dn
        D[state, state] = float(d)
    return D


def density_operator(nsites: int, sigma: int) -> np.ndarray:
    dim = 4 ** nsites
    Nop = np.zeros((dim, dim), dtype=np.float64)
    for state in range(dim):
        n = 0
        for i in range(nsites):
            n += (state >> (2 * i + sigma)) & 1
        Nop[state, state] = float(n)
    return Nop


def compute_ground_state_properties(nsites: int, neighbors: list, t: float, U: float, mu: float = 0.0) -> dict:
    H = build_hubbard_hamiltonian(nsites, neighbors, t, U, mu)
    evals, evecs = exact_diagonalization(H)
    gs = evecs[:, 0]
    E0 = evals[0]
    D = double_occupancy_operator(nsites)
    d_occ = float(np.vdot(gs, D @ gs).real)
    Nup = density_operator(nsites, 0)
    Ndn = density_operator(nsites, 1)
    n_up = float(np.vdot(gs, Nup @ gs).real)
    n_dn = float(np.vdot(gs, Ndn @ gs).real)
    return {
        "E0": E0,
        "double_occupancy": d_occ,
        "n_up": n_up,
        "n_dn": n_dn,
        "n_total": n_up + n_dn,
        "energy_gap": evals[1] - evals[0] if len(evals) > 1 else 0.0,
    }


if __name__ == "__main__":

    nsites = 2
    neighbors = [[1], [0]]
    props = compute_ground_state_properties(nsites, neighbors, t=1.0, U=4.0)
    print(props)
