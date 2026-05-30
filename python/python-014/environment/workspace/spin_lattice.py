
import numpy as np
from typing import Tuple, List, Optional
from utils import is_prime, primes_up_to, EPS_MACHINE


class PyrochloreLattice:

    def __init__(self, L: int = 4, J1: float = 1.0, J2: float = 0.0, disorder_std: float = 0.15):
        if L < 1:
            raise ValueError("L must be >= 1")
        self.L = L
        self.N = 4 * L * L * L
        self.J1 = J1
        self.J2 = J2
        self.disorder_std = disorder_std
        self._build_sites()
        self._build_bonds()

    def _build_sites(self):
        basis = np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ], dtype=float)
        sites = []
        for ix in range(self.L):
            for iy in range(self.L):
                for iz in range(self.L):
                    origin = np.array([ix, iy, iz], dtype=float) / self.L
                    for b in basis:
                        sites.append(origin + b / self.L)
        self.sites = np.array(sites, dtype=float)

        self.sites %= 1.0

    def _periodic_distance_sq(self, i: int, j: int) -> float:
        diff = np.abs(self.sites[i] - self.sites[j])
        diff = np.minimum(diff, 1.0 - diff)
        return np.sum(diff ** 2)

    def _build_bonds(self):
        self.bonds = []
        self.J = np.zeros((self.N, self.N), dtype=float)
        d_nn_sq = (np.sqrt(2.0) / (2.0 * self.L)) ** 2
        tol = 1e-6
        for i in range(self.N):
            for j in range(i + 1, self.N):
                d2 = self._periodic_distance_sq(i, j)
                if np.abs(d2 - d_nn_sq) < tol:
                    disorder = np.random.normal(0.0, self.disorder_std)
                    jval = self.J1 + disorder
                    self.bonds.append((i, j, jval))
                    self.J[i, j] = jval
                    self.J[j, i] = jval
                elif self.J2 != 0.0:
                    d_nnn_sq = 2.0 * d_nn_sq
                    if np.abs(d2 - d_nnn_sq) < tol:
                        disorder = np.random.normal(0.0, self.disorder_std)
                        jval = self.J2 + disorder
                        self.bonds.append((i, j, jval))
                        self.J[i, j] = jval
                        self.J[j, i] = jval

    def adjacency_matrix(self) -> np.ndarray:
        A = (np.abs(self.J) > EPS_MACHINE).astype(int)
        return A

    def degree_matrix(self) -> np.ndarray:
        degrees = np.sum(np.abs(self.J) > EPS_MACHINE, axis=1)
        return np.diag(degrees)

    def graph_laplacian(self) -> np.ndarray:
        return self.degree_matrix() - self.adjacency_matrix()


class PrimeFrustratedLattice:

    def __init__(self, L: int = 20, J0: float = 1.0, alpha: float = 0.3, seed: int = 42):
        np.random.seed(seed)
        if L < 2:
            raise ValueError("L must be >= 2")
        self.L = L
        self.N = L * L
        self.J0 = J0
        self.alpha = alpha
        self.primes = primes_up_to(max(100, 2 * L))
        self._build_couplings()

    def _build_couplings(self):
        self.J_h = np.zeros((self.L, self.L - 1), dtype=float)
        self.J_v = np.zeros((self.L - 1, self.L), dtype=float)
        pmax = self.primes[-1] if self.primes else 1.0
        for i in range(self.L):
            for j in range(self.L - 1):
                idx = (i + j) % max(len(self.primes), 1)
                p = self.primes[idx] if self.primes else 1
                phase = np.sin(np.pi * p / pmax)
                sign = 1.0 if np.random.rand() > 0.3 else -1.0
                self.J_h[i, j] = sign * self.J0 * (1.0 + self.alpha * phase)
        for i in range(self.L - 1):
            for j in range(self.L):
                idx = (i * 3 + j * 7) % max(len(self.primes), 1)
                p = self.primes[idx] if self.primes else 1
                phase = np.cos(np.pi * p / pmax)
                sign = 1.0 if np.random.rand() > 0.3 else -1.0
                self.J_v[i, j] = sign * self.J0 * (1.0 + self.alpha * phase)

    def to_full_matrix(self) -> np.ndarray:
        N = self.N
        J = np.zeros((N, N), dtype=float)
        for i in range(self.L):
            for j in range(self.L - 1):
                idx1 = i * self.L + j
                idx2 = i * self.L + (j + 1)
                J[idx1, idx2] = self.J_h[i, j]
                J[idx2, idx1] = self.J_h[i, j]
        for i in range(self.L - 1):
            for j in range(self.L):
                idx1 = i * self.L + j
                idx2 = (i + 1) * self.L + j
                J[idx1, idx2] = self.J_v[i, j]
                J[idx2, idx1] = self.J_v[i, j]
        return J


def connected_components_2d_spin_map(spin_map: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if spin_map.ndim != 2:
        raise ValueError("spin_map must be 2D")
    m, n = spin_map.shape
    A = (np.abs(spin_map) > threshold).astype(int)
    C = np.zeros((m, n), dtype=int)
    component_index = 0

    for i2 in range(m):
        for j2 in range(n):
            if A[i2, j2] != 0 and C[i2, j2] == 0:
                plist = [(i2, j2)]
                component_index += 1
                while plist:
                    i, j = plist.pop()
                    if C[i, j] != 0:
                        continue
                    C[i, j] = component_index

                    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < m and 0 <= nj < n:
                            if A[ni, nj] != 0 and C[ni, nj] == 0:
                                plist.append((ni, nj))
    return C
