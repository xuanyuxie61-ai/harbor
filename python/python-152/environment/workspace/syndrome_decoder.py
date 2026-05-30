import numpy as np
from utils import hamming_weight, binary_gaussian_elimination


class MWPMBruteDecoder:

    def __init__(self, H: np.ndarray):
        self.H = H.copy().astype(int) % 2
        self.m, self.n = H.shape

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = syndrome.astype(int) % 2
        best_weight = self.n + 1
        best_error = np.zeros(self.n, dtype=int)

        gray = 0
        for subset in range(1 << self.n):

            if subset > 0:
                gray = subset ^ (subset >> 1)

            e = np.array([(gray >> i) & 1 for i in range(self.n)], dtype=int)
            s = (self.H @ e) % 2
            if np.array_equal(s, syndrome):
                w = hamming_weight(e)
                if w < best_weight:
                    best_weight = w
                    best_error = e.copy()
        return best_error

    def decode_pauli(self, syndrome: np.ndarray, n_qubits: int) -> tuple:
        sx = syndrome[:self.m // 2] if syndrome.size > self.m // 2 else syndrome
        sz = syndrome[self.m // 2:] if syndrome.size > self.m // 2 else np.zeros_like(sx)


        e_x = self.decode(sx)
        e_z = self.decode(sz)
        return e_x, e_z


class BeliefPropagationDecoder:

    def __init__(self, H: np.ndarray, max_iter: int = 100):
        self.H = H.copy().astype(int) % 2
        self.m, self.n = H.shape
        self.max_iter = max_iter

    def decode(self, syndrome: np.ndarray, llr: np.ndarray = None) -> np.ndarray:






        raise NotImplementedError("Hole 2: BeliefPropagationDecoder.decode to be implemented.")


class KnapsackLikeLogicalDecoder:

    def __init__(self, H: np.ndarray):
        self.H = H.copy().astype(int) % 2
        self.m, self.n = H.shape
        self._precompute_basis()

    def _precompute_basis(self):
        rref, rank, pivots = binary_gaussian_elimination(self.H)
        free_cols = [c for c in range(self.n) if c not in pivots]
        self.kernel_basis = []
        for fc in free_cols:
            vec = np.zeros(self.n, dtype=int)
            vec[fc] = 1
            for i, p in enumerate(pivots):
                vec[p] = rref[i, fc]
            self.kernel_basis.append(vec % 2)
        self.kernel_basis = np.array(self.kernel_basis, dtype=int)
        self.n_kernel = len(self.kernel_basis)

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        syndrome = syndrome.astype(int) % 2

        rref, rank, pivots = binary_gaussian_elimination(
            np.hstack([self.H, syndrome.reshape(-1, 1)])
        )
        e0 = np.zeros(self.n, dtype=int)
        for i in range(rank - 1, -1, -1):
            p = pivots[i]
            val = rref[i, -1]
            for j in range(p + 1, self.n):
                if rref[i, j]:
                    val = (val + e0[j]) % 2
            e0[p] = val % 2

        if not np.array_equal((self.H @ e0) % 2, syndrome):

            return MWPMBruteDecoder(self.H).decode(syndrome)
        best = e0.copy()
        best_w = hamming_weight(best)

        if self.n_kernel <= 15:
            for mask in range(1, 1 << self.n_kernel):
                l = np.zeros(self.n, dtype=int)
                for b in range(self.n_kernel):
                    if (mask >> b) & 1:
                        l = (l + self.kernel_basis[b]) % 2
                candidate = (e0 + l) % 2
                w = hamming_weight(candidate)
                if w < best_w:
                    best_w = w
                    best = candidate.copy()
        return best


class UnionFindDecoder:

    def __init__(self, L: int, boundary: str = "toric"):
        self.L = L
        self.boundary = boundary
        self.n_vertices = L * L
        self.parent = np.arange(self.n_vertices)
        self.size = np.ones(self.n_vertices, dtype=int)

    def _find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def _union(self, x, y):
        rx, ry = self._find(x), self._find(y)
        if rx == ry:
            return
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]

    def decode_syndrome(self, syndrome_vertices: list) -> np.ndarray:

        n = 2 * self.L * self.L if self.boundary == "toric" else 2 * self.L * (self.L + 1)
        recovery = np.zeros(n, dtype=int)

        defects = list(syndrome_vertices)
        paired = set()
        for i in range(len(defects)):
            if i in paired:
                continue
            best_j = -1
            best_d = 2 * self.L
            for j in range(i + 1, len(defects)):
                if j in paired:
                    continue
                vi, vj = defects[i], defects[j]
                ri, ci = vi // self.L, vi % self.L
                rj, cj = vj // self.L, vj % self.L

                dr = min(abs(ri - rj), self.L - abs(ri - rj))
                dc = min(abs(ci - cj), self.L - abs(ci - cj))
                d = dr + dc
                if d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                paired.add(i)
                paired.add(best_j)

                recovery[defects[i] % n] = 1
                recovery[defects[best_j] % n] = 1
        return recovery
