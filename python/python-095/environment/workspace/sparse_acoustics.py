
import numpy as np
from collections import defaultdict


class SparseAcousticMatrix:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.st_rows = []
        self.st_cols = []
        self.st_vals = []
        self.ccs_ready = False
        self.ccc = None
        self.icc = None
        self.acc = None

    def add_entry(self, i, j, val):
        if not (0 <= i < self.m and 0 <= j < self.n):
            raise IndexError("Sparse index out of bounds")
        self.st_rows.append(i)
        self.st_cols.append(j)
        self.st_vals.append(val)

    def st_to_ccs(self):
        nst = len(self.st_vals)
        if nst == 0:
            self.ccc = np.zeros(self.n + 1, dtype=int)
            self.icc = np.array([], dtype=int)
            self.acc = np.array([], dtype=float)
            self.ccs_ready = True
            return


        data = list(zip(self.st_cols, self.st_rows, self.st_vals))
        data.sort(key=lambda x: (x[0], x[1]))


        col_counts = defaultdict(int)
        for col, row, val in data:
            col_counts[col] += 1

        self.ccc = np.zeros(self.n + 1, dtype=int)
        for j in range(1, self.n + 1):
            self.ccc[j] = self.ccc[j - 1] + col_counts.get(j - 1, 0)

        self.icc = np.zeros(nst, dtype=int)
        self.acc = np.zeros(nst, dtype=float)


        next_pos = self.ccc[:-1].copy()
        for col, row, val in data:
            pos = next_pos[col]

            if pos > self.ccc[col] and self.icc[pos - 1] == row:
                self.acc[pos - 1] += val
            else:
                self.icc[pos] = row
                self.acc[pos] = val
                next_pos[col] += 1


        actual_nnz = next_pos[-1] if self.n > 0 else 0
        if actual_nnz < nst:
            self.icc = self.icc[:actual_nnz]
            self.acc = self.acc[:actual_nnz]
            self.ccc[-1] = actual_nnz

        self.ccs_ready = True

    def ccs_mv(self, x):
        if not self.ccs_ready:
            self.st_to_ccs()
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.m, dtype=float)
        for j in range(self.n):
            clo = self.ccc[j]
            chi = self.ccc[j + 1]
            for k in range(clo, chi):
                i = self.icc[k]
                y[i] += self.acc[k] * x[j]
        return y

    def st_mv(self, x):
        x = np.asarray(x, dtype=float)
        y = np.zeros(self.m, dtype=float)
        for k in range(len(self.st_vals)):
            i = self.st_rows[k]
            j = self.st_cols[k]
            y[i] += self.st_vals[k] * x[j]
        return y

    def to_dense(self):
        A = np.zeros((self.m, self.n), dtype=float)
        for k in range(len(self.st_vals)):
            A[self.st_rows[k], self.st_cols[k]] += self.st_vals[k]
        return A


def generate_room_coupling_graph(n_nodes, connection_prob=0.15, seed=42):
    rng = np.random.default_rng(seed)
    S = SparseAcousticMatrix(n_nodes, n_nodes)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                S.add_entry(i, j, 1.0)
            elif rng.random() < connection_prob:
                val = rng.normal(0.0, 0.3)
                S.add_entry(i, j, val)
                S.add_entry(j, i, val)
    return S


def acoustic_transfer_matrix_sparse(sensor_positions, source_positions,
                                    k, reflection_coeff=0.8, max_order=2):




    raise NotImplementedError("Hole 2: acoustic_transfer_matrix_sparse 待实现")
