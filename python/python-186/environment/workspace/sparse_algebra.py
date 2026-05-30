
import numpy as np
from typing import Tuple, List, Optional


class SparseCCS:

    def __init__(self, n_rows: int, n_cols: int,
                 colptr: np.ndarray, rowind: np.ndarray, values: np.ndarray):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.colptr = colptr.copy()
        self.rowind = rowind.copy()
        self.values = values.copy()
        self.nnz = len(values)

        self._validate()

    def _validate(self):
        assert self.colptr.shape[0] == self.n_cols + 1
        assert self.colptr[0] == 0
        assert self.colptr[-1] == self.nnz
        assert len(self.rowind) == self.nnz
        assert len(self.values) == self.nnz
        assert np.all(self.rowind >= 0) and np.all(self.rowind < self.n_rows)

        for j in range(self.n_cols):
            start = self.colptr[j]
            end = self.colptr[j + 1]
            if end > start:
                rows = self.rowind[start:end]
                assert np.all(np.diff(rows) > 0), f"Column {j} rows not sorted"

    @classmethod
    def from_dense(cls, dense: np.ndarray) -> 'SparseCCS':
        n_rows, n_cols = dense.shape
        colptr = [0]
        rowind = []
        values = []

        for j in range(n_cols):
            col_nonzeros = []
            for i in range(n_rows):
                if abs(dense[i, j]) > 1e-15:
                    col_nonzeros.append((i, dense[i, j]))
            col_nonzeros.sort(key=lambda x: x[0])
            for i, v in col_nonzeros:
                rowind.append(i)
                values.append(v)
            colptr.append(len(values))

        return cls(n_rows, n_cols,
                   np.array(colptr, dtype=np.int32),
                   np.array(rowind, dtype=np.int32),
                   np.array(values, dtype=np.float64))

    def to_dense(self) -> np.ndarray:
        dense = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
        for j in range(self.n_cols):
            for idx in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[idx]
                dense[i, j] = self.values[idx]
        return dense

    def get(self, i: int, j: int) -> float:
        if i < 0 or i >= self.n_rows or j < 0 or j >= self.n_cols:
            raise IndexError(f"Index ({i},{j}) out of bounds")
        start = self.colptr[j]
        end = self.colptr[j + 1]
        pos = np.searchsorted(self.rowind[start:end], i)
        if pos < (end - start) and self.rowind[start + pos] == i:
            return self.values[start + pos]
        return 0.0

    def set(self, i: int, j: int, value: float):
        dense = self.to_dense()
        dense[i, j] = value
        new_sparse = SparseCCS.from_dense(dense)
        self.__dict__.update(new_sparse.__dict__)

    def mv(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.n_cols:
            raise ValueError(f"Dimension mismatch: A is {self.n_rows}x{self.n_cols}, x has {x.shape[0]}")

        y = np.zeros(self.n_rows, dtype=np.float64)
        for j in range(self.n_cols):
            xj = x[j]
            for idx in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[idx]
                y[i] += self.values[idx] * xj

        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.n_rows:
            raise ValueError(f"Dimension mismatch")

        y = np.zeros(self.n_cols, dtype=np.float64)
        for j in range(self.n_cols):
            dot = 0.0
            for idx in range(self.colptr[j], self.colptr[j + 1]):
                i = self.rowind[idx]
                dot += self.values[idx] * x[i]
            y[j] = dot

        return y

    @classmethod
    def network_laplacian(cls, adj: np.ndarray) -> 'SparseCCS':
        n = adj.shape[0]
        degrees = np.sum(adj, axis=1)

        colptr = [0]
        rowind = []
        values = []

        for j in range(n):
            col_nonzeros = []
            for i in range(n):
                val = -adj[i, j]
                if i == j:
                    val = degrees[i] - adj[i, j]
                if abs(val) > 1e-15:
                    col_nonzeros.append((i, val))
            col_nonzeros.sort(key=lambda x: x[0])
            for i, v in col_nonzeros:
                rowind.append(i)
                values.append(v)
            colptr.append(len(values))

        return cls(n, n,
                   np.array(colptr, dtype=np.int32),
                   np.array(rowind, dtype=np.int32),
                   np.array(values, dtype=np.float64))

    def power_iteration_sparse(self, max_iter: int = 1000, tol: float = 1e-10) -> Tuple[float, np.ndarray]:
        n = self.n_cols
        x = np.random.rand(n)
        x /= np.linalg.norm(x)

        lambda_old = 0.0

        for it in range(max_iter):
            y = self.mv(x)
            norm_y = np.linalg.norm(y)
            if norm_y < 1e-15:
                break
            x_new = y / norm_y


            lambda_new = float(x_new @ self.mv(x_new))

            diff = abs(lambda_new - lambda_old)
            cos_angle = float(x @ x_new)
            sin_angle = np.sqrt(max(0.0, 1.0 - cos_angle**2))

            x = x_new
            lambda_old = lambda_new

            if diff < tol and sin_angle < tol:
                break

        return lambda_old, x
