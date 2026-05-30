
import numpy as np


class CRSMatrix:

    def __init__(self, n, nz, row, col, val):
        self.n = n
        self.nz = nz
        self.row = np.asarray(row, dtype=int)
        self.col = np.asarray(col, dtype=int)
        self.val = np.asarray(val, dtype=float)


        if len(self.row) != n + 1:
            raise ValueError(f"行指针长度应为 {n+1}，实际为 {len(self.row)}")
        if len(self.col) != nz:
            raise ValueError(f"列索引长度应为 {nz}，实际为 {len(self.col)}")
        if len(self.val) != nz:
            raise ValueError(f"非零元值长度应为 {nz}，实际为 {len(self.val)}")
        if self.row[0] != 0:
            raise ValueError("row[0] 必须为 0")
        if self.row[n] != nz:
            raise ValueError(f"row[n] 必须为 {nz}，实际为 {self.row[n]}")
        for i in range(n):
            if self.row[i] > self.row[i + 1]:
                raise ValueError(f"行指针非单调递增于索引 {i}")
        for j in range(nz):
            if self.col[j] < 0 or self.col[j] >= n:
                raise ValueError(f"列索引越界于位置 {j}: col={self.col[j]}")

    def matvec(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n:
            raise ValueError(f"向量维度不匹配: {x.shape[0]} != {self.n}")
        y = np.zeros(self.n, dtype=float)
        for i in range(self.n):
            for k in range(self.row[i], self.row[i + 1]):
                j = self.col[k]
                y[i] += self.val[k] * x[j]
        return y

    def to_dense(self):
        A = np.zeros((self.n, self.n), dtype=float)
        for i in range(self.n):
            for k in range(self.row[i], self.row[i + 1]):
                j = self.col[k]
                A[i, j] = self.val[k]
        return A

    def residual_norm(self, x, b):
        return np.linalg.norm(b - self.matvec(x))

    @staticmethod
    def from_dense(A_dense, threshold=1e-15):
        A_dense = np.asarray(A_dense, dtype=float)
        m, n = A_dense.shape
        if m != n:
            raise ValueError("仅支持方阵")
        row = [0]
        col = []
        val = []
        for i in range(n):
            count = 0
            for j in range(n):
                if abs(A_dense[i, j]) > threshold:
                    col.append(j)
                    val.append(A_dense[i, j])
                    count += 1
            row.append(row[-1] + count)
        return CRSMatrix(n, len(val), row, col, val)


def build_molecular_hessian_crs(n_atoms, coordinates, force_constant=1.0, cutoff=3.5):
    N = n_atoms
    dim = 3 * N
    coords = np.asarray(coordinates, dtype=float)


    adjacency = [[] for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            r_ij = np.linalg.norm(coords[i] - coords[j])
            if r_ij < cutoff and r_ij > 0.1:
                adjacency[i].append((j, r_ij))
                adjacency[j].append((i, r_ij))


    row_ptr = [0]
    col_idx = []
    val = []



    for i_atom in range(N):
        for alpha in range(3):
            i_global = 3 * i_atom + alpha
            entries = {}


            for j_atom, r_ij in adjacency[i_atom]:
                for beta in range(3):
                    j_global = 3 * j_atom + beta

                    dr = coords[i_atom] - coords[j_atom]
                    if r_ij > 0:
                        h_elem = force_constant * dr[alpha] * dr[beta] / (r_ij ** 2)
                    else:
                        h_elem = 0.0
                    if j_global not in entries:
                        entries[j_global] = 0.0
                    entries[j_global] += h_elem


            for beta in range(3):
                j_global = 3 * i_atom + beta
                if j_global not in entries:
                    entries[j_global] = 0.0

                entries[j_global] += force_constant * (1.0 if alpha == beta else 0.0)


            sorted_cols = sorted(entries.keys())
            for j_global in sorted_cols:
                col_idx.append(j_global)
                val.append(entries[j_global])
            row_ptr.append(len(col_idx))

    nz = len(val)
    return CRSMatrix(dim, nz, row_ptr, col_idx, val)


def lanczos_eigenvalue_solver(crs_matrix, max_iter=50, tol=1e-10):
    n = crs_matrix.n
    if max_iter > n:
        max_iter = n


    v_prev = np.zeros(n, dtype=float)
    v_curr = np.random.randn(n)
    v_curr /= np.linalg.norm(v_curr)

    alpha = []
    beta = [0.0]

    for j in range(max_iter):
        w = crs_matrix.matvec(v_curr)
        if j > 0:
            w -= beta[j] * v_prev
        alpha_j = np.dot(v_curr, w)
        alpha.append(alpha_j)
        w -= alpha_j * v_curr
        beta_next = np.linalg.norm(w)

        if beta_next < tol:
            break

        beta.append(beta_next)
        v_prev = v_curr.copy()
        v_curr = w / beta_next


    m = len(alpha)
    T = np.zeros((m, m), dtype=float)
    for i in range(m):
        T[i, i] = alpha[i]
        if i < m - 1:
            T[i, i + 1] = beta[i + 1]
            T[i + 1, i] = beta[i + 1]

    eigenvalues = np.linalg.eigvalsh(T)
    return eigenvalues
