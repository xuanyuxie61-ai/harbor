
import numpy as np
from typing import List, Tuple, Optional
from utils import compute_bandwidth






class R8NCFSparseMatrix:

    def __init__(self, n_rows: int, n_cols: int):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.nz_num = 0
        self.rowcol = np.zeros((2, 0), dtype=int)
        self.values = np.zeros(0, dtype=float)

    def add_entry(self, row: int, col: int, value: float):
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            raise IndexError("行列索引越界")
        new_rc = np.zeros((2, self.nz_num + 1), dtype=int)
        new_val = np.zeros(self.nz_num + 1, dtype=float)
        if self.nz_num > 0:
            new_rc[:, : self.nz_num] = self.rowcol
            new_val[: self.nz_num] = self.values
        new_rc[0, self.nz_num] = row
        new_rc[1, self.nz_num] = col
        new_val[self.nz_num] = value
        self.rowcol = new_rc
        self.values = new_val
        self.nz_num += 1

    def mv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_cols:
            raise ValueError("向量维度与矩阵列数不匹配")
        y = np.zeros(self.n_rows, dtype=float)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[i] += self.values[k] * x[j]
        return y

    def mtv(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape[0] != self.n_rows:
            raise ValueError("向量维度与矩阵行数不匹配")
        y = np.zeros(self.n_cols, dtype=float)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            y[j] += self.values[k] * x[i]
        return y

    def to_dense(self) -> np.ndarray:
        A = np.zeros((self.n_rows, self.n_cols), dtype=float)
        for k in range(self.nz_num):
            i = self.rowcol[0, k]
            j = self.rowcol[1, k]
            A[i, j] += self.values[k]
        return A

    def permute_rows_cols(self, perm: List[int]) -> "R8NCFSparseMatrix":
        inv = [0] * len(perm)
        for new_idx, old_idx in enumerate(perm):
            inv[old_idx] = new_idx
        B = R8NCFSparseMatrix(self.n_rows, self.n_cols)
        for k in range(self.nz_num):
            i_old = self.rowcol[0, k]
            j_old = self.rowcol[1, k]
            B.add_entry(inv[i_old], inv[j_old], self.values[k])
        return B


def build_laplacian_2d_sparse(
    nx: int, ny: int, dx: float, dy: float
) -> R8NCFSparseMatrix:
    if nx < 3 or ny < 3:
        raise ValueError("nx 和 ny 至少为 3")
    n = nx * ny
    A = R8NCFSparseMatrix(n, n)
    denom = dx * dy
    for i in range(nx):
        for j in range(ny):
            idx = i * ny + j

            A.add_entry(idx, idx, 4.0 / denom)

            if i > 0:
                A.add_entry(idx, (i - 1) * ny + j, -1.0 / denom)

            if i < nx - 1:
                A.add_entry(idx, (i + 1) * ny + j, -1.0 / denom)

            if j > 0:
                A.add_entry(idx, i * ny + (j - 1), -1.0 / denom)

            if j < ny - 1:
                A.add_entry(idx, i * ny + (j + 1), -1.0 / denom)
    return A


def build_second_difference_1d_sparse(n: int) -> R8NCFSparseMatrix:
    if n < 2:
        raise ValueError("n 至少为 2")
    A = R8NCFSparseMatrix(n, n)
    for i in range(n):
        A.add_entry(i, i, 2.0)
        if i > 0:
            A.add_entry(i, i - 1, -1.0)
        if i < n - 1:
            A.add_entry(i, i + 1, -1.0)
    return A






def build_adjacency_from_triangulation(
    triangles: List[Tuple[int, int, int]], n_nodes: int
) -> List[List[int]]:
    adj = [set() for _ in range(n_nodes)]
    for tri in triangles:
        for k in range(3):
            a = tri[k]
            b = tri[(k + 1) % 3]
            if 0 <= a < n_nodes and 0 <= b < n_nodes:
                adj[a].add(b)
                adj[b].add(a)
    return [sorted(list(s)) for s in adj]


def rcm_reorder(adj: List[List[int]]) -> List[int]:
    n = len(adj)
    if n == 0:
        return []
    visited = [False] * n
    perm = []

    for start in range(n):
        if visited[start]:
            continue

        root = _find_pseudo_peripheral(adj, start)

        level_order = []
        queue = [root]
        visited[root] = True
        while queue:

            queue.sort(key=lambda x: len(adj[x]))
            next_queue = []
            for node in queue:
                level_order.append(node)
                for nb in adj[node]:
                    if not visited[nb]:
                        visited[nb] = True
                        next_queue.append(nb)
            queue = next_queue

        perm.extend(reversed(level_order))
    return perm


def _find_pseudo_peripheral(adj: List[List[int]], start: int) -> int:
    n = len(adj)
    current = start
    while True:
        levels, _ = _build_level_structure(adj, current)
        last_level = levels[-1]

        min_deg = n + 1
        candidate = current
        for node in last_level:
            deg = len(adj[node])
            if deg < min_deg:
                min_deg = deg
                candidate = node
        if candidate == current:
            break
        new_levels, _ = _build_level_structure(adj, candidate)
        if len(new_levels) <= len(levels):
            break
        current = candidate
    return current


def _build_level_structure(
    adj: List[List[int]], root: int
) -> Tuple[List[List[int]], List[int]]:
    n = len(adj)
    level_idx = [-1] * n
    queue = [root]
    level_idx[root] = 0
    levels = [[root]]
    while queue:
        next_queue = []
        for node in queue:
            for nb in adj[node]:
                if level_idx[nb] == -1:
                    level_idx[nb] = len(levels)
                    next_queue.append(nb)
        if not next_queue:
            break
        levels.append(next_queue)
        queue = next_queue
    return levels, level_idx


def apply_rcm_to_matrix(
    A: R8NCFSparseMatrix, triangles: List[Tuple[int, int, int]], n_nodes: int
) -> Tuple[R8NCFSparseMatrix, List[int], int, int]:
    adj = build_adjacency_from_triangulation(triangles, n_nodes)
    bw_before = compute_bandwidth(adj, list(range(n_nodes)))
    perm = rcm_reorder(adj)
    bw_after = compute_bandwidth(adj, perm)
    A_perm = A.permute_rows_cols(perm)
    return A_perm, perm, bw_before, bw_after
