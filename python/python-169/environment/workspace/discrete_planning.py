
import numpy as np
from typing import List, Tuple, Optional






def _i4vec_gcd(a: np.ndarray) -> int:
    a = np.asarray(a, dtype=int)
    g = abs(a[0])
    for val in a[1:]:
        val = abs(int(val))
        while val:
            g, val = val, g % val
        if g == 1:
            break
    return g


def diophantine_nd_solutions(a: np.ndarray, b: int,
                              max_solutions: int = 10000) -> np.ndarray:
    a = np.asarray(a, dtype=int)
    b = int(b)
    n = a.size
    if n == 0:
        return np.array([])
    if b < 0:
        return np.array([])

    g = _i4vec_gcd(a)
    if b % g != 0:
        return np.array([])

    a = a // g
    b = b // g

    sort_idx = np.argsort(-a)
    a_sorted = a[sort_idx]
    solutions = []

    def backtrack(idx: int, remaining: int, current: List[int]):
        if remaining < 0:
            return
        if idx == n - 1:
            if remaining % a_sorted[idx] == 0:
                x_last = remaining // a_sorted[idx]
                sol = current + [x_last]

                full_sol = [0] * n
                for s_i, val in zip(sort_idx, sol):
                    full_sol[s_i] = val
                solutions.append(full_sol)
            return
        max_val = remaining // a_sorted[idx]
        for val in range(max_val, -1, -1):
            if len(solutions) >= max_solutions:
                return
            backtrack(idx + 1, remaining - val * a_sorted[idx], current + [val])

    backtrack(0, b, [])
    if not solutions:
        return np.array([])
    return np.array(solutions, dtype=int)


def allocate_control_cycles(total_cycles: int, joint_weights: np.ndarray,
                            max_solutions: int = 100) -> np.ndarray:
    joint_weights = np.asarray(joint_weights, dtype=int)

    joint_weights = np.maximum(joint_weights, 1)
    sols = diophantine_nd_solutions(joint_weights, total_cycles, max_solutions)
    return sols






def rref_compute(A: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    A = np.array(A, dtype=float)
    m, n = A.shape
    A = A.copy()
    row = 0
    pivot_cols = []
    for col in range(n):

        pivot_val = 0.0
        pivot_row = -1
        for r in range(row, m):
            if abs(A[r, col]) > abs(pivot_val):
                pivot_val = A[r, col]
                pivot_row = r
        if abs(pivot_val) < tol:
            continue

        A[[row, pivot_row]] = A[[pivot_row, row]]

        A[row] = A[row] / A[row, col]

        for r in range(m):
            if r != row and abs(A[r, col]) > tol:
                A[r] = A[r] - A[r, col] * A[row]
        pivot_cols.append(col)
        row += 1
        if row >= m:
            break
    return A, pivot_cols


def exact_cover_binary(A: np.ndarray, b: np.ndarray,
                       max_solutions: int = 1000) -> List[np.ndarray]:
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape
    if b.size != m:
        raise ValueError("b维度不匹配")

    Ab = np.hstack([A, b.reshape(-1, 1)])
    R, pivot_cols = rref_compute(Ab)
    free_cols = [c for c in range(n) if c not in pivot_cols]
    k = len(free_cols)
    solutions = []
    for mask in range(1 << k):
        if len(solutions) >= max_solutions:
            break
        x = np.zeros(n, dtype=float)
        for i, fc in enumerate(free_cols):
            x[fc] = 1.0 if (mask & (1 << i)) else 0.0

        valid = True
        for r in range(m):

            pc = -1
            for c in range(n):
                if abs(R[r, c] - 1.0) < 1e-8:
                    pc = c
                    break
            if pc < 0:

                if abs(R[r, -1]) > 1e-8:
                    valid = False
                    break
                continue
            x[pc] = R[r, -1]
            for fc in free_cols:
                x[pc] -= R[r, fc] * x[fc]

            if abs(x[pc] - round(x[pc])) > 1e-6:
                valid = False
                break
            x[pc] = round(x[pc])
            if x[pc] < -1e-6 or x[pc] > 1.0 + 1e-6:
                valid = False
                break
        if valid:
            solutions.append(x.astype(int))
    return solutions


def workspace_coverage_exact_cover(n_poses: int, n_cells: int,
                                    coverage_matrix: np.ndarray) -> List[np.ndarray]:
    A = np.asarray(coverage_matrix, dtype=float)
    m, n = A.shape
    b = np.ones(m, dtype=float)
    return exact_cover_binary(A, b)
