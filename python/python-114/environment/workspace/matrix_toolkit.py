
import numpy as np
import io


def assemble_hilbert_kernel_matrix(m: int, n: int,
                                    cutoff_distance: float = 5.0,
                                    coords: np.ndarray = None) -> np.ndarray:
    if m <= 0 or n <= 0:
        raise ValueError("Dimensions must be positive")

    K = np.zeros((m, n), dtype=float)
    sigma = cutoff_distance / 3.0

    for i in range(m):
        for j in range(n):
            h_ij = 1.0 / (i + j + 1.0)
            if coords is not None and coords.shape[0] > max(i, j):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < cutoff_distance:
                    K[i, j] = h_ij * np.exp(-dist ** 2 / (sigma ** 2))
                else:
                    K[i, j] = 0.0
            else:
                K[i, j] = h_ij

    return K


def matrix_chain_optimal_order(dims: list) -> tuple:
    n = len(dims) - 1
    if n < 1:
        return 0, []
    if any(d <= 0 for d in dims):
        raise ValueError("All dimensions must be positive")


    m = np.full((n, n), np.inf, dtype=float)
    s = np.zeros((n, n), dtype=int)

    for i in range(n):
        m[i, i] = 0.0

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            for k in range(i, j):
                cost = m[i, k] + m[k + 1, j] + dims[i] * dims[k + 1] * dims[j + 1]
                if cost < m[i, j]:
                    m[i, j] = cost
                    s[i, j] = k

    return int(m[0, n - 1]), s


def read_matrix_market_string(text: str) -> dict:
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("Empty matrix market data")

    header = lines[0].strip()
    parts = header.split()
    if len(parts) < 5 or parts[0] != '%%MatrixMarket' or parts[1] != 'matrix':
        raise ValueError("Invalid Matrix Market header")

    rep = parts[2].lower()
    field = parts[3].lower()
    symm = parts[4].lower()


    idx = 1
    while idx < len(lines) and lines[idx].strip().startswith('%'):
        idx += 1

    if rep == 'coordinate':
        sizeinfo = [int(x) for x in lines[idx].strip().split()]
        idx += 1
        if len(sizeinfo) != 3:
            raise ValueError("Invalid size line for coordinate format")
        rows, cols, entries = sizeinfo

        row_idx = []
        col_idx = []
        data = []

        for e in range(entries):
            if idx >= len(lines):
                break
            vals = lines[idx].strip().split()
            idx += 1
            if len(vals) >= 3:
                row_idx.append(int(vals[0]) - 1)
                col_idx.append(int(vals[1]) - 1)
                data.append(float(vals[2]))

        A = np.zeros((rows, cols), dtype=float)
        for r, c, d in zip(row_idx, col_idx, data):
            A[r, c] = d
            if symm == 'symmetric' and r != c:
                A[c, r] = d

        return {
            'A': A,
            'rows': rows,
            'cols': cols,
            'entries': len(data),
            'rep': rep,
            'field': field,
            'symm': symm
        }
    else:
        raise NotImplementedError("Only coordinate format is supported")


def write_matrix_market_string(A: np.ndarray, title: str = "force_constant") -> str:
    rows, cols = A.shape
    lines = [f"%%MatrixMarket matrix coordinate real general"]
    lines.append(f"% {title}")

    nnz = 0
    entries = []
    for i in range(rows):
        for j in range(cols):
            if abs(A[i, j]) > 1e-14:
                entries.append((i + 1, j + 1, A[i, j]))
                nnz += 1

    lines.append(f"{rows} {cols} {nnz}")
    for i, j, v in entries:
        lines.append(f"{i} {j} {v:.8e}")

    return '\n'.join(lines)


def solve_constraint_dynamics_banded(hessian: np.ndarray,
                                      gradient: np.ndarray,
                                      constraints: np.ndarray) -> np.ndarray:
    n = hessian.shape[0]
    m = constraints.shape[0] if constraints.ndim > 1 else 1

    if constraints.ndim == 1:
        constraints = constraints.reshape(1, -1)


    aug_size = n + m
    KKT = np.zeros((aug_size, aug_size), dtype=float)
    KKT[:n, :n] = hessian
    KKT[:n, n:n + m] = constraints.T
    KKT[n:n + m, :n] = constraints

    rhs = np.zeros(aug_size, dtype=float)
    rhs[:n] = -gradient

    try:
        sol = np.linalg.solve(KKT, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]

    return sol[:n]


def condition_number_estimate(A: np.ndarray) -> float:
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]
