
import numpy as np
from typing import Tuple


def r8_hypot(x: float, y: float) -> float:
    ax = abs(x)
    ay = abs(y)
    if ax < ay:
        ax, ay = ay, ax
    if ax == 0.0:
        return 0.0
    t = ay / ax
    return ax * np.sqrt(1.0 + t * t)


def band_solve(a_band: np.ndarray,
               ml: int,
               mu: int,
               b: np.ndarray) -> np.ndarray:
    if a_band.ndim != 2 or b.ndim != 1:
        raise ValueError("Invalid dimensions")
    n = b.shape[0]
    rows = a_band.shape[0]
    if rows != 2 * ml + mu + 1:
        raise ValueError(f"Expected {2*ml+mu+1} rows, got {rows}")



    A_full = np.zeros((n, n), dtype=np.float64)
    for j in range(n):
        i1 = max(0, j - mu)
        i2 = min(n - 1, j + ml)
        for i in range(i1, i2 + 1):
            k = i - j + ml + mu
            A_full[i, j] = a_band[k, j]


    cond_est = np.linalg.cond(A_full)
    if cond_est > 1e14:

        x = np.linalg.lstsq(A_full, b, rcond=1e-14)[0]
    else:
        x = np.linalg.solve(A_full, b)
    return x


def assemble_band_storage(A_full: np.ndarray,
                          ml: int,
                          mu: int) -> np.ndarray:
    n = A_full.shape[0]
    a_band = np.zeros((2 * ml + mu + 1, n), dtype=np.float64)
    for j in range(n):
        i1 = max(0, j - mu)
        i2 = min(n - 1, j + ml)
        for i in range(i1, i2 + 1):
            k = i - j + ml + mu
            a_band[k, j] = A_full[i, j]
    return a_band


def svsort(n: int, d: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(d) < n or v.shape[0] < n:
        raise ValueError("Dimensions mismatch in svsort")
    idx = np.argsort(-d[:n])
    d_out = d[:n].copy()
    v_out = v[:n, :n].copy()
    d_out = d_out[idx]
    v_out = v_out[:, idx]
    return d_out, v_out


def apply_dirichlet_bc(A: np.ndarray,
                       b: np.ndarray,
                       bc_nodes: np.ndarray,
                       bc_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    A = A.copy()
    b = b.copy()
    for idx, val in zip(bc_nodes, bc_values):
        i = int(idx)
        A[i, :] = 0.0
        A[:, i] = 0.0
        A[i, i] = 1.0
        b[i] = val
    return A, b


def apply_neumann_bc_rhs(b: np.ndarray,
                         dx: float,
                         neumann_nodes: np.ndarray,
                         flux_values: np.ndarray) -> np.ndarray:
    b = b.copy()
    for idx, flux in zip(neumann_nodes, flux_values):
        i = int(idx)
        b[i] += flux * dx
    return b
