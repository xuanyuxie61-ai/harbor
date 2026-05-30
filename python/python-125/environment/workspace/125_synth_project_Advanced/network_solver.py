
import numpy as np
from typing import Tuple






def band_lu_factorize(
    A_band: np.ndarray,
    n: int,
    ml: int,
    mu: int
) -> Tuple[np.ndarray, int]:
    info = 0
    nrow = ml + mu + 1
    
    for k in range(n):

        pivot = A_band[mu, k]
        if abs(pivot) < 1e-14:
            info = k + 1
            pivot = 1e-14 if pivot >= 0 else -1e-14
            A_band[mu, k] = pivot
        


        for i in range(k + 1, min(k + ml + 1, n)):

            row_ik = mu + (i - k)
            if row_ik >= nrow:
                continue
            
            factor = A_band[row_ik, k] / pivot
            A_band[row_ik, k] = factor
            


            for j in range(k + 1, min(k + mu + 1, n)):

                row_ij = mu + (i - j)

                row_kj = mu + (k - j)
                
                if row_ij >= 0 and row_ij < nrow and row_kj >= 0 and row_kj < nrow:
                    A_band[row_ij, j] -= factor * A_band[row_kj, j]
    
    return A_band, info


def band_lu_solve(
    A_band_lu: np.ndarray,
    b: np.ndarray,
    n: int,
    ml: int,
    mu: int
) -> np.ndarray:
    x = b.copy().astype(np.float64)
    nrow = A_band_lu.shape[0]
    


    for i in range(1, n):
        for j in range(max(0, i - ml), i):
            row_ij = mu + (i - j)
            if row_ij < nrow:
                x[i] -= A_band_lu[row_ij, j] * x[j]
    


    for i in range(n - 1, -1, -1):
        pivot = A_band_lu[mu, i]
        if abs(pivot) < 1e-14:
            pivot = 1e-14
        
        for j in range(i + 1, min(i + mu + 1, n)):
            row_ij = mu + (i - j)
            if row_ij >= 0 and row_ij < nrow:
                x[i] -= A_band_lu[row_ij, j] * x[j]
        
        x[i] /= pivot
    
    return x






def dense_lu_factorize(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    n = A.shape[0]
    L = np.eye(n, dtype=np.float64)
    U = A.copy()
    info = 0
    
    for k in range(n):
        if abs(U[k, k]) < 1e-14:
            info = k + 1
            U[k, k] = 1e-14 if U[k, k] >= 0 else -1e-14
        
        for i in range(k + 1, n):
            factor = U[i, k] / U[k, k]
            L[i, k] = factor
            U[i, k:] -= factor * U[k, k:]
    
    return L, U, info


def dense_lu_solve(L: np.ndarray, U: np.ndarray, b: np.ndarray) -> np.ndarray:
    n = len(b)
    y = np.zeros(n, dtype=np.float64)
    x = np.zeros(n, dtype=np.float64)
    

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])
    

    for i in range(n - 1, -1, -1):
        denom = U[i, i]
        if abs(denom) < 1e-14:
            denom = 1e-14
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / denom
    
    return x






def solve_cbb_system(
    A1_band: np.ndarray,
    A2: np.ndarray,
    A3: np.ndarray,
    A4: np.ndarray,
    b: np.ndarray,
    n1: int,
    n2: int,
    ml: int,
    mu: int
) -> np.ndarray:
    b1 = b[:n1].copy()
    b2 = b[n1:].copy()
    

    A1_lu, info1 = band_lu_factorize(A1_band.copy(), n1, ml, mu)
    if info1 != 0:
        print(f"  Warning: A1 band LU factorization info={info1}")
    

    Y = np.zeros((n1, n2), dtype=np.float64)
    for j in range(n2):
        Y[:, j] = band_lu_solve(A1_lu, A2[:, j].copy(), n1, ml, mu)
    

    S = A4 - A3 @ Y
    

    L_s, U_s, info_s = dense_lu_factorize(S)
    if info_s != 0:
        print(f"  Warning: Schur complement LU factorization info={info_s}")
    

    A1_inv_b1 = band_lu_solve(A1_lu, b1.copy(), n1, ml, mu)
    

    rhs2 = b2 - A3 @ A1_inv_b1
    x2 = dense_lu_solve(L_s, U_s, rhs2)
    

    rhs1 = b1 - A2 @ x2
    x1 = band_lu_solve(A1_lu, rhs1, n1, ml, mu)
    
    return np.concatenate([x1, x2])






def band_to_dense(A_band: np.ndarray, n: int, ml: int, mu: int) -> np.ndarray:
    A_dense = np.zeros((n, n), dtype=np.float64)
    nrow = A_band.shape[0]
    
    for j in range(n):
        for i in range(max(0, j - mu), min(n, j + ml + 1)):
            row = mu + (i - j)
            if 0 <= row < nrow:
                A_dense[i, j] = A_band[row, j]
    
    return A_dense






def build_retinal_network_matrix(
    n_local: int,
    n_long_range: int,
    connectivity_radius: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ml = int(connectivity_radius)
    mu = int(connectivity_radius)
    nrow = ml + mu + 1
    A1_band = np.zeros((nrow, n_local), dtype=np.float64)
    
    for j in range(n_local):
        for i in range(max(0, j - ml), min(n_local, j + mu + 1)):
            dist = abs(i - j)
            if dist == 0:
                val = 2.0
            else:
                val = -0.3 * np.exp(-dist ** 2 / (2.0 * connectivity_radius ** 2))
            
            row = mu + (i - j)
            if 0 <= row < nrow:
                A1_band[row, j] = val
    

    np.random.seed(123)
    A2 = np.random.random((n_local, n_long_range)) * 0.1
    

    A3 = np.random.random((n_long_range, n_local)) * 0.05
    

    A4 = np.eye(n_long_range, dtype=np.float64) * 2.0
    for i in range(n_long_range):
        for j in range(n_long_range):
            if i != j:
                A4[i, j] = -0.05 * np.random.random()
    
    return A1_band, A2, A3, A4
