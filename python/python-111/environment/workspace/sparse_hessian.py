
import numpy as np
from typing import Tuple, List


def hb_to_msm(hb_data_lines: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    if len(hb_data_lines) < 3:
        raise ValueError("Invalid HB data: too few lines")
    

    header = hb_data_lines[0].strip()
    line2 = hb_data_lines[1].strip().split()
    line3 = hb_data_lines[2].strip().split()
    

    nrow = int(line2[0])
    ncol = int(line2[1])
    nnzero = int(line2[2])
    


    all_numbers = []
    for line in hb_data_lines[3:]:
        parts = line.strip().split()
        for p in parts:
            try:
                all_numbers.append(float(p))
            except ValueError:
                pass
    

    A = np.zeros((nrow, ncol))
    ptr_start = 0

    if len(all_numbers) >= nnzero:


        vals = np.array(all_numbers[:nnzero])

        count = 0
        for j in range(ncol):
            for i in range(nrow):
                if count < nnzero:
                    A[i, j] = vals[count]
                    count += 1
    
    rhs = np.array(all_numbers[nnzero:nnzero + nrow]) if len(all_numbers) > nnzero else np.array([])
    return A, rhs


def r8ss_mv(n: int, na: int, diag: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    if len(diag) != n:
        raise ValueError("diag length must equal n")
    if len(a) != na:
        raise ValueError("a length must equal na")
    if len(x) != n:
        raise ValueError("x length must equal n")
    
    b = np.zeros(n)
    for j in range(n):

        bandwidth = diag[j] - (diag[j - 1] + 1) if j > 0 else diag[j]
        start_row = j - bandwidth
        
        for k in range(start_row, j + 1):
            idx = diag[j] - (j - k)
            if 0 <= idx < na:
                val = a[idx]
                b[k] += val * x[j]
                if k != j:
                    b[j] += val * x[k]
    return b


def r8ss_from_dense(dense: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    n = dense.shape[0]
    diag = np.zeros(n, dtype=int)
    a_list = []
    
    for j in range(n):

        first_nonzero = j
        for i in range(j, -1, -1):
            if abs(dense[i, j]) > 1e-14:
                first_nonzero = i
        
        col_vals = dense[first_nonzero:j + 1, j]
        diag[j] = len(a_list) + len(col_vals) - 1
        a_list.extend(col_vals.tolist())
    
    a = np.array(a_list)
    na = len(a)
    return na, diag, a


def r8ss_to_r8ge(n: int, na: int, diag: np.ndarray, a: np.ndarray) -> np.ndarray:
    dense = np.zeros((n, n))
    for j in range(n):
        bandwidth = diag[j] - (diag[j - 1] + 1) if j > 0 else diag[j]
        start_row = j - bandwidth
        for k in range(start_row, j + 1):
            idx = diag[j] - (j - k)
            if 0 <= idx < na:
                dense[k, j] = a[idx]
                dense[j, k] = a[idx]
    return dense


def build_elastic_network_matrix(coords: np.ndarray, cutoff: float = 1.5,
                                  spring_constant: float = 1.0) -> np.ndarray:
    N = coords.shape[0]
    gamma = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(coords[i] - coords[j])
            if dist < cutoff:
                gamma[i, j] = -spring_constant
                gamma[j, i] = -spring_constant
                gamma[i, i] += spring_constant
                gamma[j, j] += spring_constant
    
    return gamma


def normal_mode_analysis(gamma: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:

    gamma = 0.5 * (gamma + gamma.T)
    eigvals, eigvecs = np.linalg.eigh(gamma)
    

    nonzero_mask = np.abs(eigvals) > 1e-8
    eigvals = eigvals[nonzero_mask]
    eigvecs = eigvecs[:, nonzero_mask]
    
    if len(eigvals) < n_modes:
        n_modes = len(eigvals)
    
    return eigvals[:n_modes], eigvecs[:, :n_modes]


def compute_mean_square_fluctuation(gamma: np.ndarray, kT: float = 1.0) -> np.ndarray:

    gamma_inv = np.linalg.pinv(gamma, rcond=1e-10)
    msf = kT * np.diag(gamma_inv)
    return msf
