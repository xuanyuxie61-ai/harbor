
import numpy as np
from typing import Tuple, Callable


def assemble_fem_tridiagonal(
    x_nodes: np.ndarray,
    D_func: Callable[[np.ndarray], np.ndarray],
    dV_func: Callable[[np.ndarray], np.ndarray],
    T: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(x_nodes) - 1
    if n < 1:
        raise ValueError("need at least one element")
    
    nu = n - 1
    adiag = np.zeros(nu)
    aleft = np.zeros(nu)
    arite = np.zeros(nu)
    rhs_base = np.zeros(nu)
    
    for e in range(n):
        xL = x_nodes[e]
        xR = x_nodes[e + 1]
        h = xR - xL
        if h <= 0:
            continue
        
        x_mid = 0.5 * (xL + xR)
        D_mid = D_func(np.array([x_mid]))[0]
        dV_mid = dV_func(np.array([x_mid]))[0]
        

        k_local = (D_mid / h) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        


        peclet = abs(dV_mid * h / (T * D_mid + 1e-12))
        alpha_upwind = 0.5 * np.tanh(peclet / 2.0)
        b_local = (dV_mid / (2.0 * T)) * np.array([[-1.0, 1.0], [-1.0, 1.0]])
        b_local += alpha_upwind * abs(dV_mid / T) * np.array([[1.0, -1.0], [-1.0, 1.0]])
        
        a_local = k_local + b_local
        

        for i_local in range(2):
            i_global = e + i_local - 1
            if 0 <= i_global < nu:
                for j_local in range(2):
                    j_global = e + j_local - 1
                    if 0 <= j_global < nu:
                        val = a_local[i_local, j_local]
                        if i_global == j_global:
                            adiag[i_global] += val
                        elif j_global < i_global:
                            aleft[i_global] += val
                        else:
                            arite[i_global] += val
    
    return adiag, aleft, arite, rhs_base


def solve_tridiagonal(adiag: np.ndarray, aleft: np.ndarray, arite: np.ndarray,
                      f: np.ndarray) -> np.ndarray:
    nu = len(adiag)
    if nu == 0:
        return np.array([])
    if len(aleft) != nu or len(arite) != nu or len(f) != nu:
        raise ValueError("array length mismatch")
    
    ad = adiag.copy()
    al = aleft.copy()
    ar = arite.copy()
    
    ar[0] = ar[0] / ad[0]
    for i in range(1, nu - 1):
        ad[i] = ad[i] - al[i] * ar[i - 1]
        if abs(ad[i]) < 1e-15:
            ad[i] = 1e-15
        ar[i] = ar[i] / ad[i]
    if nu > 1:
        ad[nu - 1] = ad[nu - 1] - al[nu - 1] * ar[nu - 2]
        if abs(ad[nu - 1]) < 1e-15:
            ad[nu - 1] = 1e-15
    
    u = np.zeros(nu)
    u[0] = f[0] / ad[0]
    for i in range(1, nu):
        u[i] = (f[i] - al[i] * u[i - 1]) / ad[i]
    
    for i in range(nu - 2, -1, -1):
        u[i] = u[i] - ar[i] * u[i + 1]
    
    return u


def fokker_planck_steady_state(
    x_nodes: np.ndarray,
    V_func: Callable[[np.ndarray], np.ndarray],
    T: float,
    D_const: float = 1.0,
) -> np.ndarray:
    n = len(x_nodes)
    V_vals = V_func(x_nodes)

    P_unnorm = np.exp(-V_vals / (T + 1e-12))

    P_unnorm = np.clip(P_unnorm, 1e-300, 1e300)
    Z = np.trapezoid(P_unnorm, x_nodes)
    if Z <= 0 or not np.isfinite(Z):
        Z = 1.0
    P = P_unnorm / Z
    return P


def fokker_planck_time_stepping(
    x_nodes: np.ndarray,
    P0: np.ndarray,
    D_func: Callable[[np.ndarray], np.ndarray],
    dV_func: Callable[[np.ndarray], np.ndarray],
    T: float,
    dt: float,
    n_steps: int,
) -> np.ndarray:
    n = len(x_nodes)
    if len(P0) != n:
        raise ValueError("P0 length mismatch")
    
    adiag, aleft, arite, _ = assemble_fem_tridiagonal(x_nodes, D_func, dV_func, T)
    nu = len(adiag)
    

    dx = np.diff(x_nodes)
    m_diag = 0.5 * (dx[:-1] + dx[1:]) if n > 2 else np.array([1.0])
    if nu == 1 and len(m_diag) == 0:
        m_diag = np.array([1.0])
    
    P = P0.copy()

    P[0] = 0.0
    P[-1] = 0.0
    
    for _ in range(n_steps):

        f = m_diag * P[1:-1]

        Ad = m_diag + dt * adiag
        Al = dt * aleft
        Ar = dt * arite
        
        P_inner = solve_tridiagonal(Ad, Al, Ar, f)
        P[1:-1] = P_inner

        total = np.trapezoid(P, x_nodes)
        if total > 0:
            P = P / total
    
    return P


def sparse_matrix_vector_product(col_ptr: np.ndarray, row_ind: np.ndarray,
                                  values: np.ndarray, vec: np.ndarray) -> np.ndarray:
    ncol = len(col_ptr) - 1
    nrow = len(vec)
    out = np.zeros(nrow)
    for col in range(ncol):
        for k in range(col_ptr[col], col_ptr[col + 1]):
            row = row_ind[k]
            if 0 <= row < nrow and 0 <= col < len(vec):
                out[row] += values[k] * vec[col]
    return out
