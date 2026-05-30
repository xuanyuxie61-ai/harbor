
import numpy as np
from typing import Tuple, Callable
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


def solve_helmholtz_1d(n: int, k: float, f: np.ndarray,
                       xlim: Tuple[float, float] = (0.0, 1.0),
                       bc_left: str = 'dirichlet',
                       bc_right: str = 'abc') -> Tuple[np.ndarray, np.ndarray]:
    a, b = xlim
    h = (b - a) / (n + 1)
    x = np.linspace(a, b, n + 2)
    










    
    p = np.zeros(n + 2, dtype=complex)
    return x, p


def solve_helmholtz_2d_dirichlet(nx: int, ny: int, k: float,
                                 f: np.ndarray,
                                 xlim: Tuple[float, float] = (0.0, 1.0),
                                 ylim: Tuple[float, float] = (0.0, 1.0)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a, b = xlim
    c, d = ylim
    hx = (b - a) / (nx + 1)
    hy = (d - c) / (ny + 1)
    
    x = np.linspace(a, b, nx + 2)
    y = np.linspace(c, d, ny + 2)
    
    N = nx * ny
    


    main_val = -2.0 / hx**2 - 2.0 / hy**2 + k**2
    off_x = 1.0 / hx**2
    off_y = 1.0 / hy**2
    

    main_diag = np.full(N, main_val)
    off_x_diag = np.full(N - 1, off_x)
    off_y_diag = np.full(N - nx, off_y)
    

    for j in range(1, ny):
        off_x_diag[j * nx - 1] = 0.0
    
    A = diags([off_y_diag, off_x_diag, main_diag, off_x_diag, off_y_diag],
              [-nx, -1, 0, 1, nx], format='csr')
    

    rhs = -f.flatten()
    

    p_flat = spsolve(A, rhs)
    p = p_flat.reshape(ny, nx)
    
    return x, y, p


def pwl_interp_1d(xd: np.ndarray, yd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    if len(xd) != len(yd):
        raise ValueError("xd和yd长度必须相同")
    
    if len(xd) < 2:
        raise ValueError("至少需要2个数据点")
    

    if not np.all(np.diff(xd) > 0):

        sort_idx = np.argsort(xd)
        xd = xd[sort_idx]
        yd = yd[sort_idx]
    
    n = len(xd)
    yi = np.zeros(len(xi))
    
    for i, x in enumerate(xi):

        if x <= xd[0]:
            k = 0
        elif x >= xd[n - 2]:
            k = n - 2
        else:

            k = np.searchsorted(xd, x) - 1
            k = max(0, min(k, n - 2))
        

        dx = xd[k + 1] - xd[k]
        if abs(dx) < 1e-14:
            yi[i] = yd[k]
        else:
            t = (x - xd[k]) / dx
            t = max(0.0, min(1.0, t))
            yi[i] = (1.0 - t) * yd[k] + t * yd[k + 1]
    
    return yi


def pwl_basis_1d(xd: np.ndarray, xi: np.ndarray) -> np.ndarray:
    if not np.all(np.diff(xd) > 0):
        sort_idx = np.argsort(xd)
        xd = xd[sort_idx]
    
    n_basis = len(xd)
    n_eval = len(xi)
    B = np.zeros((n_eval, n_basis))
    
    for i, x in enumerate(xi):

        if x <= xd[0]:
            B[i, 0] = 1.0
        elif x >= xd[-1]:
            B[i, -1] = 1.0
        else:
            k = np.searchsorted(xd, x) - 1
            k = max(0, min(k, n_basis - 2))
            
            dx = xd[k + 1] - xd[k]
            if abs(dx) < 1e-14:
                B[i, k] = 1.0
            else:
                t = (x - xd[k]) / dx
                B[i, k] = 1.0 - t
                B[i, k + 1] = t
    
    return B


def interpolate_acoustic_pressure(xd: np.ndarray, pd: np.ndarray,
                                  xi: np.ndarray) -> np.ndarray:




    
    return np.zeros_like(xi, dtype=complex)
