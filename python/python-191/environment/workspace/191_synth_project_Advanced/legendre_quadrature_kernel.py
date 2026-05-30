
import numpy as np
from typing import Tuple


def _legendre_poly_and_deriv(n: int, x: float) -> Tuple[float, float]:
    if n == 0:
        return 1.0, 0.0
    if n == 1:
        return x, 1.0
    
    pm2 = 1.0
    pm1 = x
    for k in range(1, n):
        p = ((2 * k + 1) * x * pm1 - k * pm2) / (k + 1)
        pm2 = pm1
        pm1 = p
    

    if abs(x) >= 1.0 - 1e-15:

        dp = n * (pm1 - x * pm2) / (1e-15 if abs(1 - x * x) < 1e-15 else 1 - x * x)
    else:
        dp = n * (pm2 - x * pm1) / (1 - x * x)
    
    return pm1, dp


def legendre_compute_glr(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return np.array([0.0]), np.array([2.0])
    
    x = np.zeros(n)
    w = np.zeros(n)
    

    m = n // 2
    
    for k in range(1, m + 1):

        theta = np.pi * (4 * k - 1) / (4 * n + 2)
        xk = (1.0 - (n - 1) / (8.0 * n ** 3) -
              1.0 / (384.0 * n ** 4) * (39.0 - 28.0 / (np.sin(theta) ** 2))) * np.cos(theta)
        

        for _ in range(20):
            p, dp = _legendre_poly_and_deriv(n, xk)
            if abs(dp) < 1e-300:
                break
            dx = p / dp
            xk -= dx
            if abs(dx) < 1e-15:
                break
        
        idx = k - 1
        x[idx] = -xk
        x[n - 1 - idx] = xk
        

        _, dp = _legendre_poly_and_deriv(n, xk)
        w_val = 2.0 / ((1.0 - xk * xk) * dp * dp)
        w[idx] = w_val
        w[n - 1 - idx] = w_val
    

    if n % 2 == 1:
        mid = m
        x[mid] = 0.0
        _, dp = _legendre_poly_and_deriv(n, 0.0)
        if abs(dp) < 1e-300:
            dp = 1e-300
        w[mid] = 2.0 / (dp * dp)
    

    w_sum = np.sum(w)
    if w_sum > 0:
        w *= 2.0 / w_sum
    
    return x, w


def rescale_quadrature(
    x: np.ndarray,
    w: np.ndarray,
    a: float,
    b: float
) -> Tuple[np.ndarray, np.ndarray]:
    if a >= b:
        raise ValueError("Require a < b")
    t = 0.5 * ((x + 1.0) * b - (x - 1.0) * a)
    w_scaled = 0.5 * (b - a) * w
    return t, w_scaled


def disk01_rule(nr: int, nt: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if nr < 1 or nt < 1:
        raise ValueError("nr and nt must be >= 1")
    
    xr, wr = legendre_compute_glr(nr)

    xr = (xr + 1.0) / 2.0
    wr = wr / 2.0
    
    t = 2.0 * np.pi * np.arange(nt) / nt
    

    r = np.sqrt(np.clip(xr, 0.0, 1.0))
    w = wr / nt
    
    return w, r, t


def integrate_disk_kernel(
    kernel: callable,
    nr: int = 16,
    nt: int = 32
) -> float:
    w, r, t = disk01_rule(nr, nt)
    total = 0.0
    for it in range(nt):
        cos_t = np.cos(t[it])
        sin_t = np.sin(t[it])
        for ir in range(nr):
            x = r[ir] * cos_t
            y = r[ir] * sin_t
            total += w[ir] * kernel(x, y)
    return np.pi * total


def construct_kernel_matrix_1d(
    nodes: np.ndarray,
    kernel_func: callable,
    quadrature_order: int = 16
) -> np.ndarray:
    n = len(nodes)
    if n == 0:
        return np.zeros((0, 0))
    
    xq, wq = legendre_compute_glr(quadrature_order)
    

    Phi = np.zeros((n, quadrature_order))
    for i in range(n):
        Phi[i, :] = kernel_func(xq, nodes[i])
    

    K = Phi @ np.diag(wq) @ Phi.T
    return K


def construct_kernel_matrix_2d_disk(
    nodes: np.ndarray,
    kernel_func: callable,
    nr: int = 16,
    nt: int = 32
) -> np.ndarray:
    n = nodes.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    
    w, r, t = disk01_rule(nr, nt)
    

    nq = nr * nt
    xq = np.zeros(nq)
    yq = np.zeros(nq)
    wq = np.zeros(nq)
    idx = 0
    for it in range(nt):
        for ir in range(nr):
            xq[idx] = r[ir] * np.cos(t[it])
            yq[idx] = r[ir] * np.sin(t[it])
            wq[idx] = w[ir]
            idx += 1
    
    Phi = np.zeros((n, nq))
    for i in range(n):
        Phi[i, :] = kernel_func(xq, yq, nodes[i, 0], nodes[i, 1])
    
    K = np.pi * Phi @ np.diag(wq) @ Phi.T
    return K


if __name__ == "__main__":

    x, w = legendre_compute_glr(8)

    val = np.sum(w * x ** 6)
    print("GL test (should be ~0.2857):", val)
    

    k = lambda x, y: x * x + y * y
    val = integrate_disk_kernel(k, nr=16, nt=32)
    print("Disk integral (should be ~pi/2):", val)
