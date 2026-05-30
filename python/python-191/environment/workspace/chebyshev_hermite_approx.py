
import numpy as np
from typing import Tuple, Callable, Optional


def chebyshev_nodes(a: float, b: float, N: int) -> np.ndarray:
    if a >= b:
        raise ValueError(f"Require a < b, got a={a}, b={b}")
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    k = np.arange(N + 1)
    xi = np.cos(k * np.pi / N)
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    return x


def chebyshev_coefficients(f_vals: np.ndarray) -> np.ndarray:
    N = len(f_vals) - 1
    if N < 1:
        return f_vals.copy()
    
    k = np.arange(N + 1)
    pj = np.ones(N + 1)
    pj[0] = 2.0
    pj[N] = 2.0
    

    c = np.zeros(N + 1)
    for j in range(N + 1):
        c[j] = (2.0 / N) * np.sum(
            f_vals * np.cos(j * k * np.pi / N) / pj
        )
    return c


def chebyshev_companion_matrix(c: np.ndarray, epscutoff: float = 1e-13) -> np.ndarray:
    N = len(c) - 1
    if N < 1:
        raise ValueError("Need at least degree 1 polynomial")
    

    cmax = np.max(np.abs(c))
    Nt = N
    tailnorm = 0.0
    for k in range(N, 0, -1):
        tailnorm += np.abs(c[k])
        if tailnorm < epscutoff * cmax and Nt > 1:
            Nt -= 1
        else:
            break
    
    if Nt < 1:
        Nt = 1
    
    A = np.zeros((Nt, Nt))
    if Nt > 1:
        A[0, 1] = 1.0
        for j in range(1, Nt - 1):
            A[j, j - 1] = 0.5
            A[j, j + 1] = 0.5
        
        denom = 2.0 * c[Nt]
        if abs(denom) < 1e-300:
            denom = 1e-300
        A[Nt - 1, :Nt] = -c[:Nt] / denom
        A[Nt - 1, Nt - 2] += 0.5
    
    return A


def cpr_roots(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    N: int = 64,
    tau: float = 1e-8,
    sigma: float = 1e-6
) -> Tuple[np.ndarray, float]:
    if a >= b:
        raise ValueError("Require a < b")
    if N < 2:
        raise ValueError("N must be >= 2")
    
    t = np.arange(N + 1) * np.pi / N
    xi = np.cos(t)
    x = 0.5 * (b - a) * xi + 0.5 * (b + a)
    
    try:
        fa = f(x)
    except Exception:
        fa = np.array([f(xi_val) for xi_val in x])
    fa = np.asarray(fa, dtype=np.float64)
    

    c = chebyshev_coefficients(fa)
    

    A = chebyshev_companion_matrix(c)
    

    all_roots = np.linalg.eigvals(A)
    

    roots_list = []
    for ev in all_roots:
        if np.abs(ev.imag) < tau * max(1.0, np.abs(ev.real)):
            if np.abs(ev.real) <= 1.0 + sigma:
                root_physical = 0.5 * (b - a) * ev.real + 0.5 * (b + a)
                if a - sigma * (b - a) <= root_physical <= b + sigma * (b - a):
                    roots_list.append(root_physical)
    
    roots = np.sort(np.array(roots_list))
    

    tinter = t[:N] + 0.5 / N
    xiint = np.cos(tinter)
    xinter = 0.5 * (b - a) * xiint + 0.5 * (b + a)
    try:
        fainter = f(xinter)
    except Exception:
        fainter = np.array([f(xi_val) for xi_val in xinter])
    

    taall = np.concatenate([t, tinter])
    faall = np.concatenate([fa, fainter])
    
    fpoly = np.zeros(len(taall))
    for j in range(len(c)):
        fpoly += c[j] * np.cos(j * taall)
    
    Einter = np.max(np.abs(faall[N + 1:] - fpoly[N + 1:]))
    
    return roots, Einter


def hermite_divided_differences(
    x: np.ndarray,
    y: np.ndarray,
    yp: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    yp = np.asarray(yp).ravel()
    
    n = len(x)
    if len(y) != n or len(yp) != n:
        raise ValueError("x, y, yp must have same length")
    if n == 0:
        return np.array([]), np.array([])
    

    if len(np.unique(x)) != n:
        raise ValueError("Hermite nodes must be distinct")
    
    nd = 2 * n
    z = np.zeros(nd)
    z[0::2] = x
    z[1::2] = x
    
    d = np.zeros(nd)
    d[0] = y[0]
    d[2::2] = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    d[1::2] = yp
    






    raise NotImplementedError("Hole 2: hermite divided differences not implemented")


def hermite_evaluate(
    z: np.ndarray,
    d: np.ndarray,
    x_eval: np.ndarray
) -> np.ndarray:
    x_eval = np.asarray(x_eval)
    nd = len(d)
    if nd == 0:
        return np.zeros_like(x_eval)
    
    result = d[-1] * np.ones_like(x_eval, dtype=np.float64)
    for i in range(nd - 2, -1, -1):
        result = result * (x_eval - z[i]) + d[i]
    return result


def approximate_matrix_element(
    kernel: Callable[[float, float], float],
    xi: float,
    xj: float,
    order: int = 8,
    method: str = "chebyshev"
) -> float:
    r = abs(xi - xj)
    eps = 1e-12
    
    if method == "chebyshev":
        a = max(0.0, r - 0.5)
        b = r + 0.5
        nodes = chebyshev_nodes(a, b, order)
        vals = np.array([kernel(xi, xj + (t - r)) for t in nodes])

        c = chebyshev_coefficients(vals)

        xi_mapped = 2.0 * (r - a) / (b - a) - 1.0
        if abs(xi_mapped) > 1.0 + eps:
            xi_mapped = np.clip(xi_mapped, -1.0, 1.0)
        result = np.sum(c * np.cos(np.arange(len(c)) * np.arccos(np.clip(xi_mapped, -1.0, 1.0))))
        return result
    
    elif method == "hermite":

        h = 0.01
        nodes = np.linspace(max(0.0, r - 0.5), r + 0.5, min(order, 4))
        vals = np.array([kernel(xi, xj + (t - r)) for t in nodes])
        dvals = np.array([
            (kernel(xi, xj + (t - r) + h) - kernel(xi, xj + (t - r) - h)) / (2 * h)
            for t in nodes
        ])
        z, d = hermite_divided_differences(nodes, vals, dvals)
        return float(hermite_evaluate(z, d, np.array([r]))[0])
    
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":

    f = lambda x: np.cos(3 * x)
    roots, err = cpr_roots(f, 0.0, 2.0, N=32)
    print("CPR roots:", roots)
    print("Interstitial error:", err)
    

    x = np.array([0.0, 1.0, 2.0])
    y = np.sin(x)
    yp = np.cos(x)
    z, d = hermite_divided_differences(x, y, yp)
    print("Hermite at 0.5:", hermite_evaluate(z, d, np.array([0.5])))
