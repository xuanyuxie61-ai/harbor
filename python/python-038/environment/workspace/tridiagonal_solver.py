
import numpy as np


def r83_cg_solve(A_r83, b, x0=None, tol=1e-12, max_iter=None):
    b = np.asarray(b, dtype=float)
    N = b.size
    if A_r83.shape != (3, N):
        raise ValueError(f"A_r83 shape {A_r83.shape} incompatible with b size {N}")
    
    if max_iter is None:
        max_iter = N
    
    if x0 is None:
        x = np.zeros(N, dtype=float)
    else:
        x = np.asarray(x0, dtype=float).copy()
    

    def matvec(v):
        v = np.asarray(v, dtype=float)
        out = A_r83[1, :] * v
        if N > 1:
            out[:-1] += A_r83[0, :-1] * v[1:]
            out[1:] += A_r83[2, 1:] * v[:-1]
        return out
    
    r = b - matvec(x)
    p = r.copy()
    rs_old = float(np.dot(r, r))
    rs0 = rs_old if rs_old > 0 else 1.0
    
    converged = False
    k = 0
    for k in range(max_iter):
        Ap = matvec(p)
        pAp = float(np.dot(p, Ap))
        if abs(pAp) < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = float(np.dot(r, r))
        if np.sqrt(rs_new / rs0) < tol:
            converged = True
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new
    
    info = {
        'iterations': k + 1,
        'residual': np.sqrt(rs_old / rs0) if rs0 > 0 else 0.0,
        'converged': converged
    }
    return x, info


def r83_cyclic_reduction(A_r83, b):
    b = np.asarray(b, dtype=float)
    N = b.size
    if A_r83.shape != (3, N):
        raise ValueError("Shape mismatch in r83_cyclic_reduction")
    
    if N == 0:
        return np.array([], dtype=float)
    if N == 1:
        return np.array([b[0] / A_r83[1, 0]])
    

    c = A_r83[2, 1:].copy()
    d = A_r83[1, :].copy()
    e = A_r83[0, :-1].copy()
    rhs = b.copy()
    

    for i in range(1, N):
        if abs(d[i - 1]) < 1e-30:
            raise RuntimeError("Zero pivot in cyclic reduction")
        w = c[i - 1] / d[i - 1]
        d[i] -= w * e[i - 1]
        rhs[i] -= w * rhs[i - 1]
    

    x = np.zeros(N, dtype=float)
    x[-1] = rhs[-1] / d[-1]
    for i in range(N - 2, -1, -1):
        x[i] = (rhs[i] - e[i] * x[i + 1]) / d[i]
    
    return x


def build_dif2_r83(N):
    A = np.zeros((3, N), dtype=float)
    A[1, :] = 2.0
    if N > 1:
        A[0, :-1] = -1.0
        A[2, 1:] = -1.0
    return A


def solve_diffusion_1d(u0, D, dt, dx, n_steps, solver='cyclic'):
    N = u0.size
    r = D * dt / (dx * dx)
    if r > 10.0:

        pass
    

    A = np.zeros((3, N), dtype=float)
    A[1, :] = 1.0 + 2.0 * r
    if N > 1:
        A[0, :-1] = -r
        A[2, 1:] = -r
    

    A[1, 0] = 1.0
    A[0, 0] = 0.0
    A[2, 0] = 0.0
    A[1, -1] = 1.0
    A[0, -1] = 0.0
    A[2, -1] = 0.0
    
    u = np.asarray(u0, dtype=float).copy()
    
    for _ in range(n_steps):
        rhs = u.copy()
        rhs[0] = 0.0
        rhs[-1] = 0.0
        
        if solver == 'cyclic':
            u = r83_cyclic_reduction(A, rhs)
        elif solver == 'cg':
            u, _ = r83_cg_solve(A, rhs)
        else:
            raise ValueError("solver must be 'cyclic' or 'cg'")
    
    return u


def test_tridiagonal_solvers():
    N = 100
    A = build_dif2_r83(N)
    x_exact = np.sin(np.linspace(0, np.pi, N))

    b = np.zeros(N)
    b[0] = A[1, 0] * x_exact[0] + A[0, 0] * x_exact[1]
    for i in range(1, N - 1):
        b[i] = A[2, i] * x_exact[i - 1] + A[1, i] * x_exact[i] + A[0, i] * x_exact[i + 1]
    b[-1] = A[2, -1] * x_exact[-2] + A[1, -1] * x_exact[-1]
    
    x_cg, info_cg = r83_cg_solve(A, b)
    x_cr = r83_cyclic_reduction(A, b)
    
    err_cg = np.max(np.abs(x_cg - x_exact))
    err_cr = np.max(np.abs(x_cr - x_exact))
    
    if err_cg > 1e-8 or err_cr > 1e-8:
        raise RuntimeError(f"Tridiagonal solver test failed: CG err={err_cg}, CR err={err_cr}")
    
    return {'cg_error': err_cg, 'cr_error': err_cr, 'cg_info': info_cg}


if __name__ == "__main__":
    results = test_tridiagonal_solvers()
    print("Tridiagonal solver tests passed.")
    print(f"  CG max error: {results['cg_error']:.2e}")
    print(f"  CR max error: {results['cr_error']:.2e}")
