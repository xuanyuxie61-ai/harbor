
import numpy as np






def detq_orthogonal(a):
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    tol = 0.0001
    ifault = 0
    d = 0.0

    if n <= 0:
        ifault = 1
        return d, ifault

    a2 = a.flatten()
    d = 1.0
    r = 0

    for k in range(1, n + 1):
        q = r
        x = a2[r]
        y = np.sign(x)
        if abs(y) < 1e-15:
            y = 1.0 if x >= 0 else -1.0
        d = d * y
        denom = x + y
        if abs(denom) < 1e-15:
            denom = 1e-15 if denom >= 0 else -1e-15
        y = -1.0 / denom
        x = abs(x) - 1.0

        if tol < abs(x):
            if 0.0 < x:
                ifault = 1
                return d, ifault
            if k == n:
                ifault = 1
                return d, ifault
            for i in range(k, n):
                q = q + n
                x_val = a2[q] * y
                p = r
                s = q
                for j in range(k, n):
                    p = p + 1
                    s = s + 1
                    a2[s] = a2[s] + x_val * a2[p]
        r = r + n + 1

    return d, ifault


def check_mesh_transformation_orthogonality(Jac):
    Jac = np.asarray(Jac, dtype=float)
    n = Jac.shape[0]
    I = np.eye(n)
    JTJ = Jac.T @ Jac
    err = np.linalg.norm(JTJ - I, ord='fro')
    d, fault = detq_orthogonal(Jac)
    is_ortho = (fault == 0) and (err < 0.1)
    return is_ortho, d, err






def power_iteration_eigenvector(A, max_iter=200, tol=1e-10, damping=0.85,
                                verbose=False):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]


    col_sums = A.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    T = A / col_sums


    G = damping * T + (1.0 - damping) / n * np.ones((n, n))

    x = np.ones(n) / n
    for it in range(1, max_iter + 1):
        x_new = G @ x
        x_new = x_new / np.linalg.norm(x_new, ord=1)
        diff = np.linalg.norm(x_new - x, ord=np.inf)
        x = x_new
        if diff < tol:
            if verbose:
                print(f"[Power] Converged in {it} iterations, diff={diff:.3e}")
            eigenvalue = float((x.T @ G @ x) / (x.T @ x))
            return eigenvalue, x, it, True

    if verbose:
        print(f"[Power] Max iter reached, diff={diff:.3e}")
    eigenvalue = float((x.T @ G @ x) / (x.T @ x))
    return eigenvalue, x, max_iter, False


def steady_state_concentration_solver(K, b, alpha_relax=0.8, max_iter=500,
                                      tol=1e-8):
    K = np.asarray(K, dtype=float)
    b = np.asarray(b, dtype=float)
    n = K.shape[0]


    diag = np.diag(K)
    diag = np.where(np.abs(diag) < 1e-12, 1e-10, diag)
    D_inv = np.diag(1.0 / diag)


    M = np.eye(n) - alpha_relax * D_inv @ K
    f = alpha_relax * D_inv @ b

    c = np.zeros(n)
    for it in range(1, max_iter + 1):
        c_new = M @ c + f
        diff = np.linalg.norm(c_new - c, ord=np.inf)
        c = c_new
        if diff < tol:
            res = np.linalg.norm(K @ c - b, ord=np.inf)
            return c, res, it, True

    res = np.linalg.norm(K @ c - b, ord=np.inf)
    return c, res, max_iter, False






def estimate_condition_number(A):
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    if s[-1] < 1e-15:
        return np.inf
    return s[0] / s[-1]
