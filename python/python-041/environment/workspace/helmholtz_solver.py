
import numpy as np


def givens_rotation(v1, v2):
    if abs(v1) < 1e-15:
        cs = 0.0
        sn = 1.0
    else:
        t = np.sqrt(v1 ** 2 + v2 ** 2)
        cs = abs(v1) / t
        sn = cs * v2 / v1
    return cs, sn


def apply_givens_rotation(h, cs, sn, k):
    for i in range(k - 1):
        temp = cs[i] * h[i] + sn[i] * h[i + 1]
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1]
        h[i] = temp
    cs_k, sn_k = givens_rotation(h[k - 1], h[k])
    h[k - 1] = cs_k * h[k - 1] + sn_k * h[k]
    h[k] = 0.0
    return h, cs_k, sn_k


def arnoldi(A, Q, k):
    q = A @ Q[:, k - 1]
    h = np.zeros(k + 1)
    for i in range(k):
        h[i] = np.dot(q, Q[:, i])
        q = q - h[i] * Q[:, i]
    h[k] = np.linalg.norm(q)
    if abs(h[k]) < 1e-14:
        q = np.zeros_like(q)
    else:
        q = q / h[k]
    return h, q


def gmres_solve(A, b, x0=None, max_iterations=100, threshold=1e-6):
    n = len(b)
    if x0 is None:
        x0 = np.zeros(n)
    m = min(max_iterations, n)
    r = b - A @ x0
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-14:
        b_norm = 1.0
    error = np.linalg.norm(r) / b_norm
    sn = np.zeros(m)
    cs = np.zeros(m)
    e1 = np.zeros(n)
    e1[0] = 1.0
    errors = [error]
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-14:
        return x0.copy(), np.array(errors)
    Q = np.zeros((n, m + 1))
    Q[:, 0] = r / r_norm
    beta = r_norm * e1
    H = np.zeros((m + 1, m))
    for k in range(1, m + 1):
        h, q = arnoldi(A, Q, k)
        H[:k + 1, k - 1] = h
        Q[:, k] = q

        H[:k + 1, k - 1], cs[k - 1], sn[k - 1] = apply_givens_rotation(
            H[:k + 1, k - 1].copy(), cs, sn, k
        )
        beta[k] = -sn[k - 1] * beta[k - 1]
        beta[k - 1] = cs[k - 1] * beta[k - 1]
        error = abs(beta[k]) / b_norm
        errors.append(error)
        if error <= threshold:
            break

    k_eff = k if error <= threshold else m
    y = np.linalg.solve(H[:k_eff, :k_eff], beta[:k_eff])
    x = x0 + Q[:, :k_eff] @ y
    return x, np.array(errors)


def build_helmholtz_matrix_1d(nx, dx, c, omega, pml_width=10, pml_sigma_max=1000.0):
    c = np.asarray(c, dtype=float)
    k2 = (omega / c) ** 2

    sigma = np.zeros(nx)
    for i in range(nx):
        if i < pml_width:
            d = (pml_width - i) / pml_width
            sigma[i] = pml_sigma_max * d ** 2
        elif i >= nx - pml_width:
            d = (i - (nx - pml_width - 1)) / pml_width
            sigma[i] = pml_sigma_max * d ** 2
    s = 1.0 + 1j * sigma / omega








    pass


def solve_helmholtz_1d(nx, dx, c, omega, source_pos, source_amp=1.0,
                        max_iter=200, tol=1e-8):
    A = build_helmholtz_matrix_1d(nx, dx, c, omega)
    b = np.zeros(nx, dtype=complex)
    if 0 <= source_pos < nx:
        b[source_pos] = source_amp
    x0 = np.zeros(nx, dtype=complex)



    n = nx
    A_real = np.zeros((2 * n, 2 * n))
    A_real[:n, :n] = A.real
    A_real[:n, n:] = -A.imag
    A_real[n:, :n] = A.imag
    A_real[n:, n:] = A.real
    b_real = np.zeros(2 * n)
    b_real[:n] = b.real
    b_real[n:] = b.imag
    x_sol, residuals = gmres_solve(A_real, b_real, x0=np.zeros(2 * n),
                                    max_iterations=max_iter, threshold=tol)
    u = x_sol[:n] + 1j * x_sol[n:]
    return u, residuals
