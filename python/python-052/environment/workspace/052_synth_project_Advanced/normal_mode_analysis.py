
import numpy as np

def jacobi_eigenvalue(A, tol=1e-12, max_iter=1000):
    n = A.shape[0]
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    V = np.eye(n, dtype=np.float64)
    M = A.copy().astype(np.float64)

    for sweep in range(max_iter):
        off_norm = 0.0
        for p in range(n):
            for q in range(p + 1, n):
                off_norm += M[p, q]**2
        off_norm = np.sqrt(off_norm)
        if off_norm < tol:
            break

        threshold = off_norm / (n * (n - 1))

        for p in range(n):
            for q in range(p + 1, n):
                if abs(M[p, q]) > threshold:

                    tau = (M[q, q] - M[p, p]) / (2.0 * M[p, q])
                    if tau >= 0:
                        t = 1.0 / (tau + np.sqrt(1.0 + tau**2))
                    else:
                        t = 1.0 / (tau - np.sqrt(1.0 + tau**2))
                    c = 1.0 / np.sqrt(1.0 + t**2)
                    s = t * c


                    M_pp = M[p, p]
                    M_qq = M[q, q]
                    M[p, p] = c**2 * M_pp - 2.0 * c * s * M[p, q] + s**2 * M_qq
                    M[q, q] = s**2 * M_pp + 2.0 * c * s * M[p, q] + c**2 * M_qq
                    M[p, q] = 0.0
                    M[q, p] = 0.0

                    for j in range(n):
                        if j != p and j != q:
                            M_pj = M[p, j]
                            M_qj = M[q, j]
                            M[p, j] = c * M_pj - s * M_qj
                            M[j, p] = M[p, j]
                            M[q, j] = s * M_pj + c * M_qj
                            M[j, q] = M[q, j]


                    V_p = V[:, p].copy()
                    V_q = V[:, q].copy()
                    V[:, p] = c * V_p - s * V_q
                    V[:, q] = s * V_p + c * V_q

    eigvals = np.diag(M)
    idx = np.argsort(eigvals)[::-1]
    return eigvals[idx], V[:, idx]


def qg_normal_mode_stability(Ny, Ly, U_profile, beta, Ld, k_zonal):
    dy = Ly / (Ny - 1)
    y = np.linspace(0, Ly, Ny)

    U = U_profile(y)

    U_pp = np.zeros(Ny)
    U_pp[1:-1] = (U[2:] - 2.0 * U[1:-1] + U[:-2]) / dy**2
    U_pp[0] = U_pp[1]
    U_pp[-1] = U_pp[-2]


    Qy = beta - U_pp







    A = np.zeros((Ny, Ny), dtype=np.float64)
    for j in range(Ny):
        A[j, j] = -2.0 / dy**2 - k_zonal**2 - 1.0 / (Ld**2)
        if j > 0:
            A[j, j - 1] = 1.0 / dy**2
        if j < Ny - 1:
            A[j, j + 1] = 1.0 / dy**2









    LHS = np.zeros((Ny, Ny), dtype=np.float64)
    RHS = np.zeros((Ny, Ny), dtype=np.float64)
    for j in range(Ny):
        LHS[j, :] = U[j] * A[j, :]
        LHS[j, j] += Qy[j]
        RHS[j, :] = A[j, :]



    A_std = np.linalg.solve(RHS, LHS)


    A_sym = 0.5 * (A_std + A_std.T)
    eigvals, eigvecs = jacobi_eigenvalue(A_sym, tol=1e-10, max_iter=2000)

    c = eigvals.astype(np.complex128)
    phi = eigvecs
    return c, phi


def compute_growth_rate_spectrum(Ny, Ly, U_profile, beta, Ld, k_vals):
    sigma_max = np.zeros_like(k_vals)
    for i, k in enumerate(k_vals):
        c, _ = qg_normal_mode_stability(Ny, Ly, U_profile, beta, Ld, k)
        ci = np.imag(c)
        if np.any(ci > 0):
            sigma_max[i] = k * np.max(ci)
        else:
            sigma_max[i] = 0.0
    return k_vals, sigma_max
