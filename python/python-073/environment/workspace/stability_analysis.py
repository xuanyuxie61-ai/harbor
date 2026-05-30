# -*- coding: utf-8 -*-

import numpy as np
from utils import chebyshev_diff_matrix, chebyshev_nodes, safe_divide


class CompressibleLST:

    def __init__(self, Ma=6.0, Re=1e6, Pr=0.72, gamma=1.4, N=120):
        self.Ma = Ma
        self.Re = Re
        self.Pr = Pr
        self.gamma = gamma
        self.N = N


        self.eta_cheb = chebyshev_nodes(N, a=0.0, b=12.0)
        self.D = chebyshev_diff_matrix(N, a=0.0, b=12.0)
        self.D2 = self.D @ self.D

    def set_baseflow(self, eta, u, T, mu):
        self.u_base = np.interp(self.eta_cheb, eta, u)
        self.T_base = np.interp(self.eta_cheb, eta, T)
        self.mu_base = np.interp(self.eta_cheb, eta, mu)


        self.du_deta = self.D @ self.u_base
        self.dT_deta = self.D @ self.T_base
        self.dmu_deta = self.D @ self.mu_base


        self.rho_base = safe_divide(1.0, self.T_base, fill_value=1.0)

    def build_stability_operator(self, alpha, beta=0.0):
        N = self.N + 1
        n_eq = 4 * N


        I = np.eye(N)
        D1 = self.D
        D2 = self.D2


        U = np.diag(self.u_base)
        T = np.diag(self.T_base)
        Mu = np.diag(self.mu_base)
        Rho = np.diag(self.rho_base)
        dU = np.diag(self.du_deta)
        dT = np.diag(self.dT_deta)
        dMu = np.diag(self.dmu_deta)


        k2 = alpha**2 + beta**2


        A = np.zeros((n_eq, n_eq), dtype=complex)






        row_u = slice(0, N)
        row_v = slice(N, 2 * N)
        row_T = slice(2 * N, 3 * N)
        row_p = slice(3 * N, 4 * N)

        col_u = slice(0, N)
        col_v = slice(N, 2 * N)
        col_T = slice(2 * N, 3 * N)
        col_p = slice(3 * N, 4 * N)


        A[row_u, col_u] = 1j * alpha * Rho @ U - Mu @ (D2 - k2 * I) - dMu @ D1
        A[row_u, col_v] = Rho @ dU - 1j * alpha * dMu
        A[row_u, col_p] = 1j * alpha * I











        raise NotImplementedError("build_stability_operator: 请完成动量-y、能量与连续性方程的矩阵块组装")




        bc_rows = []


        bc_rows.append((0, col_u, np.zeros(N), 1.0))
        bc_rows.append((N, col_v, np.zeros(N), 1.0))
        bc_rows.append((2 * N, col_T, np.zeros(N), 1.0))



        bc_rows.append((N - 1, col_u, np.zeros(N), 1.0))
        bc_rows.append((2 * N - 1, col_v, np.zeros(N), 1.0))
        bc_rows.append((3 * N - 1, col_T, np.zeros(N), 1.0))
        bc_rows.append((4 * N - 1, col_p, np.zeros(N), 1.0))

        B = np.eye(n_eq, dtype=complex)

        B[row_u, col_u] = 1j * alpha * Rho
        B[row_v, col_v] = 1j * alpha * Rho
        B[row_T, col_T] = 1j * alpha * Rho
        B[row_p, col_p] = 1j * alpha * Rho @ U / self.gamma


        for r, c, vec, diag_val in bc_rows:
            A[r, :] = 0.0
            A[r, c] = vec
            A[r, r] = diag_val
            B[r, :] = 0.0
            B[r, r] = 1.0


        r = 3 * N
        if r < n_eq:
            A[r, :] = 0.0
            A[r, row_p] = D1[0, :]
            B[r, :] = 0.0

        return A, B

    def temporal_eigenvalues(self, alpha, beta=0.0):
        A, B = self.build_stability_operator(alpha, beta)
        try:
            eigvals, eigvecs = np.linalg.eig(np.linalg.solve(B, A))
        except np.linalg.LinAlgError:

            B_reg = B + 1e-12 * np.eye(B.shape[0], dtype=complex)
            eigvals = np.linalg.eigvals(np.linalg.solve(B_reg, A))



        idx = np.argsort(-np.imag(eigvals))
        return eigvals[idx]

    def spatial_eigenvalues(self, omega_real, beta=0.0, alpha_guess=0.5):

        alphas = []
        for guess in [alpha_guess, alpha_guess * 1j, -alpha_guess, 0.1 + 0.1j]:
            alpha = self._newton_spatial(guess, omega_real, beta)
            if alpha is not None:
                alphas.append(alpha)
        return alphas

    def _newton_spatial(self, alpha0, omega_r, beta, max_iter=30, tol=1e-8):
        alpha = complex(alpha0)
        for _ in range(max_iter):
            A, B = self.build_stability_operator(alpha.real, beta)

            try:
                ev = np.linalg.eigvals(A, B)
            except Exception:
                return None
            distances = np.abs(ev - omega_r)
            k = np.argmin(distances)
            residual = ev[k] - omega_r
            if abs(residual) < tol:
                return alpha


            h = 1e-6
            Aph, Bph = self.build_stability_operator(alpha.real + h, beta)
            try:
                evp = np.linalg.eigvals(Aph, Bph)
            except Exception:
                return None
            dk_dar = (evp[np.argmin(np.abs(evp - omega_r))] - ev[np.argmin(distances)]) / h

            if abs(dk_dar) < 1e-12:
                break
            alpha = alpha - residual / dk_dar
        return None if abs(residual) >= tol else alpha

    def jordan_analysis(self, alpha, beta=0.0):
        A, B = self.build_stability_operator(alpha, beta)
        M = np.linalg.solve(B + 1e-12 * np.eye(B.shape[0]), A)


        eigvals, eigvecs = np.linalg.eig(M)


        cond_num = np.linalg.cond(eigvecs)


        tol = 1e-4
        clusters = []
        used = set()
        for i in range(len(eigvals)):
            if i in used:
                continue
            cluster = [i]
            used.add(i)
            for j in range(i + 1, len(eigvals)):
                if j not in used and abs(eigvals[i] - eigvals[j]) < tol:
                    cluster.append(j)
                    used.add(j)
            clusters.append(cluster)


        max_block_size = max(len(c) for c in clusters) if clusters else 1


        omega_max = np.max(np.imag(eigvals))

        return {
            'eigenvalues': eigvals,
            'condition_number': cond_num,
            'clusters': clusters,
            'max_jordan_block': max_block_size,
            'transient_growth_bound': cond_num,
            'max_temporal_growth_rate': omega_max
        }


def track_eigenvalue_mode(alpha_list, lst_solver, beta=0.0):
    tracked = []
    prev_omega = None

    for alpha in alpha_list:
        omegas = lst_solver.temporal_eigenvalues(alpha, beta)
        if len(omegas) == 0:
            tracked.append(np.nan)
            continue

        if prev_omega is None:

            choice = omegas[0]
        else:

            distances = np.abs(omegas - prev_omega)
            choice = omegas[np.argmin(distances)]

        tracked.append(choice)
        prev_omega = choice

    return tracked
