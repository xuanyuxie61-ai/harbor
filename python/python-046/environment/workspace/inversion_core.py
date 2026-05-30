
import numpy as np
from regularization import tikhonov_solve, build_laplacian_2d
from utils import check_finite


class NelderMeadOptimizer:

    def __init__(self, rho=1.0, chi=2.0, gamma=0.5, sigma=0.5,
                 tol=1e-6, max_iter=500):
        self.rho = rho
        self.chi = chi
        self.gamma = gamma
        self.sigma = sigma
        self.tol = tol
        self.max_iter = max_iter

    def optimize(self, objective_func, x0):
        x0 = np.asarray(x0, dtype=float)
        if x0.ndim == 1:
            n_dim = len(x0)

            simplex = np.zeros((n_dim + 1, n_dim))
            simplex[0] = x0
            for i in range(n_dim):
                simplex[i + 1] = x0.copy()
                if x0[i] != 0:
                    simplex[i + 1, i] *= 1.05
                else:
                    simplex[i + 1, i] = 0.05
        else:
            simplex = x0.copy()
            n_dim = simplex.shape[1]


        f_vals = np.array([objective_func(simplex[i]) for i in range(n_dim + 1)])
        n_eval = n_dim + 1

        for iteration in range(self.max_iter):

            order = np.argsort(f_vals)
            simplex = simplex[order]
            f_vals = f_vals[order]


            if f_vals[-1] - f_vals[0] < self.tol:
                break


            x_bar = np.mean(simplex[:-1], axis=0)


            x_r = (1.0 + self.rho) * x_bar - self.rho * simplex[-1]
            f_r = objective_func(x_r)
            n_eval += 1

            if f_vals[0] <= f_r < f_vals[-2]:

                simplex[-1] = x_r
                f_vals[-1] = f_r
            elif f_r < f_vals[0]:

                x_e = (1.0 + self.rho * self.chi) * x_bar - self.rho * self.chi * simplex[-1]
                f_e = objective_func(x_e)
                n_eval += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_vals[-2] <= f_r < f_vals[-1]:

                x_c = (1.0 + self.rho * self.gamma) * x_bar - self.rho * self.gamma * simplex[-1]
                f_c = objective_func(x_c)
                n_eval += 1
                if f_c <= f_r:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    simplex, f_vals = self._shrink(simplex, f_vals, objective_func)
                    n_eval += n_dim
            else:

                x_c = (1.0 - self.gamma) * x_bar + self.gamma * simplex[-1]
                f_c = objective_func(x_c)
                n_eval += 1
                if f_c < f_vals[-1]:
                    simplex[-1] = x_c
                    f_vals[-1] = f_c
                else:
                    simplex, f_vals = self._shrink(simplex, f_vals, objective_func)
                    n_eval += n_dim

        return simplex[0], f_vals[0], n_eval

    def _shrink(self, simplex, f_vals, objective_func):
        n_dim = simplex.shape[1]
        for i in range(1, n_dim + 1):
            simplex[i] = self.sigma * simplex[i] + (1.0 - self.sigma) * simplex[0]
            f_vals[i] = objective_func(simplex[i])
        return simplex, f_vals


class FaultSlipInversion:

    def __init__(self, G, W, d, lam, L=None):
        self.G = np.asarray(G, dtype=float)
        self.W = np.asarray(W, dtype=float)
        self.d = np.asarray(d, dtype=float)
        self.lam = lam
        self.M, self.N = self.G.shape

        if L is None:

            nx = int(np.sqrt(self.N))
            ny = nx
            if nx * ny != self.N:

                from regularization import build_laplacian_1d
                self.L = build_laplacian_1d(self.N)
            else:
                self.L = build_laplacian_2d(nx, ny)
        else:
            self.L = np.asarray(L, dtype=float)

    def linear_inversion(self):




        raise NotImplementedError("linear_inversion: 待实现线性 Tikhonov 反演调用")

    def nonlinear_l1_inversion(self, m0=None, gamma=0.01):
        if m0 is None:
            m0 = np.zeros(self.N)

        W_sqrt = np.sqrt(np.diag(self.W)) if self.W.ndim == 2 else np.sqrt(self.W)

        def objective(m):
            residual = self.G @ m - self.d
            data_fit = 0.5 * np.sum((W_sqrt * residual) ** 2)
            reg_tik = 0.5 * (self.lam ** 2) * np.sum((self.L @ m) ** 2)
            reg_l1 = gamma * np.sum(np.abs(m))
            return data_fit + reg_tik + reg_l1

        optimizer = NelderMeadOptimizer(tol=1e-5, max_iter=800)
        m_opt, f_opt, n_eval = optimizer.optimize(objective, m0)
        check_finite(m_opt, "nonlinear_l1_inversion m_opt")
        return m_opt, f_opt, n_eval

    def compute_misfit(self, m):
        residual = self.G @ m - self.d
        misfit = np.sqrt(np.mean(residual ** 2))
        return misfit

    def compute_model_norm(self, m):
        return np.linalg.norm(self.L @ m)

    def jackknife_uncertainty(self, m):
        m_jack = np.zeros((self.M, self.N))
        for i in range(self.M):

            G_i = np.delete(self.G, i, axis=0)
            d_i = np.delete(self.d, i)
            W_i = np.delete(np.delete(self.W, i, axis=0), i, axis=1)
            inv_i = FaultSlipInversion(G_i, W_i, d_i, self.lam, self.L)
            m_i, _ = inv_i.linear_inversion()
            m_jack[i] = m_i

        m_mean = np.mean(m_jack, axis=0)
        m_std = np.sqrt((self.M - 1) / self.M * np.sum((m_jack - m_mean) ** 2, axis=0))
        return m_std
