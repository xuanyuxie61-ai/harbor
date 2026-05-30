
import numpy as np
import stochastic_dynamics as sd
import polynomial_chaos_uq as pc
import special_functions as sf
import time_stepping as ts


class HJMMultiFactorModel:

    def __init__(self, n_factors=3, sigma0=0.02, kappa=None,
                 lorenz_n=8, lorenz_force=8.0,
                 duffing_alpha=1.0, duffing_beta=5.0,
                 duffing_gamma=8.0, duffing_delta=0.02, duffing_omega=0.5,
                 oregonator_f=1.0,
                 pc_degree=3, pc_dim=3):
        if sigma0 < 0.0:
            raise ValueError("HJMMultiFactorModel: sigma0 必须非负")
        if n_factors < 1:
            raise ValueError("HJMMultiFactorModel: n_factors 必须至少为 1")

        self.n_factors = n_factors
        self.sigma0 = sigma0
        self.kappa = kappa if kappa is not None else np.array([0.1, 0.3, 0.5])
        self.kappa = np.asarray(self.kappa, dtype=float)
        if self.kappa.shape[0] < n_factors:
            self.kappa = np.pad(self.kappa, (0, n_factors - self.kappa.shape[0]),
                                constant_values=0.5)


        self.lorenz_n = lorenz_n
        self.lorenz_force = lorenz_force
        _, _, _, _, self.lorenz_y0, _ = sd.lorenz96_parameters(
            n=lorenz_n, force=lorenz_force)


        self.duffing_alpha = duffing_alpha
        self.duffing_beta = duffing_beta
        self.duffing_gamma = duffing_gamma
        self.duffing_delta = duffing_delta
        self.duffing_omega = duffing_omega
        _, _, _, _, _, _, self.duffing_y0, _ = sd.duffing_parameters(
            alpha=duffing_alpha, beta=duffing_beta, gamma=duffing_gamma,
            delta=duffing_delta, omega=duffing_omega)


        self.oregonator_f = oregonator_f
        _, _, _, _, _, self.oregonator_y0, _ = sd.oregonator_parameters(f=oregonator_f)


        self.pc_degree = pc_degree
        self.pc_dim = pc_dim
        self.multi_indices = pc.generate_multi_indices(pc_dim, pc_degree)

    def volatility_structure(self, t, s, lorenz_state, duffing_state, oregonator_state):
        if s < 0.0:
            raise ValueError("volatility_structure: s 必须非负")

        sigma = np.zeros(self.n_factors, dtype=float)


        sigma[0] = self.sigma0 * np.exp(-self.kappa[0] * s)
        if self.n_factors > 1:
            sigma[1] = self.sigma0 * s * np.exp(-self.kappa[1] * s)


        if self.n_factors > 2:
            sigma_chaos = sd.multi_factor_coupling(
                t, lorenz_state, duffing_state, oregonator_state,
                n_factors=1, coupling_matrix=np.array([[0.0, 0.0, 1.0]]))

            sigma[2] = sigma_chaos[0] * self.sigma0 * np.exp(-self.kappa[2] * s)


        sigma = np.clip(sigma, 0.0, 1.0)
        return sigma

    def drift_term(self, t, s, lorenz_state, duffing_state, oregonator_state):
        if s <= 0.0:
            return 0.0

        import term_structure_pde as tsp
        sigma_funcs = []
        for i in range(self.n_factors):
            def make_sigma(i_factor):
                def sigma_i(t_, s_):
                    return self.volatility_structure(t_, s_, lorenz_state,
                                                      duffing_state, oregonator_state)[i_factor]
                return sigma_i
            sigma_funcs.append(make_sigma(i))

        return tsp.musiela_drift(sigma_funcs, s, t)

    def evolve_stochastic_dynamics(self, t, dt, lorenz_y, duffing_y, oregonator_y):

        def lorenz_rhs(tt, yy):
            return sd.lorenz96_deriv(tt, yy, force=self.lorenz_force)
        lorenz_new = ts.rk3_step(lorenz_rhs, t, lorenz_y, dt)


        def duffing_rhs(tt, yy):
            return sd.duffing_deriv(tt, yy,
                                    alpha=self.duffing_alpha,
                                    beta=self.duffing_beta,
                                    gamma=self.duffing_gamma,
                                    delta=self.duffing_delta,
                                    omega=self.duffing_omega)
        duffing_new = ts.rk3_step(duffing_rhs, t, duffing_y, dt)


        eta1, eta2, q, f, _, _, _ = sd.oregonator_parameters(f=self.oregonator_f)

        def oregonator_rhs(tt, yy):
            return sd.oregonator_deriv(tt, yy, eta1, eta2, q, f)

        oregonator_new = oregonator_y.copy()
        n_sub = max(1, int(np.ceil(dt / 0.01)))
        dt_sub = dt / n_sub
        for _ in range(n_sub):
            oregonator_new = ts.rk3_step(oregonator_rhs, t, oregonator_new, dt_sub)

            if np.any(np.isnan(oregonator_new)) or np.any(np.isinf(oregonator_new)):
                oregonator_new = np.array([1.0, 1.0, 1.0], dtype=float)
                break
            oregonator_new = np.clip(oregonator_new, -50.0, 50.0)


        lorenz_new = np.nan_to_num(lorenz_new, nan=0.0, posinf=50.0, neginf=-50.0)
        duffing_new = np.nan_to_num(duffing_new, nan=0.0, posinf=50.0, neginf=-50.0)
        oregonator_new = np.nan_to_num(oregonator_new, nan=1.0, posinf=50.0, neginf=-50.0)

        return lorenz_new, duffing_new, oregonator_new

    def simulate_path(self, T_grid, f_init, t_max, dt,
                      nu=0.001, mu_func=None, forcing_func=None):
        import term_structure_pde as tsp
        from scipy import sparse as sp
        from scipy.sparse.linalg import spsolve

        T_grid = np.asarray(T_grid, dtype=float)
        f_init = np.asarray(f_init, dtype=float)
        N = len(T_grid)
        n_steps = int(np.ceil(t_max / dt))
        dt = t_max / n_steps

        if mu_func is None:
            def mu_func_default(T):
                return -0.01
            mu_func = mu_func_default

        if forcing_func is None:
            def forcing_default(t, T):
                return 0.0
            forcing_func = forcing_default

        t_history = np.zeros(n_steps + 1, dtype=float)
        f_history = np.zeros((n_steps + 1, N), dtype=float)
        dynamics_history = []

        f = f_init.copy()
        lorenz_y = self.lorenz_y0.copy()
        duffing_y = self.duffing_y0.copy()
        oregonator_y = self.oregonator_y0.copy()

        t_history[0] = 0.0
        f_history[0, :] = f
        dynamics_history.append((lorenz_y.copy(), duffing_y.copy(), oregonator_y.copy()))

        for step in range(n_steps):
            t = step * dt
            t_new = (step + 1) * dt


            lorenz_y, duffing_y, oregonator_y = self.evolve_stochastic_dynamics(
                t, dt, lorenz_y, duffing_y, oregonator_y)
            dynamics_history.append((lorenz_y.copy(), duffing_y.copy(), oregonator_y.copy()))


            def sigma_factory_update(i):
                def sigma_i(t_, s_):
                    return self.volatility_structure(t_, s_, lorenz_y, duffing_y, oregonator_y)[i]
                return sigma_i

            sigma_funcs = [sigma_factory_update(i) for i in range(self.n_factors)]


            _, A_fd, rhs_f = tsp.forward_rate_pde_rhs(
                t, T_grid, f, nu, mu_func, forcing_func, sigma_funcs)










            raise NotImplementedError("HOLE_2: 隐式时间推进格式尚未实现")

            t_history[step + 1] = t_new
            f_history[step + 1, :] = f

        return t_history, f_history, dynamics_history
