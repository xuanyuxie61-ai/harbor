
import numpy as np
from typing import Callable, Tuple, Optional


class ReactionDiffusion1D:

    def __init__(self, x_grid: np.ndarray, diffusivity: float,
                 bc_type: str = "dirichlet"):
        self.x = np.asarray(x_grid, dtype=float)
        self.n = len(self.x)
        self.D = diffusivity
        self.bc_type = bc_type

    def solve_steady_state(self, reaction_func: Callable[[np.ndarray], np.ndarray],
                           bc_values: Tuple[float, float]) -> np.ndarray:
        if self.bc_type == "dirichlet":
            return self._solve_steady_dirichlet(reaction_func, bc_values)
        elif self.bc_type == "neumann":
            return self._solve_steady_neumann(reaction_func, bc_values)
        else:
            raise ValueError(f"不支持的边界条件类型: {self.bc_type}")

    def _solve_steady_dirichlet(self, reaction_func, bc_values):
        c = np.linspace(bc_values[0], bc_values[1], self.n)
        tol = 1e-10
        max_iter = 100

        for _ in range(max_iter):

            A = np.zeros((self.n, self.n))
            rhs = np.zeros(self.n)

            A[0, 0] = 1.0
            rhs[0] = bc_values[0]
            A[self.n - 1, self.n - 1] = 1.0
            rhs[self.n - 1] = bc_values[1]

            for i in range(1, self.n - 1):
                dx_l = self.x[i] - self.x[i - 1]
                dx_r = self.x[i + 1] - self.x[i]
                dx = self.x[i + 1] - self.x[i - 1]


                A[i, i - 1] = -2.0 * self.D / (dx_l * dx)
                A[i, i] = 2.0 * self.D / (dx_l * dx_r)
                A[i, i + 1] = -2.0 * self.D / (dx_r * dx)


                r_val = reaction_func(c[i])


                rhs[i] = -r_val

                dc = 1e-8
                dr_dc = (reaction_func(c[i] + dc) - r_val) / dc
                A[i, i] += dr_dc

            delta_c = np.linalg.solve(A, rhs)
            c = c + delta_c
            if np.linalg.norm(delta_c) < tol:
                break

        return c

    def _solve_steady_neumann(self, reaction_func, bc_values):
        c = np.ones(self.n) * 0.5
        tol = 1e-10
        max_iter = 100

        for _ in range(max_iter):
            A = np.zeros((self.n, self.n))
            rhs = np.zeros(self.n)


            A[0, 0] = 1.0
            A[0, 1] = -1.0
            rhs[0] = 0.0
            A[self.n - 1, self.n - 1] = 1.0
            A[self.n - 1, self.n - 2] = -1.0
            rhs[self.n - 1] = 0.0

            for i in range(1, self.n - 1):
                dx_l = self.x[i] - self.x[i - 1]
                dx_r = self.x[i + 1] - self.x[i]
                dx = self.x[i + 1] - self.x[i - 1]

                A[i, i - 1] = -2.0 * self.D / (dx_l * dx)
                A[i, i] = 2.0 * self.D / (dx_l * dx_r)
                A[i, i + 1] = -2.0 * self.D / (dx_r * dx)

                r_val = reaction_func(c[i])
                rhs[i] = -r_val
                dc = 1e-8
                dr_dc = (reaction_func(c[i] + dc) - r_val) / dc
                A[i, i] += dr_dc

            delta_c = np.linalg.solve(A, rhs)
            c = c + delta_c
            if np.linalg.norm(delta_c) < tol:
                break

        return c

    def solve_time_dependent(self, c0: np.ndarray, t_end: float,
                             reaction_func: Callable[[np.ndarray], np.ndarray],
                             n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        c = np.asarray(c0, dtype=float).copy()
        dt = t_end / n_steps
        dx_min = np.min(np.diff(self.x))
        dt_max = dx_min ** 2 / (2.0 * self.D)
        if dt > dt_max:

            n_steps = int(np.ceil(t_end / (0.9 * dt_max)))
            dt = t_end / n_steps

        trajectory = [c.copy()]
        for _ in range(n_steps):

            laplacian = np.zeros(self.n)
            for i in range(1, self.n - 1):
                dx_l = self.x[i] - self.x[i - 1]
                dx_r = self.x[i + 1] - self.x[i]
                laplacian[i] = 2.0 * (
                    (c[i + 1] - c[i]) / dx_r - (c[i] - c[i - 1]) / dx_l
                ) / (dx_l + dx_r)


            if self.bc_type == "neumann":
                laplacian[0] = laplacian[1]
                laplacian[self.n - 1] = laplacian[self.n - 2]
            else:
                laplacian[0] = 0.0
                laplacian[self.n - 1] = 0.0

            reaction = reaction_func(c)
            c = c + dt * (self.D * laplacian + reaction)


            c = np.clip(c, 0.0, 1.0)
            trajectory.append(c.copy())

        return np.array(trajectory), np.linspace(0, t_end, n_steps + 1)


class LangmuirHinshelwoodKinetics:

    def __init__(self, temperature_k: float = 500.0,
                 p_co_pa: float = 1.0e3,
                 p_o2_pa: float = 5.0e2):
        self.T = temperature_k
        self.p_co = p_co_pa
        self.p_o2 = p_o2_pa
        from utils import kb_t_ev
        self.kb_t = kb_t_ev(temperature_k)


        self.a_ads_co = 1.0e6
        self.ea_ads_co = 0.0
        self.a_des_co = 1.0e13
        self.ea_des_co = 1.3

        self.a_ads_o2 = 5.0e5
        self.ea_ads_o2 = 0.0
        self.a_des_o = 1.0e13
        self.ea_des_o = 2.0

        self.a_rxn = 1.0e13
        self.ea_rxn = 0.8

    def _rate_constants(self) -> dict:
        return {
            'k_ads_co': self.a_ads_co * self.p_co * np.exp(-self.ea_ads_co / self.kb_t),
            'k_des_co': self.a_des_co * np.exp(-self.ea_des_co / self.kb_t),
            'k_ads_o2': self.a_ads_o2 * self.p_o2 * np.exp(-self.ea_ads_o2 / self.kb_t),
            'k_des_o': self.a_des_o * np.exp(-self.ea_des_o / self.kb_t),
            'k_rxn': self.a_rxn * np.exp(-self.ea_rxn / self.kb_t),
        }

    def rhs(self, theta: np.ndarray) -> np.ndarray:
        theta = np.clip(theta, 0.0, 1.0)
        th_co = theta[0]
        th_o = theta[1]
        th_free = max(0.0, 1.0 - th_co - th_o)

        k = self._rate_constants()

        dth_co_dt = (k['k_ads_co'] * th_free
                     - k['k_des_co'] * th_co
                     - k['k_rxn'] * th_co * th_o)

        dth_o_dt = (2.0 * k['k_ads_o2'] * th_free ** 2
                    - k['k_des_o'] * th_o
                    - k['k_rxn'] * th_co * th_o)

        return np.array([dth_co_dt, dth_o_dt])

    def integrate_ode(self, theta0: np.ndarray, t_end: float,
                      n_steps: int = 10000) -> Tuple[np.ndarray, np.ndarray]:
        theta = np.asarray(theta0, dtype=float).copy()
        dt = t_end / n_steps
        trajectory = [theta.copy()]
        times = [0.0]

        for _ in range(n_steps):
            k1 = self.rhs(theta)
            k2 = self.rhs(theta + 0.5 * dt * k1)
            k3 = self.rhs(theta + 0.5 * dt * k2)
            k4 = self.rhs(theta + dt * k3)
            theta = theta + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            theta = np.clip(theta, 0.0, 1.0)
            trajectory.append(theta.copy())
            times.append(times[-1] + dt)

        return np.array(trajectory), np.array(times)

    def steady_state_coverage(self) -> np.ndarray:
        theta = np.array([0.3, 0.3])
        dt = 1e-12
        for step in range(5000000):
            dth = self.rhs(theta)

            max_dth = np.max(np.abs(dth))
            if max_dth > 1e-300:
                dt_safe = min(dt, 0.01 / max_dth)
            else:
                dt_safe = dt
            theta_new = theta + dt_safe * dth
            theta_new = np.clip(theta_new, 0.0, 1.0)
            if np.linalg.norm(dth) < 1e-14:
                break
            theta = theta_new

            if step % 1000 == 0 and dt < 1e-6:
                dt *= 2.0
        return theta
