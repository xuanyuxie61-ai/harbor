
import numpy as np
from utils import safe_divide, check_bounds, cond_number_estimate


class BandedMatrixSolver:

    def __init__(self, n, ml, mu):
        self.n = n
        self.ml = ml
        self.mu = mu
        self.m = mu + 1

    def factor(self, a):
        a_lu = np.array(a, dtype=np.float64, copy=True)
        n = self.n
        ml = self.ml
        mu = self.mu
        m = self.m
        info = 0
        ju = 0

        for k in range(n - 1):
            if abs(a_lu[m - 1, k]) < 1e-15:
                info = k + 1
                raise ValueError(f"零主元出现在第 {k} 步，矩阵可能奇异")

            lm = min(ml, n - k - 1)
            a_lu[m:m + lm, k] = -a_lu[m:m + lm, k] / a_lu[m - 1, k]

            ju = min(max(ju, mu + k), n - 1)
            mm = m
            for j in range(k + 1, ju + 1):
                mm -= 1
                a_lu[mm:mm + lm, j] += a_lu[mm - 1, j] * a_lu[m:m + lm, k]

        if abs(a_lu[m - 1, n - 1]) < 1e-15:
            info = n
            raise ValueError("最后一个主元为零，矩阵奇异")

        return a_lu, info

    def solve(self, a_lu, b):
        n = self.n
        ml = self.ml
        mu = self.mu
        m = self.m
        x = np.array(b, dtype=np.float64, copy=True)



        for k in range(n - 1):
            lm = min(ml, n - k - 1)
            if lm > 0:
                x[k + 1:k + 1 + lm] += a_lu[m:m + lm, k] * x[k]



        for k in range(n - 1, -1, -1):

            um = min(mu, n - k - 1)
            for j in range(k + 1, k + 1 + um):
                row = k - j + mu
                x[k] -= a_lu[row, j] * x[j]
            x[k] /= a_lu[m - 1, k]

        return x


class ThermalReactorModel:

    def __init__(self, L=1.0, nx=50, rho=200.0, Cp=1500.0, k_eff=0.15, u=0.05):
        self.L = L
        self.nx = nx
        self.dx = L / (nx - 1)
        self.rho = rho
        self.Cp = Cp
        self.k_eff = k_eff
        self.u = u
        self.alpha = k_eff / (rho * Cp)
        self.solver = BandedMatrixSolver(nx, ml=1, mu=1)

    def build_system_matrix(self, dt):
        nx = self.nx
        dx = self.dx
        alpha = self.alpha
        u = self.u

        a = np.zeros((3, nx), dtype=np.float64)

        r = alpha * dt / (dx * dx)
        p = u * dt / (2.0 * dx)


        a[1, 0] = 1.0
        a[1, nx - 1] = 1.0



        p_up = self.u * dt / self.dx
        for i in range(1, nx - 1):

            a[2, i - 1] = -(r + p_up)
            a[1, i] = 1.0 + 2.0 * r + p_up
            a[0, i + 1] = -r

        return a

    def solve_timestep(self, T_old, dt, Q_source, T_inlet=300.0):
        T_old = np.asarray(T_old, dtype=np.float64)
        Q_source = np.asarray(Q_source, dtype=np.float64)
        nx = self.nx


        b = T_old.copy()
        b[1:nx - 1] += dt * Q_source[1:nx - 1] / (self.rho * self.Cp)
        b[0] = T_inlet
        b[-1] = 350.0


        a = self.build_system_matrix(dt)
        a_lu, info = self.solver.factor(a)


        T_new = self.solver.solve(a_lu, b)
        T_new = check_bounds(T_new, 250.0, 1500.0, name="temperature")
        return T_new

    def simulate(self, T_init, dt, n_steps, Q_func, T_inlet=300.0):
        T = np.asarray(T_init, dtype=np.float64)
        t_history = np.zeros(n_steps + 1, dtype=np.float64)
        T_history = np.zeros((n_steps + 1, self.nx), dtype=np.float64)
        T_history[0, :] = T

        for n in range(n_steps):
            t = n * dt
            Q = Q_func(t, self.dx * np.arange(self.nx))
            T = self.solve_timestep(T, dt, Q, T_inlet)
            t_history[n + 1] = t + dt
            T_history[n + 1, :] = T

        return t_history, T_history


def compute_reaction_heat_source(x, T, kinetics, y_mass, reaction_enthalpy=-500e3):

    raise NotImplementedError("Hole 2: 请补全 compute_reaction_heat_source 函数")
