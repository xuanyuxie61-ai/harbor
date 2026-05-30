
import numpy as np


class ODESolver:

    @staticmethod
    def explicit_euler(f, t0, y0, t_end, h):
        n_steps = int(np.ceil((t_end - t0) / h))
        t = np.linspace(t0, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1,) + np.shape(y0))
        y[0] = y0

        for i in range(n_steps):
            y[i + 1] = y[i] + h * f(t[i], y[i])

        return t, y

    @staticmethod
    def trapezoidal_implicit(f, t0, y0, t_end, h, max_iter=10, tol=1e-10):
        n_steps = int(np.ceil((t_end - t0) / h))
        t = np.linspace(t0, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1,) + np.shape(y0))
        y[0] = y0

        for i in range(n_steps):
            f_n = f(t[i], y[i])
            z = y[i].copy()

            for _ in range(max_iter):
                z_new = y[i] + 0.5 * h * (f_n + f(t[i + 1], z))
                if np.max(np.abs(z_new - z)) < tol:
                    z = z_new
                    break
                z = z_new

            y[i + 1] = z

        return t, y

    @staticmethod
    def runge_kutta4(f, t0, y0, t_end, h):
        n_steps = int(np.ceil((t_end - t0) / h))
        t = np.linspace(t0, t_end, n_steps + 1)
        y = np.zeros((n_steps + 1,) + np.shape(y0))
        y[0] = y0

        for i in range(n_steps):
            k1 = h * f(t[i], y[i])
            k2 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k1)
            k3 = h * f(t[i] + 0.5 * h, y[i] + 0.5 * k2)
            k4 = h * f(t[i] + h, y[i] + k3)
            y[i + 1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return t, y

    @staticmethod
    def adaptive_rk45(f, t0, y0, t_end, h0, tol=1e-6, safety=0.9):

        a = [0.0, 0.2, 0.3, 0.6, 1.0, 0.875]
        b = [
            [],
            [0.2],
            [3.0/40.0, 9.0/40.0],
            [3.0/10.0, -9.0/10.0, 6.0/5.0],
            [-11.0/54.0, 5.0/2.0, -70.0/27.0, 35.0/27.0],
            [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]
        ]
        c = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]
        c_star = [2825.0/27648.0, 0.0, 18575.0/48384.0, 13525.0/55296.0, 277.0/14336.0, 0.25]

        t_list = [t0]
        y_list = [y0.copy()]
        t = t0
        y = y0.copy()
        h = h0

        while t < t_end:
            if t + h > t_end:
                h = t_end - t

            k = []
            for i in range(6):
                ti = t + a[i] * h
                yi = y.copy()
                for j in range(i):
                    yi += h * b[i][j] * k[j]
                k.append(f(ti, yi))

            y4 = y.copy()
            y5 = y.copy()
            for i in range(6):
                y4 += h * c[i] * k[i]
                y5 += h * c_star[i] * k[i]

            error = np.max(np.abs(y5 - y4))
            if error < 1e-14:
                error = tol * 0.1

            if error <= tol:
                t += h
                y = y5.copy()
                t_list.append(t)
                y_list.append(y.copy())


            h = safety * h * (tol / error) ** 0.2
            h = max(h, h0 * 0.01)
            h = min(h, h0 * 10.0)

        return np.array(t_list), np.array(y_list)


class CoupledOscillator:

    def __init__(self, m1=3.0, m2=5.0, k1=1.0, k2=10.0):
        self.m1 = m1
        self.m2 = m2
        self.k1 = k1
        self.k2 = k2

    def rhs(self, t, y):
        u1, v1, u2, v2 = y

        du1dt = v1
        dv1dt = (-self.k1 * u1 + self.k2 * (u2 - u1)) / self.m1
        du2dt = v2
        dv2dt = -self.k2 * (u2 - u1) / self.m2

        return np.array([du1dt, dv1dt, du2dt, dv2dt])

    def solve(self, t0, y0, t_end, h, method='rk4'):
        solver = ODESolver()
        if method == 'euler':
            return solver.explicit_euler(self.rhs, t0, y0, t_end, h)
        elif method == 'trapezoidal':
            return solver.trapezoidal_implicit(self.rhs, t0, y0, t_end, h)
        elif method == 'rk4':
            return solver.runge_kutta4(self.rhs, t0, y0, t_end, h)
        else:
            raise ValueError(f"不支持的方法: {method}")


class PhaseFieldTimeStepper:

    def __init__(self, dt, dx, dy, epsilon, tau, diffusion_coeff=1.0):
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon
        self.tau = tau
        self.diffusion_coeff = diffusion_coeff


        dt_diff_limit = 0.25 * min(dx ** 2, dy ** 2) / max(diffusion_coeff, 1e-14)
        if dt > dt_diff_limit:

            self.dt = 0.5 * dt_diff_limit

    def explicit_step(self, phi, rhs_func):
        return phi + self.dt * rhs_func(phi)

    def semi_implicit_step(self, phi, rhs_nonlinear):

        phi_new = phi + self.dt * rhs_nonlinear


        lap_phi = np.zeros_like(phi)
        lap_phi[1:-1, 1:-1] = (
            (phi_new[2:, 1:-1] - 2.0 * phi_new[1:-1, 1:-1] + phi_new[:-2, 1:-1]) / (self.dx ** 2) +
            (phi_new[1:-1, 2:] - 2.0 * phi_new[1:-1, 1:-1] + phi_new[1:-1, :-2]) / (self.dy ** 2)
        )


        phi_new += 0.1 * self.dt * self.diffusion_coeff * lap_phi

        return phi_new

    def runge_kutta_step(self, phi, rhs_func):
        k1 = self.dt * rhs_func(phi)
        k2 = self.dt * rhs_func(phi + 0.5 * k1)
        k3 = self.dt * rhs_func(phi + 0.5 * k2)
        k4 = self.dt * rhs_func(phi + k3)

        return phi + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
