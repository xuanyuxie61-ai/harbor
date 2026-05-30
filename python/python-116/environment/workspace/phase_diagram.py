
import numpy as np
from scipy.special import i0e, i1e


class NewtonMaehlySolver:

    def __init__(self, coeffs, max_iter=100, tol=1e-12):
        self.coeffs = np.asarray(coeffs, dtype=complex)
        self.max_iter = max_iter
        self.tol = tol

    def poly_and_derivative(self, z):
        z = complex(z)
        p = self.coeffs[-1]
        dp = 0.0 + 0.0j
        for c in self.coeffs[-2::-1]:
            dp = dp * z + p
            p = p * z + c
        return p, dp

    def solve(self):
        d = len(self.coeffs) - 1
        if d <= 0:
            return np.array([])


        cd = self.coeffs[-1]
        if abs(cd) < 1e-15:
            raise ValueError("最高次项系数接近零。")
        radius = 1.0 + np.max(np.abs(self.coeffs[:-1] / cd))


        theta = np.linspace(0.0, 2.0 * np.pi, d, endpoint=False)
        roots = radius * np.exp(1j * theta)

        for iteration in range(self.max_iter):
            roots_old = roots.copy()
            for i in range(d):
                pz, dpz = self.poly_and_derivative(roots[i])
                s = 0.0 + 0.0j
                for j in range(d):
                    if j != i:
                        diff = roots[i] - roots[j]
                        if abs(diff) > 1e-15:
                            s += 1.0 / diff
                denom = dpz - pz * s
                if abs(denom) < 1e-15:
                    continue
                roots[i] = roots[i] - pz / denom

            max_change = np.max(np.abs(roots - roots_old))
            max_poly = np.max(np.abs([self.poly_and_derivative(r)[0] for r in roots]))
            if max_change < self.tol and max_poly < self.tol * 10:
                return roots

        return roots


class SelfConsistentTransition:

    def __init__(self, J_coupling=2.5, kb=0.008314):
        self.J = J_coupling
        self.kb = kb

    def sc_equation(self, S, T):
        if T <= 0 or S < 0:
            return np.inf
        beta = 1.0 / (self.kb * T)
        x = beta * self.J * S

        if abs(x) > 700:

            B = 1.0 - 1.0 / (2.0 * x)
        else:
            B = i1e(x) / i0e(x)
        return S - B

    def find_roots_vs_temperature(self, T_values):
        roots = []
        for T in T_values:

            a, b = 0.01, 1.0
            fa = self.sc_equation(a, T)
            fb = self.sc_equation(b, T)
            if fa * fb > 0:

                roots.append(0.0)
                continue
            for _ in range(80):
                c = (a + b) / 2.0
                fc = self.sc_equation(c, T)
                if abs(fc) < 1e-12:
                    break
                if fa * fc <= 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
            roots.append((a + b) / 2.0)
        return np.array(roots)

    def critical_temperature(self):


        raise NotImplementedError("critical_temperature 方法需要补全")


class DuffingMembraneDynamics:

    def __init__(self, delta=0.3, alpha=1.0, beta=-1.0, gamma=0.5,
                 omega=1.2, noise_amp=0.1, seed=None):
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.omega = omega
        self.noise_amp = noise_amp
        self.rng = np.random.default_rng(seed)

    def deriv(self, t, y):
        y1, y2 = y
        noise = self.noise_amp * self.rng.normal()
        dy1dt = y2
        dy2dt = (-self.delta * y2 - self.alpha * y1 -
                 self.beta * y1 ** 3 + self.gamma * np.cos(self.omega * t) +
                 noise)
        return np.array([dy1dt, dy2dt])

    def integrate_rk4(self, y0, t_span, n_steps=5000):
        t0, tf = t_span
        dt = (tf - t0) / n_steps
        t_values = np.linspace(t0, tf, n_steps + 1)
        y_values = np.zeros((n_steps + 1, 2))
        y_values[0] = y0
        y = np.array(y0, dtype=float)

        for i in range(n_steps):
            t = t_values[i]
            k1 = self.deriv(t, y)
            k2 = self.deriv(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.deriv(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.deriv(t + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            y_values[i + 1] = y

        return t_values, y_values

    def lyapunov_exponent_estimate(self, y0, t_span, n_steps=5000, perturbation=1e-8):
        t, y = self.integrate_rk4(y0, t_span, n_steps)
        y_perturbed = y0 + np.array([perturbation, 0.0])
        yp = y_perturbed.copy()
        dt = (t_span[1] - t_span[0]) / n_steps

        for i in range(n_steps):
            t_i = t[i]
            k1 = self.deriv(t_i, yp)
            k2 = self.deriv(t_i + 0.5 * dt, yp + 0.5 * dt * k1)
            k3 = self.deriv(t_i + 0.5 * dt, yp + 0.5 * dt * k2)
            k4 = self.deriv(t_i + dt, yp + dt * k3)
            yp = yp + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        delta_init = perturbation
        delta_final = np.linalg.norm(yp - y[-1])
        if delta_final <= 0 or delta_init <= 0:
            return -np.inf
        lam = np.log(delta_final / delta_init) / (t_span[1] - t_span[0])
        return float(lam)


class PhaseDiagramBuilder:

    def __init__(self, J=2.5, kb=0.008314):
        self.J = J
        self.kb = kb
        self.sc = SelfConsistentTransition(J, kb)

    def build_diagram(self, T_range=(250, 400), n_T=50):
        T_vals = np.linspace(T_range[0], T_range[1], n_T)
        S_vals = self.sc.find_roots_vs_temperature(T_vals)



        A0 = 0.64
        kappa = 25.0
        A_vals = A0 * (1.0 + 0.1 * (1.0 - S_vals))
        P_vals = self.kb * T_vals / A_vals - kappa * (A_vals - A0) / A0

        return T_vals, S_vals, P_vals

    def latent_heat(self, Tc, S_gel, S_fluid):
        return 0.5 * self.J * (S_gel ** 2 - S_fluid ** 2)
