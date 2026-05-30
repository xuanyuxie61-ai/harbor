
import numpy as np


E_CHARGE = 1.602176634e-19
EV_TO_J = 1.602176634e-19
HBAR = 1.054571817e-34
K_B = 1.380649e-23


class NonlinearSolver:

    def __init__(self, max_iter=100, tol=1e-12):
        self.max_iter = max_iter
        self.tol = tol

    def snyder_method(self, f, a, b):
        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs.")

        it = 0
        while abs(b - a) > self.tol and it < self.max_iter:

            if abs(fb - fa) < 1e-30:
                break
            c = (a * fb - b * fa) / (fb - fa)
            fc = f(c)

            if fb * fc < 0.0:
                a = b
                fa = fb
            else:
                fa = fa / 2.0

            b = c
            fb = fc
            it += 1

        root = (a + b) / 2.0
        return root, it

    def bisection(self, f, a, b):
        fa = f(a)
        fb = f(b)
        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs.")

        it = 0
        while abs(b - a) > self.tol and it < self.max_iter:
            c = (a + b) / 2.0
            fc = f(c)
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            it += 1

        return (a + b) / 2.0, it

    def secant_method(self, f, x0, x1):
        f0 = f(x0)
        f1 = f(x1)
        it = 0

        for _ in range(self.max_iter):
            if abs(f1 - f0) < 1e-30:
                break
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            if abs(x2 - x1) < self.tol:
                return x2, it
            x0, f0 = x1, f1
            x1, f1 = x2, f(x2)
            it += 1

        return x1, it

    def find_fermi_level(self, carrier_density, temperature, hamiltonian,
                         method='snyder'):
        from dirac_surface import DiracSurfaceHamiltonian

        prefactor = 1.0 / (2.0 * np.pi * (hamiltonian.hbar * hamiltonian.v_F) ** 2)

        def density_integral(E_F_eV):
            E_F_J = E_F_eV * EV_TO_J

            E_max = max(10.0 * abs(E_F_J), 10.0 * abs(hamiltonian.Delta), 1e-20)
            n_E = 500
            E_vals = np.linspace(0.0, E_max, n_E)
            dE = E_vals[1] - E_vals[0]

            if temperature < 1e-6:

                N_E = E_vals * prefactor
                f_diff = np.where(E_vals <= abs(E_F_J), 1.0, 0.0)
            else:
                beta = 1.0 / (K_B * temperature)
                N_E = E_vals * prefactor
                f_diff = 1.0 / (1.0 + np.exp(beta * (E_vals - E_F_J))) \
                         - 1.0 / (1.0 + np.exp(beta * (E_vals + E_F_J)))

            n_computed = np.sum(N_E * f_diff) * dE
            return n_computed - carrier_density


        E_min = -5.0
        E_max = 5.0
        f_min = density_integral(E_min)
        f_max = density_integral(E_max)


        attempts = 0
        while f_min * f_max > 0 and attempts < 50:
            E_min *= 2.0
            E_max *= 2.0
            f_min = density_integral(E_min)
            f_max = density_integral(E_max)
            attempts += 1

        if f_min * f_max > 0:

            return self.secant_method(density_integral, 0.01, 0.5)[0]

        if method == 'snyder':
            E_F, it = self.snyder_method(density_integral, E_min, E_max)
        elif method == 'bisection':
            E_F, it = self.bisection(density_integral, E_min, E_max)
        else:
            E_F, it = self.secant_method(density_integral, E_min, E_max)

        return E_F

    def self_consistent_scattering_time(self, E_F, disorder, temperature=0.0):
        E_F_J = E_F * EV_TO_J

        def tau_equation(tau_guess):



            rate_born = disorder.born_scattering_rate(E_F_J)
            if rate_born < 1e-30:
                return 1.0 / tau_guess
            rate_sc = 1.0 / tau_guess


            return rate_sc - rate_born


        tau0 = disorder.transport_scattering_time(E_F_J)
        if tau0 < 1e-20 or tau0 > 1e20:
            tau0 = 1e-14

        root, _ = self.secant_method(tau_equation, tau0 * 0.5, tau0 * 2.0)
        tau_sc = max(1e-20, root)
        return tau_sc

    def find_gap_from_magnetization(self, T_C, M_sat, M, J_ex):
        Delta = J_ex * (M / M_sat)
        return Delta

    def solve_for_t_matrix_pole(self, G0_func, V0, E_guess):
        def objective(E):
            return 1.0 - V0 * G0_func(E)

        E_real, _ = self.snyder_method(lambda e: np.real(objective(e)),
                                        E_guess - 1.0, E_guess + 1.0)
        return E_real
