"""
Nonlinear Root-Finding for Self-Consistent Equations
=====================================================
Implements bracketing and root-finding methods for solving
self-consistent equations in transport theory.

Based on nonlin_snyder (project 811), this module provides:
- Snyder's method for root finding with sign-change bracketing
- Bisection with secant acceleration
- Self-consistent solutions for the Fermi level, scattering time,
  and T-matrix poles.

Key equations solved:
1. Charge neutrality: n(E_F) = n_dop + n_intrinsic
2. Self-energy: Sigma(E) = n_imp * V_0 / (1 - V_0 * G_0(E))
3. Fermi level from carrier density:
     n = int dE N(E) f(E - E_F)
"""

import numpy as np

# Physical constants
E_CHARGE = 1.602176634e-19
EV_TO_J = 1.602176634e-19
HBAR = 1.054571817e-34
K_B = 1.380649e-23


class NonlinearSolver:
    """
    Nonlinear equation solvers for self-consistent transport problems.
    """

    def __init__(self, max_iter=100, tol=1e-12):
        self.max_iter = max_iter
        self.tol = tol

    def snyder_method(self, f, a, b):
        """
        Snyder's bracketing method for finding a root in [a, b].

        Algorithm (from project 811):
            c = (a*f(b) - b*f(a)) / (f(b) - f(a))   [secant step]
            if f(b)*f(c) < 0:
                a = b, f(a) = f(b)
            else:
                f(a) = f(a) / 2.0   [dampen]
            b = c, f(b) = f(c)

        Parameters
        ----------
        f : callable
            Target function.
        a, b : float
            Bracket endpoints with f(a)*f(b) < 0.

        Returns
        -------
        root : float
        it : int
            Number of iterations.
        """
        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            raise ValueError("f(a) and f(b) must have opposite signs.")

        it = 0
        while abs(b - a) > self.tol and it < self.max_iter:
            # Secant intersection
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
        """
        Standard bisection method.

        Parameters
        ----------
        f : callable
        a, b : float

        Returns
        -------
        root : float
        it : int
        """
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
        """
        Secant method without bracketing.

        Parameters
        ----------
        f : callable
        x0, x1 : float
            Initial guesses.

        Returns
        -------
        root : float
        it : int
        """
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
        """
        Solve for the Fermi level given a target carrier density.

        n = int_0^inf dE N(E) [f(E - E_F) - f(E + E_F)]

        For the massive Dirac cone:
            N(E) = |E| / (2*pi*hbar^2*v_F^2)

        Parameters
        ----------
        carrier_density : float
            Target carrier density in m^{-2}.
        temperature : float
            Temperature in K.
        hamiltonian : DiracSurfaceHamiltonian
        method : str
            'snyder', 'bisection', or 'secant'.

        Returns
        -------
        E_F : float
            Fermi level in eV.
        """
        from dirac_surface import DiracSurfaceHamiltonian

        prefactor = 1.0 / (2.0 * np.pi * (hamiltonian.hbar * hamiltonian.v_F) ** 2)

        def density_integral(E_F_eV):
            E_F_J = E_F_eV * EV_TO_J
            # Numerical integration over energy
            E_max = max(10.0 * abs(E_F_J), 10.0 * abs(hamiltonian.Delta), 1e-20)
            n_E = 500
            E_vals = np.linspace(0.0, E_max, n_E)
            dE = E_vals[1] - E_vals[0]

            if temperature < 1e-6:
                # T=0: integrate up to E_F
                N_E = E_vals * prefactor
                f_diff = np.where(E_vals <= abs(E_F_J), 1.0, 0.0)
            else:
                beta = 1.0 / (K_B * temperature)
                N_E = E_vals * prefactor
                f_diff = 1.0 / (1.0 + np.exp(beta * (E_vals - E_F_J))) \
                         - 1.0 / (1.0 + np.exp(beta * (E_vals + E_F_J)))

            n_computed = np.sum(N_E * f_diff) * dE
            return n_computed - carrier_density

        # Bracket the root
        E_min = -5.0
        E_max = 5.0
        f_min = density_integral(E_min)
        f_max = density_integral(E_max)

        # Expand bracket if needed
        attempts = 0
        while f_min * f_max > 0 and attempts < 50:
            E_min *= 2.0
            E_max *= 2.0
            f_min = density_integral(E_min)
            f_max = density_integral(E_max)
            attempts += 1

        if f_min * f_max > 0:
            # Fallback: use secant with initial guess
            return self.secant_method(density_integral, 0.01, 0.5)[0]

        if method == 'snyder':
            E_F, it = self.snyder_method(density_integral, E_min, E_max)
        elif method == 'bisection':
            E_F, it = self.bisection(density_integral, E_min, E_max)
        else:
            E_F, it = self.secant_method(density_integral, E_min, E_max)

        return E_F

    def self_consistent_scattering_time(self, E_F, disorder, temperature=0.0):
        """
        Solve the self-consistent equation for the scattering time:

            1/tau = (2*pi / hbar) * n_i * V_0^2 * N(E_F) * <1 - cos theta>

        with self-energy correction:
            E_F -> E_F + Re[Sigma(E_F)]
            1/tau = -2 * Im[Sigma(E_F)] / hbar

        Parameters
        ----------
        E_F : float
            Fermi energy in eV.
        disorder : DisorderScattering
        temperature : float

        Returns
        -------
        tau_sc : float
            Self-consistent scattering time in seconds.
        """
        E_F_J = E_F * EV_TO_J

        def tau_equation(tau_guess):
            # Self-energy at energy E_F with lifetime tau_guess
            # Approximate: Im[Sigma] = -hbar / (2*tau)
            # Born rate: 1/tau_born = disorder.born_scattering_rate(E_F)
            rate_born = disorder.born_scattering_rate(E_F_J)
            if rate_born < 1e-30:
                return 1.0 / tau_guess
            rate_sc = 1.0 / tau_guess
            # Self-consistency: tau_sc = tau_born (for Born approximation)
            # For T-matrix, add corrections
            return rate_sc - rate_born

        # Initial guess
        tau0 = disorder.transport_scattering_time(E_F_J)
        if tau0 < 1e-20 or tau0 > 1e20:
            tau0 = 1e-14

        root, _ = self.secant_method(tau_equation, tau0 * 0.5, tau0 * 2.0)
        tau_sc = max(1e-20, root)
        return tau_sc

    def find_gap_from_magnetization(self, T_C, M_sat, M, J_ex):
        """
        Compute the exchange gap from the magnetization using the
        mean-field relation:

            Delta = J_ex * M

        with M(T) given by the Brillouin function or mean-field approximation.

        Parameters
        ----------
        T_C : float
            Curie temperature in K.
        M_sat : float
            Saturation magnetization in A/m.
        M : float
            Current magnetization.
        J_ex : float
            Exchange coupling in eV.

        Returns
        -------
        Delta : float
            Exchange gap in eV.
        """
        Delta = J_ex * (M / M_sat)
        return Delta

    def solve_for_t_matrix_pole(self, G0_func, V0, E_guess):
        """
        Find the pole of the T-matrix by solving:

            1 - V0 * G0(E) = 0

        Parameters
        ----------
        G0_func : callable
            G0(E) returning complex value.
        V0 : float
            Potential strength.
        E_guess : float
            Initial energy guess in eV.

        Returns
        -------
        E_pole : complex
            Pole energy in eV.
        """
        def objective(E):
            return 1.0 - V0 * G0_func(E)

        E_real, _ = self.snyder_method(lambda e: np.real(objective(e)),
                                        E_guess - 1.0, E_guess + 1.0)
        return E_real
