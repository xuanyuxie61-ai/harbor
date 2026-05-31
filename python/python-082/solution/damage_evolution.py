"""
damage_evolution.py
===================
Nonlinear ODE models for cyclic damage accumulation in composites.

Incorporates core algorithms from:
- 511_heartbeat_ode : Nonlinear oscillator with fast-slow dynamics
    (van der Pol type) adapted to model hysteretic damage accumulation.
- 1387_vanderpol_ode_period : Asymptotic period estimation for nonlinear
    oscillators, used to predict fatigue life cycles.

Scientific role:
    Models the time-dependent evolution of damage variables under
    cyclic mechanical loading. The damage rate is governed by a
    coupled nonlinear ODE system inspired by van der Pol dynamics,
    capturing the hysteretic energy dissipation characteristic of
    composite materials under fatigue.

Key formulas:
-----------
1. Damage evolution ODE (adapted van der Pol heartbeat model):
   d(d_f)/dt = -1/epsilon * (d_f^3 - a*d_f + d_m) + omega * cos(omega*t)
   d(d_m)/dt = d_f - gamma

   Here d_f is the fiber damage, d_m is the matrix damage proxy,
   epsilon << 1 controls the sharpness of damage onset,
   a is the damage threshold, gamma is the steady-state damage rate.

2. Fatigue damage accumulation (Paris-like law with nonlinear correction):
   dd/dN = C * (Delta_K / K_c)^m * (1 - d)^{-q}
   where N is the number of cycles, Delta_K is the stress intensity
   range, and C, m, q are material constants.

3. Cyclic stress intensity:
   Delta_K = Y * sigma_max * sqrt(pi * a)
   where a is the crack length, Y is the geometry factor.

4. Van der Pol period (asymptotic, large mu):
   T ~ (3 - 2*ln(2)) * mu + 3*alpha/mu^{1/3} - (1/3)*ln(mu)/mu + ...
   where alpha = 2.338107 is the first zero of the Airy function.

5. Energy-based damage:
   dW/dN = integral_0^{1/f} sigma(t) * deps/dt dt
   The accumulated plastic work drives damage growth.
"""

import numpy as np


class CyclicDamageModel:
    """
    Coupled ODE model for fatigue damage evolution.
    """

    def __init__(self, epsilon=0.001, a_param=0.81, gamma=0.45,
                 omega=7.85, C_paris=1e-10, m_paris=3.5,
                 K_c=30.0e6, sigma_max=100.0e6, Y_geom=1.12):
        self.epsilon = epsilon
        self.a_param = a_param
        self.gamma = gamma
        self.omega = omega
        self.C_paris = C_paris
        self.m_paris = m_paris
        self.K_c = K_c
        self.sigma_max = sigma_max
        self.Y_geom = Y_geom

    def derivatives(self, t, y):
        """
        Compute dy/dt for the coupled damage ODE.

        Parameters
        ----------
        t : float
            Time (s).
        y : ndarray, shape (2,)
            [d_f, d_m] damage variables.

        Returns
        -------
        dydt : ndarray, shape (2,)
        """
        d_f, d_m = y[0], y[1]
        # Clip to physical range [0, 1]
        d_f = np.clip(d_f, 0.0, 0.999)
        d_m = np.clip(d_m, 0.0, 0.999)

        # Fast-slow dynamics (heartbeat/van der Pol adaptation)
        dd_f = -(1.0 / self.epsilon) * (d_f ** 3 - self.a_param * d_f + d_m)
        dd_m = d_f - self.gamma + 0.1 * np.cos(self.omega * t)

        return np.array([dd_f, dd_m])

    def rk4_integrate(self, y0, t0, tstop, n_steps=10000):
        """
        Integrate damage ODE using classical 4th-order Runge-Kutta.

        Parameters
        ----------
        y0 : ndarray
            Initial condition.
        t0, tstop : float
            Time span.
        n_steps : int

        Returns
        -------
        t_array : ndarray
        y_array : ndarray, shape (n_steps+1, 2)
        """
        y0 = np.asarray(y0, dtype=float)
        dt = (tstop - t0) / n_steps
        t_array = np.linspace(t0, tstop, n_steps + 1)
        y_array = np.zeros((n_steps + 1, len(y0)))
        y_array[0] = y0

        for i in range(n_steps):
            t = t_array[i]
            y = y_array[i]
            k1 = self.derivatives(t, y)
            k2 = self.derivatives(t + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = self.derivatives(t + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = self.derivatives(t + dt, y + dt * k3)
            y_array[i + 1] = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            # Clip after update
            y_array[i + 1] = np.clip(y_array[i + 1], 0.0, 1.0)

        return t_array, y_array

    def paris_law_cycles(self, a0, a_crit, n_cycles=100000):
        """
        Integrate Paris law for crack growth per cycle.

        da/dN = C * (Delta_K / K_c)^m
        Delta_K = Y * sigma_max * sqrt(pi * a)

        Parameters
        ----------
        a0 : float
            Initial crack length (m).
        a_crit : float
            Critical crack length (m).
        n_cycles : int
            Maximum number of cycles.

        Returns
        -------
        cycles : ndarray
            Cycle numbers.
        crack_lengths : ndarray
            Crack length at each cycle.
        life : int
            Cycles to failure (a >= a_crit).
        """
        a = float(a0)
        crack_lengths = [a]
        cycles = [0]

        for N in range(1, n_cycles + 1):
            if a >= a_crit:
                break
            delta_K = self.Y_geom * self.sigma_max * np.sqrt(np.pi * a)
            da_dN = self.C_paris * (delta_K / self.K_c) ** self.m_paris
            a += da_dN
            cycles.append(N)
            crack_lengths.append(a)

        life = len(cycles) - 1
        return np.array(cycles), np.array(crack_lengths), life

    def vanderpol_period_estimate(self, mu):
        """
        Estimate the period of a van der Pol oscillator with parameter mu
        using the asymptotic formula of Urabe.

        For large mu:
        T = (3 - 2*ln(2)) * mu + 3*alpha/mu^{1/3}
            - (1/3)*ln(mu)/mu + (3*ln(2) - ln(3) - 1.5 + b0 - 2*d)/mu

        where alpha = 2.338107, b0 = 0.1723, d = 0.4889.

        Parameters
        ----------
        mu : float
            Nonlinearity parameter.

        Returns
        -------
        T : float
            Estimated period.
        """
        if mu == 0.0:
            return 2.0 * np.pi

        alpha = 2.338107
        b0 = 0.1723
        d = 0.4889

        T = ((3.0 - 2.0 * np.log(2.0)) * mu
             + 3.0 * alpha / (mu ** (1.0 / 3.0))
             - (1.0 / 3.0) * np.log(mu) / mu
             + (3.0 * np.log(2.0) - np.log(3.0) - 1.5 + b0 - 2.0 * d) / mu)
        return T

    def hysteresis_energy_per_cycle(self, stress_amplitude, strain_amplitude,
                                    n_points=100):
        """
        Compute dissipated energy per cycle from stress-strain hysteresis loop.

        W = integral sigma d(epsilon) over one cycle

        Parameters
        ----------
        stress_amplitude : float
            Max stress (Pa).
        strain_amplitude : float
            Max strain.
        n_points : int

        Returns
        -------
        energy : float
            Dissipated energy per unit volume (J/m^3).
        """
        theta = np.linspace(0, 2.0 * np.pi, n_points)
        # Ramberg-Osgood type hysteresis with phase lag
        phi = 0.15  # phase lag (radians)
        sigma = stress_amplitude * np.sin(theta)
        epsilon = strain_amplitude * np.sin(theta - phi)
        de = np.gradient(epsilon)
        energy = np.trapezoid(sigma, epsilon)
        return abs(energy)


def cumulative_damage_miner(stress_history, S_n_curve, N_f_curve):
    """
    Miner's rule for cumulative fatigue damage.

    D = sum_i (n_i / N_f_i)

    where n_i is the number of cycles at stress level S_i,
    and N_f_i is the cycles to failure at S_i from the S-N curve.

    Parameters
    ----------
    stress_history : list of tuple (S_i, n_i)
        Stress amplitude and cycle count pairs.
    S_n_curve : callable
        N_f = S_n_curve(S) gives fatigue life for stress S.
    N_f_curve : callable or None
        Alternative life curve.

    Returns
    -------
    D : float
        Cumulative damage ratio (D >= 1 implies failure).
    """
    D = 0.0
    for S, n in stress_history:
        N_f = S_n_curve(S)
        if N_f <= 0:
            continue
        D += n / N_f
    return D
