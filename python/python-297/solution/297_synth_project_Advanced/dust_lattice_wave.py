"""
dust_lattice_wave.py
=======================
Dust lattice wave (DLW) propagation in the dusty plasma crystal
using spectral representation of wave modes.

Physical motivation:
    Dust lattice waves are collective oscillations of dust grains
    in a strongly coupled dusty plasma crystal. They are the
    dusty plasma analog of phonons in solid-state physics.

    The dispersion relation for longitudinal DLW in a 1D chain:
        omega_L(q) = 2 * omega_pd * sqrt(sum_{n=1}^{inf} sin^2(nqa/2) *
                     [1/(n^3) * (1 + kappa*n + kappa^2*n^2/2) * exp(-kappa*n)])

    The transverse mode:
        omega_T(q) = 2 * omega_pd * sqrt(sum_{n=1}^{inf} sin^2(nqa/2) *
                     [1/(n^3) * (1 + kappa*n) * exp(-kappa*n)])

    For the wave representation learning approach (from 1102_ryan597):
    We represent the wave field as a superposition of modes and track
    the amplitude evolution using a spectral decomposition.

References:
    - Melandso, "Lattice waves in dust plasma crystals",
      Phys. Plasmas 6, 1738 (1999)
    - From 1102_ryan597_RepresentationLearningWaves: spectral wave representation
"""

import numpy as np
from typing import Tuple, Optional, Dict


class DustLatticeWaveField:
    """
    Spectral representation of dust lattice wave field.

    The displacement field u_n(t) is represented as:
        u_n(t) = (1/N) * sum_q A(q,t) * exp(i*q*n*a)

    where A(q,t) is the spectral amplitude at wavevector q.

    The time evolution follows:
        d^2 A(q)/dt^2 = -omega(q)^2 * A(q) + F(q,t)

    where omega(q) is the DLW dispersion relation and F is an
    external driving force (e.g., from a modulated laser).
    """

    def __init__(
        self,
        kappa: float,
        omega_pd: float,
        N: int = 64,
        n_modes: int = 32,
    ):
        """
        Parameters
        ----------
        kappa : float
            Screening parameter.
        omega_pd : float
            Dust plasma frequency [rad/s].
        N : int
            Number of grains in the chain.
        n_modes : int
            Number of spectral modes to retain.
        """
        self.kappa = kappa
        self.omega_pd = omega_pd
        self.N = N
        self.n_modes = n_modes
        self.a = 1.0  # Normalized lattice constant

        # Wavevectors
        self.q_values = np.linspace(0, np.pi / self.a, n_modes)

        # Dispersion relation
        self.omega_L, self.omega_T = self._compute_dispersion()

        # Spectral amplitudes
        self.A_L = np.zeros(n_modes, dtype=complex)  # Longitudinal
        self.A_T = np.zeros(n_modes, dtype=complex)  # Transverse
        self.dA_L_dt = np.zeros(n_modes, dtype=complex)
        self.dA_T_dt = np.zeros(n_modes, dtype=complex)

    def _compute_dispersion(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute DLW dispersion relations.

        Longitudinal:
            omega_L^2(q) = 2*omega_pd^2 * sum_{n=1}^{N/2} sin^2(nqa/2) *
                          f_L(n, kappa)

        Transverse:
            omega_T^2(q) = 2*omega_pd^2 * sum_{n=1}^{N/2} sin^2(nqa/2) *
                          f_T(n, kappa)

        where:
            f_L(n, kappa) = (1+kappa*n+kappa^2*n^2/2)*exp(-kappa*n)/n^3
            f_T(n, kappa) = (1+kappa*n)*exp(-kappa*n)/(2*n^3)

        Returns
        -------
        omega_L, omega_T : np.ndarray, shape (n_modes,)
            Longitudinal and transverse mode frequencies.
        """
        omega_L = np.zeros(self.n_modes)
        omega_T = np.zeros(self.n_modes)
        n_max = max(1, self.N // 2)

        for iq, q in enumerate(self.q_values):
            sum_L = 0.0
            sum_T = 0.0
            for n in range(1, n_max + 1):
                sin2 = np.sin(n * q * self.a / 2.0)**2
                # Longitudinal coupling
                f_L = (1.0 + self.kappa * n + 0.5 * (self.kappa * n)**2) * \
                      np.exp(-self.kappa * n) / n**3
                # Transverse coupling
                f_T = (1.0 + self.kappa * n) * \
                      np.exp(-self.kappa * n) / (2.0 * n**3)
                sum_L += sin2 * f_L
                sum_T += sin2 * f_T

            omega_L[iq] = self.omega_pd * np.sqrt(max(0.0, 2.0 * sum_L))
            omega_T[iq] = self.omega_pd * np.sqrt(max(0.0, 2.0 * sum_T))

        return omega_L, omega_T

    def initialize_wave_packet(
        self,
        q_center: float,
        sigma_q: float,
        amplitude: float = 1.0,
        mode_type: str = "longitudinal",
    ) -> None:
        """
        Initialize a Gaussian wave packet in spectral space.

        A(q) = amplitude * exp(-(q - q_center)^2 / (2*sigma_q^2))

        Parameters
        ----------
        q_center : float
            Center wavevector of the packet.
        sigma_q : float
            Width in q-space.
        amplitude : float
            Peak amplitude.
        mode_type : str
            "longitudinal" or "transverse".
        """
        A0 = amplitude * np.exp(
            -(self.q_values - q_center)**2 / (2.0 * sigma_q**2)
        )
        if mode_type == "longitudinal":
            self.A_L = A0.astype(complex)
            self.dA_L_dt = np.zeros_like(self.A_L)
        else:
            self.A_T = A0.astype(complex)
            self.dA_T_dt = np.zeros_like(self.A_T)

    def spectral_to_spatial(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform spectral amplitudes to spatial displacement field.

        u_n = (1/N) * sum_q A(q) * exp(i*q*n*a)

        Returns
        -------
        u_L, u_T : np.ndarray, shape (N,)
            Longitudinal and transverse displacements.
        """
        n_grid = np.arange(self.N)
        u_L = np.zeros(self.N)
        u_T = np.zeros(self.N)

        for n_idx in range(self.N):
            phase = np.exp(1j * self.q_values * n_idx * self.a)
            u_L[n_idx] = np.real(np.sum(self.A_L * phase)) / self.N
            u_T[n_idx] = np.real(np.sum(self.A_T * phase)) / self.N

        return u_L, u_T

    def evolve_time(
        self,
        dt: float,
        n_steps: int,
        driving_force: Optional[np.ndarray] = None,
        damping: float = 0.0,
    ) -> Dict[str, np.ndarray]:
        """
        Time-evolve the wave field using velocity Verlet integration.

        d^2A/dt^2 = -omega^2 * A - gamma * dA/dt + F(t)

        Parameters
        ----------
        dt : float
            Time step [s].
        n_steps : int
            Number of time steps.
        driving_force : np.ndarray, optional
            External driving force F(q, t). Shape: (n_steps, n_modes).
        damping : float
            Damping coefficient gamma [1/s].

        Returns
        -------
        results : dict
            'time': time array,
            'u_L_history': displacement history,
            'u_T_history': transverse displacement history,
            'energy_history': total energy history.
        """
        times = np.arange(n_steps + 1) * dt
        u_L_history = np.zeros((n_steps + 1, self.N))
        u_T_history = np.zeros((n_steps + 1, self.N))
        energy_history = np.zeros(n_steps + 1)

        for step in range(n_steps + 1):
            # Record state
            u_L, u_T = self.spectral_to_spatial()
            u_L_history[step] = u_L
            u_T_history[step] = u_T

            # Compute energy
            E_kin_L = 0.5 * np.sum(np.abs(self.dA_L_dt)**2)
            E_pot_L = 0.5 * np.sum(self.omega_L**2 * np.abs(self.A_L)**2)
            E_kin_T = 0.5 * np.sum(np.abs(self.dA_T_dt)**2)
            E_pot_T = 0.5 * np.sum(self.omega_T**2 * np.abs(self.A_T)**2)
            energy_history[step] = E_kin_L + E_pot_L + E_kin_T + E_pot_T

            if step < n_steps:
                # Driving force
                F_L = driving_force[step] if driving_force is not None else np.zeros(self.n_modes)
                F_T = np.zeros(self.n_modes)

                # Velocity Verlet (longitudinal)
                accel_L = -self.omega_L**2 * self.A_L - damping * self.dA_L_dt + F_L
                self.A_L += self.dA_L_dt * dt + 0.5 * accel_L * dt**2
                accel_L_new = -self.omega_L**2 * self.A_L - damping * self.dA_L_dt + F_L
                self.dA_L_dt += 0.5 * (accel_L + accel_L_new) * dt

                # Velocity Verlet (transverse)
                accel_T = -self.omega_T**2 * self.A_T - damping * self.dA_T_dt + F_T
                self.A_T += self.dA_T_dt * dt + 0.5 * accel_T * dt**2
                accel_T_new = -self.omega_T**2 * self.A_T - damping * self.dA_T_dt + F_T
                self.dA_T_dt += 0.5 * (accel_T + accel_T_new) * dt

        return {
            "time": times,
            "u_L_history": u_L_history,
            "u_T_history": u_T_history,
            "energy_history": energy_history,
        }

    def group_velocity(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute group velocities v_g = domega/dq.

        Returns
        -------
        v_g_L, v_g_T : np.ndarray, shape (n_modes,)
            Group velocities for longitudinal and transverse modes.
        """
        dq = np.gradient(self.q_values)
        # Handle zero or very small dq values
        dq = np.where(np.abs(dq) < 1e-30, 1e-30, dq)
        v_g_L = np.gradient(self.omega_L) / dq
        v_g_T = np.gradient(self.omega_T) / dq
        # Replace NaN with 0
        v_g_L = np.nan_to_num(v_g_L, nan=0.0, posinf=0.0, neginf=0.0)
        v_g_T = np.nan_to_num(v_g_T, nan=0.0, posinf=0.0, neginf=0.0)
        return v_g_L, v_g_T

    def density_of_states(self, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute the phonon density of states g(omega).

        g(omega) = (1/N) * sum_q delta(omega - omega(q))

        Approximated with a histogram.

        Parameters
        ----------
        n_bins : int
            Number of frequency bins.

        Returns
        -------
        omega_bins : np.ndarray
            Frequency bin centers.
            g_L, g_T : np.ndarray
            Density of states for each branch.
        """
        all_omega = np.concatenate([self.omega_L, self.omega_T])
        omega_min, omega_max = all_omega.min(), all_omega.max()
        if omega_max - omega_min < 1e-15:
            omega_max = omega_min + 1.0

        bins = np.linspace(omega_min, omega_max, n_bins + 1)
        g_L, _ = np.histogram(self.omega_L, bins=bins, density=True)
        g_T, _ = np.histogram(self.omega_T, bins=bins, density=True)
        omega_bins = 0.5 * (bins[:-1] + bins[1:])

        return omega_bins, g_L, g_T
