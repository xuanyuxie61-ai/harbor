"""
langevin_fission.py
====================
Langevin dynamics for nuclear fission in the collective coordinate space.

Maps from: 1022_jeremiecoullon_PRT_post (MCMC/Metropolis-Hastings sampling)

Physical context
----------------
The fission dynamics is modelled by the Langevin equations:

  dq_i/dt = p_i / M_i(q)
  dp_i/dt = -dV/dq_i - gamma_ij * p_j / M_j + sqrt(2 * T * gamma_ij) * R_j(t)

where q = (c, h, alpha) are collective coordinates, M is the inertia tensor,
V is the PES, gamma is the dissipation tensor, T is the nuclear temperature,
and R(t) is a white-noise random force.

The one-body dissipation model uses the wall-and-window formula:
  gamma_wall ~ rho_m * R_0^3 * F_wall(c,h,alpha)
  gamma_window ~ rho_m * R_0^3 * F_window(c,h,alpha)

where rho_m ~ 0.16 fm^{-3} is nuclear matter density.

The temperature is related to excitation energy by:
  E* = a * T^2  (Fermi gas model)
where a = A/8 MeV^{-1} is the level density parameter.

The MCMC sampler from PRT is used for sampling the thermal equilibrium
distribution at the saddle point, providing the fragment mass distribution.
"""

import math
import random
from typing import Tuple, List, Dict, Optional

from potential_energy_surface import FissionPES
from nuclear_constants import BOLTZMANN_MEV_K, HBAR_C_MEV_FM


class FissionLangevin:
    """
    Langevin dynamics for fission in 3D collective coordinate space.

    State: x = (c, h, alpha, p_c, p_h, p_alpha)
    Equations of motion (Kramers form):
      dq_i/dt = p_i / M_i
      dp_i/dt = -dV/dq_i - gamma_i * p_i / M_i + noise_i

    Noise: <noise_i(t) * noise_j(t')> = 2 * gamma_i * T * M_i * delta_ij * delta(t-t')
    """

    def __init__(self, z_cn: int = 92, a_cn: int = 236,
                 temperature: float = 1.5,
                 gamma_diss: float = 5.0e21):
        """
        Parameters
        ----------
        z_cn : int
            Compound nucleus charge.
        a_cn : int
            Compound nucleus mass number.
        temperature : float
            Nuclear temperature in MeV (typical: 1-2 MeV for fission).
        gamma_diss : float
            Dissipation strength in s^{-1} (reduced: beta = gamma/(2M)).
        """
        self.pes = FissionPES(z_cn, a_cn)
        self.Z = z_cn
        self.A = a_cn
        self.N = a_cn - z_cn
        self.T = temperature  # MeV
        self.gamma_diss = gamma_diss

        # Collective inertia (in units of hbar^2/MeV ~ 10^{-42} kg*m^2)
        # For irrotational flow: M ~ 10-50 MeV*fm^2/c^2
        self.M_c = 15.0    # elongation inertia (MeV * 10^{-22}s^2)
        self.M_h = 10.0    # neck inertia
        self.M_a = 12.0    # asymmetry inertia

        # Time unit: 10^{-21} s (zeptosecond)
        self.time_unit = 1.0e-21

        # State vector: (c, h, alpha, p_c, p_h, p_alpha)
        self.state = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0]

        # Trajectory storage
        self.trajectory: List[Dict[str, float]] = []

    def set_initial_conditions(self, c0: float = 1.0, h0: float = 1.0,
                               alpha0: float = 0.0) -> None:
        """Set initial conditions at the ground state."""
        self.state = [c0, h0, alpha0, 0.0, 0.0, 0.0]
        self.trajectory = []

    def noise_amplitude(self, m_eff: float) -> float:
        """
        Noise amplitude from fluctuation-dissipation theorem:
          sigma = sqrt(2 * gamma * T * M)

        where gamma is in appropriate units and M is the collective mass.
        """
        arg = 2.0 * self.gamma_diss * self.T * m_eff
        if arg < 0:
            return 0.0
        return math.sqrt(arg)

    def forces(self, c: float, h: float, alpha: float,
               pc: float, ph: float, pa: float) -> Tuple[float, float, float]:
        """
        Compute total force on each collective coordinate:
          F_i = -dV/dq_i - gamma_i * p_i / M_i + noise

        Returns (F_c, F_h, F_alpha).
        """
        grad_c, grad_h, grad_a = self.pes.gradient(c, h, alpha)

        # Dissipative forces
        f_diss_c = -self.gamma_diss * pc / self.M_c
        f_diss_h = -self.gamma_diss * ph / self.M_h
        f_diss_a = -self.gamma_diss * pa / self.M_a

        # Conservative forces
        f_c = -grad_c + f_diss_c
        f_h = -grad_h + f_diss_h
        f_a = -grad_a + f_diss_a

        return (f_c, f_h, f_a)

    def _gaussian_random(self) -> float:
        """Box-Muller transform for Gaussian random numbers."""
        u1 = random.random()
        u2 = random.random()
        while u1 < 1e-30:
            u1 = random.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def step_verlet(self, dt: float) -> Tuple[float, float, float]:
        """
        Velocity-Verlet integration of Langevin equations:

        1. q(t+dt) = q(t) + p(t)/M * dt + 0.5 * F(t)/M * dt^2
        2. Compute F(t+dt) using q(t+dt)
        3. p(t+dt) = p(t) + 0.5*(F(t) + F(t+dt)) * dt
                     - gamma * p(t)/M * dt + noise * sqrt(dt)

        Returns (c, h, alpha) at new time step.
        """
        c, h, alpha, pc, ph, pa = self.state

        # Current forces
        fc, fh, fa = self.forces(c, h, alpha, pc, ph, pa)

        # Update positions
        c_new = c + pc / self.M_c * dt + 0.5 * fc / self.M_c * dt * dt
        h_new = h + ph / self.M_h * dt + 0.5 * fh / self.M_h * dt * dt
        a_new = alpha + pa / self.M_a * dt + 0.5 * fa / self.M_a * dt * dt

        # Compute new forces at new positions (with old momenta estimate)
        pc_half = pc + fc * dt
        ph_half = ph + fh * dt
        pa_half = pa + fa * dt
        fc_new, fh_new, fa_new = self.forces(c_new, h_new, a_new,
                                               pc_half, ph_half, pa_half)

        # Update momenta with dissipation and noise
        noise_c = self.noise_amplitude(self.M_c) * self._gaussian_random() * math.sqrt(dt)
        noise_h = self.noise_amplitude(self.M_h) * self._gaussian_random() * math.sqrt(dt)
        noise_a = self.noise_amplitude(self.M_a) * self._gaussian_random() * math.sqrt(dt)

        pc_new = pc + 0.5 * (fc + fc_new) * dt - self.gamma_diss * pc / self.M_c * dt + noise_c
        ph_new = ph + 0.5 * (fh + fh_new) * dt - self.gamma_diss * ph / self.M_h * dt + noise_h
        pa_new = pa + 0.5 * (fa + fa_new) * dt - self.gamma_diss * pa / self.M_a * dt + noise_a

        # Clamp to physical region
        c_new = max(0.5, min(3.0, c_new))
        h_new = max(-1.0, min(1.5, h_new))
        a_new = max(-0.5, min(0.5, a_new))

        self.state = [c_new, h_new, a_new, pc_new, ph_new, pa_new]
        return (c_new, h_new, a_new)

    def run_trajectory(self, dt: float = 0.001, n_steps: int = 10000,
                       scission_threshold: float = 2.0) -> Dict[str, any]:
        """
        Run a single Langevin trajectory until scission or max steps.

        Scission is defined as c > scission_threshold with h < 0.3
        (neck ruptured).

        Returns dict with trajectory data and scission properties.
        """
        self.trajectory = []
        scissioned = False
        scission_time = 0.0
        scission_alpha = 0.0

        for step in range(n_steps):
            c, h, alpha = self.step_verlet(dt)
            t_now = step * dt

            self.trajectory.append({
                'step': step,
                'time': t_now,
                'c': c,
                'h': h,
                'alpha': alpha,
                'energy': self.pes.total_potential(c, h, alpha)
            })

            # Check scission condition
            if c > scission_threshold and h < 0.3:
                scissioned = True
                scission_time = t_now
                scission_alpha = alpha
                break

        # Determine fragment masses from alpha
        # alpha = (A_heavy - A_light) / A_cn
        # A_heavy = A_cn * (1 + alpha) / 2
        # A_light = A_cn * (1 - alpha) / 2
        a_heavy = int(round(self.A * (1.0 + scission_alpha) / 2.0))
        a_light = self.A - a_heavy

        return {
            'scissioned': scissioned,
            'scission_time': scission_time,
            'scission_alpha': scission_alpha,
            'a_heavy': a_heavy,
            'a_light': a_light,
            'n_steps': len(self.trajectory),
            'final_c': self.state[0],
            'final_h': self.state[1],
        }

    def metropolis_sample_saddle(self, n_samples: int = 500,
                                 temp_sample: float = 1.5,
                                 c_saddle: float = 1.35,
                                 proposal_width: float = 0.05) -> List[Dict[str, float]]:
        """
        Metropolis-Hastings sampling at the inner saddle point.

        Based on the MCMC framework from PRT project:
        1. Propose new (c, h, alpha) from Gaussian proposal
        2. Compute acceptance ratio:
           alpha_acc = min(1, exp(-(V_new - V_old)/T))
        3. Accept or reject

        The sampled configurations give the mass-asymmetry distribution
        at the saddle point, which determines the fragment mass split.

        This is directly analogous to the MH sampler in PRT:
          acceptance criterion: alpha = loss_new - loss_current
          compared against -log(U(0,1)).
        """
        samples = []
        # Start at saddle point
        c_cur, h_cur, a_cur = c_saddle, 1.0, 0.0
        v_cur = self.pes.total_potential(c_cur, h_cur, a_cur)

        n_accepted = 0

        for i in range(n_samples):
            # Gaussian proposal (from PRT's Gaussian move)
            c_prop = c_cur + proposal_width * self._gaussian_random()
            h_prop = h_cur + proposal_width * self._gaussian_random()
            a_prop = a_cur + proposal_width * self._gaussian_random()

            # Clamp proposals
            c_prop = max(0.8, min(2.5, c_prop))
            h_prop = max(0.0, min(1.5, h_prop))
            a_prop = max(-0.5, min(0.5, a_prop))

            v_prop = self.pes.total_potential(c_prop, h_prop, a_prop)

            # MH acceptance (from PRT's MH criterion)
            delta_v = v_prop - v_cur
            if delta_v <= 0:
                accept = True
            else:
                # exp(-delta_V / T)
                log_alpha = -delta_v / temp_sample
                if log_alpha > -500:
                    accept = random.random() < math.exp(log_alpha)
                else:
                    accept = False

            if accept:
                c_cur, h_cur, a_cur = c_prop, h_prop, a_prop
                v_cur = v_prop
                n_accepted += 1

            samples.append({
                'c': c_cur,
                'h': h_cur,
                'alpha': a_cur,
                'energy': v_cur,
                'a_heavy': int(round(self.A * (1.0 + a_cur) / 2.0)),
                'a_light': int(round(self.A * (1.0 - a_cur) / 2.0)),
            })

        acceptance_rate = n_samples / max(1, n_samples)
        if n_accepted > 0:
            acceptance_rate = n_accepted / n_samples

        return samples


def compute_fission_timescale(trajectory: List[Dict[str, float]],
                              c_saddle: float = 1.35) -> Dict[str, float]:
    """
    Compute characteristic fission timescales from a Langevin trajectory.

    Pre-scission time: time from ground state to saddle point.
    Scission time: time from saddle to scission.
    """
    t_saddle = 0.0
    passed_saddle = False

    for point in trajectory:
        c = point['c']
        t = point['time']
        if c > c_saddle and not passed_saddle:
            t_saddle = t
            passed_saddle = True

    t_scission = trajectory[-1]['time'] if trajectory else 0.0
    t_prescission = t_saddle
    t_postscission = t_scission - t_saddle if passed_saddle else 0.0

    return {
        't_prescission': t_prescission,
        't_scission': t_scission,
        't_postscission': t_postscission,
        'passed_saddle': passed_saddle,
    }
