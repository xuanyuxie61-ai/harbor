"""
duration_feedback.py
====================

Duration-modulated stellar feedback with switching dynamics.

The seed project 1239_ynffsy_DurationModulatedDynamics studies neural
population dynamics where the *duration* spent in each discrete brain
state modulates the continuous dynamics within that state.  We lift
this idea to star formation feedback: the ISM switches between two
macroscopic states (ACTIVE star-forming, QUIESCENT) and the duration
tau spent in each state modulates the strength of mechanical and
radiative energy injection.

Key mappings
------------
* Discrete latent state z_t in {ACTIVE, QUIESCENT}
    -> star formation mode (bursting / quiescent)
* Duration tau_t = t - t_last_switch
    -> duration of the current star formation episode
* Continuous state x_t (SLDS emission)
    -> thermodynamic state (rho, e_th) of the gas
* Switching kernel P(switch | tau)
    -> star formation efficiency dependence on episode duration

The duration-modulated feedback strength is

    F(t) = F_base(z_t) * f_mod(tau_t)

with f_mod(tau) = 1 - exp(-tau / tau_rise)  (rise phase)
             * exp(-(tau - tau_peak)^2 / (2 tau_width^2))  (decay)

giving a pulse-like profile in each active episode.

Switching dynamics
------------------
We model the state switching as a Poisson process with time-varying
rate lambda(tau) that depends on the episode duration:

    lambda(tau) = lambda_0 + lambda_1 exp(-tau / tau_sat)

When lambda(tau) dt > Uniform(0, 1) we trigger a switch.  This is the
duration-modulated analogue of the SLDS switching kernel.

Feedback channels
-----------------
For each supernova event the mechanical + thermal energy injection is

    E_SN = 1.0e51 erg        (canonical SN energy)
    M_ej = 10 M_sun          (ejecta mass)
    v_ej = sqrt(2 E_SN / M_ej) ~ 4400 km/s

The volumetric heating rate in an ACTIVE region is

    Gamma_SN = n_SN * E_SN * f_mod(tau)

with n_SN the supernova rate per unit volume set by the star formation
rate density via the Kennicutt-Schmidt law:

    Sigma_SFR = A_KS * (Sigma_gas / 1 M_sun pc^{-2})^{1.4}
              [M_sun yr^{-1} kpc^{-2}]

References
----------
- Kennicutt, R. C. 1998, ApJ 498, 541 (KS law)
- Kim, C.-G., & Ostriker, E. C. 2015, ApJ 802, 127 (SN feedback model)
- DurationModulatedDynamics (seed project 1239)
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Any

from astro_constants import (
    SOLAR_MASS_CGS, YEAR_CGS, MEWAYEAR_CGS, KILOPARSEC_CGS,
    PROTON_MASS_CGS, BOLTZMANN_CGS, MEAN_MOLECULAR_WEIGHT_NEUTRAL,
    DOMAIN_SIZE_KPC,
)


# Convenient alias to handle possible constant-name typo
_MYR = MEWAYEAR_CGS if "MEWAYEAR" in dir() else 3.15576e13


# =====================================================================
#                   STAR-FORMATION STATE MACHINE
# =====================================================================

class StarFormationState:
    """Discrete state: ACTIVE = 1, QUIESCENT = 0."""
    QUIESCENT = 0
    ACTIVE = 1

    def __init__(self) -> None:
        self.current = self.QUIESCENT
        self.duration_myr = 0.0
        self.n_switches = 0
        self.switch_times_myr: List[float] = []

    def update(self, dt_myr: float) -> None:
        self.duration_myr += dt_myr

    def trigger_switch(self, t_myr: float) -> None:
        self.current = 1 - self.current
        self.duration_myr = 0.0
        self.n_switches += 1
        self.switch_times_myr.append(t_myr)

    def __repr__(self) -> str:
        s = "ACTIVE" if self.current == self.ACTIVE else "QUIESCENT"
        return (f"StarFormationState({s}, duration={self.duration_myr:.2f} Myr, "
                f"n_switches={self.n_switches})")


# =====================================================================
#                 DURATION-MODULATED SWITCHING KERNEL
# =====================================================================

def switching_rate(tau_myr: float, lambda_0: float = 0.05,
                    lambda_1: float = 0.20, tau_sat: float = 10.0) -> float:
    """
    Time-varying switching rate (events per Myr):

        lambda(tau) = lambda_0 + lambda_1 exp(-tau / tau_sat)

    The rate is highest at the start of an episode and decays to a
    baseline as the episode duration grows.
    """
    return lambda_0 + lambda_1 * math.exp(-tau_myr / tau_sat)


def maybe_switch(state: StarFormationState, dt_myr: float, t_myr: float,
                  rng: Optional[np.random.Generator] = None) -> bool:
    """
    Stochastically switch state with probability  lambda(tau) * dt.
    """
    if rng is None:
        rng = np.random.default_rng()
    lam = switching_rate(state.duration_myr)
    prob = 1.0 - math.exp(-lam * dt_myr)
    if rng.random() < prob:
        state.trigger_switch(t_myr)
        return True
    return False


# =====================================================================
#               DURATION-MODULATED FEEDBACK STRENGTH
# =====================================================================

def feedback_modulation(tau_myr: float, tau_rise: float = 1.0,
                         tau_peak: float = 3.0,
                         tau_width: float = 2.0) -> float:
    """
    Pulse-like modulation of feedback strength with episode duration.

        f_mod(tau) = (1 - exp(-tau / tau_rise))
                     * exp(-(tau - tau_peak)^2 / (2 tau_width^2))

    This produces a gradual rise, a peak at tau ~ tau_peak, and a
    gradual decay, matching the observed time profile of stellar
    feedback in young star clusters (Kim & Ostriker 2015).
    """
    rise = 1.0 - math.exp(-tau_myr / max(tau_rise, 1.0e-6))
    decay = math.exp(-((tau_myr - tau_peak) ** 2)
                     / max(2.0 * tau_width * tau_width, 1.0e-6))
    return rise * decay


# =====================================================================
#                   SUPERNOVA FEEDBACK HEATING
# =====================================================================

E_SN_ERG = 1.0e51                       # erg per supernova
M_EJ_SOLAR = 10.0                       # solar masses per SN ejecta
N_SN_PER_MSTAR = 0.01                   # ~1 SN per 100 M_sun formed


class SupernovaFeedback:
    """
    Compute volumetric heating rate from supernova feedback with
    duration-modulated strength.
    """

    def __init__(
        self,
        A_KS: float = 2.5e-4,            # Kennicutt-Schmidt coefficient
        n_KS: float = 1.4,               # KS exponent
        E_SN: float = E_SN_ERG,
        n_sn_per_mstar: float = N_SN_PER_MSTAR,
    ) -> None:
        self.A_KS = A_KS
        self.n_KS = n_KS
        self.E_SN = E_SN
        self.n_sn_per_mstar = n_sn_per_mstar
        self.heating_history: List[Dict[str, float]] = []

    # -----------------------------------------------------------------
    #  Kennicutt-Schmidt star formation rate
    # -----------------------------------------------------------------
    def star_formation_rate_density(
        self,
        sigma_gas_msun_pc2: np.ndarray,
    ) -> np.ndarray:
        """
        Sigma_SFR = A_KS * (Sigma_gas)^n   [M_sun yr^{-1} kpc^{-2}].
        """
        return self.A_KS * np.power(np.maximum(sigma_gas_msun_pc2, 0.0),
                                    self.n_KS)

    # -----------------------------------------------------------------
    #  volumetric heating rate
    # -----------------------------------------------------------------
    def heating_rate(
        self,
        rho_cgs: np.ndarray,
        dx_cgs: float,
        state: StarFormationState,
    ) -> np.ndarray:
        """
        Compute Gamma_SN (erg s^{-1} cm^{-3}) for each cell.

        Only ACTIVE cells receive feedback; the strength is modulated
        by f_mod(tau).  The rate is

            Gamma = rho / (mu m_p) * SFR / (H_height) * n_sn * E_SN * f_mod / V
        """
        heating = np.zeros_like(rho_cgs)
        if state.current != StarFormationState.ACTIVE:
            return heating
        fmod = feedback_modulation(state.duration_myr)
        # convert gas density to column density proxy (rho * dx)
        sigma_cgs = rho_cgs * dx_cgs
        sigma_msun_pc2 = sigma_cgs / (SOLAR_MASS_CGS / (3.0856776e18) ** 2)
        sfr_density = self.star_formation_rate_density(sigma_msun_pc2)
        # volumetric rate: SFR_density / dx in M_sun yr^{-1} kpc^{-3}
        vol_rate = sfr_density / (dx_cgs / KILOPARSEC_CGS)
        # convert to number of SN per second per cm^3
        sn_rate_cgs = (vol_rate * self.n_sn_per_mstar
                       / (SOLAR_MASS_CGS * YEAR_CGS)
                       * (KILOPARSEC_CGS ** 3))
        heating = sn_rate_cgs * self.E_SN * fmod
        self.heating_history.append({
            "t_myr": state.duration_myr,
            "f_mod": fmod,
            "max_heating": float(np.max(heating)),
            "state": state.current,
        })
        return heating


# =====================================================================
#                 DURATION-MODULATED DYNAMICS SIMULATOR
# =====================================================================

class DurationModulatedSimulator:
    """
    Top-level driver that advances the star-formation state alongside
    the hydrodynamics: at each time step we

      1. update the state duration
      2. probabilistically trigger a switch
      3. compute the duration-modulated heating rate
      4. return the heating as a source term for the energy equation
    """

    def __init__(self, seed: int = 42) -> None:
        self.state = StarFormationState()
        self.feedback = SupernovaFeedback()
        self.rng = np.random.default_rng(seed)
        self.time_myr = 0.0
        self.history: List[Dict[str, Any]] = []

    def step(self, dt_myr: float, rho_cgs: np.ndarray, dx_cgs: float
             ) -> np.ndarray:
        """Advance the state machine and return the heating source term."""
        self.state.update(dt_myr)
        self.time_myr += dt_myr
        maybe_switch(self.state, dt_myr, self.time_myr, self.rng)
        heating = self.feedback.heating_rate(rho_cgs, dx_cgs, self.state)
        self.history.append({
            "t_myr": self.time_myr,
            "state": self.state.current,
            "duration_myr": self.state.duration_myr,
            "f_mod": feedback_modulation(self.state.duration_myr),
            "max_heating": float(np.max(heating)),
        })
        return heating

    def summary(self) -> str:
        n_active = sum(1 for h in self.history if h["state"] == 1)
        n_total = max(len(self.history), 1)
        return (f"DurationModulatedSimulator summary:\n"
                f"  total time:      {self.time_myr:.2f} Myr\n"
                f"  active fraction: {n_active / n_total:.3f}\n"
                f"  number of switches: {self.state.n_switches}\n"
                f"  mean f_mod:      "
                f"{np.mean([h['f_mod'] for h in self.history]):.3f}")
