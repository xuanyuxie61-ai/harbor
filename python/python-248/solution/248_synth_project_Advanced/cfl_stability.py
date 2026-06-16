"""
cfl_stability.py
================

CFL (Courant-Friedrichs-Lewy) stability control and non-linear
time-dependent constraint handling for the compressible-Euler +
self-gravity + cooling system.

The seed project 1217_CardiacModelling_nonlinear-time-dependent-leak
studies a non-linear and time-dependent leak current whose evolution is

    I_leak(t) = I_ss + (I_0 - I_ss) exp(-t / tau)

with tau = tau(V, t) depending non-linearly on voltage and time.  In
our astrophysical analogue the gas cooling rate has exactly the same
mathematical structure: the cooling time

    t_cool(rho, T, Z, t) = e_th / Lambda(rho, T, Z, t)

depends non-linearly on density, temperature, metallicity and time
through ionisation state and supernova feedback.  The exponential
tail fit used in the cardiac project is here repurposed to estimate
the effective cooling time from discrete samples of the cooling rate.

Key formulae
------------
CFL condition for the compressible Euler equations in 3-D:

    dt <= CFL * dx / max_i (|v_i| + c_{s,i} + c_{A,i})

where c_A = B / sqrt(4 pi rho) is the Alfven speed (zero in our
pure-hydro runs, but the form is kept for MHD extension).

With self-gravity, a Jeans-length constraint appears:

    dt <= dt_Jeans = min_i sqrt(pi dx_i^2 / (G rho_i))

With radiative cooling, a cooling-time constraint:

    dt <= dt_cool = min_i e_th,i / |Lambda_i|

The overall stable dt is the minimum of all three.

Non-linear cooling-time fit (from cardiac leak-current analogue)
----------------------------------------------------------------
Given discrete samples (t_k, Lambda_k) of the cooling rate we fit

    Lambda(t) ~ Lambda_ss + (Lambda_0 - Lambda_ss) exp(-t / tau_cool)

by least-squares minimisation of the residual sum

    R = sum_k (Lambda_k - Lambda_fit(t_k))^2.

The fitted tau_cool is then used as the sub-cycling step for the
cooling source term when the global dt exceeds tau_cool.
"""

from __future__ import annotations
import math
import numpy as np
from typing import Tuple, Optional, Callable, List

from astro_constants import (
    RK_CFL_NUMBER, RK_CFL_MAX, GRAVITATIONAL_CGS,
)


# =====================================================================
#                         CLASSICAL CFL NUMBER
# =====================================================================

def cfl_number(dx: float, dt: float, vmax_cgs: float) -> float:
    """
    CFL number  nu = dt * v_max / dx.
    """
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    return dt * abs(vmax_cgs) / dx


def max_signal_speed(rho_cgs: np.ndarray, p_cgs: np.ndarray,
                     vx: np.ndarray, vy: np.ndarray, vz: np.ndarray,
                     gamma: float = 5.0 / 3.0) -> float:
    """
    Maximum signal speed across all cells:

        s_max = max_i (|v_i| + c_{s,i})

    where c_s = sqrt(gamma P / rho) is the adiabatic sound speed and
    |v| = sqrt(vx^2 + vy^2 + vz^2).
    """
    cs = np.sqrt(gamma * p_cgs / np.maximum(rho_cgs, 1.0e-60))
    v_abs = np.sqrt(vx * vx + vy * vy + vz * vz)
    return float(np.max(v_abs + cs))


def cfl_dt(dx_cgs: float, s_max: float,
           cfl_target: float = RK_CFL_NUMBER) -> float:
    """
    Time step allowed by the CFL condition:

        dt_CFL = cfl_target * dx / s_max

    Clipped to dt_max = RK_CFL_MAX * dx / s_max to avoid blowup when
    s_max is very small (e.g. at cold, stagnant cells).
    """
    if s_max <= 1.0e-60:
        return RK_CFL_MAX * dx_cgs / 1.0e-60
    dt = cfl_target * dx_cgs / s_max
    return min(dt, RK_CFL_MAX * dx_cgs / max(s_max, 1.0e-60))


# =====================================================================
#                    JEANS-LENGTH STABILITY CONSTRAINT
# =====================================================================

def jeans_dt(dx_cgs: float, rho_cgs: np.ndarray,
             gamma: float = 5.0 / 3.0, safety: float = 0.4) -> float:
    """
    Time step to resolve the Jeans length:

        dt_Jeans = safety * min_i sqrt(pi dx^2 / (G rho_i))

    When rho_i is large the Jeans length becomes small and a standard
    CFL based on sound speed under-resolves the gravitational collapse;
    this additional constraint enforces dx > L_J / 4.
    """
    rho_safe = np.maximum(rho_cgs, 1.0e-60)
    dt_arr = safety * np.sqrt(math.pi * dx_cgs * dx_cgs
                              / (GRAVITATIONAL_CGS * rho_safe))
    return float(np.min(dt_arr))


# =====================================================================
#                  COOLING-TIME STABILITY CONSTRAINT
# =====================================================================

def cooling_dt(e_th_cgs: np.ndarray, Lambda_cgs: np.ndarray,
               safety: float = 0.3) -> float:
    """
    Time step to resolve the cooling time:

        dt_cool = safety * min_i (e_th,i / |Lambda_i|)

    The safety factor reflects the non-linearity of the cooling
    function (at constant density the cooling equation is
    de/dt = -Lambda(rho, e) and an explicit scheme requires
    dt < e / |dLambda/de| ~ e / |Lambda|).
    """
    Lam_abs = np.maximum(np.abs(Lambda_cgs), 1.0e-60)
    e_safe = np.maximum(e_th_cgs, 1.0e-60)
    dt_arr = safety * e_safe / Lam_abs
    return float(np.min(dt_arr))


# =====================================================================
#          EXPONENTIAL FIT OF TIME-DEPENDENT COOLING RATE
#          (directly lifted from the cardiac leak-current fit)
# =====================================================================

def fit_exponential_leak(times: np.ndarray, values: np.ndarray
                         ) -> Tuple[float, float, float]:
    """
    Fit the non-linear time-dependent model

        v(t) = a exp(-b t) + c

    to a time series (times, values).  The cardiac project uses this
    to fit leak-current tails; we use it for cooling-rate tails.

    Implementation: a three-stage least-squares procedure.

    1. Estimate c = mean(values[-N/5:])  (steady-state tail value)
    2. Linearise: log(|v - c|) = log(a) - b t  -> linear fit for a, b.
    3. Refine all three parameters via scipy.optimize.curve_fit.

    Returns (a, b, c) with b = 1/tau the decay rate.
    """
    if times.size < 5:
        raise ValueError("need at least 5 samples for exponential fit")
    # Stage 1: estimate c
    n_tail = max(times.size // 5, 2)
    c_est = float(np.mean(values[-n_tail:]))
    # Stage 2: linear fit on log residual
    shifted = values - c_est
    positive = shifted > 1.0e-60
    if np.sum(positive) < 3:
        # fallback: zero decay rate
        return (float(values[0] - c_est), 0.0, c_est)
    log_shift = np.log(shifted[positive])
    t_pos = times[positive]
    # least-squares: log_shift = log(a) - b * t
    A = np.vstack([np.ones_like(t_pos), t_pos]).T
    sol, _res, _rank, _sv = np.linalg.lstsq(A, log_shift, rcond=None)
    log_a_est, mb_est = sol
    a_est = math.exp(log_a_est)
    b_est = -mb_est
    # Stage 3: refine
    try:
        from scipy.optimize import curve_fit
        def func(t, a, b, c):
            return a * np.exp(-b * t) + c
        popt, _pcov = curve_fit(func, times, values,
                                p0=[a_est, b_est, c_est],
                                maxfev=2000)
        a, b, c = popt
    except Exception:
        a, b, c = a_est, b_est, c_est
    return (float(a), float(b), float(c))


def effective_cooling_time(a: float, b: float, c: float) -> float:
    """
    From the exponential fit  Lambda(t) = a exp(-b t) + c,
    the effective cooling time is

        tau_eff = 1 / b      if b > 0 and |c| << |a|
               or infinity    otherwise (steady-state regime).
    """
    if b <= 1.0e-12:
        return float("inf")
    return 1.0 / b


# =====================================================================
#                  COMBINED STABILITY CONTROLLER
# =====================================================================

class StabilityController:
    """
    Master stability controller that combines the CFL, Jeans, and
    cooling constraints into a single dt choice.

    The controller also monitors the *dominant* constraint at each
    step (for diagnostics) and warns if the ratio dt/dt_min exceeds
    a safety threshold.
    """

    def __init__(self, dx_cgs: float, cfl_target: float = RK_CFL_NUMBER,
                 jeans_safety: float = 0.4, cooling_safety: float = 0.3):
        self.dx_cgs = dx_cgs
        self.cfl_target = cfl_target
        self.jeans_safety = jeans_safety
        self.cooling_safety = cooling_safety
        self.last_dt_cfl = 0.0
        self.last_dt_jeans = 0.0
        self.last_dt_cooling = 0.0
        self.last_dominant = "cfl"
        self.history: List[dict] = []

    def compute_dt(
        self,
        rho: np.ndarray,
        p: np.ndarray,
        vx: np.ndarray,
        vy: np.ndarray,
        vz: np.ndarray,
        e_th: np.ndarray,
        Lambda: Optional[np.ndarray] = None,
        gamma: float = 5.0 / 3.0,
    ) -> float:
        """Compute the maximum stable dt."""
        s_max = max_signal_speed(rho, p, vx, vy, vz, gamma)
        dt_cfl = cfl_dt(self.dx_cgs, s_max, self.cfl_target)
        dt_jeans = jeans_dt(self.dx_cgs, rho, gamma, self.jeans_safety)
        self.last_dt_cfl = dt_cfl
        self.last_dt_jeans = dt_jeans
        candidates = {"cfl": dt_cfl, "jeans": dt_jeans}
        if Lambda is not None:
            dt_cool = cooling_dt(e_th, Lambda, self.cooling_safety)
            self.last_dt_cooling = dt_cool
            candidates["cooling"] = dt_cool
        dt_min_name = min(candidates, key=lambda k: candidates[k])
        dt_min = candidates[dt_min_name]
        self.last_dominant = dt_min_name
        self.history.append({
            "dt_cfl": dt_cfl, "dt_jeans": dt_jeans,
            "dt_cooling": candidates.get("cooling", None),
            "dominant": dt_min_name,
            "dt": dt_min,
        })
        return dt_min

    def summary(self) -> str:
        if not self.history:
            return "no stability history"
        last = self.history[-1]
        return (f"StabilityController: last dt = {last['dt']:.3e} "
                f"(dominant: {last['dominant']})")
