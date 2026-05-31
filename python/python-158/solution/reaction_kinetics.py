"""
reaction_kinetics.py
====================
Time integration of stiff NOx reaction kinetics ODE system.

Incorporates:
- normal_ode (818): exact solution of a linear ODE for code verification.
- ode_euler_backward (826): backward Euler with Picard iteration for stiff systems.

The governing equations are:
    dY_i/dt = (1/rho) * omega_i(Y, T)   for i = 1,...,N_species

where omega_i is the mass production rate [kg/(m^3·s)] from reaction_mechanism.py.

This is a stiff system because chemical time scales span ~10 orders of magnitude
(fast radical equilibria vs. slow NO formation).

We solve using:
1. Backward Euler (A-stable, first-order):
        Y^{n+1} = Y^n + dt * f(Y^{n+1})
   solved by Newton iteration (improved over Picard).

2. Verification against exact solution of the normal ODE:
        dy/dt = -t*y,   y(t) = exp(-t^2/2)/sqrt(2*pi)
"""

import numpy as np
from reaction_mechanism import (
    compute_production_rates, NSPEC, MW, SPECIES_NAMES, get_pathway_contributions
)
from utils import newton_raphson_scalar, condition_estimate


# ======================================================================
# 1. Exact solution verification (from normal_ode)
# ======================================================================

def normal_ode_rhs(t: float, y: float) -> float:
    """dy/dt = -t * y"""
    return -t * y


def normal_ode_exact(t: float) -> float:
    """Exact solution: y(t) = exp(-t^2/2) / sqrt(2*pi)"""
    return np.exp(-t * t / 2.0) / np.sqrt(2.0 * np.pi)


def verify_integrator(integrator_func, t0: float, tf: float, n_steps: int) -> dict:
    """
    Verify an ODE integrator against the normal ODE exact solution.
    Returns L2 error and maximum error.
    """
    y0 = normal_ode_exact(t0)
    t_vals = np.linspace(t0, tf, n_steps + 1)
    y_num = integrator_func(normal_ode_rhs, y0, t_vals)
    y_exact = np.array([normal_ode_exact(t) for t in t_vals])
    errors = np.abs(y_num - y_exact)
    return {
        "L2_error": np.sqrt(np.mean(errors ** 2)),
        "max_error": np.max(errors),
        "final_relative_error": abs(y_num[-1] - y_exact[-1]) / max(abs(y_exact[-1]), 1e-30)
    }


# ======================================================================
# 2. Backward Euler with damped Newton solver
# ======================================================================

def backward_euler_step(
    f, y0: np.ndarray, dt: float, jac=None, max_newton: int = 20,
    newton_tol: float = 1e-8
) -> np.ndarray:
    """
    Single step of backward Euler:
        y = y0 + dt * f(y)
    Solved via damped Newton iteration:
        (I - dt * J) * dy = -(y - y0 - dt * f(y))
    """
    y = y0.copy()
    n = len(y0)
    I = np.eye(n)
    
    for _ in range(max_newton):
        fy = f(y)
        residual = y - y0 - dt * fy
        if np.linalg.norm(residual) < newton_tol:
            break
        
        if jac is not None:
            J = jac(y)
        else:
            # Finite difference Jacobian
            J = np.zeros((n, n))
            eps = 1e-8
            f0 = fy
            for j in range(n):
                yp = y.copy()
                dy = max(eps * abs(y[j]), eps)
                yp[j] += dy
                fp = f(yp)
                J[:, j] = (fp - f0) / dy
        
        # Solve (I - dt*J) * delta = -residual
        M = I - dt * J
        try:
            delta = np.linalg.solve(M, -residual)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse
            delta = np.linalg.lstsq(M, -residual, rcond=None)[0]
        
        # Damping
        alpha = 1.0
        for _ in range(10):
            y_new = y + alpha * delta
            # Enforce non-negativity
            y_new = np.maximum(y_new, 0.0)
            res_new = y_new - y0 - dt * f(y_new)
            if np.linalg.norm(res_new) < np.linalg.norm(residual):
                y = y_new
                break
            alpha *= 0.5
        else:
            y = np.maximum(y + 0.1 * delta, 0.0)
    
    # Final normalization to ensure mass fractions sum to 1.0
    y_sum = np.sum(y)
    if y_sum > 0.0:
        y = y / y_sum
    return y


def integrate_backward_euler(
    f, y0: np.ndarray, t_end: float, dt_init: float = 1e-6,
    dt_min: float = 1e-12, dt_max: float = 1.0, atol: float = 1e-8,
    jac=None, dense_output: bool = False
) -> dict:
    """
    Adaptive backward Euler integration for stiff ODE:
        dy/dt = f(y), y(0) = y0, integrate to t_end.
    
    Time step adaptation based on error estimate from step-doubling.
    """
    y = y0.copy().astype(float)
    t = 0.0
    dt = dt_init
    trajectory = [(t, y.copy())] if dense_output else None
    n_steps = 0
    n_rejected = 0
    
    while t < t_end:
        dt = min(dt, t_end - t)
        if dt < dt_min:
            dt = dt_min
        
        # Full step
        y_full = backward_euler_step(f, y, dt, jac=jac)
        
        # Two half steps for error estimate
        y_half = backward_euler_step(f, y, dt / 2.0, jac=jac)
        y_half = backward_euler_step(f, y_half, dt / 2.0, jac=jac)
        
        err = np.linalg.norm(y_full - y_half) / (atol + np.linalg.norm(y_half) * 1e-4)
        
        if err <= 1.0 or dt <= dt_min * 1.1:
            # Accept step
            y = y_half
            t += dt
            n_steps += 1
            if dense_output:
                trajectory.append((t, y.copy()))
            
            # Increase step
            if err < 0.1:
                dt = min(dt * 2.0, dt_max)
        else:
            # Reject and reduce
            dt = max(dt * 0.5, dt_min)
            n_rejected += 1
        
        if n_steps > 100000:
            break
    
    result = {
        "y_final": y,
        "t_final": t,
        "n_steps": n_steps,
        "n_rejected": n_rejected,
    }
    if dense_output:
        result["trajectory"] = trajectory
    return result


# ======================================================================
# 3. Single-reactor combustion ODE wrapper
# ======================================================================

class ReactorODE:
    """
    Defines the ODE system for a constant-pressure batch reactor
    with NOx chemistry.
    """
    
    def __init__(self, T: float, P: float = 101325.0, MW_mix: float = 0.029):
        self.T = T
        self.P = P
        self.MW_mix = MW_mix
        self.rho = P * MW_mix / (8.314462618 * T) if T > 0 else 1.2
    
    def rhs(self, Y: np.ndarray) -> np.ndarray:
        """Return dY/dt [1/s]."""
        omega = compute_production_rates(Y, self.T, self.rho)
        return omega / max(self.rho, 1e-30)
    
    def jacobian(self, Y: np.ndarray) -> np.ndarray:
        """Finite-difference Jacobian of rhs."""
        n = len(Y)
        J = np.zeros((n, n))
        f0 = self.rhs(Y)
        eps = 1e-8
        for j in range(n):
            Yp = Y.copy()
            dy = max(eps * abs(Y[j]), eps)
            Yp[j] += dy
            fp = self.rhs(Yp)
            J[:, j] = (fp - f0) / dy
        return J


def simulate_batch_reactor(
    Y0: np.ndarray, T: float, t_end: float, P: float = 101325.0
) -> dict:
    """
    Simulate a constant-temperature batch reactor from t=0 to t_end.
    Returns final composition and pathway analysis.
    """
    reactor = ReactorODE(T, P)
    
    # Set initial density
    reactor.rho = P * reactor.MW_mix / (8.314462618 * T)
    
    result = integrate_backward_euler(
        reactor.rhs, Y0, t_end, dt_init=1e-7, dt_min=1e-12,
        dt_max=1e-3, atol=1e-10, jac=reactor.jacobian, dense_output=True
    )
    
    # Pathway analysis at final state
    Y_final = result["y_final"]
    contributions = get_pathway_contributions(Y_final, T, reactor.rho)
    
    result["pathways"] = contributions
    result["NO_ppm"] = Y_final[SPECIES_NAMES.index("NO")] * 1e6
    result["NO2_ppm"] = Y_final[SPECIES_NAMES.index("NO2")] * 1e6
    result["N2O_ppm"] = Y_final[SPECIES_NAMES.index("N2O")] * 1e6
    return result
