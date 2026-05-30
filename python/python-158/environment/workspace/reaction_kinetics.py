
import numpy as np
from reaction_mechanism import (
    compute_production_rates, NSPEC, MW, SPECIES_NAMES, get_pathway_contributions
)
from utils import newton_raphson_scalar, condition_estimate






def normal_ode_rhs(t: float, y: float) -> float:
    return -t * y


def normal_ode_exact(t: float) -> float:
    return np.exp(-t * t / 2.0) / np.sqrt(2.0 * np.pi)


def verify_integrator(integrator_func, t0: float, tf: float, n_steps: int) -> dict:
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






def backward_euler_step(
    f, y0: np.ndarray, dt: float, jac=None, max_newton: int = 20,
    newton_tol: float = 1e-8
) -> np.ndarray:
    y = y0.copy()
    n = len(y0)
    I = np.eye(n)
    








    pass
    

    y_sum = np.sum(y)
    if y_sum > 0.0:
        y = y / y_sum
    return y


def integrate_backward_euler(
    f, y0: np.ndarray, t_end: float, dt_init: float = 1e-6,
    dt_min: float = 1e-12, dt_max: float = 1.0, atol: float = 1e-8,
    jac=None, dense_output: bool = False
) -> dict:
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
        

        y_full = backward_euler_step(f, y, dt, jac=jac)
        

        y_half = backward_euler_step(f, y, dt / 2.0, jac=jac)
        y_half = backward_euler_step(f, y_half, dt / 2.0, jac=jac)
        
        err = np.linalg.norm(y_full - y_half) / (atol + np.linalg.norm(y_half) * 1e-4)
        
        if err <= 1.0 or dt <= dt_min * 1.1:

            y = y_half
            t += dt
            n_steps += 1
            if dense_output:
                trajectory.append((t, y.copy()))
            

            if err < 0.1:
                dt = min(dt * 2.0, dt_max)
        else:

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






class ReactorODE:
    
    def __init__(self, T: float, P: float = 101325.0, MW_mix: float = 0.029):
        self.T = T
        self.P = P
        self.MW_mix = MW_mix
        self.rho = P * MW_mix / (8.314462618 * T) if T > 0 else 1.2
    
    def rhs(self, Y: np.ndarray) -> np.ndarray:
        omega = compute_production_rates(Y, self.T, self.rho)
        return omega / max(self.rho, 1e-30)
    
    def jacobian(self, Y: np.ndarray) -> np.ndarray:
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
    reactor = ReactorODE(T, P)
    

    reactor.rho = P * reactor.MW_mix / (8.314462618 * T)
    
    result = integrate_backward_euler(
        reactor.rhs, Y0, t_end, dt_init=1e-7, dt_min=1e-12,
        dt_max=1e-3, atol=1e-10, jac=reactor.jacobian, dense_output=True
    )
    

    Y_final = result["y_final"]
    contributions = get_pathway_contributions(Y_final, T, reactor.rho)
    
    result["pathways"] = contributions
    result["NO_ppm"] = Y_final[SPECIES_NAMES.index("NO")] * 1e6
    result["NO2_ppm"] = Y_final[SPECIES_NAMES.index("NO2")] * 1e6
    result["N2O_ppm"] = Y_final[SPECIES_NAMES.index("N2O")] * 1e6
    return result
