
import numpy as np
from scipy.optimize import fsolve


def implicit_midpoint_step(
    y: np.ndarray,
    t: float,
    dt: float,
    f: callable,
    theta: float = 0.5
) -> np.ndarray:
    def residual(yp):
        y_mid = (1.0 - theta) * y + theta * yp
        return yp - y - dt * f(t + theta * dt, y_mid)

    y_next = fsolve(residual, y, full_output=False)
    return y_next


def milne_lte_estimate(
    y_mid: np.ndarray,
    y_pred: np.ndarray,
    dt: float,
    dt_prev: float,
    dt_prev2: float
) -> float:

    lte = np.linalg.norm(y_mid - y_pred) / (dt ** 2 + 1e-14)
    return lte


def adaptive_midpoint_solve(
    f: callable,
    y0: np.ndarray,
    t_span: tuple[float, float],
    dt_init: float = 0.1,
    abstol: float = 1e-6,
    reltol: float = 1e-4,
    theta: float = 0.5
) -> dict:
    t0, tf = t_span
    t = t0
    y = np.asarray(y0, dtype=float)
    dt = dt_init

    t_history = [t]
    y_history = [y.copy()]

    n_steps = 0
    n_rejected = 0
    n_fsolve = 0

    dt_prev = dt
    dt_prev2 = dt


    kappa = 0.9
    min_dt = 1e-10
    max_dt = (tf - t0) / 10.0

    while t < tf:
        dt = min(dt, tf - t)


        def residual(yp):
            y_mid = (1.0 - theta) * y + theta * yp
            return yp - y - dt * f(t + theta * dt, y_mid)

        y_next = fsolve(residual, y, full_output=False)
        n_fsolve += 1


        y_pred = y + dt * f(t, y)
        err = np.linalg.norm(y_next - y_pred)
        y_scale = abstol + reltol * np.maximum(np.abs(y), np.abs(y_next))
        err_max = err / (np.linalg.norm(y_scale) + 1e-14)

        if err_max <= 1.0 or dt <= min_dt:

            t = t + dt
            y = y_next
            t_history.append(t)
            y_history.append(y.copy())
            n_steps += 1


            if err_max > 1e-14:
                dt_new = kappa * (1.0 / err_max) ** (1.0 / 3.0) * dt
            else:
                dt_new = 1.5 * dt
            dt_new = np.clip(dt_new, 0.1 * dt, 1.5 * dt)
            dt_new = np.clip(dt_new, min_dt, max_dt)

            dt_prev2 = dt_prev
            dt_prev = dt
            dt = dt_new
        else:

            n_rejected += 1
            dt = max(0.5 * dt, min_dt)

    return {
        't': np.array(t_history),
        'y': np.array(y_history),
        'n_steps': n_steps,
        'n_rejected': n_rejected,
        'n_fsolve': n_fsolve,
    }


def mean_field_eco_epi_ode(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    S1, I1, R1, S2, I2, R2 = y
    N1 = S1 + I1 + R1
    N2 = S2 + I2 + R2

    K = params.get('K_mean', 100.0)
    r = params.get('r_mean', 1.0)
    beta11 = params.get('beta11', 0.3)
    beta12 = params.get('beta12', 0.1)
    beta21 = params.get('beta21', 0.1)
    beta22 = params.get('beta22', 0.3)
    gamma1 = params.get('gamma1', 0.1)
    gamma2 = params.get('gamma2', 0.1)
    mu1 = params.get('mu1', 0.02)
    mu2 = params.get('mu2', 0.02)
    alpha12 = params.get('alpha12', 0.5)
    alpha21 = params.get('alpha21', 0.5)

    growth1 = r * S1 * (1.0 - (N1 + alpha12 * N2) / K)
    growth2 = r * S2 * (1.0 - (N2 + alpha21 * N1) / K)

    foi1 = beta11 * S1 * I1 + beta12 * S1 * I2
    foi2 = beta21 * S2 * I1 + beta22 * S2 * I2

    dS1 = growth1 - foi1
    dI1 = foi1 - (gamma1 + mu1) * I1
    dR1 = gamma1 * I1
    dS2 = growth2 - foi2
    dI2 = foi2 - (gamma2 + mu2) * I2
    dR2 = gamma2 * I2

    return np.array([dS1, dI1, dR1, dS2, dI2, dR2])
