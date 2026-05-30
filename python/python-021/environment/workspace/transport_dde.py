
import numpy as np
from parameters import get_transport_params


def interpolate_history(t_query, t_hist, y_hist):
    if t_query <= t_hist[0]:
        return float(y_hist[0])
    if t_query >= t_hist[-1]:
        return float(y_hist[-1])

    idx = np.searchsorted(t_hist, t_query)
    if idx == 0:
        return float(y_hist[0])
    t1, t2 = t_hist[idx - 1], t_hist[idx]
    y1, y2 = y_hist[idx - 1], y_hist[idx]
    if abs(t2 - t1) < 1e-15:
        return float(y1)
    return float(y1 + (y2 - y1) * (t_query - t1) / (t2 - t1))


def transport_dde_rhs(t, W, W_delayed, gamma, beta, n, W0,
                       P_heat, tau_E, P_loss_coeff):
    W = float(W)
    W_delayed = float(W_delayed)
    if W_delayed < 0.0:
        W_delayed = 0.0


    denom = W0 ** n + W_delayed ** n
    if denom < 1e-30:
        feedback = 0.0
    else:
        feedback = beta * (W_delayed ** n) / denom

    dWdt = P_heat * feedback - gamma * W / tau_E - P_loss_coeff * W
    return dWdt


def rk4_dde_step(t, W, h, tau, gamma, beta, n, W0,
                 P_heat, tau_E, P_loss_coeff, t_hist, W_hist):
    def rhs_now(tn, Wn, Wd):
        return transport_dde_rhs(tn, Wn, Wd, gamma, beta, n, W0,
                                 P_heat, tau_E, P_loss_coeff)

    Wd1 = interpolate_history(t - tau, t_hist, W_hist)
    k1 = h * rhs_now(t, W, Wd1)

    Wd2 = interpolate_history(t + 0.5 * h - tau, t_hist, W_hist)
    k2 = h * rhs_now(t + 0.5 * h, W + 0.5 * k1, Wd2)

    Wd3 = interpolate_history(t + 0.5 * h - tau, t_hist, W_hist)
    k3 = h * rhs_now(t + 0.5 * h, W + 0.5 * k2, Wd3)

    Wd4 = interpolate_history(t + h - tau, t_hist, W_hist)
    k4 = h * rhs_now(t + h, W + k3, Wd4)

    return W + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0


def simulate_energy_transport(transport_params=None, n_steps=2000):
    if transport_params is None:
        transport_params = get_transport_params()

    gamma = transport_params["gamma"]
    beta = transport_params["beta"]
    n = transport_params["n"]
    tau = transport_params["tau"]
    t0 = transport_params["t0"]
    y0 = transport_params["y0"]
    tstop = transport_params["tstop"]


    W0 = 5.0e4
    P_heat = 2.0e5
    tau_E = 1.5
    P_loss_coeff = 0.05

    h = (tstop - t0) / n_steps
    if h <= 0:
        raise ValueError("步长必须为正")


    n_hist = max(int(np.ceil(tau / h)) + 10, 100)
    t_hist = np.linspace(t0 - tau, t0, n_hist)
    W_hist = np.full(n_hist, float(y0[0]))

    t_arr = np.zeros(n_steps + 1)
    W_arr = np.zeros(n_steps + 1)
    t_arr[0] = t0
    W_arr[0] = float(y0[0])

    for step in range(n_steps):
        t_curr = t_arr[step]
        W_curr = W_arr[step]

        W_next = rk4_dde_step(
            t_curr, W_curr, h, tau, gamma, beta, n, W0,
            P_heat, tau_E, P_loss_coeff, t_hist, W_hist
        )


        W_next = max(W_next, 0.0)

        W_arr[step + 1] = W_next
        t_arr[step + 1] = t_curr + h


        t_hist = np.append(t_hist[1:], t_curr + h)
        W_hist = np.append(W_hist[1:], W_next)

    P_loss_arr = P_loss_coeff * W_arr



    if len(W_arr) > 100:
        dW = np.diff(W_arr)
        lyap_approx = np.mean(np.log(np.abs(dW[1:] / (dW[:-1] + 1e-20)) + 1e-20))
    else:
        lyap_approx = 0.0

    info = {
        "delay_tau": tau,
        "mackey_glass_n": n,
        "lyapunov_approx": lyap_approx,
        "mean_energy_density": float(np.mean(W_arr)),
        "max_energy_density": float(np.max(W_arr)),
    }
    return t_arr, W_arr, P_loss_arr, info


def compute_confinement_time_scaling(I_p, B_t, n_e20, P_loss, R, a, kappa, M=2.5):
    if P_loss <= 0:
        P_loss = 1e-6
    tau_E = (0.048 * (I_p ** 0.85) * (B_t ** 0.2) *
             (n_e20 ** 0.1) * (P_loss ** (-0.5)) *
             (R ** 1.5) * (a ** 0.3) * (kappa ** 0.5) * (M ** 0.5))
    return tau_E


def compute_particle_diffusivity(q, R0, a, nu_ei, rho_i):
    epsilon = a / (R0 + 1e-20)
    epsilon_safe = max(epsilon, 1e-6)
    return (q ** 2) * nu_ei * (rho_i ** 2) / (epsilon_safe ** 1.5)
