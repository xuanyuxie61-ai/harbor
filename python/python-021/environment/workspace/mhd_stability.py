
import numpy as np
from parameters import R0, a_minor, B0, q0, q_edge, MU0






def build_mhd_transition_matrix(p_stable=0.85, p_elm=0.08,
                                 p_tearing=0.04, p_rwm=0.02,
                                 p_disruption=0.01):
    labels = [
        "Confined", "ELM", "Tearing", "RWM",
        "Pre-Disruption", "Disruption", "VDE", "Recovery"
    ]

    P = np.zeros((8, 8))


    P[0, 0] = p_stable
    P[1, 0] = p_elm
    P[2, 0] = p_tearing
    P[3, 0] = p_rwm
    P[4, 0] = p_disruption


    P[0, 1] = 0.9
    P[1, 1] = 0.05
    P[4, 1] = 0.05


    P[0, 2] = 0.6
    P[2, 2] = 0.2
    P[4, 2] = 0.15
    P[5, 2] = 0.05


    P[0, 3] = 0.5
    P[3, 3] = 0.2
    P[4, 3] = 0.2
    P[5, 3] = 0.1


    P[4, 4] = 0.3
    P[5, 4] = 0.4
    P[6, 4] = 0.2
    P[7, 4] = 0.1


    P[5, 5] = 1.0


    P[5, 6] = 0.7
    P[6, 6] = 0.2
    P[7, 6] = 0.1


    P[0, 7] = 0.8
    P[7, 7] = 0.2


    col_sums = P.sum(axis=0)
    col_sums = np.where(col_sums < 1e-15, 1.0, col_sums)
    P = P / col_sums

    return P, labels


def mhd_markov_evolution(P, initial_state, n_steps=100):
    n_states = P.shape[0]
    state = np.asarray(initial_state, dtype=float)
    state /= (state.sum() + 1e-30)

    history = np.zeros((n_steps + 1, n_states))
    history[0, :] = state

    absorption_step = None
    for step in range(n_steps):
        state = P @ state
        history[step + 1, :] = state
        if absorption_step is None and state[5] > 0.5:
            absorption_step = step + 1

    if absorption_step is not None:
        absorption_time = float(absorption_step)
    else:

        Q = np.delete(np.delete(P, 5, axis=0), 5, axis=1)
        I = np.eye(n_states - 1)
        try:
            N_fund = np.linalg.inv(I - Q)
            t_expect = N_fund.sum(axis=1)
            absorption_time = float(t_expect[0])
        except np.linalg.LinAlgError:
            absorption_time = float(n_steps)

    return history, absorption_time


def compute_ideal_mhd_delta_w(m_mode, n_mode, q_profile, r_grid, p_profile,
                               B_theta, B_phi, R=R0, gamma=5.0 / 3.0):
    r = np.asarray(r_grid)
    q = np.asarray(q_profile)
    p = np.asarray(p_profile)
    Bt = np.asarray(B_theta)

    if len(r) < 3:
        return 0.0, "stable"

    dr = r[1] - r[0] if len(r) > 1 else 1.0
    dq_dr = np.gradient(q, dr)
    dp_dr = np.gradient(p, dr)


    resonant_idx = np.argmin(np.abs(m_mode - n_mode * q))


    m_minus_nq = m_mode - n_mode * q
    f = (r ** 3 / (R ** 2)) * (Bt ** 2) * (m_minus_nq ** 2)
    g = ((2.0 * r / (R ** 2)) * (Bt ** 2) * m_minus_nq *
         (m_minus_nq - (r / (q + 1e-20)) * dq_dr))
    g += (2.0 * MU0 * r * dp_dr * (m_mode ** 2 - 1.0) / (m_mode ** 2 + 1e-20))


    a = r[-1]
    xi = 1.0 - (r / a) ** 2
    dxi_dr = -2.0 * r / (a ** 2)

    integrand = f * (dxi_dr ** 2) + g * (xi ** 2)
    delta_w = np.pi * np.trapezoid(integrand, r)

    stability = "unstable" if delta_w < 0 else "stable"
    return float(delta_w), stability


def compute_mercier_criterion(q_profile, r_grid, p_profile, B_phi, B_theta, R=R0):
    from parameters import MU0
    r = np.asarray(r_grid)
    q = np.asarray(q_profile)
    p = np.asarray(p_profile)
    Bp = np.asarray(B_phi)
    Bt = np.asarray(B_theta)

    dr = r[1] - r[0] if len(r) > 1 else 1.0
    dq_dr = np.gradient(q, dr)
    dp_dr = np.gradient(p, dr)


    term1 = (r ** 4 / (4.0 * R ** 2 * (q ** 4) + 1e-30)) * (1.0 - q ** 2) ** 2
    term2 = ((2.0 * MU0 * R ** 2 * (q ** 2)) / (Bp ** 2 + 1e-30)) * dp_dr * (1.0 - 1.0 / (q ** 2 + 1e-30))
    D_M = ((q / (r + 1e-20)) ** 2) * (term1 - term2)


    unstable = []
    in_unstable = False
    r_start = None
    for i in range(len(r)):
        if D_M[i] < 0:
            if not in_unstable:
                r_start = r[i]
                in_unstable = True
        else:
            if in_unstable:
                unstable.append((float(r_start), float(r[i])))
                in_unstable = False
    if in_unstable:
        unstable.append((float(r_start), float(r[-1])))

    return D_M, unstable


def compute_critical_beta(q_profile, r_grid, B_phi, R=R0, a=a_minor):
    epsilon = a / R
    q_edge_val = q_profile[-1] if len(q_profile) > 0 else 3.0
    beta_c = 3.5 * epsilon / (q_edge_val + 1e-10)
    return float(beta_c * 100.0)
