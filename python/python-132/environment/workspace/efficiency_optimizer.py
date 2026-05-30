
import numpy as np
from utils import ensure_positive, clip_with_warning






def fermat_optimize_trays(N_min, R_min, target_cost_func, N_max=200, tol=1e-4):
    N_min = max(int(N_min), 2)
    N_max = max(int(N_max), N_min + 1)
    R_min = max(float(R_min), 1.01)


    N_start = int(np.sqrt(N_max * N_min))
    N_start = clip_with_warning(N_start, N_min, N_max, "N_start")

    C_min = float('inf')
    N_opt = N_min
    R_opt = R_min
    history = []

    for N in range(N_min, N_max + 1):


        R_start = max(R_min, np.sqrt(N) * 0.1)
        R_end = max(R_start + 5.0, R_min * 10.0)
        R_grid = np.linspace(R_start, R_end, 50)

        for R in R_grid:
            C = target_cost_func(N, R)
            history.append((N, R, C))
            if C < C_min:
                C_min = C
                N_opt = N
                R_opt = R

    return N_opt, R_opt, C_min, history






def gilliland_correlation(R, R_min, N, N_min):
    R = max(R, 1.001)
    R_min = max(R_min, 1.0001)
    N = max(int(N), 2)
    N_min = max(int(N_min), 1)

    X = (R - R_min) / (R + 1.0)
    X = max(X, 1e-6)

    Y_calc = 1.0 - np.exp(
        (1.0 + 54.4 * X) * (X - 1.0) / (11.0 + 117.2 * X) * np.sqrt(X)
    )
    Y_actual = (N - N_min) / (N + 1.0)

    return Y_actual - Y_calc


def estimate_N_from_R(R, R_min, N_min):
    R = max(R, 1.001)
    R_min = max(R_min, 1.0001)
    N_min = max(int(N_min), 1)

    X = (R - R_min) / (R + 1.0)
    X = max(X, 1e-6)

    Y = 1.0 - np.exp(
        (1.0 + 54.4 * X) * (X - 1.0) / (11.0 + 117.2 * X) * np.sqrt(X)
    )

    N = int(np.ceil((Y * (N_min + 1.0) + N_min) / (1.0 - Y)))
    return max(N, N_min + 1)






def reboiler_duty(R, D, q_cond, lambda_vap, feed_rate, z_F, x_D, x_B):
    R = max(R, 0.0)
    D = max(D, 1e-12)
    V = (R + 1.0) * D
    Q_R = V * lambda_vap + q_cond
    return Q_R


def total_cost_model(N, R, D, q_cond, lambda_vap, feed_rate,
                     z_F, x_D, x_B, c_steam, t_op, a_cap, b_cap, column_diameter):
    Q_R = reboiler_duty(R, D, q_cond, lambda_vap, feed_rate, z_F, x_D, x_B)


    C_op = c_steam * (Q_R / 2.8e6) * t_op


    C_cap = (a_cap + b_cap * (N ** 0.8) * (column_diameter ** 1.5)) / 10.0

    return C_op + C_cap


def optimize_distillation_cost(N_min, R_min, D, q_cond, lambda_vap,
                                feed_rate, z_F, x_D, x_B,
                                c_steam, t_op, a_cap, b_cap, column_diameter):
    def cost_func(N, R):
        return total_cost_model(
            int(N), R, D, q_cond, lambda_vap, feed_rate,
            z_F, x_D, x_B, c_steam, t_op, a_cap, b_cap, column_diameter
        )

    N_opt, R_opt, C_min, history = fermat_optimize_trays(
        N_min, R_min, cost_func, N_max=max(N_min + 100, 200)
    )

    return N_opt, R_opt, C_min, history
