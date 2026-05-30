
import numpy as np


def smoluchowski_1d_sliding_step(P, U, D_s, kT, dt, dx, boundary="reflecting"):
    N = len(P)
    if N < 3:
        return P.copy()

    P = np.asarray(P, dtype=float)
    U = np.asarray(U, dtype=float)


    F_half = np.zeros(N - 1)
    for i in range(N - 1):
        dU = U[i + 1] - U[i]
        F_half[i] = -dU / dx

    P_new = np.zeros(N)
    alpha = D_s * dt / (dx * dx)

    for i in range(N):

        if i == 0:
            P_ip1 = P[i + 1]
            P_im1 = P[i + 1] if boundary == "reflecting" else 0.0
        elif i == N - 1:
            P_ip1 = P[i - 1] if boundary == "reflecting" else 0.0
            P_im1 = P[i - 1]
        else:
            P_ip1 = P[i + 1]
            P_im1 = P[i - 1]

        diff = alpha * (P_ip1 - 2.0 * P[i] + P_im1)


        drift = 0.0
        if i < N - 1:
            drift -= (dt / (2.0 * dx * kT)) * F_half[i] * (P[i + 1] + P[i])
        if i > 0:
            drift += (dt / (2.0 * dx * kT)) * F_half[i - 1] * (P[i] + P[i - 1])

        P_new[i] = P[i] + diff + drift


    P_new = np.maximum(P_new, 0.0)
    total = np.sum(P_new)
    if total > 0:
        P_new /= total
    else:
        P_new = np.ones(N) / N

    return P_new


def porous_medium_step_1d(C, m, dt, dx, source=None, boundary="neumann"):
    N = len(C)
    if N < 3:
        return C.copy()

    C = np.maximum(np.asarray(C, dtype=float), 0.0)
    Cm = C ** m

    C_new = np.zeros(N)
    coeff = dt / (dx * dx)

    for i in range(N):
        if i == 0:
            Cm_im1 = Cm[1] if boundary == "neumann" else 0.0
            Cm_ip1 = Cm[1]
        elif i == N - 1:
            Cm_im1 = Cm[N - 2]
            Cm_ip1 = Cm[N - 2] if boundary == "neumann" else 0.0
        else:
            Cm_im1 = Cm[i - 1]
            Cm_ip1 = Cm[i + 1]

        laplace = Cm_ip1 - 2.0 * Cm[i] + Cm_im1
        C_new[i] = C[i] + coeff * laplace

    if source is not None:
        C_new += dt * np.asarray(source)


    C_new = np.maximum(C_new, 0.0)
    return C_new


def porous_medium_barenblatt_solution(x, t, m, C0=1.0, delta=0.01):
    alpha = 1.0 / (m - 1.0)
    beta = 1.0 / (m + 1.0)
    gamma = (m - 1.0) / (2.0 * m * (m + 1.0))

    bot = (t + delta) ** beta
    A = C0
    factor = A - gamma * (x / bot) ** 2
    C = np.zeros_like(x, dtype=float)
    mask = factor > 0

    with np.errstate(invalid="ignore"):
        C[mask] = (t + delta) ** (-beta) * factor[mask] ** alpha
    return C


def normal_mode_relaxation(P0, D, k, t):
    return P0 * np.exp(-D * k * k * t)


def sliding_search_time(dna_length, D_s, target_size):
    if dna_length <= 0 or D_s <= 0:
        return float("inf")
    a = target_size / dna_length
    if a >= 1.0:
        return 0.0
    mean_time = (dna_length ** 2) / (2.0 * D_s) * (1.0 - a) ** 2
    return mean_time
