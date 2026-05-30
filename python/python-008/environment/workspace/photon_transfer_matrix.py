
import numpy as np


def incidence_to_transition(A):
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    s = np.sum(A, axis=0)
    T = np.zeros((n, n), dtype=float)
    for i in range(n):
        if s[i] > 0.0:
            T[:, i] = A[:, i] / s[i]
        else:
            T[i, i] = 1.0
    return T


def power_rank(A, max_iter=200, tol=1e-12):
    T = incidence_to_transition(A)
    n = T.shape[0]
    x = np.ones(n, dtype=float) / n

    for it in range(max_iter):
        x_new = T @ x


        norm = np.sum(x_new)
        if norm > 0.0:
            x_new = x_new / norm

        if np.linalg.norm(x_new - x, ord=1) < tol:
            x = x_new
            break
        x = x_new

    return x, it + 1


def build_compton_transfer_matrix(n_bins, T_e, tau_es, p_scatter=0.9):
    k_B = 1.380649e-16
    m_e_c2 = 8.18710565e-7
    eps = 4.0 * k_B * T_e / m_e_c2
    eps = max(eps, 1e-6)

    n = n_bins + 1
    A = np.zeros((n, n), dtype=float)

    for i in range(n_bins):

        for j in range(i + 1, n_bins):

            delta = j - i
            prob = np.exp(-delta / (eps * (i + 1))) / (delta + 1.0) ** 2
            A[j, i] = prob


        for j in range(max(0, i - 3), i):
            A[j, i] = 0.1 / (i - j + 1.0)


        A[i, i] = 0.2


        A[n_bins, i] = (1.0 - p_scatter) * np.exp(-tau_es)


    A[n_bins, n_bins] = 1.0


    for i in range(n_bins):
        col_sum = np.sum(A[:, i])
        if col_sum > 0.0:
            A[:, i] /= col_sum

    return A


def compute_photon_stats(A):
    n_ss, iters = power_rank(A)
    n_bins = A.shape[0] - 1


    p_escape = A[n_bins, :n_bins]
    mean_scat = np.sum(n_ss[:n_bins]) / (np.sum(p_escape * n_ss[:n_bins]) + 1e-30)


    y_param = np.sum(n_ss[:n_bins] * np.arange(1, n_bins + 1))

    return {
        "steady_state": n_ss,
        "mean_scatterings": mean_scat,
        "y_param": y_param,
        "iterations": iters,
    }
