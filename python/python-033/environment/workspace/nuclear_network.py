
import numpy as np
from circulant_solver import circulant_solve


def build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx=0):
    n_nuc = len(nuclides)
    J = np.zeros((n_nuc, n_nuc), dtype=float)
    S = np.zeros(n_nuc, dtype=float)

    N_A = 6.02214076e23

    for i, (z_i, n_i, a_i) in enumerate(nuclides):
        key_i = (z_i, a_i)


        outflow = 0.0


        cap_rate = rates['capture'].get(key_i, 0.0)
        if np.ndim(cap_rate) == 0:
            cap = float(cap_rate)
        else:


            cap = 0.0
        outflow += n_n * cap


        phot_rate = rates['photodis'].get(key_i, 0.0)
        if np.ndim(phot_rate) == 0:
            phot = float(phot_rate)
        else:


            phot = 0.0
        outflow += phot


        beta = float(rates['beta'].get(key_i, 0.0))
        outflow += beta


        alpha = float(rates['alpha'].get(key_i, 0.0))
        outflow += alpha


        fiss = float(rates['fission'].get(key_i, 0.0))
        outflow += fiss

        J[i, i] -= outflow


        for j, (z_j, n_j, a_j) in enumerate(nuclides):
            if i == j:
                continue
            key_j = (z_j, a_j)

            inflow = 0.0


            if z_j == z_i + 1 and n_j == n_i - 1 and a_j == a_i:
                inflow += float(rates['beta'].get(key_j, 0.0))


            if z_j == z_i and n_j == n_i - 1 and a_j == a_i - 1:
                cap_j = rates['capture'].get(key_j, 0.0)
                if np.ndim(cap_j) == 0:
                    cap_j = float(cap_j)
                else:

                    cap_j = 0.0
                inflow += n_n * cap_j


            if z_j == z_i and n_j == n_i + 1 and a_j == a_i + 1:
                phot_j = rates['photodis'].get(key_j, 0.0)
                if np.ndim(phot_j) == 0:
                    phot_j = float(phot_j)
                else:

                    phot_j = 0.0
                inflow += phot_j


            if z_j == z_i + 2 and n_j == n_i + 2 and a_j == a_i + 4:
                inflow += float(rates['alpha'].get(key_j, 0.0))


            if a_j > 220 and a_i < a_j and a_i > 80:
                fiss_j = float(rates['fission'].get(key_j, 0.0))

                inflow += fiss_j * 0.1

            if inflow > 0:
                J[i, j] += inflow



        S[i] = 0.0

    return J, S


def solve_network_implicit_euler(nuclides, rates, rho, n_n, Y0, t_end, n_steps=1000, temp_profile=None):
    n_nuc = len(nuclides)
    Y = np.asarray(Y0, dtype=float).copy()
    dt = t_end / n_steps

    t_history = np.zeros(n_steps + 1)
    Y_history = np.zeros((n_steps + 1, n_nuc))
    t_history[0] = 0.0
    Y_history[0] = Y

    for step in range(n_steps):
        temp_idx = temp_profile[step] if temp_profile is not None else 0
        J, S = build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx)
        M = np.eye(n_nuc) - dt * J


        rhs = Y + dt * S

        try:
            Y_new = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:

            Y_new = np.linalg.lstsq(M, rhs, rcond=None)[0]


        Y_new = np.maximum(Y_new, 0.0)

        total = np.sum(Y_new)
        if total > 0:
            Y_new = Y_new / total

        Y = Y_new
        t_history[step + 1] = (step + 1) * dt
        Y_history[step + 1] = Y

    return t_history, Y_history


def solve_network_bdf2(nuclides, rates, rho, n_n, Y0, t_end, n_steps=500, temp_profile=None):
    n_nuc = len(nuclides)
    Y = np.asarray(Y0, dtype=float).copy()
    dt = t_end / n_steps

    t_history = np.zeros(n_steps + 1)
    Y_history = np.zeros((n_steps + 1, n_nuc))
    t_history[0] = 0.0
    Y_history[0] = Y


    temp_idx = temp_profile[0] if temp_profile is not None else 0
    J, S = build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx)
    M = np.eye(n_nuc) - dt * J
    rhs = Y + dt * S
    try:
        Y_prev = Y.copy()
        Y = np.linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        Y_prev = Y.copy()
        Y = np.linalg.lstsq(M, rhs, rcond=None)[0]
    Y = np.maximum(Y, 0.0)
    total = np.sum(Y)
    if total > 0:
        Y = Y / total
    Y_history[1] = Y
    t_history[1] = dt

    for step in range(1, n_steps):


        temp_idx = 0
        J, S = build_reaction_matrix(nuclides, rates, rho, n_n, Y, temp_idx)
        M = 3.0 * np.eye(n_nuc) - 2.0 * dt * J
        rhs = 4.0 * Y - Y_prev + 2.0 * dt * S

        try:
            Y_new = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            Y_new = np.linalg.lstsq(M, rhs, rcond=None)[0]

        Y_new = np.maximum(Y_new, 0.0)
        total = np.sum(Y_new)
        if total > 0:
            Y_new = Y_new / total

        Y_prev = Y.copy()
        Y = Y_new
        t_history[step + 1] = (step + 1) * dt
        Y_history[step + 1] = Y

    return t_history, Y_history


def compute_abundance_peaks(Y_final, nuclides, A_bins=None):
    if A_bins is None:
        A_bins = np.arange(70, 251, 5)
    A_centers = 0.5 * (A_bins[:-1] + A_bins[1:])
    abundances = np.zeros(len(A_centers))

    for i, (z, n, a) in enumerate(nuclides):
        idx = np.searchsorted(A_bins, a) - 1
        idx = np.clip(idx, 0, len(A_centers) - 1)
        abundances[idx] += Y_final[i]

    return A_centers, abundances


def test_nuclear_network():
    from reaction_rates import build_reaction_rate_table
    nuclides = [(26, 30, 56), (26, 31, 57), (27, 30, 57), (27, 31, 58), (28, 30, 58)]
    T9_range = np.array([1.0, 1.5, 2.0])
    S_n_table = {(26, 56): 8.0, (26, 57): 7.5, (27, 57): 8.2, (27, 58): 7.8, (28, 58): 8.5}
    T_half_table = {(26, 56): 1e10, (26, 57): 1.5, (27, 57): 272.0, (27, 58): 70.8, (28, 58): 1e10}

    rates = build_reaction_rate_table(nuclides, T9_range, S_n_table, T_half_table)
    Y0 = np.ones(len(nuclides)) / len(nuclides)
    t, Y_hist = solve_network_implicit_euler(nuclides, rates, rho=1e8, n_n=1e30,
                                              Y0=Y0, t_end=10.0, n_steps=100)
    print(f"[nuclear_network] Final abundances: {Y_hist[-1]}")
    print(f"[nuclear_network] Sum of abundances: {np.sum(Y_hist[-1]):.6f}")
    assert np.sum(Y_hist[-1]) > 0.99, "Total abundance conservation violated"


if __name__ == "__main__":
    test_nuclear_network()
