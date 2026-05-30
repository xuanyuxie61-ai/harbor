
import numpy as np


def subset_sum_swap_anc(candidate_powers, desired_power_budget):
    n = len(candidate_powers)
    a = np.array(candidate_powers, dtype=float)
    sum_desired = float(desired_power_budget)


    order = np.argsort(-a)
    a_sorted = a[order]

    selected = np.zeros(n, dtype=bool)
    sum_achieved = 0.0

    while True:
        nmove = 0
        for idx in range(n):
            i = order[idx]
            if not selected[i]:
                if sum_achieved + a_sorted[idx] <= sum_desired + 1e-9:
                    selected[i] = True
                    sum_achieved += a_sorted[idx]
                    nmove += 1
                    continue

            if not selected[i]:

                for jdx in range(n):
                    j = order[jdx]
                    if selected[j]:
                        new_sum = sum_achieved + a_sorted[idx] - a_sorted[jdx]
                        if sum_achieved < new_sum <= sum_desired + 1e-9:
                            selected[j] = False
                            selected[i] = True
                            sum_achieved = new_sum
                            nmove += 2
                            break

        if nmove == 0:
            break

    return selected, sum_achieved


def greedy_source_selection(H, d, max_sources, power_budget, source_powers):
    M, N = H.shape
    H = np.asarray(H, dtype=complex)
    d = np.asarray(d, dtype=complex)

    selected = np.zeros(N, dtype=bool)
    total_power = 0.0


    for _ in range(min(max_sources, N)):
        best_j = -1
        best_cost = np.inf
        best_s = None

        for j in range(N):
            if selected[j]:
                continue
            if total_power + source_powers[j] > power_budget + 1e-9:
                continue


            temp_sel = selected.copy()
            temp_sel[j] = True
            H_sub = H[:, temp_sel]
            try:
                s = np.linalg.lstsq(H_sub, -d, rcond=None)[0]
                residual = d + H_sub @ s
                cost = np.vdot(residual, residual).real
            except np.linalg.LinAlgError:
                cost = np.inf

            if cost < best_cost:
                best_cost = cost
                best_j = j
                best_s = s

        if best_j < 0:
            break

        selected[best_j] = True
        total_power += source_powers[best_j]


    if np.any(selected):
        H_sel = H[:, selected]
        try:
            filters = np.linalg.lstsq(H_sel, -d, rcond=None)[0]
        except np.linalg.LinAlgError:
            filters = np.zeros(np.sum(selected), dtype=complex)
    else:
        filters = np.array([], dtype=complex)

    return selected, filters
