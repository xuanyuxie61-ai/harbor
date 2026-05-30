
import numpy as np


def subset_sum_table(target, weights):
    n = weights.size
    table = np.zeros(target, dtype=int)

    for i in range(n):
        w = int(weights[i])
        if w <= 0 or w > target:
            continue
        for j in range(target - w, -1, -1):
            if j == 0:
                if table[w - 1] == 0:
                    table[w - 1] = w
            elif table[j - 1] != 0 and table[j + w - 1] == 0:
                table[j + w - 1] = w

    return table


def subset_sum_find(target, weights):
    table = subset_sum_table(target, weights)
    if target <= 0 or target > table.size or table[target - 1] == 0:
        return None

    subset = []
    remaining = target
    while remaining > 0:
        w = table[remaining - 1]
        if w == 0:
            return None
        subset.append(w)
        remaining -= w
        if remaining < 0:
            return None

    return subset


def discrete_ic_cascade(epsilon_0, gamma_discrete, n_scatter_max=5):

    log_factors = np.round(100.0 * np.log(gamma_discrete ** 2)).astype(int)
    log_factors = np.clip(log_factors, 1, None)

    energies_list = []
    current_log = np.round(100.0 * np.log(epsilon_0)).astype(int)

    for _ in range(n_scatter_max):

        target_max = current_log + int(np.sum(log_factors)) + 1
        target_max = min(target_max, 50000)
        table = subset_sum_table(target_max, log_factors)

        achievable = []
        for s in range(target_max):
            if table[s] != 0:
                achievable.append(current_log + s + 1)

        if len(achievable) == 0:
            break

        energies = np.exp(np.array(achievable, dtype=float) / 100.0)
        energies_list.append(energies)
        current_log = int(np.median(achievable))

    return energies_list


def cascade_compactness(L, R):
    sigma_T = 6.6524587158e-25
    m_e = 9.10938356e-28
    c = 2.99792458e10
    ell = (L * sigma_T) / (R * m_e * c ** 3)
    return ell
