
import numpy as np






def knapsack_brute_force(values, weights, capacity):
    n = len(values)
    vmax = 0.0
    wmax = 0.0
    smax = np.zeros(n, dtype=int)


    total_subsets = 1 << n
    for s in range(total_subsets):
        selection = np.array([(s >> i) & 1 for i in range(n)], dtype=int)
        w_test = np.dot(selection, weights)
        if w_test <= capacity:
            v_test = np.dot(selection, values)
            if v_test > vmax:
                vmax = v_test
                wmax = w_test
                smax = selection.copy()

    return vmax, wmax, smax


def subset_next(s):
    s = np.asarray(s, dtype=int).copy()
    n = len(s)
    for i in range(n):
        if s[i] == 0:
            s[i] = 1
            return s
        else:
            s[i] = 0
    return s






def diophantine_bounded_solutions(a, b, m):
    a = np.asarray(a, dtype=int)
    m = np.asarray(m, dtype=int)
    n = len(a)
    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0

    while True:
        r = b - np.dot(a[:j], y[:j])
        if j < n:
            j += 1
            y[j - 1] = min(r // a[j - 1], m[j - 1])
        else:
            if r == 0:
                solutions.append(y.copy())

            while j > 0:
                if y[j - 1] > 0:
                    y[j - 1] -= 1
                    break
                j -= 1
            if j == 0:
                break

    if not solutions:
        return np.empty((0, n), dtype=int)
    return np.array(solutions, dtype=int)






def catalyst_value_per_segment(W_cat, T_segment, Q_gas,
                                k0=1.2e-3, Ea=85000.0, R=8.314,
                                yield_heavy=0.75, rho_wax=780.0):
    W_cat = np.asarray(W_cat, dtype=float)
    T_segment = np.asarray(T_segment, dtype=float)


    k = k0 * np.exp(-Ea / (R * T_segment))


    X = 1.0 - np.exp(-k * W_cat / max(Q_gas, 1e-12))
    X = np.clip(X, 0.0, 0.999)


    values = X * yield_heavy * rho_wax * W_cat
    weights = W_cat.copy()
    return values, weights


def optimize_catalyst_loading(W_total, n_segments, T_profile,
                              Q_gas=0.01, method='brute_force'):
    if n_segments <= 0:
        raise ValueError("n_segments must be positive")


    n_candidates = min(n_segments * 2, 20)
    W_candidates = np.linspace(0, W_total, n_candidates)

    if method == 'brute_force':
        values, weights = catalyst_value_per_segment(
            W_candidates, np.interp(np.linspace(0, 1, n_candidates),
                                    np.linspace(0, 1, len(T_profile)), T_profile),
            Q_gas)
        vmax, wmax, smax = knapsack_brute_force(values, weights, W_total)
        selected_W = W_candidates[smax == 1]
        return {
            'method': 'brute_force',
            'max_value': vmax,
            'total_weight': wmax,
            'selection': smax,
            'selected_weights': selected_W,
        }

    elif method == 'diophantine':

        m_particle = 0.1
        N_total = int(W_total / m_particle)
        a = np.ones(n_segments, dtype=int)
        m = np.full(n_segments, N_total, dtype=int)
        solutions = diophantine_bounded_solutions(a, N_total, m)

        if solutions.shape[0] == 0:
            return {
                'method': 'diophantine',
                'max_value': 0.0,
                'best_solution': np.zeros(n_segments, dtype=int),
            }

        best_val = -1.0
        best_sol = solutions[0]
        for sol in solutions:
            W_seg = sol * m_particle
            vals, _ = catalyst_value_per_segment(W_seg, T_profile, Q_gas)
            total_val = np.sum(vals)
            if total_val > best_val:
                best_val = total_val
                best_sol = sol.copy()

        return {
            'method': 'diophantine',
            'max_value': best_val,
            'best_solution': best_sol,
            'particle_mass': m_particle,
        }
    else:
        raise ValueError(f"Unknown method: {method}")
