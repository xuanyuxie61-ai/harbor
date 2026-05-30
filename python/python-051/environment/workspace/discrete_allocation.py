
import numpy as np






def diophantine_nd_nonnegative(a, b):
    a = np.asarray(a).flatten()
    n = len(a)
    if b < 0:
        return np.array([]).reshape(0, n)
    if np.any(a <= 0):
        raise ValueError("系数 a 必须全为正整数")

    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0

    while True:
        r = b - np.dot(a[:j], y[:j])
        if j < n:
            y[j] = r // a[j]
            j += 1
        else:
            if r == 0:
                solutions.append(y.copy())

            while j > 0:
                j -= 1
                if y[j] > 0:
                    y[j] -= 1
                    j += 1
                    break
            else:
                break

    if len(solutions) == 0:
        return np.array([]).reshape(0, n)
    return np.array(solutions)


def allocate_nutrient_budget(budget, demand_coeffs, objective="min_variance"):
    sols = diophantine_nd_nonnegative(demand_coeffs, budget)
    if sols.shape[0] == 0:
        return np.zeros(len(demand_coeffs), dtype=int)

    if objective == "min_variance":

        variances = np.var(sols, axis=1)
        idx = np.argmin(variances)
    elif objective == "max_even":

        mins = np.min(sols, axis=1)
        idx = np.argmax(mins)
    else:
        idx = 0

    return sols[idx]






def dictionary_encode(state_vectors, tol=1e-8):
    n_snapshots, n_features = state_vectors.shape


    dictionary = []
    indices = np.zeros(n_snapshots, dtype=int)

    for i in range(n_snapshots):
        vec = state_vectors[i]
        found = False
        for j, ref in enumerate(dictionary):
            if np.linalg.norm(vec - ref) <= tol * max(1.0, np.linalg.norm(ref)):
                indices[i] = j
                found = True
                break
        if not found:
            dictionary.append(vec.copy())
            indices[i] = len(dictionary) - 1

    dictionary = np.array(dictionary)
    n_unique = dictionary.shape[0]


    original_size = n_snapshots * n_features
    encoded_size = n_unique * n_features + n_snapshots * np.log2(max(n_unique, 2)) / 8.0
    compression_ratio = original_size / max(encoded_size, 1.0)

    return dictionary, indices, compression_ratio


def dictionary_decode(dictionary, indices):
    return dictionary[indices]






def snapshot_matrix(ocean_fields):
    parts = []
    for key in sorted(ocean_fields.keys()):
        parts.append(ocean_fields[key].ravel())
    return np.concatenate(parts)


def pack_snapshots(snapshots_list):
    return np.vstack(snapshots_list)
