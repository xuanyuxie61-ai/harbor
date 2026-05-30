
import numpy as np
from scipy.linalg import eig


def build_msm_transition_matrix(n_states, temperature=1.0, seed=None):
    if seed is not None:
        np.random.seed(seed)


    E = np.random.randn(n_states)

    pi = np.exp(-E / temperature)
    pi /= np.sum(pi)


    K = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(i + 1, n_states):
            rate = np.exp(-abs(E[j] - E[i]) / temperature)
            K[i, j] = rate
            K[j, i] = rate


    row_sums = np.sum(K, axis=1)
    T = np.zeros_like(K)
    for i in range(n_states):
        if row_sums[i] > 0:
            T[i, :] = K[i, :] / row_sums[i]
        else:
            T[i, i] = 1.0



    for i in range(n_states):
        T[i, i] += 1e-6
        T[i, :] /= np.sum(T[i, :])

    return T, pi


def msm_eigenvalues_timescales(T, tau=1.0):
    mu, _ = eig(T, left=False, right=True)
    mu = np.sort(mu)[::-1]
    mu_real = np.real(mu)

    timescales = []
    for k in range(1, len(mu_real)):
        val = mu_real[k]
        if val > 0 and val < 1.0:
            t_k = -tau / np.log(val)
            timescales.append(t_k)
        else:
            timescales.append(np.inf)

    return mu_real, np.array(timescales)


def msm_propagate(T, p0, n_steps):
    p = np.array(p0, dtype=float)
    p /= np.sum(p)
    for _ in range(n_steps):
        p = p @ T

        p = np.maximum(p, 0.0)
        s = np.sum(p)
        if s > 0:
            p /= s
        else:
            p = np.ones(len(p0)) / len(p0)
    return p


def msm_mfpt(T, target_state, start_state=None):
    n = T.shape[0]
    if target_state < 0 or target_state >= n:
        raise ValueError("Invalid target state.")


    indices = [i for i in range(n) if i != target_state]
    T_red = T[np.ix_(indices, indices)]
    I = np.eye(len(indices))
    b = np.ones(len(indices))

    try:
        m = np.linalg.solve(I - T_red, b)
    except np.linalg.LinAlgError:
        m = np.linalg.lstsq(I - T_red, b, rcond=None)[0]

    if start_state is None:
        return m
    if start_state == target_state:
        return 0.0
    idx = indices.index(start_state)
    return m[idx]


def optimal_tensor_contraction_path(dims, n_contract):
    from matrix_utils import matrix_chain_optimal_order, reconstruct_optimal_order

    dims_sub = dims[: n_contract + 1]
    min_cost, s = matrix_chain_optimal_order(dims_sub)
    if s.size == 0:
        path = "A0"
    else:
        path = reconstruct_optimal_order(s, 0, n_contract - 1)
    return min_cost, path
