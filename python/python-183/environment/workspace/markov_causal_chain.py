
import numpy as np
from typing import Tuple, List, Optional


def build_causal_markov_chain(p: int,
                               causal_edges: List[Tuple[int, int, float]],
                               n_states_per_var: int = 3,
                               absorption_threshold: float = 0.9) -> Tuple[np.ndarray, List[int], List[int]]:
    if p <= 0:
        raise ValueError("变量数 p 必须为正。")
    if not (0.0 < absorption_threshold <= 1.0):
        raise ValueError("absorption_threshold 必须在 (0,1] 内。")

    n_total = p * n_states_per_var
    P = np.zeros((n_total, n_total))


    for var in range(p):
        base = var * n_states_per_var
        for s in range(n_states_per_var):
            idx = base + s
            probs = np.zeros(n_states_per_var)
            if s == 0:
                probs[0] = 0.5
                probs[1] = 0.5
            elif s == n_states_per_var - 1:
                probs[s] = 0.6
                probs[s - 1] = 0.4
            else:
                probs[s - 1] = 0.25
                probs[s] = 0.5
                probs[s + 1] = 0.25

            probs = probs / np.sum(probs)
            for sp in range(n_states_per_var):
                P[idx, base + sp] = probs[sp]


    for i, j, w in causal_edges:
        if i == j:
            continue
        base_i = i * n_states_per_var
        base_j = j * n_states_per_var
        for si in range(n_states_per_var):
            idx_i = base_i + si

            boost = float(si) / max(n_states_per_var - 1, 1) * abs(w)
            for sj in range(n_states_per_var):
                idx_j_from = base_j + sj

                if sj < n_states_per_var - 1:
                    P[idx_j_from, base_j + min(sj + 1, n_states_per_var - 1)] += boost * 0.1
                if sj > 0:
                    P[idx_j_from, base_j + max(sj - 1, 0)] -= boost * 0.05


    P = np.maximum(P, 0.0)
    row_sums = P.sum(axis=1)
    row_sums[row_sums == 0.0] = 1.0
    P = P / row_sums[:, np.newaxis]



    absorbing_states = []
    transient_states = []
    for var in range(p):
        base = var * n_states_per_var

        absorbing_states.append(base + n_states_per_var - 1)
        for s in range(n_states_per_var - 1):
            transient_states.append(base + s)


    for ab in absorbing_states:
        P[ab, :] = 0.0
        P[ab, ab] = 1.0

    return P, transient_states, absorbing_states


def canonical_form(P: np.ndarray,
                   transient_states: List[int],
                   absorbing_states: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t_states = list(transient_states)
    a_states = list(absorbing_states)
    n_t = len(t_states)
    n_a = len(a_states)
    state_map = np.array(t_states + a_states)

    P_canon = np.zeros_like(P)
    for new_i, old_i in enumerate(state_map):
        for new_j, old_j in enumerate(state_map):
            P_canon[new_i, new_j] = P[old_i, old_j]

    Q = P_canon[:n_t, :n_t]
    R = P_canon[:n_t, n_t:]
    return P_canon, Q, R, state_map


def fundamental_matrix(Q: np.ndarray) -> np.ndarray:
    n_t = Q.shape[0]
    I = np.eye(n_t)

    eigvals = np.linalg.eigvals(Q)
    spectral_radius = np.max(np.abs(eigvals))
    if spectral_radius >= 1.0:

        Q = Q * 0.999 / spectral_radius
    N = np.linalg.inv(I - Q)
    return N


def absorption_probabilities_and_times(Q: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = fundamental_matrix(Q)
    B = N @ R
    t = N @ np.ones(N.shape[0])
    return B, t


def intervene_do_state(P: np.ndarray,
                       state_idx: int,
                       new_value_state: int,
                       n_states_per_var: int) -> np.ndarray:
    P_do = P.copy()

    var_id = state_idx // n_states_per_var
    base = var_id * n_states_per_var
    for s in range(n_states_per_var):
        row = base + s
        P_do[row, :] = 0.0
        P_do[row, new_value_state] = 1.0
    return P_do


def demo():
    np.random.seed(7)
    p = 6
    causal_edges = [(0, 1, 0.5), (1, 2, 0.4), (2, 3, 0.3),
                    (0, 3, 0.2), (4, 5, 0.6), (3, 5, 0.35)]
    P, trans, absorb = build_causal_markov_chain(p, causal_edges, n_states_per_var=3)
    P_canon, Q, R, state_map = canonical_form(P, trans, absorb)
    B, t = absorption_probabilities_and_times(Q, R)
    print(f"[markov_causal_chain] 状态数={P.shape[0]}, 瞬态={len(trans)}, 吸收态={len(absorb)}")
    print(f"[markov_causal_chain] 吸收概率矩阵 B 维度={B.shape}, 期望时间范围=[{t.min():.3f}, {t.max():.3f}]")


    P_do = intervene_do_state(P, state_idx=0, new_value_state=2, n_states_per_var=3)
    P_canon_do, Q_do, R_do, _ = canonical_form(P_do, trans, absorb)
    B_do, t_do = absorption_probabilities_and_times(Q_do, R_do)
    effect = np.linalg.norm(B_do - B, 'fro')
    print(f"[markov_causal_chain] do-干预因果效应 (Frobenius 范数差): {effect:.4f}")
    return P, B, t


if __name__ == "__main__":
    demo()
