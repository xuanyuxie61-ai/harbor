
import numpy as np
from typing import List, Dict, Tuple






def extract_event_features(events: List[Dict]) -> np.ndarray:
    if not events:
        return np.zeros((0, 4))

    energies = np.array([ev.get("energy_obs", 0.0) for ev in events])
    zs = np.array([ev.get("z", 0.0) for ev in events])
    ts = np.array([ev.get("time_day", 0.0) for ev in events])
    rs = np.sqrt(np.array([ev.get("x", 0.0) ** 2 + ev.get("y", 0.0) ** 2 for ev in events]))


    E_max = np.max(energies) if np.max(energies) > 0 else 1.0
    z_max = np.max(zs) if np.max(zs) > 0 else 1.0
    t_max = 365.25
    R_max = np.max(rs) if np.max(rs) > 0 else 1.0

    X = np.column_stack([
        energies / E_max,
        zs / z_max,
        ts / t_max,
        rs / R_max,
    ])
    return X






def build_distance_matrix(X: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
    N = X.shape[0]
    if N < 2:
        return np.zeros((N, N))
    if weights is None:
        weights = np.ones(X.shape[1])
    weights = np.asarray(weights)

    Dmat = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            diff = X[i] - X[j]
            dist = np.sqrt(np.sum(weights * diff ** 2))
            Dmat[i, j] = dist
            Dmat[j, i] = dist
    return Dmat


def symmetrize_distance_matrix(Dmat: np.ndarray) -> np.ndarray:
    return 0.5 * (Dmat + Dmat.T)


def single_linkage_clustering(Dmat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    N = Dmat.shape[0]
    if N < 2:
        return np.zeros((0, 3)), np.zeros(N, dtype=int)


    clusters = [{i} for i in range(N)]
    linkage = []
    active = list(range(N))


    cluster_dist = {}
    for i in range(N):
        for j in range(i + 1, N):
            cluster_dist[(i, j)] = Dmat[i, j]

    next_id = N
    while len(active) > 1:

        min_dist = np.inf
        pair = (active[0], active[1])
        for i_idx in range(len(active)):
            for j_idx in range(i_idx + 1, len(active)):
                a = active[i_idx]
                b = active[j_idx]
                key = (min(a, b), max(a, b))
                d = cluster_dist.get(key, np.inf)
                if d < min_dist:
                    min_dist = d
                    pair = (a, b)

        a, b = pair
        linkage.append([a, b, min_dist])


        new_cluster = next_id
        next_id += 1


        for c in active:
            if c == a or c == b:
                continue
            key_a = (min(a, c), max(a, c))
            key_b = (min(b, c), max(b, c))
            da = cluster_dist.get(key_a, np.inf)
            db = cluster_dist.get(key_b, np.inf)
            key_new = (min(new_cluster, c), max(new_cluster, c))
            cluster_dist[key_new] = min(da, db)

        active = [c for c in active if c != a and c != b]
        active.append(new_cluster)

    linkage_matrix = np.array(linkage)


    if len(linkage) > 0:
        threshold = np.percentile(linkage_matrix[:, 2], 50.0)
        labels = cut_dendrogram(linkage_matrix, N, threshold)
    else:
        labels = np.zeros(N, dtype=int)

    return linkage_matrix, labels


def cut_dendrogram(linkage: np.ndarray, n_leaves: int, threshold: float) -> np.ndarray:
    labels = np.zeros(n_leaves, dtype=int)
    cluster_id = 0

    def assign(node: int, cid: int):
        if node < n_leaves:
            labels[node] = cid
        else:
            row = int(node - n_leaves)
            if row < len(linkage):
                left, right, dist = linkage[row]
                if dist > threshold:
                    nonlocal cluster_id
                    cluster_id += 1
                    assign(int(left), cluster_id)
                    cluster_id += 1
                    assign(int(right), cluster_id)
                else:
                    assign(int(left), cid)
                    assign(int(right), cid)
            else:
                labels[node % n_leaves] = cid

    assign(2 * n_leaves - 2, cluster_id)
    return labels






def fisher_discriminant(
    X_signal: np.ndarray,
    X_background: np.ndarray,
) -> Tuple[np.ndarray, float, float]:
    mu_s = np.mean(X_signal, axis=0)
    mu_b = np.mean(X_background, axis=0)

    cov_s = np.cov(X_signal, rowvar=False, bias=True)
    cov_b = np.cov(X_background, rowvar=False, bias=True)

    if cov_s.ndim == 0:
        cov_s = np.atleast_2d(cov_s)
    if cov_b.ndim == 0:
        cov_b = np.atleast_2d(cov_b)


    if X_signal.shape[1] == 1:
        cov_s = cov_s.reshape(1, 1) if cov_s.size == 1 else cov_s
        cov_b = cov_b.reshape(1, 1) if cov_b.size == 1 else cov_b

    Sw = cov_s + cov_b

    Sw += 1.0e-6 * np.eye(Sw.shape[0])

    try:
        w = np.linalg.solve(Sw, mu_s - mu_b)
    except np.linalg.LinAlgError:
        w = mu_s - mu_b


    norm = np.linalg.norm(w)
    if norm > 1.0e-15:
        w = w / norm


    proj_s = X_signal @ w
    proj_b = X_background @ w


    threshold = 0.5 * (np.mean(proj_s) + np.mean(proj_b))
    separation = abs(np.mean(proj_s) - np.mean(proj_b)) / np.sqrt(np.var(proj_s) + np.var(proj_b))

    return w, float(threshold), float(separation)


def apply_discriminant_cut(
    X: np.ndarray,
    w: np.ndarray,
    threshold: float,
    direction: str = ">",
) -> np.ndarray:
    proj = X @ w
    if direction == ">":
        return proj > threshold
    else:
        return proj < threshold






def evaluate_background_rejection(
    signal_events: List[Dict],
    background_events: List[Dict],
    w: np.ndarray,
    threshold: float,
    target_efficiency: float = 0.9,
) -> Dict:
    X_s = extract_event_features(signal_events)
    X_b = extract_event_features(background_events)

    mask_s = apply_discriminant_cut(X_s, w, threshold, ">")
    mask_b = apply_discriminant_cut(X_b, w, threshold, ">")

    n_s_pass = int(np.sum(mask_s))
    n_b_pass = int(np.sum(mask_b))
    n_s_total = len(signal_events)
    n_b_total = len(background_events)

    efficiency = n_s_pass / n_s_total if n_s_total > 0 else 0.0
    rejection = n_b_total / n_b_pass if n_b_pass > 0 else np.inf
    purity = n_s_pass / (n_s_pass + n_b_pass) if (n_s_pass + n_b_pass) > 0 else 0.0

    return {
        "signal_efficiency": float(efficiency),
        "background_rejection": float(rejection),
        "purity": float(purity),
        "n_signal_pass": n_s_pass,
        "n_background_pass": n_b_pass,
        "target_efficiency": target_efficiency,
    }






if __name__ == "__main__":

    np.random.seed(42)
    X_s = np.random.randn(50, 4) + np.array([2.0, 0.0, 0.0, 0.0])
    X_b = np.random.randn(50, 4)


    D = build_distance_matrix(X_s[:5])
    assert D.shape == (5, 5)
    assert np.allclose(D, D.T), "距离矩阵不对称"
    assert np.all(np.diag(D) == 0.0), "对角线应为零"


    linkage, labels = single_linkage_clustering(D)
    assert len(linkage) == 4, "连接矩阵长度应为 N-1"


    w, thr, sep = fisher_discriminant(X_s, X_b)
    assert sep > 0.5, f"分离度过低: {sep}"

    mask = apply_discriminant_cut(np.vstack([X_s, X_b]), w, thr, ">")
    assert np.sum(mask[:50]) > np.sum(mask[50:]), "信号应主要落在阈值上方"

    print("event_reconstruction.py: 所有自测通过")
