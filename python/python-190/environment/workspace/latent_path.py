
import numpy as np


def path_cost(n: int, distance: np.ndarray, p: np.ndarray) -> float:
    cost = 0.0
    i1 = n - 1
    for i2 in range(n):
        idx1 = int(p[i1]) - 1
        idx2 = int(p[i2]) - 1
        cost += distance[idx1, idx2]
        i1 = i2
    return float(cost)


def perm1_next3(n: int, p: np.ndarray, more: bool, rank: int) -> tuple:
    if not more:
        p = np.arange(1, n + 1)
        more = True
        rank = 1
        return p, more, rank

    n2 = n
    m2 = rank
    s = n

    while True:
        q = m2 % n2
        t = m2 % (2 * n2)
        if q != 0:
            break
        if t == 0:
            s -= 1
        m2 = m2 // n2
        n2 -= 1
        if n2 == 0:
            p = np.arange(1, n + 1)
            more = False
            rank = 1
            return p, more, rank

    if n2 != 0:
        if q == t:
            s -= q
        else:
            s = s + q - n2

        idx1 = s - 1
        idx2 = s
        tmp = p[idx1]
        p[idx1] = p[idx2]
        p[idx2] = tmp
        rank += 1

    return p, more, rank


def tsp_brute(distance: np.ndarray) -> tuple:
    distance = np.asarray(distance, dtype=float)
    n = distance.shape[0]
    if n < 2:
        return np.array([1]), 0.0, 0.0

    total_max = -np.inf
    total_min = np.inf
    total_ave = 0.0
    paths = 0

    p = np.arange(1, n + 1)
    more = False
    rank = 0

    while True:
        p, more, rank = perm1_next3(n, p, more, rank)
        if not more:
            break
        paths += 1
        total = path_cost(n, distance, p)
        total_ave += total
        if total > total_max:
            total_max = total
        if total < total_min:
            total_min = total
            p_min = np.copy(p)

    if paths == 0:
        return np.array([1]), 0.0, 0.0
    total_ave /= paths
    return p_min, float(total_min), float(total_ave)


def latent_space_tsp_path(vectors: np.ndarray) -> tuple:
    n = vectors.shape[0]
    if n <= 2:
        return vectors, 0.0

    diff = vectors[:, None, :] - vectors[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    if n > 8:
        return _greedy_tsp_path(vectors, dist)
    p_min, total_min, _ = tsp_brute(dist)
    ordered = vectors[p_min - 1]
    return ordered, float(total_min)


def _greedy_tsp_path(vectors: np.ndarray, dist: np.ndarray) -> tuple:
    n = vectors.shape[0]
    visited = [False] * n
    current = 0
    path = [current]
    visited[current] = True
    total = 0.0
    for _ in range(n - 1):

        nearest = -1
        min_dist = np.inf
        for j in range(n):
            if not visited[j] and dist[current, j] < min_dist:
                min_dist = dist[current, j]
                nearest = j
        if nearest == -1:
            break
        total += min_dist
        visited[nearest] = True
        path.append(nearest)
        current = nearest

    total += dist[current, path[0]]
    ordered = vectors[path]
    return ordered, float(total)


def slerp(z1: np.ndarray, z2: np.ndarray, t: float) -> np.ndarray:
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    z1 = z1 / (np.linalg.norm(z1) + 1e-15)
    z2 = z2 / (np.linalg.norm(z2) + 1e-15)
    dot = np.clip(np.dot(z1, z2), -1.0, 1.0)
    theta = np.arccos(abs(dot))
    if theta < 1e-10:
        return (1.0 - t) * z1 + t * z2

    if dot < 0.0:
        z2 = -z2
        dot = -dot
        theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    if sin_theta < 1e-10:
        return (1.0 - t) * z1 + t * z2
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * z1 + b * z2


def latent_interpolation_sequence(ordered_vectors: np.ndarray,
                                  steps: int = 10) -> np.ndarray:
    n = ordered_vectors.shape[0]
    if n < 2:
        return ordered_vectors
    seq = []
    for i in range(n - 1):
        for s in range(steps):
            t = s / steps
            seq.append(slerp(ordered_vectors[i], ordered_vectors[i + 1], t))
    seq.append(ordered_vectors[-1])
    return np.array(seq)
