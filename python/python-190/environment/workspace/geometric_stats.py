
import numpy as np


def triangle_sample(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                    n: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    r1 = rng.random(n)
    r2 = rng.random(n)

    mask = r1 + r2 > 1.0
    r1[mask] = 1.0 - r1[mask]
    r2[mask] = 1.0 - r2[mask]
    r3 = 1.0 - r1 - r2
    pts = (r1[:, None] * v1.reshape(1, -1)
           + r2[:, None] * v2.reshape(1, -1)
           + r3[:, None] * v3.reshape(1, -1))
    return pts


def triangle_distance_stats(v1: np.ndarray, v2: np.ndarray, v3: np.ndarray,
                            n_samples: int = 5000, seed: int = None) -> dict:
    rng = np.random.default_rng(seed)
    pts1 = triangle_sample(v1, v2, v3, n_samples, rng.integers(0, 2**31))
    pts2 = triangle_sample(v1, v2, v3, n_samples, rng.integers(0, 2**31))
    dists = np.sqrt(np.sum((pts1 - pts2) ** 2, axis=1))
    return {
        "mean": float(np.mean(dists)),
        "variance": float(np.var(dists, ddof=1)),
        "std": float(np.std(dists, ddof=1)),
    }


def equilateral_distance_pdf(d: np.ndarray, side: float = 1.0) -> np.ndarray:
    d = np.asarray(d, dtype=float)
    s = float(side)
    if s <= 0.0:
        raise ValueError("边长必须为正。")
    r = s / np.sqrt(3.0)
    pdf = np.zeros_like(d)

    mask1 = (d > 0.0) & (d <= 1.5 * r)
    pdf[mask1] = (2.0 * d[mask1] / (s * s)) * (
        (2.0 * np.pi / 3.0) - (np.sqrt(3.0) / 2.0) * (d[mask1] / r) ** 2
    )

    mask2 = (d > 1.5 * r) & (d <= s)
    ratio = d[mask2] / r
    term1 = 2.0 * np.arcsin(np.clip(s / (2.0 * d[mask2]), -1.0, 1.0))
    term2 = (np.sqrt(3.0) / 2.0) * ratio ** 2
    term3 = np.sqrt(np.clip(ratio ** 2 - 2.25, 0.0, None))
    pdf[mask2] = (2.0 * d[mask2] / (s * s)) * (term1 - term2 + term3)

    pdf = np.where(d <= 0.0, 0.0, pdf)
    pdf = np.where(d > s, 0.0, pdf)
    return pdf


def wasserstein_approx_mc(samples_p: np.ndarray, samples_q: np.ndarray) -> float:

    n = min(samples_p.shape[0], samples_q.shape[0])
    if n < 2:
        return 0.0
    rng = np.random.default_rng()
    idx_p = rng.choice(samples_p.shape[0], size=n, replace=False)
    idx_q = rng.choice(samples_q.shape[0], size=n, replace=False)
    sp = samples_p[idx_p]
    sq = samples_q[idx_q]

    dp = np.sqrt(np.sum((sp[:-1] - sp[1:]) ** 2, axis=1))
    dq = np.sqrt(np.sum((sq[:-1] - sq[1:]) ** 2, axis=1))
    mean_p = float(np.mean(dp))
    mean_q = float(np.mean(dq))
    std_p = float(np.std(dp, ddof=1))
    std_q = float(np.std(dq, ddof=1))

    w1 = abs(mean_p - mean_q) + 0.5 * abs(std_p - std_q)
    return w1


def mesh_distance_distribution(nodes: np.ndarray, triangles: list,
                               n_samples: int = 2000, seed: int = None) -> dict:
    rng = np.random.default_rng(seed)
    from triangle_quadrature import triangle_area
    areas = []
    for tri in triangles:
        areas.append(triangle_area(nodes[tri[0]], nodes[tri[1]], nodes[tri[2]]))
    areas = np.array(areas)
    total_area = np.sum(areas)
    if total_area < 1e-15:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0, "n_pairs": 0}


    probs = areas / total_area
    n_tri_samples = min(len(triangles), max(10, n_samples // 10))
    chosen = rng.choice(len(triangles), size=n_tri_samples, p=probs)

    all_pts = []
    for idx in chosen:
        tri = triangles[idx]
        pts = triangle_sample(nodes[tri[0]], nodes[tri[1]], nodes[tri[2]], 20, rng.integers(0, 2**31))
        all_pts.append(pts)
    all_pts = np.vstack(all_pts)


    n = all_pts.shape[0]
    if n < 2:
        return {"mean": 0.0, "variance": 0.0, "std": 0.0, "n_pairs": 0}
    idx1 = rng.choice(n, size=min(n_samples, n * (n - 1) // 2), replace=False)
    idx2 = rng.choice(n, size=idx1.size, replace=False)

    mask_same = idx1 == idx2
    if np.any(mask_same):
        idx2[mask_same] = (idx2[mask_same] + 1) % n
    dists = np.sqrt(np.sum((all_pts[idx1] - all_pts[idx2]) ** 2, axis=1))
    return {
        "mean": float(np.mean(dists)),
        "variance": float(np.var(dists, ddof=1)),
        "std": float(np.std(dists, ddof=1)),
        "n_pairs": int(dists.size),
    }
