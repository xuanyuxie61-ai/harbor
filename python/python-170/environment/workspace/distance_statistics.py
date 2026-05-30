
import numpy as np


def circle_unit_sample(n: int = 1):
    theta = 2.0 * np.pi * np.random.rand(n)
    p = np.column_stack((np.cos(theta), np.sin(theta)))
    return p


def circle_distance_stats(n: int, radius: float = 1.0):
    if n <= 0:
        raise ValueError("n must be positive.")
    p = circle_unit_sample(n)
    q = circle_unit_sample(n)
    d = np.linalg.norm(p - q, axis=1)
    mu = np.mean(d)
    if n > 1:
        var = np.sum((d - mu) ** 2) / (n - 1)
    else:
        var = 0.0
    return mu, var


def circle_distance_pdf(d: np.ndarray, radius: float = 1.0):
    d = np.asarray(d, dtype=float)
    pdf = np.zeros_like(d)
    mask = (d > 0) & (d < 2.0 * radius)
    pdf[mask] = (1.0 / np.pi) / np.sqrt(1.0 - 0.25 * (d[mask] / radius) ** 2)

    pdf = np.clip(pdf, 0.0, 10.0)
    return pdf


def multivariate_distance_stats(dim: int, n: int):
    if dim <= 0 or n <= 0:
        raise ValueError("dim and n must be positive.")
    p = np.random.randn(n, dim)
    q = np.random.randn(n, dim)
    d = np.linalg.norm(p - q, axis=1)
    mu = np.mean(d)
    if n > 1:
        var = np.sum((d - mu) ** 2) / (n - 1)
    else:
        var = 0.0
    return mu, var


def kl_divergence_empirical_vs_uniform(p_hist: np.ndarray, q_hist: np.ndarray, eps: float = 1e-10):
    p = np.asarray(p_hist, dtype=float) + eps
    q = np.asarray(q_hist, dtype=float) + eps
    p /= np.sum(p)
    q /= np.sum(q)
    kl = np.sum(p * np.log(p / q))
    return float(kl)


def compute_emergence_index(positions: np.ndarray, arena_radius: float = 1.0, n_bins: int = 20):
    n = positions.shape[0]
    if n < 2:
        return 0.0, 0.0, 0.0


    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    idx = np.triu_indices(n, k=1)
    dists = dists[idx]

    dist_mean = float(np.mean(dists))
    dist_var = float(np.var(dists))


    hist_emp, bin_edges = np.histogram(dists, bins=n_bins, range=(0.0, 2.0 * arena_radius))


    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_vals = circle_distance_pdf(centers, arena_radius)

    bin_width = bin_edges[1] - bin_edges[0]
    hist_theo = pdf_vals * bin_width
    hist_theo = np.clip(hist_theo, 0.0, None)

    emergence_index = kl_divergence_empirical_vs_uniform(hist_emp, hist_theo)
    return emergence_index, dist_mean, dist_var
