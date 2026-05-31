"""
distance_statistics.py
======================
Statistical analysis of inter-robot distances for emergent behavior detection.

Incorporates:
  - circle_distance_stats / circle_distance_pdf (from 178_circle_distance)
  - normal01_multivariate_distance_stats (from 819_normal01_multivariate_distance)

Scientific role:
  In a circular patrol arena, the distance distribution between randomly
  chosen robot pairs reveals the degree of spatial organization. We compare
  the empirical distance histogram against the theoretical PDF for uniformly
  random points on a circle,
      pdf(d) = (1/pi) / sqrt(1 - 0.25*d^2/R^2)
  and compute the Kullback-Leibler divergence to quantify emergence.
  Multivariate distance statistics are used for the high-dimensional
  state-space embedding of the swarm.
"""

import numpy as np


def circle_unit_sample(n: int = 1):
    """
    Sample n points uniformly on the unit circle.

    Parameters
    ----------
    n : int
        Number of samples.

    Returns
    -------
    p : ndarray, shape (n, 2)
        Sample points.
    """
    theta = 2.0 * np.pi * np.random.rand(n)
    p = np.column_stack((np.cos(theta), np.sin(theta)))
    return p


def circle_distance_stats(n: int, radius: float = 1.0):
    """
    Estimate mean and variance of chord lengths for random points on a circle.

    The exact mean for a unit circle is 4/pi ≈ 1.2732.

    Parameters
    ----------
    n : int
        Number of sample pairs.
    radius : float
        Circle radius.

    Returns
    -------
    mu : float
        Estimated mean distance.
    var : float
        Estimated variance.
    """
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
    """
    Theoretical PDF of chord length for two random points on a circle.

        pdf(d) = (1/pi) * 1/sqrt(1 - 0.25*d^2/R^2),   0 <= d <= 2R

    Parameters
    ----------
    d : ndarray
        Distances.
    radius : float
        Circle radius.

    Returns
    -------
    pdf : ndarray
        Probability density values.
    """
    d = np.asarray(d, dtype=float)
    pdf = np.zeros_like(d)
    mask = (d > 0) & (d < 2.0 * radius)
    pdf[mask] = (1.0 / np.pi) / np.sqrt(1.0 - 0.25 * (d[mask] / radius) ** 2)
    # handle boundary singularity numerically
    pdf = np.clip(pdf, 0.0, 10.0)
    return pdf


def multivariate_distance_stats(dim: int, n: int):
    """
    Estimate mean and variance of Euclidean distances between two independent
    N(0, I_d) random vectors.

    The exact mean is sqrt(2) * Gamma((d+1)/2) / Gamma(d/2).

    Parameters
    ----------
    dim : int
        Spatial dimension.
    n : int
        Number of sample pairs.

    Returns
    -------
    mu : float
        Estimated mean.
    var : float
        Estimated variance.
    """
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
    """
    Compute KL divergence D_KL(P || Q) between two normalized histograms.

    Parameters
    ----------
    p_hist : ndarray
        Empirical histogram (observed robot distances).
    q_hist : ndarray
        Theoretical histogram (uniform random points).
    eps : float
        Floor to avoid log(0).

    Returns
    -------
    kl : float
        KL divergence.
    """
    p = np.asarray(p_hist, dtype=float) + eps
    q = np.asarray(q_hist, dtype=float) + eps
    p /= np.sum(p)
    q /= np.sum(q)
    kl = np.sum(p * np.log(p / q))
    return float(kl)


def compute_emergence_index(positions: np.ndarray, arena_radius: float = 1.0, n_bins: int = 20):
    """
    Compute an emergence index based on distance distribution divergence.

    A low KL divergence from the uniform random distribution indicates
    disordered behavior; a high divergence indicates structure.

    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Robot positions in 2D.
    arena_radius : float
        Radius of the circular arena.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    emergence_index : float
        KL divergence (higher = more structured).
    dist_mean : float
        Mean inter-robot distance.
    dist_var : float
        Variance of inter-robot distances.
    """
    n = positions.shape[0]
    if n < 2:
        return 0.0, 0.0, 0.0

    # compute all pairwise distances
    diffs = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    # extract upper triangle without diagonal
    idx = np.triu_indices(n, k=1)
    dists = dists[idx]

    dist_mean = float(np.mean(dists))
    dist_var = float(np.var(dists))

    # empirical histogram
    hist_emp, bin_edges = np.histogram(dists, bins=n_bins, range=(0.0, 2.0 * arena_radius))

    # theoretical pdf histogram
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    pdf_vals = circle_distance_pdf(centers, arena_radius)
    # pdf_vals is a density; convert to histogram proportions
    bin_width = bin_edges[1] - bin_edges[0]
    hist_theo = pdf_vals * bin_width
    hist_theo = np.clip(hist_theo, 0.0, None)

    emergence_index = kl_divergence_empirical_vs_uniform(hist_emp, hist_theo)
    return emergence_index, dist_mean, dist_var
