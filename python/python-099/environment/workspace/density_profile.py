"""
density_profile.py
------------------
Piecewise-linear (PWL) approximation of plasma electron density profiles
and Monte-Carlo statistical analysis of electron spatial distributions
using sphere-distance sampling.

Incorporates core ideas from:
  - 925_pwl_approx_1d   (1D piecewise linear approximation)
  - 1125_sphere_positive_distance (Monte Carlo distance statistics)
"""

import numpy as np
import math
from utils import clamp, ensure_positive


def pwl_approx_1d(
    xd: np.ndarray,
    yd: np.ndarray,
    xc: np.ndarray,
) -> np.ndarray:
    """
    Construct a piecewise-linear approximation of data (xd, yd) using
    control points xc by solving a least-squares projection.

    For each data point xd[i], it falls in some interval [xc[j], xc[j+1]].
    The basis value at xd[i] for control point j is the linear hat function:
        phi_j(x) = (xc[j+1] - x) / (xc[j+1] - xc[j])   if x in [xc[j], xc[j+1]]
        phi_j(x) = (x - xc[j-1]) / (xc[j] - xc[j-1])   if x in [xc[j-1], xc[j]]

    We build an ND x NC matrix A and solve  yc = (A^T A)^{-1} A^T yd
    (normal equations) for the control values yc.

    Parameters
    ----------
    xd : (ND,) ndarray
        Data abscissas.
    yd : (ND,) ndarray
        Data ordinates.
    xc : (NC,) ndarray
        Control points (must be strictly increasing, NC >= 2).

    Returns
    -------
    yc : (NC,) ndarray
        Control values at xc.
    """
    xd = np.asarray(xd, dtype=float)
    yd = np.asarray(yd, dtype=float)
    xc = np.asarray(xc, dtype=float)

    if xc.size < 2:
        raise ValueError("At least 2 control points are required.")
    if not np.all(np.diff(xc) > 0):
        raise ValueError("xc must be strictly increasing.")

    ND = xd.size
    NC = xc.size

    # Build projection matrix A[ND, NC]
    A = np.zeros((ND, NC), dtype=float)
    for i in range(ND):
        x = xd[i]
        if x <= xc[0]:
            A[i, 0] = 1.0
        elif x >= xc[-1]:
            A[i, -1] = 1.0
        else:
            # Find interval
            j = int(np.searchsorted(xc, x) - 1)
            j = clamp(j, 0, NC - 2)
            h = xc[j + 1] - xc[j]
            if abs(h) < 1e-14:
                A[i, j] = 1.0
            else:
                w = (x - xc[j]) / h
                A[i, j] = 1.0 - w
                A[i, j + 1] = w

    # Solve normal equations: (A^T A) yc = A^T yd
    AtA = A.T @ A
    Aty = A.T @ yd

    # Add small regularization for numerical stability
    reg = 1e-12 * np.eye(NC)
    AtA_reg = AtA + reg

    yc = np.linalg.solve(AtA_reg, Aty)
    return yc


def pwl_interp_1d(
    xc: np.ndarray,
    yc: np.ndarray,
    xi: np.ndarray,
) -> np.ndarray:
    """
    Piecewise-linear interpolation from control points (xc, yc) to query points xi.

    Parameters
    ----------
    xc : (NC,) ndarray
        Control abscissas (strictly increasing).
    yc : (NC,) ndarray
        Control ordinates.
    xi : (Nq,) ndarray
        Query points.

    Returns
    -------
    yi : (Nq,) ndarray
    """
    xc = np.asarray(xc, dtype=float)
    yc = np.asarray(yc, dtype=float)
    xi = np.asarray(xi, dtype=float)

    if not np.all(np.diff(xc) > 0):
        raise ValueError("xc must be strictly increasing.")

    yi = np.empty_like(xi)
    for i in range(xi.size):
        x = xi[i]
        if x <= xc[0]:
            yi[i] = yc[0]
        elif x >= xc[-1]:
            yi[i] = yc[-1]
        else:
            j = int(np.searchsorted(xc, x) - 1)
            j = clamp(j, 0, xc.size - 2)
            h = xc[j + 1] - xc[j]
            if abs(h) < 1e-14:
                yi[i] = yc[j]
            else:
                w = (x - xc[j]) / h
                yi[i] = (1.0 - w) * yc[j] + w * yc[j + 1]
    return yi


def generate_density_profile(
    z: np.ndarray,
    peak_density: float,
    width: float,
    profile_type: str = "gaussian",
) -> np.ndarray:
    """
    Generate a synthetic electron density profile n_e(z).

    Parameters
    ----------
    z : (Nz,) ndarray
        Depth coordinates [m].
    peak_density : float
        Peak electron density [m^-3].
    width : float
        Characteristic width [m].
    profile_type : str
        "gaussian", "exponential", "linear", or "step".

    Returns
    -------
    n_e : (Nz,) ndarray
    """
    z = np.asarray(z, dtype=float)
    peak_density = max(float(peak_density), 1.0)
    width = max(float(width), 1e-6)

    z_mid = 0.5 * (z.min() + z.max())
    if profile_type == "gaussian":
        n_e = peak_density * np.exp(-0.5 * ((z - z_mid) / width) ** 2)
    elif profile_type == "exponential":
        n_e = peak_density * np.exp(-(z - z.min()) / width)
    elif profile_type == "linear":
        n_e = peak_density * np.maximum(0.0, 1.0 - (z - z.min()) / width)
    elif profile_type == "step":
        n_e = np.where(np.abs(z - z_mid) <= width / 2.0, peak_density, 0.0)
    else:
        raise ValueError("profile_type must be 'gaussian', 'exponential', 'linear', or 'step'.")

    return ensure_positive(n_e)


def sphere_positive_sample(n: int, seed: int = None) -> np.ndarray:
    """
    Sample n points uniformly on the positive octant of the unit sphere
    (x>0, y>0, z>0) using the normal-distribution trick.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int, optional

    Returns
    -------
    points : (n, 3) ndarray
    """
    n = max(int(n), 1)
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    pts = rng.normal(size=(n, 3))
    pts = np.abs(pts)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    pts = pts / norms
    return pts


def sphere_positive_distance_stats(n: int, seed: int = None) -> dict:
    """
    Compute statistical properties of pairwise distances between points
    sampled on the positive unit sphere.

    Parameters
    ----------
    n : int
        Number of sample points.
    seed : int

    Returns
    -------
    stats : dict
        { "mean": float, "std": float, "min": float, "max": float }
    """
    pts = sphere_positive_sample(n, seed=seed)
    # Pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            dists.append(float(d))

    if len(dists) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

    dists = np.array(dists)
    return {
        "mean": float(np.mean(dists)),
        "std": float(np.std(dists)),
        "min": float(np.min(dists)),
        "max": float(np.max(dists)),
    }


def electron_scattering_angle_distribution(
    n_samples: int = 1000,
    seed: int = 42,
) -> tuple:
    """
    Model the angular distribution of scattered electromagnetic waves
    by plasma electrons as a proxy for scattering anisotropy.

    We sample random directions on the positive sphere and compute the
    cosine of the scattering angle relative to the incident direction
    (assumed along +z).  The mean cosine quantifies forward scattering.

    Parameters
    ----------
    n_samples : int
    seed : int

    Returns
    -------
    (mean_cos_theta, std_cos_theta, cos_samples)
    """
    pts = sphere_positive_sample(n_samples, seed=seed)
    cos_theta = pts[:, 2]  # z-component is cos(theta) for unit vectors
    return (
        float(np.mean(cos_theta)),
        float(np.std(cos_theta)),
        cos_theta,
    )
