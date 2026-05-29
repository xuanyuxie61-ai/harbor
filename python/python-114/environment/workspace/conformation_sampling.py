"""
conformation_sampling.py
High-dimensional conformational sampling and dimensionality reduction
for repair protein dynamics.

Derived from: 1052_sammon_data + 517_henon_orbit

Repair proteins like PARP1 and DNA-PKcs undergo large conformational
changes upon DNA damage recognition. This module provides:

  1. Manifold learning via Sammon mapping for visualizing high-dimensional
     conformational landscapes in low dimensions.
  2. Chaotic orbit generators (Henon map) for modeling the non-linear
     search dynamics of proteins in conformational space.
  3. Simplex-based sampling for multi-state equilibrium distributions.

Key formulas:
  - Sammon stress: E = (1/sum_{i<j} d*_{ij}) * sum_{i<j} (d*_{ij} - d_{ij})^2 / d*_{ij}
  - Henon map:  x_{n+1} = x_n * c - (y_n - x_n^2) * s
                y_{n+1} = x_n * s + (y_n - x_n^2) * c
    where c = cos(alpha), s = sin(alpha).
  - Simplex vertices in R^n: v_j satisfying centroid=0, ||v_j||=1,
    and v_i . v_j = -1/n for i!=j.
"""

import numpy as np


def generate_helix_conformations(n_points=30, radius=1.0, pitch=1.0):
    """
    Generate points along a helix in 3D, modeling the helical path
    of a repair protein scanning along the DNA major groove.

    Parameters
    ----------
    n_points : int
    radius : float
        Helix radius (nm).
    pitch : float
        Rise per turn (nm).

    Returns
    -------
    X : ndarray, shape (n_points, 3)
    """
    t = np.linspace(0, 4 * np.pi, n_points)
    X = np.zeros((n_points, 3))
    X[:, 0] = radius * np.cos(t)
    X[:, 1] = radius * np.sin(t)
    X[:, 2] = pitch * t / (2 * np.pi)
    return X


def generate_circle_conformations(n_points=20, radius=1.0, center=None):
    """
    Generate points around a circle in 2D, modeling a closed-loop
    conformational change (e.g., DNA-PKcs ring closure).

    Parameters
    ----------
    n_points : int
    radius : float
    center : ndarray, shape (2,)

    Returns
    -------
    X : ndarray, shape (n_points, 2)
    """
    if center is None:
        center = np.zeros(2)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    X = np.zeros((n_points, 2))
    X[:, 0] = center[0] + radius * np.cos(theta)
    X[:, 1] = center[1] + radius * np.sin(theta)
    return X


def simplex_vertices(n_dim):
    """
    Compute Cartesian coordinates of the vertices of a regular simplex
    in n_dim dimensions, centered at the origin with unit vertex norm.

    Vertices satisfy:
      - centroid = 0
      - ||v_j|| = 1
      - v_i . v_j = -1/n_dim for i != j

    Parameters
    ----------
    n_dim : int

    Returns
    -------
    V : ndarray, shape (n_dim, n_dim+1)
    """
    V = np.zeros((n_dim, n_dim + 1))
    for j in range(n_dim):
        V[j, j] = 1.0
    a = (1.0 - np.sqrt(1.0 + n_dim)) / n_dim
    V[:, n_dim] = a
    # Center at origin
    centroid = np.mean(V, axis=1, keepdims=True)
    V -= centroid
    # Normalize
    s = np.linalg.norm(V[:, 0])
    if s > 0:
        V /= s
    return V


def sample_simplex_mixture(n_dim, n_points, std=0.2):
    """
    Sample points near the vertices of a regular simplex, modeling
    a multi-state conformational equilibrium (e.g., open/closed/ intermediate).

    Parameters
    ----------
    n_dim : int
    n_points : int
    std : float
        Gaussian noise standard deviation.

    Returns
    -------
    X : ndarray, shape (n_points, n_dim)
    labels : ndarray, shape (n_points,)
        Vertex assignment for each point.
    """
    V = simplex_vertices(n_dim)
    X = np.zeros((n_points, n_dim))
    labels = np.zeros(n_points, dtype=int)
    n_vertices = n_dim + 1
    for p in range(n_points):
        k = np.random.randint(0, n_vertices)
        X[p, :] = V[:, k] + std * np.random.randn(n_dim)
        labels[p] = k
    return X, labels


def sammon_mapping(X, n_components=2, max_iter=300, tol=1e-5, alpha=0.3):
    """
    Sammon's nonlinear mapping for dimensionality reduction.

    Minimizes the stress:
        E = (1 / sum_{i<j} d*_{ij}) * sum_{i<j} (d*_{ij} - d_{ij})^2 / d*_{ij}

    where d*_{ij} is the distance in original space, d_{ij} in reduced space.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
    n_components : int
    max_iter : int
    tol : float
    alpha : float
        Learning rate.

    Returns
    -------
    Y : ndarray, shape (n_samples, n_components)
        Low-dimensional embedding.
    stress_history : list
    """
    n_samples = X.shape[0]
    # Compute original distances
    D_star = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            d = np.linalg.norm(X[i] - X[j])
            if d < 1e-10:
                d = 1e-10
            D_star[i, j] = d
            D_star[j, i] = d

    c = np.sum(D_star) / 2.0
    if c == 0:
        c = 1.0

    # Initialize with PCA-like random projection
    Y = np.random.randn(n_samples, n_components) * 0.01
    # Better init: use first n_components columns scaled
    stds = np.std(X, axis=0)
    if len(stds) >= n_components:
        Y = X[:, :n_components] / (stds[:n_components] + 1e-10) * 0.1

    stress_history = []
    for it in range(max_iter):
        # Compute reduced distances
        D = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                d = np.linalg.norm(Y[i] - Y[j])
                if d < 1e-10:
                    d = 1e-10
                D[i, j] = d
                D[j, i] = d

        # Compute stress
        stress = 0.0
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                stress += ((D_star[i, j] - D[i, j]) ** 2) / D_star[i, j]
        stress /= c
        stress_history.append(stress)

        if it > 0 and abs(stress_history[-1] - stress_history[-2]) < tol:
            break

        # Gradient descent update
        for i in range(n_samples):
            delta = np.zeros(n_components)
            for j in range(n_samples):
                if i == j:
                    continue
                diff = Y[i] - Y[j]
                denom = D[i, j] * D_star[i, j]
                if denom < 1e-14:
                    continue
                factor = (D_star[i, j] - D[i, j]) / denom
                delta += factor * diff
            Y[i] -= alpha * delta / c

    return Y, stress_history


def henon_orbit_trajectory(x0, y0, n_steps, alpha_angle=0.4):
    """
    Generate a trajectory of Henon's area-preserving map, modeling
    the chaotic search dynamics of a repair protein in conformational space.

    Map equations:
        x_{n+1} = x_n * cos(alpha) - (y_n - x_n^2) * sin(alpha)
        y_{n+1} = x_n * sin(alpha) + (y_n - x_n^2) * cos(alpha)

    Parameters
    ----------
    x0, y0 : float
        Initial condition.
    n_steps : int
    alpha_angle : float
        Dynamical parameter (radians).

    Returns
    -------
    trajectory : ndarray, shape (n_steps, 2)
    """
    c = np.cos(alpha_angle)
    s = np.sin(alpha_angle)
    traj = np.zeros((n_steps, 2))
    x, y = x0, y0
    for k in range(n_steps):
        if abs(x) < 1.0 and abs(y) < 1.0:
            x_new = x * c - (y - x * x) * s
            y_new = x * s + (y - x * x) * c
            x, y = x_new, y_new
        traj[k] = [x, y]
    return traj


def lyapunov_exponent_henon(x0, y0, n_steps, alpha_angle=0.4, delta0=1e-8):
    """
    Estimate the largest Lyapunov exponent of the Henon map via
    trajectory divergence, quantifying the chaoticity of the protein
    conformational search.

    lambda_1 = lim_{n->inf} (1/n) * sum_{k=1}^n log(||delta_k|| / ||delta_{k-1}||)

    Parameters
    ----------
    x0, y0 : float
    n_steps : int
    alpha_angle : float
    delta0 : float

    Returns
    -------
    lambda1 : float
    """
    c = np.cos(alpha_angle)
    s = np.sin(alpha_angle)

    # Perturbed initial condition
    x, y = x0, y0
    x_p, y_p = x0 + delta0, y0

    lyap_sum = 0.0
    count = 0
    for _ in range(n_steps):
        if abs(x) < 1.0 and abs(y) < 1.0:
            x_new = x * c - (y - x * x) * s
            y_new = x * s + (y - x * x) * c
            x, y = x_new, y_new

            x_p_new = x_p * c - (y_p - x_p * x_p) * s
            y_p_new = x_p * s + (y_p - x_p * x_p) * c
            x_p, y_p = x_p_new, y_p_new

            d = np.sqrt((x - x_p) ** 2 + (y - y_p) ** 2)
            if d > 0 and delta0 > 0:
                lyap_sum += np.log(d / delta0)
                count += 1
                # Renormalize
                x_p = x + delta0 * (x_p - x) / d
                y_p = y + delta0 * (y_p - y) / d

    if count == 0:
        return 0.0
    return lyap_sum / count
