"""
rbf_kernel.py
=============
Radial Basis Function (RBF) kernels and RBF-enhanced neural network layers.

RBF networks have the form:
    f(x) = \sum_{j=1}^N w_j \phi( ||x - c_j|| / r_0 )

where \phi is a radial kernel.  Common choices include:
  - Multiquadric:   \phi_1(r) = sqrt(r^2 + r_0^2)
  - Inverse MQ:     \phi_2(r) = 1 / sqrt(r^2 + r_0^2)
  - Thin-plate:     \phi_3(r) = r^2 log(r / r_0)
  - Gaussian:       \phi_4(r) = exp(-0.5 r^2 / r_0^2)

In the PINN context, RBF kernels serve two roles:
  1. As a baseline interpolation method for comparison
  2. As a kernel layer within the deep network for local feature enhancement

Adapted from seed project 1013_rbf_interp_1d.
"""

import numpy as np


def rbf_phi1(r, r0):
    """Multiquadric: sqrt(r^2 + r0^2)"""
    return np.sqrt(r ** 2 + r0 ** 2)


def rbf_phi2(r, r0):
    """Inverse multiquadric: 1 / sqrt(r^2 + r0^2)"""
    return 1.0 / np.sqrt(r ** 2 + r0 ** 2 + 1e-12)


def rbf_phi3(r, r0):
    """Thin-plate spline: r^2 * log(r / r0)"""
    v = np.zeros_like(r)
    mask = r > 1e-12
    v[mask] = r[mask] ** 2 * np.log(r[mask] / r0 + 1e-12)
    return v


def rbf_phi4(r, r0):
    """Gaussian: exp(-0.5 * r^2 / r0^2)"""
    return np.exp(-0.5 * r ** 2 / (r0 ** 2 + 1e-12))


def compute_pairwise_distance(X1, X2):
    """
    Compute pairwise Euclidean distances between rows of X1 and X2.

    For X1 shape (n1, d) and X2 shape (n2, d), returns matrix D of shape
    (n1, n2) where D[i,j] = ||X1[i] - X2[j]||_2.

    Computation uses the identity:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a^T b
    """
    if X1.ndim != 2 or X2.ndim != 2:
        raise ValueError("X1 and X2 must be 2D arrays")
    if X1.shape[1] != X2.shape[1]:
        raise ValueError("X1 and X2 must have the same number of columns")

    # Add small epsilon for numerical stability
    sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
    sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
    cross = X1 @ X2.T
    D2 = sq1 + sq2 - 2.0 * cross
    # Clamp negative values caused by roundoff
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)


def rbf_interpolation_weights(X_data, f_data, r0, phi_type='gaussian'):
    """
    Compute RBF interpolation weights w by solving:
        A w = f
    where A[i,j] = \phi( ||x_i - x_j|| / r0 ).

    Parameters
    ----------
    X_data : ndarray, shape (nd, d)
        Data points.
    f_data : ndarray, shape (nd,)
        Function values at data points.
    r0 : float
        RBF scale parameter.
    phi_type : str
        'multiquadric', 'inverse_mq', 'thin_plate', or 'gaussian'.

    Returns
    -------
    w : ndarray, shape (nd,)
        Weights solving the linear system.
    condition_number : float
        Condition number of the interpolation matrix.
    """
    phi_map = {
        'multiquadric': rbf_phi1,
        'inverse_mq': rbf_phi2,
        'thin_plate': rbf_phi3,
        'gaussian': rbf_phi4,
    }
    if phi_type not in phi_map:
        raise ValueError(f"Unknown phi_type: {phi_type}")
    phi = phi_map[phi_type]

    nd = X_data.shape[0]
    D = compute_pairwise_distance(X_data, X_data)
    A = phi(D, r0)

    # Regularize for numerical stability
    A += 1e-10 * np.eye(nd)
    cond_num = np.linalg.cond(A)
    w = np.linalg.solve(A, f_data)
    return w, cond_num


def rbf_interpolate(X_data, w, r0, X_query, phi_type='gaussian'):
    """
    Evaluate RBF interpolant at query points.

    Parameters
    ----------
    X_data : ndarray, shape (nd, d)
        Data points (centers).
    w : ndarray, shape (nd,)
        Weights.
    r0 : float
        Scale parameter.
    X_query : ndarray, shape (nq, d)
        Query points.
    phi_type : str
        Kernel type.

    Returns
    -------
    f_query : ndarray, shape (nq,)
        Interpolated values.
    """
    phi_map = {
        'multiquadric': rbf_phi1,
        'inverse_mq': rbf_phi2,
        'thin_plate': rbf_phi3,
        'gaussian': rbf_phi4,
    }
    phi = phi_map[phi_type]
    D = compute_pairwise_distance(X_query, X_data)
    Aq = phi(D, r0)
    return Aq @ w


class RBFKernelLayer:
    """
    An RBF kernel layer that can be inserted into a neural network.

    Computes:
        h_i = \sum_j w_{ij} \phi( ||x - c_j|| / r0 )

    where centers c_j are learnable or fixed.
    """

    def __init__(self, n_centers, input_dim, r0=1.0, phi_type='gaussian',
                 learnable_centers=False, seed=42):
        rng = np.random.default_rng(seed)
        self.n_centers = n_centers
        self.input_dim = input_dim
        self.r0 = float(r0)
        self.phi_type = phi_type
        self.learnable_centers = learnable_centers

        # Initialize centers uniformly in [-1, 1]^d
        self.centers = rng.uniform(-1.0, 1.0, size=(n_centers, input_dim))
        # Output weights
        self.W = rng.normal(0.0, 1.0 / np.sqrt(n_centers), size=(n_centers, 1))
        self.b = np.zeros(1)

    def forward(self, X):
        """
        Parameters
        ----------
        X : ndarray, shape (n_samples, input_dim)

        Returns
        -------
        Y : ndarray, shape (n_samples, 1)
        """
        D = compute_pairwise_distance(X, self.centers)
        phi_map = {
            'multiquadric': rbf_phi1,
            'inverse_mq': rbf_phi2,
            'thin_plate': rbf_phi3,
            'gaussian': rbf_phi4,
        }
        phi = phi_map[self.phi_type]
        Phi = phi(D, self.r0)
        return Phi @ self.W + self.b
