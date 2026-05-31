"""
uq_analysis.py
==============
Uncertainty quantification analysis: statistical moments, sensitivity indices,
spectral stability, and chaotic-sensitivity diagnostics.

Fused from seed projects:
- 171_chirikov_iteration : chaotic dynamical system sensitivity (standard map)
- 131_c8lib              : complex matrix norms and spectral analysis
- 1130_sphere_triangle_quad : spherical sampling for global sensitivity

Mathematical foundation
-----------------------
1. Chaotic sensitivity via Lyapunov exponent estimation:
   For a dynamical map F, the largest Lyapunov exponent is
       \lambda = lim_{n->inf} (1/n) log ||DF^n(x_0) v_0||
   where DF is the Jacobian.  We use the Chirikov standard map as a proxy
   to study how stochastic parameter perturbations propagate in chaotic regimes.

   Standard map:
       y' = y + k sin(x)
       x' = x + y'   (mod 2\pi)
   Jacobian:
       J = [[1, k cos(x)], [1, 1 + k cos(x)]]

2. Spectral stability of gPC system matrix:
   For the Galerkin-projected system A u = b, the condition number
       \kappa(A) = \sigma_{max}(A) / \sigma_{min}(A)
   governs numerical stability.  We compute it via singular value decomposition.
   For complex matrices (Frobenius norm from c8lib):
       ||A||_F = sqrt( sum_{i,j} |a_{ij}|^2 )

3. Sobol sensitivity indices (from chaos_expansion.py) quantify how much each
   stochastic dimension contributes to output variance.

4. Convergence diagnostics:
   - L2 relative error: ||u - u_ref||_2 / ||u_ref||_2
   - Wasserstein-2 distance between empirical distributions
   - Kolmogorov-Smirnov statistic
"""

import numpy as np


# ---------------------------------------------------------------------------
# Chirikov standard map and Lyapunov exponent
# ---------------------------------------------------------------------------

def chirikov_map_step(x, y, k):
    """
    One step of the Chirikov standard map.
    Returns (x_next, y_next).
    """
    y_next = y + k * np.sin(x)
    x_next = (x + y_next) % (2.0 * np.pi)
    return x_next, y_next


def chirikov_jacobian(x, k):
    """
    Jacobian matrix of the standard map at state x (before the step).
    J = [[1, k*cos(x)], [1, 1+k*cos(x)]]
    """
    c = k * np.cos(x)
    return np.array([[1.0, c], [1.0, 1.0 + c]])


def lyapunov_exponent_standard_map(k, n_iter=5000, n_burn=100):
    """
    Estimate the largest Lyapunov exponent of the Chirikov standard map
    for perturbation strength k, using the method of Benettin et al.

    Algorithm:
        v_0 = random unit vector
        for t = 1..n_iter:
            v_t = J(x_{t-1}) v_{t-1}
            r_t = ||v_t||
            v_t = v_t / r_t
            lambda += log(r_t)
        lambda = lambda / n_iter
    """
    x = np.random.rand() * 2.0 * np.pi
    y = np.random.rand() * 2.0 * np.pi
    v = np.random.randn(2)
    v = v / np.linalg.norm(v)
    lam = 0.0
    for t in range(n_burn + n_iter):
        J = chirikov_jacobian(x, k)
        v = J @ v
        r = np.linalg.norm(v)
        v = v / r
        x, y = chirikov_map_step(x, y, k)
        if t >= n_burn:
            lam += np.log(max(r, 1e-30))
    return lam / n_iter


# ---------------------------------------------------------------------------
# Spectral / matrix analysis
# ---------------------------------------------------------------------------

def frobenius_norm(A):
    """Compute Frobenius norm sqrt(sum |a_ij|^2)."""
    A = np.asarray(A, dtype=complex)
    return np.sqrt(np.sum(np.abs(A) ** 2))


def spectral_condition_number(A):
    """Compute condition number via SVD."""
    A = np.asarray(A, dtype=float)
    s = np.linalg.svd(A, compute_uv=False)
    s_max = np.max(s)
    s_min = np.max(s[s > 1e-15])
    return s_max / s_min


def gershgorin_discs(A):
    """
    Compute Gershgorin disc centers and radii for a square matrix.
    Every eigenvalue lies in at least one disc D(a_{ii}, R_i) where
        R_i = sum_{j != i} |a_{ij}|
    Returns centers, radii.
    """
    A = np.asarray(A, dtype=complex)
    n = A.shape[0]
    centers = np.diag(A)
    radii = np.zeros(n)
    for i in range(n):
        radii[i] = np.sum(np.abs(A[i, :])) - np.abs(A[i, i])
    return centers, radii


def power_iteration_eigenvalue(A, max_iter=1000, tol=1e-12):
    """
    Approximate the dominant eigenvalue and eigenvector of a matrix
    using the power iteration method.
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[0]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)
    lam_old = 0.0
    for _ in range(max_iter):
        Av = A @ v
        lam = np.dot(v, Av)
        v = Av / np.linalg.norm(Av)
        if abs(lam - lam_old) < tol:
            break
        lam_old = lam
    return lam, v


# ---------------------------------------------------------------------------
# Statistical diagnostics
# ---------------------------------------------------------------------------

def empirical_cdf(data, x_grid):
    """
    Evaluate the empirical CDF of 1-D data at points x_grid.
    """
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)
    cdf = np.searchsorted(data, x_grid, side='right') / n
    return cdf


def kolmogorov_smirnov_stat(data1, data2):
    """
    Kolmogorov-Smirnov statistic between two 1-D samples.
    D = sup_x |F_1(x) - F_2(x)|
    """
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    all_data = np.concatenate([data1, data2])
    grid = np.linspace(np.min(all_data), np.max(all_data), 1000)
    cdf1 = empirical_cdf(data1, grid)
    cdf2 = empirical_cdf(data2, grid)
    return np.max(np.abs(cdf1 - cdf2))


def wasserstein2_distance(data1, data2):
    """
    Approximate Wasserstein-2 distance between two 1-D empirical distributions.
    For sorted samples of equal size n:
        W_2^2 = (1/n) sum_{i=1}^n (x_{(i)} - y_{(i)})^2
    """
    data1 = np.sort(np.asarray(data1, dtype=float))
    data2 = np.sort(np.asarray(data2, dtype=float))
    # Interpolate to common grid if sizes differ
    n = max(len(data1), len(data2))
    q = np.linspace(0, 1, n)
    d1_q = np.quantile(data1, q)
    d2_q = np.quantile(data2, q)
    return np.sqrt(np.mean((d1_q - d2_q) ** 2))


def moment_statistics(samples):
    """
    Compute mean, variance, skewness, and excess kurtosis of a sample.
    """
    s = np.asarray(samples, dtype=float)
    n = len(s)
    mu = np.mean(s)
    var = np.var(s, ddof=1)
    std = np.sqrt(var)
    if std < 1e-15:
        return mu, var, 0.0, -3.0
    skew = np.mean((s - mu) ** 3) / std ** 3
    kurt = np.mean((s - mu) ** 4) / std ** 4 - 3.0
    return mu, var, skew, kurt


# ---------------------------------------------------------------------------
# Spherical sampling for global sensitivity (Monte Carlo on S^{d-1})
# ---------------------------------------------------------------------------

def sample_unit_sphere_uniform(d, n_samples):
    """
    Generate n_samples uniformly distributed points on the unit sphere S^{d-1}
    using the normal distribution method.
    """
    X = np.random.randn(n_samples, d)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / norms


def spherical_triangle_area(v1, v2, v3):
    """
    Compute the area of a spherical triangle on the unit sphere given
    vertices v1, v2, v3 (normalized vectors).
    Girard's theorem: area = A + B + C - pi
    where A, B, C are the interior angles.
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)
    # Side lengths (angles between vertices)
    a = np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0))
    b = np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0))
    c = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))
    # Spherical excess using L'Huilier's theorem
    s = 0.5 * (a + b + c)
    # Guard against numerical issues
    tan_E_4 = np.sqrt(np.tan(0.5 * s) * np.tan(0.5 * (s - a)) *
                      np.tan(0.5 * (s - b)) * np.tan(0.5 * (s - c)))
    E = 4.0 * np.arctan(tan_E_4)
    return E


def test_uq_analysis():
    """Self-tests."""
    # Lyapunov exponent for small k (should be ~0 for integrable, >0 for chaotic)
    lam_small = lyapunov_exponent_standard_map(0.5, n_iter=2000, n_burn=200)
    assert lam_small < 0.1, f"Small k should be near-integrable, got lambda={lam_small}"
    lam_large = lyapunov_exponent_standard_map(2.5, n_iter=2000, n_burn=200)
    assert lam_large > 0.05, f"Large k should be chaotic, got lambda={lam_large}"
    # Frobenius norm test
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert np.isclose(frobenius_norm(A), np.sqrt(30.0), atol=1e-12)
    # Gershgorin
    centers, radii = gershgorin_discs(A)
    eigs = np.linalg.eigvals(A)
    for ev in eigs:
        in_disc = any(np.abs(ev - c) <= r + 1e-10 for c, r in zip(centers, radii))
        assert in_disc, "Gershgorin theorem violated"
    # KS stat
    d1 = np.random.randn(1000)
    d2 = np.random.randn(1000)
    ks = kolmogorov_smirnov_stat(d1, d2)
    assert ks < 0.2  # Same distribution
    # Sphere sample
    pts = sample_unit_sphere_uniform(3, 100)
    norms = np.linalg.norm(pts, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)
    print("uq_analysis: all self-tests passed")


if __name__ == "__main__":
    test_uq_analysis()
