"""
monte_carlo_scattering.py

Monte Carlo simulation of photon transport in turbid media for OCT.
Incorporates:
- sphere_monte_carlo: random sampling of scattering directions on unit sphere
- hypercube_surface_distance: parameter space exploration on hypercube surface
- Henyey-Greenstein phase function for anisotropic scattering
"""

import numpy as np


# ---------------------------------------------------------------------------
# Unit sphere sampling (from sphere_monte_carlo)
# ---------------------------------------------------------------------------

def sphere01_sample(n):
    """
    Generate n uniformly random points on the unit sphere in R^3.

    Algorithm: sample from 3D standard normal, then normalize each vector.

    Parameters
    ----------
    n : int
        Number of points.

    Returns
    -------
    x : ndarray, shape (3, n)
        Points on unit sphere.
    """
    if n < 1:
        raise ValueError("n must be >= 1.")
    x = np.random.randn(3, n)
    norms = np.sqrt(np.sum(x ** 2, axis=0))
    norms = np.where(norms < 1e-14, 1.0, norms)
    x = x / norms
    return x


def sphere01_monomial_integral(e):
    """
    Exact integral of monomial x^e1 y^e2 z^e3 over the unit sphere.

    If any e_i is odd: integral = 0.
    If all e_i are even:
        integral = 2 * Gamma((e1+1)/2) * Gamma((e2+1)/2) * Gamma((e3+1)/2)
                   / Gamma((e1+e2+e3+3)/2)

    Parameters
    ----------
    e : array_like, shape (3,)
        Exponents.

    Returns
    -------
    val : float
        Integral value.
    """
    e = np.asarray(e, dtype=int)
    if e.shape != (3,):
        raise ValueError("e must have shape (3,).")
    if np.any(e < 0):
        raise ValueError("All exponents must be non-negative.")
    if np.all(e == 0):
        val = 2.0 * np.sqrt(np.pi ** 3) / sp_gamma(1.5)
    elif np.any(e % 2 == 1):
        val = 0.0
    else:
        val = 2.0
        for i in range(3):
            val *= sp_gamma(0.5 * (e[i] + 1))
        val /= sp_gamma(0.5 * np.sum(e + 1))
    return val


# ---------------------------------------------------------------------------
# Hypercube surface sampling (from hypercube_surface_distance)
# ---------------------------------------------------------------------------

def hypercube_surface_sample(n, d):
    """
    Sample n points uniformly at random from the surface of the unit hypercube
    in d dimensions.

    Algorithm:
    1. Start with uniform random points in [0,1]^d.
    2. Randomly choose one coordinate axis per point.
    3. Set that coordinate to 0 or 1 uniformly.

    In OCT context: represents sampling on the boundary of the d-dimensional
    optical parameter space (mu_a, mu_s, g, n, ...).

    Parameters
    ----------
    n : int
        Number of samples.
    d : int
        Spatial dimension.

    Returns
    -------
    p : ndarray, shape (n, d)
        Sampled points on hypercube surface.
    """
    if n < 1 or d < 1:
        raise ValueError("n and d must be >= 1.")
    p = np.random.rand(n, d)
    i = np.random.randint(0, d, size=n)
    s = np.random.randint(0, 2, size=n)
    k = np.arange(n) + i * n
    p.flat[k] = s
    return p


def hypercube_surface_distance_stats(n, d):
    """
    Estimate mean and variance of Euclidean distance between two random points
    on the surface of the d-dimensional unit hypercube.

    Parameters
    ----------
    n : int
        Number of sample pairs.
    d : int
        Dimension.

    Returns
    -------
    dmu : float
        Estimated mean distance.
    dvar : float
        Estimated variance.
    """
    p1 = hypercube_surface_sample(n, d)
    p2 = hypercube_surface_sample(n, d)
    dists = np.linalg.norm(p1 - p2, axis=1)
    dmu = np.mean(dists)
    dvar = np.var(dists, ddof=1)
    return dmu, dvar


# ---------------------------------------------------------------------------
# Henyey-Greenstein random scattering
# ---------------------------------------------------------------------------

def hg_sample_cos_theta(n, g):
    """
    Sample scattering angle cosine from Henyey-Greenstein distribution.

    Inverse CDF method:
      cos_theta = (1 + g^2 - ((1 - g^2)/(1 - g + 2 g U))^2) / (2 g)
    for g != 0; for g = 0, uniform in cos_theta.

    Parameters
    ----------
    n : int
        Number of samples.
    g : float
        Anisotropy factor, -1 < g < 1.

    Returns
    -------
    cos_theta : ndarray, shape (n,)
    """
    if not (-1.0 < g < 1.0):
        raise ValueError("g must be in (-1, 1).")
    u = np.random.rand(n)
    if abs(g) < 1e-8:
        return 2.0 * u - 1.0
    g2 = g * g
    numerator = 1.0 - g2
    denom = 1.0 - g + 2.0 * g * u
    denom = np.where(np.abs(denom) < 1e-14, 1e-14, denom)
    cos_theta = (1.0 + g2 - (numerator / denom) ** 2) / (2.0 * g)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return cos_theta


# ---------------------------------------------------------------------------
# Monte Carlo photon packet tracking for OCT
# ---------------------------------------------------------------------------

def track_photon_packet(initial_position, initial_direction, max_steps,
                        mu_s, mu_a, g, layer_z_boundaries,
                        n_medium=1.33, step_size=None):
    """
    Track a single photon packet through layered tissue using Monte Carlo.

    At each step:
    1. Move by exponential random step with mean 1/mu_t.
    2. Check layer boundaries; if crossed, update refractive index.
    3. Decide absorption vs scattering by roulette.
    4. If scattered, sample new direction from HG.

    Parameters
    ----------
    initial_position : ndarray, shape (3,)
        Starting position [x, y, z] in microns.
    initial_direction : ndarray, shape (3,)
        Initial direction vector (will be normalized).
    max_steps : int
        Maximum number of interactions.
    mu_s, mu_a : float or callable
        Scattering and absorption coefficients (1/micron).
        If callable, signature f(z) -> coefficient.
    g : float or callable
        Anisotropy factor. If callable, signature f(z) -> g.
    layer_z_boundaries : array_like
        z-coordinates of layer interfaces.
    n_medium : float or array_like
        Refractive index per layer.
    step_size : float, optional
        Fixed step size; if None, use mean free path.

    Returns
    -------
    path : list of ndarray
        Positions at each step.
    weights : list of float
        Photon weight (survival probability) at each step.
    """
    pos = np.asarray(initial_position, dtype=float).copy()
    direc = np.asarray(initial_direction, dtype=float)
    norm = np.linalg.norm(direc)
    if norm < 1e-14:
        direc = np.array([0.0, 0.0, 1.0])
    else:
        direc = direc / norm

    layer_boundaries = np.asarray(layer_z_boundaries, dtype=float)
    if np.isscalar(n_medium):
        n_layers = len(layer_boundaries) - 1
        n_vals = np.full(n_layers, n_medium, dtype=float)
    else:
        n_vals = np.asarray(n_medium, dtype=float)

    def get_layer_idx(z):
        for idx in range(len(layer_boundaries) - 1):
            if layer_boundaries[idx] <= z < layer_boundaries[idx + 1]:
                return idx
        return len(layer_boundaries) - 2

    def get_coeff(z, coeff):
        if callable(coeff):
            return float(coeff(z))
        return float(coeff)

    path = [pos.copy()]
    weight = 1.0
    weights = [weight]

    for _ in range(max_steps):
        layer_idx = get_layer_idx(pos[2])
        mu_s_val = get_coeff(pos[2], mu_s)
        mu_a_val = get_coeff(pos[2], mu_a)
        mu_t = mu_s_val + mu_a_val
        if mu_t <= 1e-14:
            break

        if step_size is None:
            s = -np.log(max(np.random.rand(), 1e-14)) / mu_t
        else:
            s = step_size

        new_pos = pos + s * direc

        # Check boundary crossing
        crossed = False
        for bz in layer_boundaries:
            if (pos[2] - bz) * (new_pos[2] - bz) < 0:
                # Intersection with boundary
                if abs(direc[2]) > 1e-14:
                    s_boundary = (bz - pos[2]) / direc[2]
                    new_pos = pos + s_boundary * direc
                    # Snell's law simplified: reflect back if critical angle
                    n1 = n_vals[min(layer_idx, len(n_vals) - 1)]
                    n2 = n_vals[min(layer_idx + 1, len(n_vals) - 1)]
                    if n1 != n2:
                        # Simplified: specular reflection at boundary
                        direc[2] = -direc[2]
                        new_pos = pos + s_boundary * direc
                    crossed = True
                    break

        pos = new_pos.copy()
        path.append(pos.copy())

        # Absorption roulette
        albedo = mu_s_val / mu_t if mu_t > 0 else 0.0
        if np.random.rand() > albedo:
            weight = 0.0
            weights.append(weight)
            break
        weight *= albedo
        weights.append(weight)

        # Scatter
        g_val = get_coeff(pos[2], g)
        cos_theta = hg_sample_cos_theta(1, g_val)[0]
        sin_theta = np.sqrt(max(0.0, 1.0 - cos_theta * cos_theta))
        phi_angle = 2.0 * np.pi * np.random.rand()

        # Rotate direction
        if abs(direc[2]) > 0.99999:
            ux = sin_theta * np.cos(phi_angle)
            uy = sin_theta * np.sin(phi_angle)
            uz = cos_theta * np.sign(direc[2])
        else:
            denom = np.sqrt(1.0 - direc[2] ** 2)
            ux = (sin_theta * (direc[0] * direc[2] * np.cos(phi_angle) - direc[1] * np.sin(phi_angle))
                  / denom + direc[0] * cos_theta)
            uy = (sin_theta * (direc[1] * direc[2] * np.cos(phi_angle) + direc[0] * np.sin(phi_angle))
                  / denom + direc[1] * cos_theta)
            uz = -denom * sin_theta * np.cos(phi_angle) + direc[2] * cos_theta
        direc = np.array([ux, uy, uz])
        direc = direc / np.linalg.norm(direc)

    return path, weights


def simulate_oct_signal_mc(n_photons, source_z, detector_z,
                           layer_boundaries, layer_props, max_steps=100):
    """
    Monte Carlo simulation of OCT signal: count photons that return from
    a given depth to the detector.

    Parameters
    ----------
    n_photons : int
        Number of photon packets.
    source_z, detector_z : float
        Source and detector positions.
    layer_boundaries : array_like
        Layer boundaries.
    layer_props : list of dict
        Each dict with 'mu_s', 'mu_a', 'g', 'n'.
    max_steps : int
        Max interactions per photon.

    Returns
    -------
    signal : float
        Detected signal fraction.
    depths : ndarray
        Depths of last scattering events.
    """
    detected = 0
    depths = []
    mu_s_arr = np.array([p['mu_s'] for p in layer_props])
    mu_a_arr = np.array([p['mu_a'] for p in layer_props])
    g_arr = np.array([p['g'] for p in layer_props])
    n_arr = np.array([p['n'] for p in layer_props])

    def coeff_func(z, arr):
        for idx in range(len(layer_boundaries) - 1):
            if layer_boundaries[idx] <= z < layer_boundaries[idx + 1]:
                return arr[idx]
        return arr[-1]

    for _ in range(n_photons):
        pos0 = np.array([0.0, 0.0, source_z])
        dir0 = np.array([0.0, 0.0, 1.0])
        path, weights = track_photon_packet(
            pos0, dir0, max_steps,
            lambda z: coeff_func(z, mu_s_arr),
            lambda z: coeff_func(z, mu_a_arr),
            lambda z: coeff_func(z, g_arr),
            layer_boundaries,
            n_arr
        )
        if len(path) > 1 and weights[-1] > 0:
            last_z = path[-1][2]
            depths.append(last_z)
            # Detect if photon returns near detector
            if abs(last_z - detector_z) < 1.0:
                detected += weights[-1]

    signal = detected / n_photons if n_photons > 0 else 0.0
    return signal, np.array(depths)


# ---------------------------------------------------------------------------
# scipy.special import guard
# ---------------------------------------------------------------------------
try:
    from scipy.special import gamma as sp_gamma
except Exception:
    # Fallback if scipy not available: use math gamma for integers/half-integers
    import math
    def sp_gamma(x):
        return math.gamma(x)
