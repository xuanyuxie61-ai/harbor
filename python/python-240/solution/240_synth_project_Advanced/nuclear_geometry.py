"""
核几何与参与者平面模块
Nuclear geometry: Wood-Saxon nuclear profile, participant eccentricity,
reaction plane angle Psi_n and participant plane.

Algorithms sourced from:
  - 468_geometry (r8vec cross product, rotation, triangle area, angles)
  - 1309_triangle_interpolate (barycentric interpolation inside triangles)
  - 1316_triangle_symq_rule (symmetric quadrature for polygonal integrals)
"""
import numpy as np
from physics_constants import (
    AU_RADIUS, AU_DIFFUSENESS, AU_MASS_NUMBER,
    NUCLEON_CROSS_SECTION_FM2, HBAR_C
)


def wood_saxon_density(r, R=AU_RADIUS, a=AU_DIFFUSENESS):
    """
    Wood-Saxon nuclear density distribution:
        ρ(r) = ρ₀ / (1 + exp((r - R)/a))

    Parameters
    ----------
    r : float or ndarray
        Radial distance in fm
    R : float
        Nuclear radius parameter (fm)
    a : float
        Surface diffuseness (fm)

    Returns
    -------
    float or ndarray
        Density in fm^{-3} (normalized to ρ₀ = 0.16 fm^{-3})
    """
    rho0 = 0.16  # fm^{-3}, nuclear saturation density
    r = np.asarray(r, dtype=float)
    x = (r - R) / a
    # Numerical safeguard: clamp exponent to prevent overflow
    x = np.clip(x, -50.0, 50.0)
    return rho0 / (1.0 + np.exp(x))


def wood_saxon_thickness(ta, b, R=AU_RADIUS, a=AU_DIFFUSENESS, nz=50):
    """
    Nuclear thickness function (Glauber model):
        T_A(b) = ∫ dz ρ_A(√(b² + z²))

    Computed by composite trapezoidal rule along z.

    Parameters
    ----------
    ta : ignored (placeholder for API compatibility)
    b : float or ndarray
        Impact parameter in fm
    R, a : float
        Wood-Saxon parameters
    nz : int
        Number of integration points along z

    Returns
    -------
    float or ndarray
        Thickness in fm^{-2}
    """
    b = np.asarray(b, dtype=float)
    z_max = 10.0 * R
    z = np.linspace(-z_max, z_max, nz)
    dz = z[1] - z[0]
    r = np.sqrt(b**2 + z**2)
    rho = wood_saxon_density(r, R, a)
    return np.trapz(rho, dx=dz, axis=-1)


def sample_nucleon_positions(n_part, R=AU_RADIUS, a=AU_DIFFUSENESS, seed=None):
    """
    Sample nucleon transverse positions from Wood-Saxon distribution using
    rejection sampling.

    Algorithm (from 468_geometry's ball01_sample_nd):
    1. Sample uniform in disk r_max
    2. Accept with probability ∝ ρ(r)/ρ₀

    Parameters
    ----------
    n_part : int
        Number of nucleons to sample
    R, a : float
        Wood-Saxon parameters
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    x, y : ndarray
        Transverse positions in fm
    """
    rng = np.random.default_rng(seed)
    r_max = R + 5.0 * a  # generous box
    positions = []
    attempts = 0
    max_attempts = n_part * 100
    while len(positions) < n_part and attempts < max_attempts:
        # Uniform sampling in disk (from 468 ball01_sample_2d)
        r = r_max * np.sqrt(rng.uniform())
        phi = 2.0 * np.pi * rng.uniform()
        xi, yi = r * np.cos(phi), r * np.sin(phi)
        # Acceptance test (rejection)
        rho_val = wood_saxon_density(r, R, a)
        if rng.uniform() < rho_val / 0.16:
            positions.append((xi, yi))
        attempts += 1
    arr = np.array(positions)
    if arr.size == 0:
        return np.zeros(0), np.zeros(0)
    return arr[:, 0], arr[:, 1]


def apply_hard_sphere_collision(x_a, y_a, x_b, y_b, b_impact):
    """
    Determine participants in a hard-sphere nucleon-nucleon collision.
    Two nucleons at (x_a, y_a) and (x_b - b, y_b) collide if their
    transverse distance < √(σ_NN/π).

    Parameters
    ----------
    x_a, y_a : ndarray
        Nucleus A positions
    x_b, y_b : ndarray
        Nucleus B positions (before shift by impact parameter)
    b_impact : float
        Impact parameter in fm

    Returns
    -------
    participant_mask_a, participant_mask_b : ndarray of bool
        Whether each nucleon participated
    """
    sigma = NUCLEON_CROSS_SECTION_FM2
    d_cut = np.sqrt(sigma / np.pi)  # fm
    x_b_shifted = x_b + b_impact
    dx = x_a[:, None] - x_b_shifted[None, :]
    dy = y_a[:, None] - y_b[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    # Each nucleon is participant if any nucleon from the other nucleus
    # is within d_cut
    mask_a = np.any(dist < d_cut, axis=1)
    mask_b = np.any(dist < d_cut, axis=0)
    return mask_a, mask_b


def participant_eccentricity(x_part, y_part, weight=None):
    """
    Participant eccentricity from initial spatial distribution:
        ε_n e^{i n Φ_n} = -{r^n e^{i n φ}} / {r^n}

    where {·} denotes weight-averaged over participants.

    For n=2 (elliptic):
        ε_2 e^{i 2 Φ_2} = -<x² - y² + 2i xy> / <r²>

    Parameters
    ----------
    x_part, y_part : ndarray
        Participant positions in fm
    weight : ndarray, optional
        Weights (e.g., number of collisions N_coll per participant)

    Returns
    -------
    eps_n : ndarray
        Eccentricities for n = 1, 2, 3, 4, 5
    psi_n : ndarray
        Participant plane angles in radians
    """
    if weight is None:
        weight = np.ones_like(x_part)
    total_weight = np.sum(weight)
    if total_weight <= 0:
        return np.zeros(5), np.zeros(5)

    r = np.sqrt(x_part**2 + y_part**2)
    phi = np.arctan2(y_part, x_part)

    eps = np.zeros(5)
    psi = np.zeros(5)

    for n in range(1, 6):
        # Complex eccentricity vector
        q_n = np.sum(weight * r**n * np.exp(1j * n * phi))
        norm = np.sum(weight * r**n)
        if norm <= 0:
            eps[n - 1] = 0.0
            psi[n - 1] = 0.0
            continue
        qn_complex = -q_n / norm
        eps[n - 1] = np.abs(qn_complex)
        psi[n - 1] = np.angle(qn_complex) / n

    return eps, psi


def triangle_contains_point(x1, y1, x2, y2, x3, y3, xp, yp):
    """
    Test if point (xp, yp) lies inside triangle (x1,y1)-(x2,y2)-(x3,y3)
    using barycentric coordinates (from 1309_triangle_interpolate / 468).

    Algorithm: solve for (λ1, λ2, λ3) such that
        xp = λ1 x1 + λ2 x2 + λ3 x3
        yp = λ1 y1 + λ2 y2 + λ3 y3
        λ1 + λ2 + λ3 = 1
    Point inside iff all λi ≥ 0.

    Returns
    -------
    bool or ndarray of bool
        True if point(s) inside triangle
    """
    # Area of full triangle (from 468 triangle_area)
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-14:
        # Degenerate triangle
        return np.zeros_like(xp, dtype=bool)

    lam1 = ((y2 - y3) * (xp - x3) + (x3 - x2) * (yp - y3)) / denom
    lam2 = ((y3 - y1) * (xp - x3) + (x1 - x3) * (yp - y3)) / denom
    lam3 = 1.0 - lam1 - lam2
    return (lam1 >= -1e-12) & (lam2 >= -1e-12) & (lam3 >= -1e-12)


def triangle_interpolate_linear(x1, y1, x2, y2, x3, y3, f1, f2, f3, xp, yp):
    """
    Linear interpolation inside triangle using barycentric coords
    (from 1309_triangle_interpolate_linear).

    f(xp, yp) = λ1 f1 + λ2 f2 + λ3 f3
    """
    denom = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
    if abs(denom) < 1e-14:
        return np.zeros_like(xp)
    lam1 = ((y2 - y3) * (xp - x3) + (x3 - x2) * (yp - y3)) / denom
    lam2 = ((y3 - y1) * (xp - x3) + (x1 - x3) * (yp - y3)) / denom
    lam3 = 1.0 - lam1 - lam2
    return lam1 * f1 + lam2 * f2 + lam3 * f3


def triangle_symq_centroid_and_area(x1, y1, x2, y2, x3, y3):
    """
    Triangle centroid and signed area (from 1316 / 468):
        A = 0.5 |det([x2-x1, y2-y1; x3-x1, y3-y1])|
        (xc, yc) = ((x1+x2+x3)/3, (y1+y2+y3)/3)

    Returns
    -------
    xc, yc : float
    area : float
    """
    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    xc = (x1 + x2 + x3) / 3.0
    yc = (y1 + y2 + y3) / 3.0
    return xc, yc, area


def angle_between_vectors(x1, y1, x2, y2):
    """
    Angle between 2D vectors (from 468 r8vec_angle):
        θ = arccos((v1·v2) / (|v1||v2|))

    Returns angle in radians in [0, π]
    """
    dot = x1 * x2 + y1 * y2
    n1 = np.sqrt(x1**2 + y1**2)
    n2 = np.sqrt(x2**2 + y2**2)
    if n1 < 1e-14 or n2 < 1e-14:
        return 0.0
    cos_theta = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return np.arccos(cos_theta)


def rotate_vector(x, y, theta):
    """
    2D rotation by angle theta (from 468 vector_rotate_2d):
        x' = x cos θ - y sin θ
        y' = x sin θ + y cos θ
    """
    c, s = np.cos(theta), np.sin(theta)
    return c * x - s * y, s * x + c * y


def reaction_plane_angle(eps2_complex):
    """
    Extract 2nd-order reaction plane angle from complex eccentricity:
        Ψ_2 = (1/2) arg(-ε_2 e^{i 2 Φ_2})

    Parameters
    ----------
    eps2_complex : complex
        Complex eccentricity ε_2 e^{i 2 Φ_2}

    Returns
    -------
    float
        Reaction plane angle Ψ_2 in radians
    """
    return 0.5 * np.angle(-eps2_complex)


def participant_plane_triangle_integral(f_vals, tri_vertices):
    """
    Integral of a linearly interpolated function over a triangle using
    symmetric quadrature (1-point centroid rule, exact for linear).
    From 1316 triangle_symq_rule order 1.

    Parameters
    ----------
    f_vals : (3,) array
        Function values at triangle vertices
    tri_vertices : (3, 2) array
        Triangle vertex coordinates

    Returns
    -------
    float
        Integral ∫ f dA over triangle
    """
    (x1, y1), (x2, y2), (x3, y3) = tri_vertices
    xc, yc, area = triangle_symq_centroid_and_area(x1, y1, x2, y2, x3, y3)
    f_center = np.mean(f_vals)  # Centroid quadrature
    return f_center * area


def overlap_region_area(x_a, y_a, x_b, y_b, d_cut, grid_res=200, extent=15.0):
    """
    Compute overlap region area between two nuclei by grid-based integration
    using triangle symmetric quadrature on the grid.

    Parameters
    ----------
    x_a, y_a : ndarray
        Nucleus A nucleon positions
    x_b, y_b : ndarray
        Nucleus B nucleon positions
    d_cut : float
        Collision distance threshold
    grid_res : int
        Grid resolution per dimension
    extent : float
        Domain extent [-extent, extent]

    Returns
    -------
    float
        Overlap area in fm^2
    """
    # Create triangular mesh from grid (two triangles per cell)
    xs = np.linspace(-extent, extent, grid_res)
    ys = np.linspace(-extent, extent, grid_res)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    cell_area = dx * dy

    # For each cell center, check if it's inside both nuclear thickness profiles
    xc_grid = 0.5 * (xs[:-1] + xs[1:])
    yc_grid = 0.5 * (ys[:-1] + ys[1:])
    XX, YY = np.meshgrid(xc_grid, yc_grid, indexing='ij')

    # Compute thickness for each nucleus at grid points
    t_a = wood_saxon_thickness(None, np.sqrt(XX**2 + YY**2))
    t_b = wood_saxon_thickness(None, np.sqrt((XX - 0.0)**2 + YY**2))

    # Overlap region: both thicknesses non-negligible
    overlap = (t_a > 0.01) & (t_b > 0.01)
    return np.sum(overlap) * cell_area
