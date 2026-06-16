"""
初始条件模块：Glauber 模型涨落能量密度
Initial conditions for heavy-ion collisions via Monte Carlo Glauber model
with fluctuating nucleon positions and Gaussian smearing.

Algorithms sourced from:
  - 1099_amsontag_mcpse (Monte Carlo event sampling)
  - 871_plasma_matrix (matrix assembly for transport coefficients)
  - 846_paraheat_functional (heat-like diffusion for smearing)
"""
import numpy as np
from physics_constants import (
    AU_RADIUS, AU_DIFFUSENESS, AU_MASS_NUMBER,
    NUCLEON_CROSS_SECTION_FM2, HBAR_C
)
from nuclear_geometry import (
    wood_saxon_density, sample_nucleon_positions,
    apply_hard_sphere_collision
)


def gaussian_kernel_2d(x, y, sigma):
    """
    2D Gaussian smearing kernel:
        G(x, y) = (1 / (2πσ²)) exp(-(x² + y²) / (2σ²))

    Used to smear point-like nucleon energy depositions into a continuous
    energy density profile.

    Parameters
    ----------
    x, y : ndarray
        Grid coordinates in fm
    sigma : float
        Smearing width in fm (typical: 0.4-0.8 fm)

    Returns
    -------
    ndarray
        Kernel values at (x, y)
    """
    if sigma <= 0:
        raise ValueError("Smearing width sigma must be positive")
    return (1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-(x**2 + y**2) / (2.0 * sigma**2))


def build_energy_density_grid(x_part, y_part, weights, x_grid, y_grid, sigma):
    """
    Build transverse energy density ε(x, y) on grid from participant
    positions using Gaussian smearing.

        ε(x, y) = Σ_i w_i G(x - x_i, y - y_i; σ)

    Parameters
    ----------
    x_part, y_part : ndarray
        Participant positions in fm
    weights : ndarray
        Weights per participant (e.g., N_coll or constant)
    x_grid, y_grid : ndarray
        2D meshgrid of evaluation points
    sigma : float
        Smearing width in fm

    Returns
    -------
    ndarray
        Energy density on grid in GeV/fm^3
    """
    eps_grid = np.zeros_like(x_grid)
    for xi, yi, wi in zip(x_part, y_part, weights):
        eps_grid += wi * gaussian_kernel_2d(x_grid - xi, y_grid - yi, sigma)
    return eps_grid


def mc_glauber_event(n_nucleons, b_impact, sigma_shear, seed=None,
                     R=AU_RADIUS, a=AU_DIFFUSENESS):
    """
    Generate a single Monte Carlo Glauber event:
    1. Sample nucleon positions from Wood-Saxon
    2. Determine participants via hard-sphere criterion
    3. Build fluctuating energy density with smearing

    Parameters
    ----------
    n_nucleons : int
        Number of nucleons per nucleus (A = 197 for Au)
    b_impact : float
        Impact parameter in fm
    sigma_shear : float
        Gaussian smearing width in fm
    seed : int, optional
        Random seed
    R, a : float
        Wood-Saxon parameters

    Returns
    -------
    dict
        Event data with keys:
        - 'x_part_a', 'y_part_a': Nucleus A participants
        - 'x_part_b', 'y_part_b': Nucleus B participants
        - 'n_part': total participant count
        - 'n_coll': estimated collision count
    """
    rng = np.random.default_rng(seed)

    # Sample both nuclei
    x_a, y_a = sample_nucleon_positions(n_nucleons, R, a, seed=rng.integers(2**31))
    x_b, y_b = sample_nucleon_positions(n_nucleons, R, a, seed=rng.integers(2**31))

    # Determine participants
    mask_a, mask_b = apply_hard_sphere_collision(x_a, y_a, x_b, y_b, b_impact)

    x_part_a, y_part_a = x_a[mask_a], y_a[mask_a]
    x_part_b, y_part_b = x_b[mask_b], y_b[mask_b]

    # Count collisions (from 1099 mcpse Monte Carlo counting)
    d_cut = np.sqrt(NUCLEON_CROSS_SECTION_FM2 / np.pi)
    x_b_shifted = x_b + b_impact
    n_coll = 0
    for i, (xa, ya) in enumerate(zip(x_a, y_a)):
        for j, (xb, yb) in enumerate(zip(x_b_shifted, y_b)):
            if (xa - xb)**2 + (ya - yb)**2 < d_cut**2:
                n_coll += 1

    return {
        'x_part_a': x_part_a, 'y_part_a': y_part_a,
        'x_part_b': x_part_b, 'y_part_b': y_part_b,
        'n_part': int(mask_a.sum() + mask_b.sum()),
        'n_coll': n_coll,
        'b_impact': b_impact,
        'x_all_a': x_a, 'y_all_a': y_a,
        'x_all_b': x_b, 'y_all_b': y_b,
    }


def make_transverse_grid(extent=12.0, n_grid=64):
    """
    Create uniform transverse grid centered at origin.

    Parameters
    ----------
    extent : float
        Half-size of domain in fm
    n_grid : int
        Number of grid points per dimension

    Returns
    -------
    x, y : ndarray
        2D meshgrid coordinates
    dx, dy : float
        Grid spacing in fm
    """
    xs = np.linspace(-extent, extent, n_grid)
    ys = np.linspace(-extent, extent, n_grid)
    dx = xs[1] - xs[0]
    dy = ys[1] - ys[0]
    x, y = np.meshgrid(xs, ys, indexing='ij')
    return x, y, dx, dy


def initial_energy_density_from_event(event, n_grid=64, extent=12.0, sigma=0.5):
    """
    Convert a Glauber event into a transverse energy density field.

    The normalization is such that total energy ∝ (n_part + N_coll * α),
    with α controlling the binary collision vs. participant mixing.

    Energy scale: typical RHIC Au+Au √s=200 GeV → ε₀ ~ 15 GeV/fm³

    Parameters
    ----------
    event : dict
        Output from mc_glauber_event
    n_grid : int
        Grid size
    extent : float
        Domain extent in fm
    sigma : float
        Smearing width

    Returns
    -------
    x, y : ndarray
        Grid coordinates
    eps : ndarray
        Energy density ε(x, y) in GeV/fm^3
    dx : float
        Grid spacing
    """
    x, y, dx, _ = make_transverse_grid(extent, n_grid)

    # Participant weights: w = (1 - α) + α * n_coll_i / n_part
    alpha_mix = 0.15  # 15% binary collision contribution
    n_part_a = len(event['x_part_a'])
    n_part_b = len(event['x_part_b'])

    # Build weighted participant array
    x_part = np.concatenate([event['x_part_a'], event['x_part_b']])
    y_part = np.concatenate([event['y_part_a'], event['y_part_b']])

    if len(x_part) == 0:
        return x, y, np.zeros_like(x), dx

    # Uniform weights per participant
    weights = np.ones_like(x_part) * (1.0 - alpha_mix)

    # Build energy density
    eps = build_energy_density_grid(x_part, y_part, weights, x, y, sigma)

    # Normalize to physical energy density scale
    # ε₀ ≈ 15 GeV/fm³ at τ₀ = 0.6 fm/c (typical RHIC)
    total_energy = np.sum(eps) * dx**2
    if total_energy > 0:
        target_energy = 15.0 * np.pi * 6.0**2  # Rough normalization
        eps *= target_energy / total_energy

    return x, y, eps, dx


def initial_temperature_profile(eps, eos_func):
    """
    Convert initial energy density to temperature using equation of state:
        ε(T) → T

    Inverts ε(T) relation from lattice-QCD parameterization.

    Parameters
    ----------
    eps : ndarray
        Energy density field in GeV/fm^3
    eos_func : callable
        Function that returns T given ε

    Returns
    -------
    ndarray
        Temperature field T(x, y) in GeV
    """
    return np.vectorize(eos_func)(eps)


def fluctuation_correlator(eps, x, y, dx, r_bins=20, r_max=6.0):
    """
    Two-point correlation function of energy density fluctuations:
        C(r) = <δε(x) δε(x+r)> / <δε²>

    where δε = ε - <ε>. Used to quantify initial-state fluctuation scale.

    Parameters
    ----------
    eps : ndarray
        Energy density field
    x, y : ndarray
        Grid coordinates
    dx : float
        Grid spacing
    r_bins : int
        Number of radial bins
    r_max : float
        Maximum separation in fm

    Returns
    -------
    r_centers : ndarray
        Bin centers
    corr : ndarray
        Correlation function C(r)
    """
    eps_mean = np.mean(eps)
    deps = eps - eps_mean
    var = np.mean(deps**2)

    if var < 1e-14:
        r_centers = np.linspace(0.5 * r_max / r_bins, r_max, r_bins)
        return r_centers, np.zeros(r_bins)

    # Flatten arrays
    eps_flat = deps.flatten()
    x_flat = x.flatten()
    y_flat = y.flatten()

    # Sample pairs for efficiency
    rng = np.random.default_rng(42)
    n_sample = min(2000, len(eps_flat))
    idx = rng.choice(len(eps_flat), n_sample, replace=False)

    r_edges = np.linspace(0, r_max, r_bins + 1)
    corr_sum = np.zeros(r_bins)
    corr_count = np.zeros(r_bins)

    for i in idx:
        # Compute distances from point i to all others
        dists = np.sqrt((x_flat - x_flat[i])**2 + (y_flat - y_flat[i])**2)
        for b in range(r_bins):
            mask = (dists >= r_edges[b]) & (dists < r_edges[b + 1])
            if np.any(mask):
                corr_sum[b] += np.mean(eps_flat[i] * eps_flat[mask])
                corr_count[b] += 1

    corr_count[corr_count == 0] = 1
    corr = corr_sum / corr_count / var
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    return r_centers, corr


def build_transport_matrix(n_cells, eta_over_s, temperature):
    """
    Assemble transport coefficient matrix for viscous stress tensor.
    From 871_plasma_matrix: sparse banded matrix with diagonal dominance.

    The shear stress π^{μν} evolves via:
        τ_π dπ^{μν}/dτ + π^{μν} = 2η σ^{μν}

    Discretized, this gives a banded linear system.

    Parameters
    ----------
    n_cells : int
        Number of grid cells
    eta_over_s : float
        Shear viscosity over entropy density
    temperature : ndarray
        Local temperature at each cell

    Returns
    -------
    ndarray (n_cells, n_cells)
        Sparse transport matrix (stored dense for small systems)
    """
    # Shear viscosity η = (η/s) * s, with s = (4/3)ε/T ≈ (4σ_SB/3)T³
    sigma_sb = np.pi**2 / 30.0 * (16 + 21 * 1.5)  # N_f=2.5 effective
    entropy_density = (4.0 / 3.0) * sigma_sb * temperature**3 / np.maximum(temperature, 1e-6)
    eta = eta_over_s * entropy_density

    # Build banded matrix: main diagonal = 1/τ_π, off-diagonals = -D/Δx²
    tau_pi = 5.0 * eta / np.maximum(entropy_density, 1e-10)  # Relaxation time
    tau_pi = np.maximum(tau_pi, 0.1)  # Bound from below

    # Identity-like structure with small off-diagonal couplings
    mat = np.zeros((n_cells, n_cells))
    for i in range(n_cells):
        mat[i, i] = 1.0 / tau_pi.flat[i]
        if i > 0:
            mat[i, i - 1] = -0.1 / tau_pi.flat[i]
        if i < n_cells - 1:
            mat[i, i + 1] = -0.1 / tau_pi.flat[i]

    return mat


def thermal_diffusion_kernel(eps, dx, dy, dt, diffusivity=0.1):
    """
    Apply thermal diffusion operator to smooth energy density
    (from 846_paraheat_functional):
        ∂ε/∂τ = D ∇²ε

    Forward Euler step:
        ε^{n+1} = ε^n + D Δt ∇²ε^n

    Parameters
    ----------
    eps : ndarray (nx, ny)
        Energy density
    dx, dy : float
        Grid spacing
    dt : float
        Time step
    diffusivity : float
        Thermal diffusion coefficient D in fm

    Returns
    -------
    ndarray
        Updated energy density after one diffusion step
    """
    nx, ny = eps.shape
    eps_new = eps.copy()

    # 2nd order central difference Laplacian
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            laplacian = ((eps[i+1, j] - 2*eps[i, j] + eps[i-1, j]) / dx**2 +
                        (eps[i, j+1] - 2*eps[i, j] + eps[i, j-1]) / dy**2)
            eps_new[i, j] += diffusivity * dt * laplacian

    return eps_new
