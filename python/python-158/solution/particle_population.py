"""
particle_population.py
======================
Spatial distribution and aggregation of pulverized coal particle ensembles
using Centroidal Voronoi Tessellation (CVT) with mass lumping.

Incorporates cvt_3d_lumping (249): Lloyd algorithm with density-weighted
lumping for generator placement.

Scientific application: model the spatial distribution of coal particles
in a burner to study local clustering effects on combustion efficiency
and NOx formation. Clustered regions may create fuel-rich pockets that
enhance prompt NOx and reduce burnout.

Mathematical model:
    Given a density function rho(x,y,z) in domain Omega = [-1,1]^3,
    find N generators {g_i} such that each g_i is the centroid of its
    Voronoi cell V_i:
        g_i = (integral_{V_i} rho(x) * x dV) / (integral_{V_i} rho(x) dV)

Lloyd algorithm:
    1. Initialize generators g_i^{(0)}
    2. Repeat:
         a. Compute Voronoi tessellation of {g_i}
         b. Update g_i <- centroid(V_i)
       until convergence

With mass lumping (discretized):
        g_i^{new} = (sum_{s in V_i} rho(s) * s) / (sum_{s in V_i} rho(s))
    where the sum is over a fine grid of sample points.
"""

import numpy as np
from typing import Callable


# ======================================================================
# 1. Density functions for particle distribution
# ======================================================================

def density_uniform(s: np.ndarray) -> np.ndarray:
    """Uniform density: rho = 1."""
    return np.ones(s.shape[0])


def density_chebyshev_1d(s: np.ndarray) -> np.ndarray:
    """
    1D Chebyshev zero density (from mu_chebyzero):
        mu(s) = 1 / sqrt(1 - s^2)   for s in (-1,1)
    """
    val = 1.0 / np.sqrt(np.clip(1.0 - s[:, 0] ** 2, 1e-12, 1.0))
    return val


def density_chebyshev_3d(s: np.ndarray) -> np.ndarray:
    """
    3D tensor-product Chebyshev density (from mu_3d_chebyzero):
        mu(sx,sy,sz) = mu(sx) * mu(sy) * mu(sz)
    """
    if s.shape[1] < 3:
        return density_chebyshev_1d(s)
    mux = 1.0 / np.sqrt(np.clip(1.0 - s[:, 0] ** 2, 1e-12, 1.0))
    muy = 1.0 / np.sqrt(np.clip(1.0 - s[:, 1] ** 2, 1e-12, 1.0))
    muz = 1.0 / np.sqrt(np.clip(1.0 - s[:, 2] ** 2, 1e-12, 1.0))
    return mux * muy * muz


def density_burner_profile(s: np.ndarray, jet_strength: float = 5.0) -> np.ndarray:
    """
    Density profile mimicking a coaxial burner:
    High density in central jet (r < 0.3), lower in outer recirculation.
    """
    if s.shape[1] < 3:
        r = np.sqrt(np.sum(s ** 2, axis=1))
    else:
        r = np.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2)
    # Central jet + annular recirculation
    rho = jet_strength * np.exp(-r ** 2 / 0.09) + 0.5 * np.exp(-(r - 0.6) ** 2 / 0.04)
    return np.clip(rho, 0.01, 100.0)


# ======================================================================
# 2. Lloyd CVT algorithm with lumping
# ======================================================================

def cvt_lloyd_3d(
    n_generators: int,
    density_func: Callable,
    n_samples: int = 40,
    max_iter: int = 50,
    tol: float = 1e-5,
    domain: tuple = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
) -> dict:
    """
    3D Lloyd CVT with mass lumping.
    
    Args:
        n_generators: number of generators (particle clusters)
        density_func: callable mapping Nx3 array -> N array of densities
        n_samples: samples per dimension (total n_samples^3)
        max_iter: maximum Lloyd iterations
        tol: convergence tolerance on generator displacement
        domain: ((xmin,xmax), (ymin,ymax), (zmin,zmax))
    
    Returns:
        dict with generators, energies, displacements
    """
    # Sample grid
    xs = np.linspace(domain[0][0], domain[0][1], n_samples)
    ys = np.linspace(domain[1][0], domain[1][1], n_samples)
    zs = np.linspace(domain[2][0], domain[2][1], n_samples)
    
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    samples = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # Evaluate density
    rho_samples = density_func(samples)
    rho_samples = np.maximum(rho_samples, 1e-12)
    
    # Initialize generators randomly within domain (weighted by density)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(samples), size=n_generators, replace=False, p=rho_samples / rho_samples.sum())
    generators = samples[idx].copy()
    
    energies = []
    displacements = []
    
    for it in range(max_iter):
        # Assign each sample to nearest generator
        dists = np.zeros((len(samples), n_generators))
        for g in range(n_generators):
            diff = samples - generators[g]
            dists[:, g] = np.sum(diff ** 2, axis=1)
        nearest = np.argmin(dists, axis=1)
        
        # Update generators as density-weighted centroids
        new_generators = np.zeros_like(generators)
        for g in range(n_generators):
            mask = nearest == g
            if np.any(mask):
                weights = rho_samples[mask]
                new_generators[g] = np.average(samples[mask], axis=0, weights=weights)
            else:
                # Re-seed empty cell
                new_generators[g] = samples[rng.integers(len(samples))]
        
        # Compute metrics
        disp = np.mean(np.sqrt(np.sum((new_generators - generators) ** 2, axis=1)))
        displacements.append(disp)
        
        # CVT energy: sum of weighted squared distances
        energy = 0.0
        for g in range(n_generators):
            mask = nearest == g
            if np.any(mask):
                diff = samples[mask] - generators[g]
                energy += np.sum(rho_samples[mask] * np.sum(diff ** 2, axis=1))
        energies.append(energy)
        
        generators = new_generators
        
        if disp < tol:
            break
    
    return {
        "generators": generators,
        "energies": np.array(energies),
        "displacements": np.array(displacements),
        "n_iter": it + 1,
    }


# ======================================================================
# 3. Particle cluster statistics
# ======================================================================

def cluster_statistics(generators: np.ndarray, domain: tuple) -> dict:
    """
    Compute geometric statistics of generator distribution.
    """
    if len(generators) == 0:
        return {}
    
    # Pairwise distances
    n = len(generators)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(generators[i] - generators[j])
            dists.append(d)
    dists = np.array(dists)
    
    # Radial distribution
    if domain is not None:
        cx = 0.5 * (domain[0][0] + domain[0][1])
        cy = 0.5 * (domain[1][0] + domain[1][1])
        cz = 0.5 * (domain[2][0] + domain[2][1])
        radii = np.sqrt(
            (generators[:, 0] - cx) ** 2 +
            (generators[:, 1] - cy) ** 2 +
            (generators[:, 2] - cz) ** 2
        )
    else:
        radii = np.zeros(n)
    
    return {
        "mean_spacing": np.mean(dists) if len(dists) > 0 else 0.0,
        "min_spacing": np.min(dists) if len(dists) > 0 else 0.0,
        "max_spacing": np.max(dists) if len(dists) > 0 else 0.0,
        "std_spacing": np.std(dists) if len(dists) > 0 else 0.0,
        "mean_radius": np.mean(radii),
        "std_radius": np.std(radii),
    }


# ======================================================================
# 4. Coalescence / aggregation model
# ======================================================================

def aggregation_kernel(
    v_i: float, v_j: float, T: float = 1500.0, mu_g: float = 4e-5
) -> float:
    """
    Turbulent aggregation kernel for two particles of volumes v_i, v_j [m^3].
    Based on Saffman-Turner kernel:
        beta_ij = 1.3 * sqrt(epsilon/nu) * (d_i + d_j)^3
    where epsilon is turbulent dissipation rate and nu = mu_g/rho_g.
    Simplified here to:
        beta_ij = C_agg * (v_i^{1/3} + v_j^{1/3})^3 * sqrt(T)
    """
    d_i = (6.0 * v_i / np.pi) ** (1.0 / 3.0) if v_i > 0 else 0.0
    d_j = (6.0 * v_j / np.pi) ** (1.0 / 3.0) if v_j > 0 else 0.0
    C_agg = 1e12
    return C_agg * ((d_i + d_j) ** 3) * np.sqrt(max(T, 300.0))


def simulate_smoluchowski_aggregation(
    volumes: np.ndarray, n_steps: int = 100, dt: float = 1e-4
) -> np.ndarray:
    """
    Discrete Smoluchowski coagulation equation (simplified Monte-Carlo style):
        dn_k/dt = 0.5 * sum_{i+j=k} beta_{ij} n_i n_j - n_k * sum_j beta_{kj} n_j
    """
    n_bins = len(volumes)
    N = np.ones(n_bins)  # number concentration in each bin
    
    for step in range(n_steps):
        # Compute aggregation rates beta_{ij} * N_i * N_j
        rates = np.zeros((n_bins, n_bins))
        for i in range(n_bins):
            for j in range(i, n_bins):
                beta = aggregation_kernel(volumes[i], volumes[j])
                rates[i, j] = beta * N[i] * N[j]
        
        total_rate = np.sum(rates)
        if total_rate < 1e-30:
            break
        
        # Explicit Euler for Smoluchowski
        dN = np.zeros(n_bins)
        # Death terms: each bin k loses particles at rate N_k * sum_j beta_{kj} N_j
        for k in range(n_bins):
            for j in range(n_bins):
                b = aggregation_kernel(volumes[k], volumes[j])
                dN[k] -= b * N[k] * N[j] * dt
        
        # Birth terms: aggregate from i and j goes to bin min(i+j, n_bins-1)
        for i in range(n_bins):
            for j in range(i, n_bins):
                b = aggregation_kernel(volumes[i], volumes[j])
                rate = b * N[i] * N[j]
                k_agg = min(i + j, n_bins - 1)
                if i == j:
                    dN[k_agg] += 0.5 * rate * dt
                else:
                    dN[k_agg] += rate * dt
        
        N = np.maximum(N + dN, 0.0)
        if step % 10 == 0 and step > 0:
            # Adaptive stability check
            if np.sum(N) < 0.5 * n_bins:
                break
    
    return N
