"""
adaptive_mesh.py
================
Adaptive Mesh Refinement for Flame-Resolved DNS Using Centroidal Voronoi
Tessellation (CVT) and 1D Flamelet Line Grids.

Based on seed projects:
  247 (cvt_2d_lumping) - Lloyd CVT algorithm with density lumping
  680 (line_grid)      - 1D line grid generation with multiple centering options

Scientific Context:
-------------------
In turbulent combustion DNS, the flame thickness is O(δ_L) ~ 0.1-1 mm,
much smaller than turbulent integral scales. Adaptive mesh refinement (AMR)
is essential to resolve the flame while keeping computational cost manageable.

CVT Approach:
-------------
The Lloyd algorithm minimizes the energy functional:
  E(g_1,...,g_N) = Σ_i ∫_{V_i} ρ(x) ||x - g_i||² dx

where V_i is the Voronoi cell of generator g_i and ρ(x) is a density function
peaked in high-gradient (flame) regions. The optimal generator locations satisfy:
  g_i = ∫_{V_i} ρ(x) x dx / ∫_{V_i} ρ(x) dx

The iteration: g_i^{(n+1)} = centroid of V_i under density ρ.

Line Grid for Flamelet Profiles:
--------------------------------
For 1D laminar flamelet computations, a refined grid is needed near the
reaction zone (Z ≈ Z_st). We use centering option 5:
  x_j = ( (2N - 2j + 1)a + (2j - 1)b ) / (2N)
which provides the finest resolution near boundaries — adapted here to
cluster points near stoichiometric mixture fraction.
"""

import numpy as np


def line_grid(n, a, b, centering=1):
    """
    Generate 1D grid points on [a,b] with specified centering.
    Based on seed 680 (line_grid.m).

    Centering options:
      1: Uniform including endpoints
      2: Uniform excluding endpoints
      3: Uniform, a included, b excluded
      4: Uniform, a excluded, b included
      5: Staggered (finest near boundaries)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    x = np.zeros(n)
    for j in range(n):
        jj = j + 1  # 1-based index
        if centering == 1:
            if n == 1:
                x[j] = 0.5 * (a + b)
            else:
                x[j] = ((n - jj) * a + (jj - 1) * b) / (n - 1)
        elif centering == 2:
            x[j] = ((n - jj + 1) * a + jj * b) / (n + 1)
        elif centering == 3:
            x[j] = ((n - jj + 1) * a + (jj - 1) * b) / n
        elif centering == 4:
            x[j] = ((n - jj) * a + jj * b) / n
        elif centering == 5:
            x[j] = ((2 * n - 2 * jj + 1) * a + (2 * jj - 1) * b) / (2 * n)
        else:
            raise ValueError(f"Invalid centering: {centering}")
    return x


def flamelet_stretched_grid(n, z_st, stretch_factor=3.0):
    """
    Generate a 1D grid for mixture fraction Z ∈ [0,1] with clustering
    near the stoichiometric value z_st.

    Mapping: Z = z_st + (1/π) * arctan( stretch_factor * tan(π(ξ - 0.5)) )
    where ξ is uniformly distributed in [0,1].
    """
    xi = line_grid(n, 0.0, 1.0, centering=1)
    # Gudermannian-like stretching
    Z = z_st + (1.0 / np.pi) * np.arctan(
        stretch_factor * np.tan(np.pi * (xi - 0.5))
    )
    # Ensure bounds
    Z = np.clip(Z, 0.0, 1.0)
    return Z


class CVTAdaptiveMesh2D:
    """
    2D Centroidal Voronoi Tessellation for adaptive mesh generation.
    Based on seed 247 (cvt_2d_lumping.m).
    """

    def __init__(self, n_generators, n_samples, density_func=None,
                 x_bounds=(-1.0, 1.0), y_bounds=(-1.0, 1.0)):
        """
        Parameters
        ----------
        n_generators : int
            Number of CVT generators (must be >= 3).
        n_samples : int
            Number of sample points per dimension for Voronoi estimation.
        density_func : callable or None
            Function ρ(x, y) returning density values.
            If None, uniform density is used.
        x_bounds, y_bounds : tuple
            Domain boundaries.
        """
        if n_generators < 3:
            raise ValueError("n_generators must be >= 3")
        self.n = n_generators
        self.n_samples = max(10, n_samples)
        self.density_func = density_func
        self.xmin, self.xmax = x_bounds
        self.ymin, self.ymax = y_bounds

    def _generate_samples(self):
        """Generate uniform sample points avoiding boundaries."""
        eps = 1e-6
        sx = np.linspace(self.xmin + eps, self.xmax - eps, self.n_samples)
        sy = np.linspace(self.ymin + eps, self.ymax - eps, self.n_samples)
        SX, SY = np.meshgrid(sx, sy, indexing='ij')
        return SX.flatten(), SY.flatten()

    def _compute_density(self, x, y):
        """Evaluate density function with clipping for robustness."""
        if self.density_func is None:
            return np.ones_like(x)
        rho = self.density_func(x, y)
        # Clip extreme values for numerical stability
        return np.clip(rho, 0.01, 100.0)

    def lloyd_iteration(self, max_iter=50, tol=1e-6):
        """
        Run Lloyd's algorithm to compute CVT generators.

        Returns
        -------
        generators : ndarray, shape (n, 2)
            Final generator positions (x, y).
        energy_history : list
            CVT energy at each iteration.
        motion_history : list
            Average generator motion at each iteration.
        """
        # Initialize generators uniformly
        g = np.zeros((self.n, 2))
        g[:, 0] = np.random.uniform(self.xmin, self.xmax, self.n)
        g[:, 1] = np.random.uniform(self.ymin, self.ymax, self.n)

        sx, sy = self._generate_samples()
        rho = self._compute_density(sx, sy)
        # In CVT theory, the asymptotic density follows ρ^(2/3) for optimal
        # quantization in 2D. We use ρ^2 for stronger flame focusing.
        r = rho**2

        energy_history = []
        motion_history = []

        for it in range(max_iter):
            # For each sample, find nearest generator
            # Use vectorized distance computation
            dist2 = ((sx[:, None] - g[None, :, 0])**2
                     + (sy[:, None] - g[None, :, 1])**2)
            nearest = np.argmin(dist2, axis=1)

            # Compute mass-weighted centroids
            g_new = np.zeros_like(g)
            mass = np.zeros(self.n)
            for i in range(self.n):
                mask = (nearest == i)
                if np.any(mask):
                    mass[i] = np.sum(r[mask])
                    g_new[i, 0] = np.sum(r[mask] * sx[mask]) / mass[i]
                    g_new[i, 1] = np.sum(r[mask] * sy[mask]) / mass[i]
                else:
                    # Reinitialize empty cell
                    g_new[i, 0] = np.random.uniform(self.xmin, self.xmax)
                    g_new[i, 1] = np.random.uniform(self.ymin, self.ymax)

            # Compute energy (mass-weighted squared distance)
            energy = np.sum(r * dist2[np.arange(len(sx)), nearest]) / self.n_samples
            motion = np.mean(np.sum((g_new - g)**2, axis=1))

            energy_history.append(energy)
            motion_history.append(motion)

            g = g_new.copy()

            if motion < tol:
                break

        return g, energy_history, motion_history

    def extract_flame_resolved_grid(self, nx_dns, ny_dns):
        """
        Generate a uniform background grid with flagged refinement regions
        near CVT generators (flame zones).

        Returns
        -------
        x_grid, y_grid : 1D arrays
            Uniform DNS grid coordinates.
        refinement_mask : ndarray, shape (nx_dns, ny_dns)
            Boolean mask indicating cells needing refinement.
        """
        x_grid = np.linspace(self.xmin, self.xmax, nx_dns)
        y_grid = np.linspace(self.ymin, self.ymax, ny_dns)
        X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

        # Flag cells near any generator
        refinement_mask = np.zeros((nx_dns, ny_dns), dtype=bool)
        dx = (self.xmax - self.xmin) / nx_dns
        dy = (self.ymax - self.ymin) / ny_dns
        radius = 3.0 * max(dx, dy)

        for i in range(self.n):
            dist = np.sqrt((X - self.g[i, 0])**2 + (Y - self.g[i, 1])**2)
            refinement_mask |= (dist < radius)

        return x_grid, y_grid, refinement_mask


def build_density_from_scalar_gradient(X, Y, scalar_field, grad_threshold=0.1):
    """
    Construct a CVT density function ρ(x,y) that is large where the scalar
    gradient is large (i.e., in flame zones).

    ρ(x,y) = 1 + |∇Z|² / (|∇Z|²)_max * ρ_max
    """
    # Compute gradient magnitude via finite differences
    dx = X[1, 0] - X[0, 0]
    dy = Y[0, 1] - Y[0, 0]
    dZdx = np.gradient(scalar_field, axis=0) / dx
    dZdy = np.gradient(scalar_field, axis=1) / dy
    grad_mag = np.sqrt(dZdx**2 + dZdy**2)

    # Normalize and create density
    grad_max = np.max(grad_mag)
    if grad_max < 1e-12:
        return lambda x, y: np.ones_like(np.atleast_1d(x))

    # Interpolate to arbitrary points
    from scipy.interpolate import RegularGridInterpolator
    try:
        interp = RegularGridInterpolator(
            (X[:, 0], Y[0, :]), grad_mag, bounds_error=False, fill_value=0.0
        )

        def density_func(x, y):
            pts = np.column_stack([np.atleast_1d(x), np.atleast_1d(y)])
            vals = interp(pts)
            return 1.0 + 10.0 * vals / grad_max

        return density_func
    except Exception:
        # Fallback without scipy
        def density_func(x, y):
            return np.ones_like(np.atleast_1d(x))
        return density_func
