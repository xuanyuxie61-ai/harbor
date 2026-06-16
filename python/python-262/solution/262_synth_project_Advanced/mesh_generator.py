"""
mesh_generator.py
=================
Adaptive mesh generation for the magnetic reconnection simulation.
Provides structured grid generation with refinement near the current
sheet and X-points.

Maps seed projects:
  - 475_gmsh_to_fem: mesh format conversion, node/element generation
  - 905_pram: structured grid/tiling patterns → domain decomposition
  - 1394_voronoi_city: Voronoi-based spatial partitioning
  - 329_ellipse_distance: elliptical mesh stretching functions

Physical motivation:
  - The current sheet is extremely thin (delta ~ S^{-1/2} L)
  - For S ~ 10^12 (corona), delta/L ~ 10^{-6}
  - Uniform meshing would require ~10^6 points across the sheet
  - Adaptive mesh refinement (AMR) or stretched grids are essential

Key equations:
  - Tangent-hyperbolic stretching: z_new = z + alpha * tanh(beta * (z - z0))
  - Geometric stretching: dx_i = dx_0 * r^i (r = stretching ratio)
  - Monitor function for AMR: M(x) = 1 + alpha * |d^2B/dx^2|
"""

import numpy as np


# ============================================================
# 1D Mesh Generation Functions
# ============================================================

def uniform_mesh(a, b, n):
    """
    Uniform mesh on [a, b] with n cells.
    x_i = a + i * dx,  dx = (b-a)/n
    """
    dx = (b - a) / max(n, 1)
    return np.linspace(a + 0.5 * dx, b - 0.5 * dx, n), dx


def tanh_stretch_mesh(a, b, n, beta=3.0, center=0.0):
    """
    Tangent-hyperbolic stretched mesh, clustered around 'center'.

    The mapping is:
        xi in [-1, 1] (uniform)  ->  x in [a, b]
        x(xi) = center + (b-a)/2 * tanh(beta * xi) / tanh(beta)

    This creates fine resolution near z=0 (the current sheet) and
    coarser resolution near the boundaries.

    beta controls the clustering strength:
        beta=0: uniform
        beta=3: moderate clustering
        beta=5: strong clustering at center
    """
    xi = np.linspace(-1.0, 1.0, n)
    if beta > 0:
        x = center + 0.5 * (b - a) * np.tanh(beta * xi) / np.tanh(beta)
    else:
        x = center + 0.5 * (b - a) * xi

    # Cell sizes (non-uniform)
    dx = np.diff(x)
    dx = np.append(dx, dx[-1])  # approximate last cell

    return x, dx


def geometric_mesh(a, b, n, r=1.05):
    """
    Geometric (ratio) mesh: dx_{i+1} = r * dx_i

    Useful for boundary layer type problems where resolution
    is needed near one boundary.

    r > 1: cells grow from left to right
    r < 1: cells shrink from left to right
    r = 1: uniform mesh
    """
    if abs(r - 1.0) < 1e-10:
        return uniform_mesh(a, b, n)

    # Total length: L = dx_0 * (1 + r + r^2 + ... + r^{n-1})
    # dx_0 = L * (r - 1) / (r^n - 1)
    L = b - a
    if abs(r) > 1e-10:
        dx0 = L * (r - 1.0) / (r ** n - 1.0) if abs(r ** n - 1.0) > 1e-30 else L / n
    else:
        dx0 = L / n

    # Cell boundaries
    x_bnd = np.zeros(n + 1)
    x_bnd[0] = a
    dx_i = dx0
    for i in range(n):
        x_bnd[i + 1] = x_bnd[i] + dx_i
        dx_i *= r

    # Cell centers
    x = 0.5 * (x_bnd[:-1] + x_bnd[1:])
    dx = np.diff(x_bnd)

    return x, dx


def elliptic_mesh_1d(a, b, n, monitor_func, max_iter=100, tol=1e-6):
    """
    1D elliptic mesh generation with a monitor function.

    Solve the elliptic PDE for the grid mapping:
        d/dxi (M(x) * dx/dxi) = 0

    where M(x) is the monitor function that controls grid spacing.
    High M → fine grid, low M → coarse grid.

    For the reconnection problem:
        M(x) = 1 + alpha * |d^2 Bx / dz^2|
    to cluster points where the current density (J ~ dB/dz) is large.

    Iterative solution by finite differences.
    Maps to the GMSH-to-FEM mesh generation pipeline (475_gmsh_to_fem).
    """
    # Initialize with uniform mesh in computational space
    xi = np.linspace(0, 1, n)
    x = np.linspace(a, b, n)
    dxi = xi[1] - xi[0]

    for iteration in range(max_iter):
        x_old = x.copy()

        # Compute monitor function at current grid points
        M = monitor_func(x)
        M = np.maximum(M, 0.1)  # prevent zero

        # Update interior points by solving the elliptic equation
        for i in range(1, n - 1):
            M_left = 0.5 * (M[i] + M[i - 1])
            M_right = 0.5 * (M[i] + M[i + 1])

            denom = M_left + M_right
            if denom > 1e-30:
                x[i] = (M_left * x[i - 1] + M_right * x[i + 1]) / denom

        # Fix boundaries
        x[0] = a
        x[-1] = b

        # Check convergence
        diff = np.max(np.abs(x - x_old))
        if diff < tol:
            break

    dx = np.gradient(x)
    return x, dx


# ============================================================
# 2D Adaptive Mesh Construction
# ============================================================

def create_reconnection_mesh(nx, nz, Lx, Lz, stretch_z=3.0,
                               stretch_x=0.0):
    """
    Create a 2D mesh for the reconnection simulation with stretching.

    The z-direction is stretched to cluster points near z=0 (current sheet).
    The x-direction can optionally be stretched near the X-point.

    Input:
        nx, nz: number of cells in each direction
        Lx, Lz: domain size
        stretch_z: tanh stretching parameter for z
        stretch_x: tanh stretching parameter for x (0 = uniform)

    Output:
        grid: ReconnectionGrid object
    """
    from mhd_operators import ReconnectionGrid

    if stretch_z > 0:
        z_nodes, dz = tanh_stretch_mesh(0.0, Lz, nz, stretch_z, Lz / 2.0)
    else:
        z_nodes, dz = uniform_mesh(0.0, Lz, nz)

    if stretch_x > 0:
        x_nodes, dx = tanh_stretch_mesh(0.0, Lx, nx, stretch_x, Lx / 2.0)
    else:
        x_nodes, dx = uniform_mesh(0.0, Lx, nx)

    grid = ReconnectionMesh(nx, nz, Lx, Lz, x_nodes, z_nodes, dx, dz)
    return grid


class ReconnectionMesh:
    """
    2D mesh for reconnection simulation with non-uniform spacing.

    Extends ReconnectionGrid to support variable dx, dz arrays.
    """

    def __init__(self, nx, nz, Lx, Lz, x, z, dx, dz):
        self.nx = nx
        self.nz = nz
        self.Lx = Lx
        self.Lz = Lz
        self.x = x
        self.z = z
        self.dx_arr = dx
        self.dz_arr = dz
        self.dx = np.min(dx)  # minimum for CFL
        self.dz = np.min(dz)
        self.X, self.Z = np.meshgrid(x, z, indexing='ij')
        self.dA = np.min(dx) * np.min(dz)
        self.volume = Lx * Lz
        self.bc_type = 'periodic_x_dirichlet_z'

    def __repr__(self):
        return (f"ReconnectionMesh(nx={self.nx}, nz={self.nz}, "
                f"Lx={self.Lx:.2e}, Lz={self.Lz:.2e}, "
                f"dx_min={self.dx:.2e}, dz_min={self.dz:.2e})")


# ============================================================
# Mesh Quality Metrics
# ============================================================

def mesh_quality_metrics(grid):
    """
    Compute mesh quality metrics:
      - Aspect ratio: max(dx, dz) / min(dx, dz)
      - Stretching ratio: max(dx_i) / min(dx_i)
      - Smoothness: max|dx_{i+1} - dx_i| / max(dx_i)
    """
    if hasattr(grid, 'dx_arr') and grid.dx_arr is not None:
        dx = np.atleast_1d(grid.dx_arr)
        dz = np.atleast_1d(grid.dz_arr)
    else:
        dx = np.full(grid.nx, grid.dx)
        dz = np.full(grid.nz, grid.dz)

    # Stretching ratio
    sr_x = float(np.max(dx)) / max(float(np.min(dx)), 1e-30)
    sr_z = float(np.max(dz)) / max(float(np.min(dz)), 1e-30)

    # Smoothness
    if len(dx) > 1:
        smooth_x = float(np.max(np.abs(np.diff(dx)))) / max(float(np.max(dx)), 1e-30)
    else:
        smooth_x = 0.0
    if len(dz) > 1:
        smooth_z = float(np.max(np.abs(np.diff(dz)))) / max(float(np.max(dz)), 1e-30)
    else:
        smooth_z = 0.0

    return {
        'stretch_ratio_x': sr_x,
        'stretch_ratio_z': sr_z,
        'smoothness_x': smooth_x,
        'smoothness_z': smooth_z,
        'dx_min': float(np.min(dx)),
        'dx_max': float(np.max(dx)),
        'dz_min': float(np.min(dz)),
        'dz_max': float(np.max(dz)),
    }


# ============================================================
# Domain Decomposition (maps to 905_pram)
# ============================================================

def domain_decomposition_1d(n_global, n_procs, strategy='contiguous'):
    """
    Decompose a 1D domain into subdomains for parallel computation.

    Maps the PRAM (Parallel Random Access Machine) tiling patterns (905_pram)
    to domain decomposition strategies.

    Strategies:
      - 'contiguous': each proc gets n_global/n_procs consecutive cells
      - 'cyclic': proc i gets cells i, i+n_procs, i+2*n_procs, ...
      - 'block_cyclic': blocks of B cells distributed cyclically

    Returns:
        list of (start_idx, count) for each processor
    """
    if strategy == 'contiguous':
        base = n_global // n_procs
        remainder = n_global % n_procs
        decomp = []
        offset = 0
        for p in range(n_procs):
            count = base + (1 if p < remainder else 0)
            decomp.append((offset, count))
            offset += count
        return decomp

    elif strategy == 'cyclic':
        decomp = [[] for _ in range(n_procs)]
        for i in range(n_global):
            decomp[i % n_procs].append(i)
        return [(p, indices) for p, indices in enumerate(decomp)]

    else:
        # Default: contiguous
        return domain_decomposition_1d(n_global, n_procs, 'contiguous')


def pram_boundary_word(n_procs):
    """
    Generate the PRAM boundary word for the domain decomposition.

    Maps the PRAM grid word structure (905_pram) to the communication
    pattern between processors. Each "letter" represents a boundary
    data exchange direction.

    For n_procs processors in a 1D decomposition:
      - 'A' = exchange with left neighbor
      - 'B' = exchange with right neighbor
    """
    if n_procs <= 1:
        return 'A'

    word = 'A' * (n_procs - 1) + 'B' * (n_procs - 1)
    return word


# ============================================================
# Adaptive Mesh Refinement Indicator
# ============================================================

def refinement_indicator(Bx, Bz, grid, threshold=0.5):
    """
    Compute the refinement indicator based on the current density gradient.

    Cells where |dJ/dz| > threshold * max|dJ/dz| should be refined.

    This is the basis for h-adaptive mesh refinement (AMR).

    Returns:
        refine_flag: 2D boolean array
        indicator: 2D float array (normalized indicator values)
    """
    from mhd_operators import curl_2d, d_dz_4th

    Jy = curl_2d(Bx, Bz, grid)
    dJ_dz = d_dz_4th(Jy, grid.dz)

    max_dJ = np.max(np.abs(dJ_dz))
    if max_dJ > 1e-30:
        indicator = np.abs(dJ_dz) / max_dJ
    else:
        indicator = np.zeros_like(Jy)

    refine_flag = indicator > threshold

    return refine_flag, indicator
