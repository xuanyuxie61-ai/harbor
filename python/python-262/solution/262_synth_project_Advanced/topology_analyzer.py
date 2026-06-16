"""
topology_analyzer.py
====================
Analysis of magnetic field topology during reconnection:
  - X-point (null point) detection
  - Magnetic island (plasmoid) identification
  - Field line connectivity and tracing
  - Voronoi-based domain partitioning around null points
  - Squashedness (Q) analysis for quasi-separatrix layers (QSLs)

Physical context:
  - During reconnection, the magnetic topology changes: field lines
    break and reconnect, forming magnetic islands (plasmoids)
  - X-points are where B = 0 and the field has a hyperbolic structure
  - QSLs are regions of strong field line mapping gradient (finite Q)
  - The squashing factor Q = |d(x_foot1)/d(x_foot2)| characterizes QSLs

Maps seed projects:
  - 1394_voronoi_city: Voronoi diagram for domain partitioning →
    used to partition the reconnection layer around null points
  - 1151: polymer topology → field line topology / connectivity matrix
  - 1031_yd-kwon_SGBS: gradient-based search for optimal X-point locations
  - 329_ellipse_distance: distance metrics for null point classification
"""

import numpy as np


# ============================================================
# X-Point (Magnetic Null) Detection
# ============================================================

def find_null_points(Bx, Bz, grid, threshold=0.1):
    """
    Find magnetic null points (X-points) where |B| ~ 0.

    A 2D null point satisfies:
        Bx(x, z) = 0  AND  Bz(x, z) = 0

    In practice, we search for local minima of |B|^2 below a threshold.

    Classification (Priest & Titov 1996):
      - If eigenvalues of grad(B) are real, opposite sign → X-point
      - If eigenvalues are imaginary → O-point (magnetic island center)

    Input:
        Bx, Bz: 2D magnetic field arrays
        grid: ReconnectionGrid
        threshold: |B| below this is considered a null

    Output:
        nulls: list of dicts with keys 'x', 'z', 'type', 'eigenvalues'
    """
    B_sq = Bx ** 2 + Bz ** 2
    nx, nz = B_sq.shape

    nulls = []

    # Find local minima of |B|^2
    for i in range(2, nx - 2):
        for j in range(2, nz - 2):
            if B_sq[i, j] < threshold ** 2:
                # Check if it's a local minimum
                is_min = True
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        if B_sq[i + di, j + dj] <= B_sq[i, j]:
                            is_min = False
                            break
                    if not is_min:
                        break

                if is_min:
                    # Classify the null point via the Jacobian of B
                    jac = compute_B_jacobian(Bx, Bz, grid, i, j)
                    eigvals = np.linalg.eigvals(jac)

                    # X-point: real eigenvalues of opposite sign
                    # O-point: imaginary eigenvalues
                    if np.all(np.isreal(eigvals)):
                        if eigvals[0].real * eigvals[1].real < 0:
                            null_type = 'X'
                        else:
                            null_type = 'degenerate'
                    else:
                        null_type = 'O'

                    nulls.append({
                        'x': grid.x[i],
                        'z': grid.z[j],
                        'B_sq': B_sq[i, j],
                        'type': null_type,
                        'eigenvalues': eigvals,
                        'jacobian': jac,
                        'ix': i,
                        'iz': j,
                    })

    return nulls


def compute_B_jacobian(Bx, Bz, grid, i, j):
    """
    Compute the 2x2 Jacobian matrix of the magnetic field at point (i, j):

        J = [[dBx/dx, dBx/dz],
             [dBz/dx, dBz/dz]]

    Uses 2nd-order central differences.
    """
    dx = grid.dx
    dz = grid.dz

    dBx_dx = (Bx[i + 1, j] - Bx[i - 1, j]) / (2.0 * dx)
    dBx_dz = (Bx[i, j + 1] - Bx[i, j - 1]) / (2.0 * dz)
    dBz_dx = (Bz[i + 1, j] - Bz[i - 1, j]) / (2.0 * dx)
    dBz_dz = (Bz[i, j + 1] - Bz[i, j - 1]) / (2.0 * dz)

    return np.array([[dBx_dx, dBx_dz],
                     [dBz_dx, dBz_dz]])


# ============================================================
# Voronoi Domain Partitioning (maps to 1394_voronoi_city)
# ============================================================

def voronoi_partition(nulls, grid):
    """
    Partition the reconnection domain into Voronoi cells around
    each magnetic null point.

    Maps the Voronoi city concept (1394_voronoi_city): each null point
    is a "city", and the Voronoi cell defines its domain of influence.

    The perpendicular bisectors between null pairs define the boundaries
    of magnetic domains with different connectivity.

    Input:
        nulls: list of null point dicts (from find_null_points)
        grid: ReconnectionGrid

    Output:
        partition: 2D int array, partition[i,j] = index of nearest null
    """
    if len(nulls) == 0:
        return np.zeros((grid.nx, grid.nz), dtype=int)

    partition = np.zeros((grid.nx, grid.nz), dtype=int)

    for i in range(grid.nx):
        for j in range(grid.nz):
            min_dist = np.inf
            nearest = 0
            for k, null in enumerate(nulls):
                dist = ((grid.x[i] - null['x']) ** 2
                        + (grid.z[j] - null['z']) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    nearest = k
            partition[i, j] = nearest

    return partition


# ============================================================
# Field Line Tracing
# ============================================================

def trace_field_line(Bx, Bz, grid, x0, z0, direction=1.0, ds=0.1,
                     max_steps=5000):
    """
    Trace a magnetic field line from starting point (x0, z0) by
    integrating dx/ds = B / |B|.

    Uses 4th-order Runge-Kutta for the field line ODE:
        dx/ds = Bx(x,z) / |B(x,z)|
        dz/ds = Bz(x,z) / |B(x,z)|

    Stops when the field line exits the domain or max_steps is reached.

    Input:
        direction: +1 or -1 for tracing direction
        ds: step size along the field line
        max_steps: maximum number of integration steps

    Output:
        x_line, z_line: arrays of field line coordinates
    """
    x_line = [x0]
    z_line = [z0]

    x, z = x0, z0

    for _ in range(max_steps):
        # Interpolate B to current position
        Bx_interp = bilinear_interp(Bx, grid, x, z)
        Bz_interp = bilinear_interp(Bz, grid, x, z)

        B_mag = np.sqrt(Bx_interp ** 2 + Bz_interp ** 2)
        if B_mag < 1e-15:
            break

        # RK4 integration of field line ODE
        def rhs(xc, zc):
            bx = bilinear_interp(Bx, grid, xc, zc)
            bz = bilinear_interp(Bz, grid, xc, zc)
            bm = np.sqrt(bx ** 2 + bz ** 2) + 1e-30
            return direction * bx / bm, direction * bz / bm

        k1x, k1z = rhs(x, z)
        k2x, k2z = rhs(x + 0.5 * ds * k1x, z + 0.5 * ds * k1z)
        k3x, k3z = rhs(x + 0.5 * ds * k2x, z + 0.5 * ds * k2z)
        k4x, k4z = rhs(x + ds * k3x, z + ds * k3z)

        x += ds / 6.0 * (k1x + 2 * k2x + 2 * k3x + k4x)
        z += ds / 6.0 * (k1z + 2 * k2z + 2 * k3z + k4z)

        # Check domain bounds
        if (x < grid.x[0] or x > grid.x[-1]
                or z < grid.z[0] or z > grid.z[-1]):
            break

        x_line.append(x)
        z_line.append(z)

    return np.array(x_line), np.array(z_line)


def bilinear_interp(field, grid, x, z):
    """
    Bilinear interpolation of a 2D field at point (x, z).
    """
    # Find grid cell
    ix = np.searchsorted(grid.x, x) - 1
    iz = np.searchsorted(grid.z, z) - 1

    ix = np.clip(ix, 0, grid.nx - 2)
    iz = np.clip(iz, 0, grid.nz - 2)

    # Local coordinates
    fx = (x - grid.x[ix]) / max(grid.dx, 1e-30)
    fz = (z - grid.z[iz]) / max(grid.dz, 1e-30)
    fx = np.clip(fx, 0.0, 1.0)
    fz = np.clip(fz, 0.0, 1.0)

    # Bilinear interpolation
    val = ((1 - fx) * (1 - fz) * field[ix, iz]
           + fx * (1 - fz) * field[ix + 1, iz]
           + (1 - fx) * fz * field[ix, iz + 1]
           + fx * fz * field[ix + 1, iz + 1])
    return val


# ============================================================
# Connectivity Matrix (maps to 1151 polymer topology)
# ============================================================

def connectivity_matrix(Bx, Bz, grid, n_lines=20):
    """
    Compute the magnetic field line connectivity matrix.

    Maps the polymer topology analysis (1151): just as polymer chains
    have topological invariants (knots, links), magnetic field lines
    have connectivity invariants.

    Trace field lines from the z-boundaries and record where they
    connect on the opposite boundary. The connectivity matrix
    C[i,j] measures the flux connection between boundary element i
    on z_min and boundary element j on z_max.

    Analogous to the radius of gyration for polymers, we compute
    the "field line span" - the distance between footpoints.
    """
    nx_bnd = min(n_lines, grid.nx)
    z_min = grid.z[0]
    z_max = grid.z[-1]

    # Starting points along z_min boundary
    x_starts = np.linspace(grid.x[2], grid.x[-3], nx_bnd)

    connections = []
    spans = []

    for x0 in x_starts:
        # Trace upward
        x_up, z_up = trace_field_line(Bx, Bz, grid, x0, z_min,
                                       direction=1.0, max_steps=2000)
        if len(x_up) > 1:
            x_end = x_up[-1]
            span = abs(x_end - x0)
            connections.append((x0, x_end))
            spans.append(span)

    return {
        'connections': connections,
        'spans': np.array(spans),
        'mean_span': np.mean(spans) if spans else 0.0,
        'max_span': np.max(spans) if spans else 0.0,
        'n_connected': len(connections),
    }


# ============================================================
# Squashing Factor Q (for QSL detection)
# ============================================================

def compute_squashing_Q(Bx, Bz, grid, x0, z0, delta=0.01):
    """
    Compute the squashing factor Q at a field line footpoint.

    Q measures the gradient of the field line mapping between boundaries:
        Q = (|N|^2 + |M|^2 + |N x M|^2) / |B_n1 B_n2|

    where N, M are the columns of the deformation matrix:
        N = d(x2, z2) / d(x1)
        M = d(x2, z2) / d(z1)

    High Q identifies quasi-separatrix layers (QSLs) where reconnection
    can occur even without null points (Titov & Démoulin 1999).

    Input:
        delta: perturbation size for finite-difference gradient
    """
    # Trace from 4 nearby points to compute the Jacobian
    def trace_endpoint(xs, zs):
        xl, zl = trace_field_line(Bx, Bz, grid, xs, zs,
                                   direction=1.0, max_steps=2000)
        if len(xl) > 1:
            return xl[-1], zl[-1]
        return xs, zs

    # Central differences for the mapping Jacobian
    x2_p, z2_p = trace_endpoint(x0 + delta, z0)
    x2_m, z2_m = trace_endpoint(x0 - delta, z0)
    x3_p, z3_p = trace_endpoint(x0, z0 + delta)
    x3_m, z3_m = trace_endpoint(x0, z0 - delta)

    # Jacobian elements
    dx2_dx1 = (x2_p - x2_m) / (2.0 * delta)
    dx2_dz1 = (x3_p - x3_m) / (2.0 * delta)

    # Simplified Q (2D approximation)
    det_J = abs(dx2_dx1)
    trace_J = dx2_dx1 ** 2 + dx2_dz1 ** 2

    Q = trace_J / max(det_J, 1e-30)
    return Q


# ============================================================
# Reconnection Rate Measurement
# ============================================================

def measure_reconnection_rate(Bx, Bz, grid, location='x_point'):
    """
    Measure the reconnection rate via the out-of-plane electric field
    at the X-point (or the average along the current sheet).

    E_rec = |E_y| at the X-point = eta * J_y at the null

    In normalized units, this gives the inflow Alfvén Mach number:
        M_A = E_rec / (B_up * V_A) ~ reconnection rate

    Sweet-Parker: M_A ~ S^{-1/2}
    Petschek: M_A ~ pi / (8 ln S)
    Fast (plasmoid): M_A ~ 0.01 (independent of S for S > 10^4)
    """
    # Find the X-point closest to the center
    nulls = find_null_points(Bx, Bz, grid)

    if len(nulls) == 0:
        return 0.0, nulls

    # Find the null closest to domain center
    cx = 0.5 * grid.Lx
    cz = 0.5 * grid.Lz
    best_null = min(nulls,
                    key=lambda n: (n['x'] - cx) ** 2 + (n['z'] - cz) ** 2)

    # Reconnection rate = |B| at the inflow boundary of the null
    ix, iz = best_null['ix'], best_null['iz']
    B_inflow = np.sqrt(Bx[ix, iz] ** 2 + Bz[ix, iz] ** 2)

    return B_inflow, nulls
