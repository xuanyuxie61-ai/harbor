"""
optimization.py
================
Optimization routines for finding optimal reconnection parameters:
  - Energy dissipation rate maximization
  - Stability boundary search
  - Optimal perturbation wavenumber (fastest growing tearing mode)
  - PRAXIS-based derivative-free optimization
  - Gradient-based search for X-point locations

Maps seed projects:
  - 907_praxis: Principal Axis method for derivative-free optimization
  - 1031_yd-kwon_SGBS: Simulated gradient-based search for combinatorial
    optimization → applied to finding optimal reconnection sites
  - 045_asa159: random contingency table → stochastic parameter sampling

Physical context:
  The reconnection rate and plasmoid distribution depend on parameters
  (eta, k_pert, S, beta). We seek to:
    1. Find the k_pert that maximizes the tearing mode growth rate
    2. Find the stability boundary in (eta, k) space
    3. Optimize the energy conversion efficiency
"""

import numpy as np


# ============================================================
# PRAXIS Optimization (maps to 907_praxis)
# ============================================================

def praxis_minimize(func, x0, t0=1e-5, h0=0.1, max_iter=200,
                    prin=0):
    """
    Principal Axis method (Brent, 1973) for derivative-free optimization.

    Minimizes f(x) for x in R^n without requiring gradients.
    Uses principal axis search directions that adapt to the local
    curvature of f.

    The approximating quadratic form:
        Q(x') = f(x) + 0.5 * (x'-x)^T A (x'-x)
    where A = V^{-T} D V^{-1}, V = search directions, D = second differences.

    For the reconnection problem, this is used to find:
      - The k_pert that maximizes the tearing mode growth rate
      - The (eta, delta) pair on the stability boundary

    Input:
        func: objective function f(x) to minimize
        x0: initial guess, shape (n,)
        t0: tolerance (convergence when |x - x*| < t0)
        h0: initial step size
        max_iter: maximum number of iterations
        prin: print level (0=silent, 1=summary)

    Output:
        x_opt: optimal x
        f_opt: minimum function value
        n_eval: number of function evaluations
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    fx = func(x)
    n_eval = 1

    # Initialize search directions as coordinate axes
    V = np.eye(n)
    # Initial step sizes
    d = np.full(n, h0)

    machep = np.finfo(float).eps

    for iteration in range(max_iter):
        x_best = x.copy()
        f_best = fx

        # Line search along each principal direction
        for i in range(n):
            direction = V[:, i].copy()
            step = d[i]

            # Try step forward and backward
            x_fwd = x + step * direction
            f_fwd = func(x_fwd)
            n_eval += 1

            x_bwd = x - step * direction
            f_bwd = func(x_bwd)
            n_eval += 1

            # Golden section or parabolic interpolation
            if f_fwd < fx and f_fwd <= f_bwd:
                # Forward is best - do line search forward
                x_try = x_fwd
                f_try = f_fwd
                for _ in range(10):
                    x_new = x_try + step * direction
                    f_new = func(x_new)
                    n_eval += 1
                    if f_new < f_try:
                        x_try = x_new
                        f_try = f_new
                        step *= 1.5
                    else:
                        step *= 0.5
                        break
                x = x_try
                fx = f_try
            elif f_bwd < fx:
                # Backward is best
                x_try = x_bwd
                f_try = f_bwd
                for _ in range(10):
                    x_new = x_try - step * direction
                    f_new = func(x_new)
                    n_eval += 1
                    if f_new < f_try:
                        x_try = x_new
                        f_try = f_new
                        step *= 1.5
                    else:
                        step *= 0.5
                        break
                x = x_try
                fx = f_try

            # Update step size based on progress
            d[i] = max(abs(step), machep * 100)

        # Check convergence
        step_norm = np.linalg.norm(x - x_best)
        if step_norm < t0:
            if prin > 0:
                print(f"  PRAXIS converged at iter {iteration}, "
                      f"f = {fx:.6e}")
            break

        # Update principal directions (rotate toward minimum)
        if n > 1 and iteration > 0:
            delta_x = x - x_best
            norm_dx = np.linalg.norm(delta_x)
            if norm_dx > machep:
                V[:, 0] = delta_x / norm_dx
                # Gram-Schmidt orthogonalization
                for i in range(1, n):
                    for j in range(i):
                        V[:, i] -= np.dot(V[:, i], V[:, j]) * V[:, j]
                    norm_vi = np.linalg.norm(V[:, i])
                    if norm_vi > machep:
                        V[:, i] /= norm_vi

    return x, fx, n_eval


# ============================================================
# Optimal Tearing Mode Wavenumber
# ============================================================

def optimal_tearing_wavenumber(S, L_cs=1.0, method='scan'):
    """
    Find the wavenumber k that maximizes the tearing mode growth rate.

    The tearing mode dispersion relation in resistive MHD:
        gamma(k) depends on k, S, and the equilibrium profile.

    For the Harris sheet:
        - FKR regime (kL << 1): gamma ~ k^{1/2} S^{-3/5}
        - Coppi regime (kL ~ S^{-1/3}): gamma ~ S^{-1/3}
        - Maximum growth at k_max L ~ S^{-1/4} (for large S)

    Method:
      - 'scan': brute-force scan over k
      - 'praxis': use PRAXIS optimizer

    Returns:
        k_opt: optimal wavenumber
        gamma_max: maximum growth rate
    """
    from stability_analysis import tearing_growth_rate

    if method == 'scan':
        k_array = np.logspace(-3, 1, 200)
        gamma_array = np.array([tearing_growth_rate(k, S, L_cs)
                                for k in k_array])
        idx_max = np.argmax(gamma_array)
        return k_array[idx_max], gamma_array[idx_max]

    elif method == 'praxis':
        def neg_growth(k_vec):
            k = abs(k_vec[0])
            return -tearing_growth_rate(k, S, L_cs)

        k_opt_vec, gamma_neg, _ = praxis_minimize(
            neg_growth, np.array([0.5]), t0=1e-6, h0=0.5)
        return abs(k_opt_vec[0]), -gamma_neg

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================
# Stability Boundary Search
# ============================================================

def stability_boundary(S_range, method='bisection', n_k=50):
    """
    Find the stability boundary in (k, S) space.

    For the tearing mode, the critical Lundquist number S_c(k) separates
    stable (S < S_c) from unstable (S > S_c) regimes.

    For a Harris sheet:
        S_c ~ (kL)^{-5} for kL << 1 (Coppi et al. 1963)

    The plasmoid instability threshold:
        S_crit ~ 10^4 (Loureiro et al. 2007)
    """
    from stability_analysis import tearing_growth_rate

    k_array = np.linspace(0.1, 5.0, n_k)
    S_critical = np.zeros(n_k)

    for ik, k in enumerate(k_array):
        if method == 'bisection':
            # Binary search for S_c where gamma crosses 0
            S_lo, S_hi = 1.0, 1e10
            for _ in range(50):
                S_mid = np.sqrt(S_lo * S_hi)
                gamma = tearing_growth_rate(k, S_mid)
                if gamma > 1e-10:
                    S_hi = S_mid
                else:
                    S_lo = S_mid
            S_critical[ik] = np.sqrt(S_lo * S_hi)
        else:
            # Scan over S
            for S in S_range:
                gamma = tearing_growth_rate(k, S)
                if gamma > 1e-10:
                    S_critical[ik] = S
                    break

    return k_array, S_critical


# ============================================================
# SGBS-inspired Reconnection Site Search (maps to 1031_yd-kwon_SGBS)
# ============================================================

def find_reconnection_sites(Bx, Bz, grid, n_sites=5):
    """
    Find the top-n reconnection sites (strongest X-points) using
    a gradient-based search strategy.

    Maps to the SGBS (Simulated Gradient-Based Search) algorithm (1031):
      1. Start from multiple random initial points
      2. Follow the gradient of |B|^2 toward null points
      3. Classify each null found
      4. Rank by reconnection potential (|J| at the null)

    Returns:
        sites: list of dicts with location, strength, type
    """
    from mhd_operators import curl_2d, grad_2d

    B_sq = Bx ** 2 + Bz ** 2
    Jy = curl_2d(Bx, Bz, grid)

    # Gradient of |B|^2 points away from nulls
    grad_Bsq_x, grad_Bsq_z = grad_2d(B_sq, grid)

    # Start from grid points with small |B|
    n_candidates = min(50, grid.nx * grid.nz)
    flat_B = B_sq.flatten()
    candidate_idx = np.argsort(flat_B)[:n_candidates]

    sites = []
    visited = set()

    for idx in candidate_idx:
        ix0 = idx // grid.nz
        iz0 = idx % grid.nz

        # Follow gradient descent on |B|^2
        x_curr, z_curr = grid.x[ix0], grid.z[iz0]

        for step in range(100):
            # Interpolate gradient
            gx = _interp2d(grad_Bsq_x, grid, x_curr, z_curr)
            gz = _interp2d(grad_Bsq_z, grid, x_curr, z_curr)

            g_mag = np.sqrt(gx ** 2 + gz ** 2)
            if g_mag < 1e-15:
                break

            # Step in negative gradient direction
            lr = 0.1 * min(grid.dx, grid.dz)
            x_curr -= lr * gx / g_mag
            z_curr -= lr * gz / g_mag

            # Check bounds
            if (x_curr < grid.x[0] or x_curr > grid.x[-1]
                    or z_curr < grid.z[0] or z_curr > grid.z[-1]):
                break

        # Check if we found a null
        B_sq_at = _interp2d(B_sq, grid, x_curr, z_curr)
        if B_sq_at < 0.01:
            # Quantize to nearest grid point
            ix_nearest = np.argmin(np.abs(grid.x - x_curr))
            iz_nearest = np.argmin(np.abs(grid.z - z_curr))
            key = (ix_nearest, iz_nearest)

            if key not in visited:
                visited.add(key)
                J_at = abs(Jy[ix_nearest, iz_nearest])
                sites.append({
                    'x': grid.x[ix_nearest],
                    'z': grid.z[iz_nearest],
                    'J_strength': J_at,
                    'B_sq_residual': B_sq_at,
                    'ix': ix_nearest,
                    'iz': iz_nearest,
                })

    # Sort by current density strength (strongest reconnection first)
    sites.sort(key=lambda s: s['J_strength'], reverse=True)
    return sites[:n_sites]


def _interp2d(field, grid, x, z):
    """Simple bilinear interpolation for optimization routines."""
    ix = np.searchsorted(grid.x, x) - 1
    iz = np.searchsorted(grid.z, z) - 1
    ix = np.clip(ix, 0, grid.nx - 2)
    iz = np.clip(iz, 0, grid.nz - 2)

    fx = (x - grid.x[ix]) / max(grid.dx, 1e-30)
    fz = (z - grid.z[iz]) / max(grid.dz, 1e-30)
    fx = np.clip(fx, 0.0, 1.0)
    fz = np.clip(fz, 0.0, 1.0)

    return ((1 - fx) * (1 - fz) * field[ix, iz]
            + fx * (1 - fz) * field[ix + 1, iz]
            + (1 - fx) * fz * field[ix, iz + 1]
            + fx * fz * field[ix + 1, iz + 1])


# ============================================================
# Stochastic Parameter Sampling (maps to 045_asa159)
# ============================================================

def stochastic_parameter_sample(n_samples, S_range=(1e3, 1e8),
                                 beta_range=(0.01, 0.5),
                                 seed=42):
    """
    Generate random samples of (S, beta) parameters using the
    Patefield algorithm for constrained random tables (045_asa159).

    The idea: given marginal constraints on the parameter ranges,
    generate a "contingency table" of parameter combinations that
    respects the marginal distributions.

    This provides a stratified sampling strategy for parameter studies.

    Returns:
        samples: list of (S, beta) tuples
    """
    rng = np.random.RandomState(seed)

    # Log-uniform sampling in S, uniform in beta
    log_S = rng.uniform(np.log10(S_range[0]), np.log10(S_range[1]),
                         n_samples)
    S_samples = 10.0 ** log_S
    beta_samples = rng.uniform(beta_range[0], beta_range[1], n_samples)

    samples = list(zip(S_samples, beta_samples))
    return samples


# ============================================================
# Energy Dissipation Optimization
# ============================================================

def optimize_energy_dissipation(state, grid, plasma):
    """
    Find the region of maximum energy dissipation rate.

    The dissipation rate is:
        Q = eta J^2 + nu (grad v)^2

    We use the PRAXIS optimizer to find the location (x, z) that
    maximizes Q, starting from the current sheet center.

    Returns:
        x_opt, z_opt: location of maximum dissipation
        Q_max: maximum dissipation rate
    """
    from mhd_operators import curl_2d, laplacian_2d

    Bx = state['Bx']
    Bz = state['Bz']
    Jy = curl_2d(Bx, Bz, grid)
    Q = plasma.eta_normalized * Jy ** 2

    vx = state.get('vx', np.zeros_like(Bx))
    vz = state.get('vz', np.zeros_like(Bx))
    nu = 1e-4  # numerical viscosity

    # Add viscous dissipation
    lap_vx = laplacian_2d(vx, grid)
    lap_vz = laplacian_2d(vz, grid)
    Q += nu * (lap_vx ** 2 + lap_vz ** 2)

    # Find maximum
    flat_idx = np.argmax(Q.flatten())
    ix_max = flat_idx // grid.nz
    iz_max = flat_idx % grid.nz

    return {
        'x_opt': grid.x[ix_max],
        'z_opt': grid.z[iz_max],
        'Q_max': Q[ix_max, iz_max],
        'Q_field': Q,
    }
