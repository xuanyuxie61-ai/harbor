"""
Spectral Quadrature Rules for Energy Budget Diagnostics
=======================================================
Derived from seed project 939_quad_fast_rule (Clenshaw-Curtis,
Fejér, Gauss-Legendre quadrature).

In mesoscale eddy dynamics, accurate spectral integration is essential
for evaluating energy budget terms, such as:

    E(k) = ½∫_{|k'|=k} |ψ̂(k')|²  k'²  dk'

These rules provide machine-precision integration of smooth spectral
integrands on [−1, 1], which are mapped to radial wavenumber shells.

Key Formulas:
- Gauss-Legendre abscissas xᵢ are roots of Legendre polynomial Pₙ(x).
- Weights: wᵢ = 2 / [(1−xᵢ²) (Pₙ'(xᵢ))²]
- Clenshaw-Curtis: xᵢ = cos(πi/n), weights via discrete cosine transform.
- Fejér Type-1: open nodes xᵢ = cos(π(2i−1)/(2n)), closed-form weights.
"""

import numpy as np

def gauss_legendre_rule(n):
    """
    Compute Gauss-Legendre quadrature nodes and weights on [−1, 1].

    Nodes are eigenvalues of the symmetric tridiagonal Jacobi matrix:
        Jᵢ,ᵢ   = 0
        Jᵢ,ᵢ₊₁ = i / sqrt(4i² − 1)

    Parameters
    ----------
    n : int
        Number of nodes (n ≥ 1).

    Returns
    -------
    x, w : ndarray
        Nodes and weights.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    # Build symmetric tridiagonal Jacobi matrix
    i = np.arange(1.0, n, dtype=np.float64)
    beta = i / np.sqrt(4.0 * i**2 - 1.0)
    J = np.diag(beta, 1) + np.diag(beta, -1)
    eigvals, eigvecs = np.linalg.eigh(J)
    x = eigvals
    w = 2.0 * (eigvecs[0, :]**2)
    return x, w

def clenshaw_curtis_rule(n):
    """
    Compute Clenshaw-Curtis quadrature nodes and weights on [−1, 1].

    Nodes: xⱼ = cos(jπ / n),  j = 0,…,n
    Weights computed via explicit DCT-II formula:
        wⱼ = cⱼ · Σ_{k=0}^{⌊n/2⌋}  bₖ / (4k²−1) · cos(2kjπ/n)
    where b₀ = 1, bₖ = 2/(4k²−1) for k≥1, and cⱼ are endpoint factors.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    j = np.arange(n + 1)
    x = np.cos(j * np.pi / n)

    # Chebyshev weight coefficients
    k = np.arange(n // 2 + 1)
    b = np.ones_like(k, dtype=np.float64)
    b[0] = 1.0
    if len(b) > 1:
        b[1:] = 2.0 / (4.0 * k[1:]**2 - 1.0)

    # DCT-II to get weights
    w = np.zeros(n + 1, dtype=np.float64)
    for idx in range(n + 1):
        s = np.sum(b * np.cos(2.0 * k * idx * np.pi / n))
        if idx == 0 or idx == n:
            s *= 0.5
        w[idx] = s / n

    # Endpoint corrections
    w[0] *= 0.5
    w[-1] *= 0.5
    w[0] += 1.0 / (n**2 - 1.0) if n > 1 else 0.0
    w[-1] += 1.0 / (n**2 - 1.0) if n > 1 else 0.0
    return x, w

def fejer1_rule(n):
    """
    Fejér Type-1 quadrature (open, excludes endpoints).
    Nodes: xⱼ = cos( (2j−1)π / (2n) ), j = 1,…,n
    Weights: closed-form via sine series.
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    j = np.arange(1, n + 1)
    x = np.cos((2.0 * j - 1.0) * np.pi / (2.0 * n))

    w = np.zeros(n, dtype=np.float64)
    m = n // 2
    for idx in range(n):
        s = 0.0
        for k in range(1, m + 1):
            s += np.sin((2.0 * k - 1.0) * (2.0 * idx + 1.0) * np.pi / (2.0 * n)) / (2.0 * k - 1.0)
        w[idx] = (2.0 / n) * s
    return x, w

def integrate_radial_energy_spectrum(k_vals, E_vals, rule='gauss_legendre', n_quad=64):
    """
    Integrate the radial energy spectrum E(k) over a finite band
    [k_min, k_max] using high-order spectral quadrature.

    Parameters
    ----------
    k_vals, E_vals : ndarray, 1-D
        Sampled wavenumbers and energy density values.
    rule : str
        'gauss_legendre', 'clenshaw_curtis', or 'fejer1'.
    n_quad : int
        Number of quadrature points.

    Returns
    -------
    energy_band : float
        Integrated energy in the band.
    """
    k_vals = np.asarray(k_vals).ravel()
    E_vals = np.asarray(E_vals).ravel()
    if len(k_vals) < 2:
        raise ValueError("k_vals must have at least 2 points.")
    k_min, k_max = float(np.min(k_vals)), float(np.max(k_vals))
    if k_min < 0 or k_max <= k_min:
        raise ValueError("Invalid wavenumber range.")

    if rule == 'gauss_legendre':
        t, w = gauss_legendre_rule(n_quad)
    elif rule == 'clenshaw_curtis':
        t, w = clenshaw_curtis_rule(n_quad)
    elif rule == 'fejer1':
        t, w = fejer1_rule(n_quad)
    else:
        raise ValueError(f"Unknown quadrature rule: {rule}")

    # Affine map
    k_nodes = 0.5 * (k_max - k_min) * t + 0.5 * (k_max + k_min)
    jac = 0.5 * (k_max - k_min)

    # Sort and interpolate E(k) to quadrature nodes
    sort_idx = np.argsort(k_vals)
    k_sorted = k_vals[sort_idx]
    E_sorted = E_vals[sort_idx]
    k_nodes_clipped = np.clip(k_nodes, k_sorted[0], k_sorted[-1])
    E_nodes = np.interp(k_nodes_clipped, k_sorted, E_sorted)

    energy_band = float(np.sum(w * E_nodes) * jac)
    return energy_band

def compute_spectral_inner_product(spec1, spec2, ksq, dx, dy, Nx, Ny):
    """
    Compute L² inner product in spectral space using quadrature weights:
        ⟨f, g⟩ = ∫ f̂(k) · ĝ*(k) dk
    Discretized with spectral quadrature on radial shells.
    """
    dA_spec = (2 * np.pi / (Nx * dx)) * (2 * np.pi / (Ny * dy))
    integrand = np.real(spec1 * np.conj(spec2))
    return np.sum(integrand) * dA_spec
