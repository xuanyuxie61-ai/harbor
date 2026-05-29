"""
volume_integrator.py
====================
3D Gauss-Legendre quadrature for integrating electromagnetic quantities
over nanostructure volumes.

The integral of a function f(x,y,z) over a rectangular box
[a₁,b₁] × [a₂,b₂] × [a₃,b₃] is approximated by the tensor-product rule:

    I ≈ Σ_{i=1}^{n_x} Σ_{j=1}^{n_y} Σ_{k=1}^{n_z}
        w_i^{(x)} w_j^{(y)} w_k^{(z)} f(x_i, y_j, z_k)

where {x_i, w_i^{(x)}} are the Gauss-Legendre nodes and weights on [a₁,b₁],
obtained by mapping the standard nodes ξ_i ∈ [−1,1]:

    x_i = ((1−ξ_i) a₁ + (1+ξ_i) b₁) / 2
    w_i^{(x)} = w_i^{(std)} (b₁ − a₁) / 2

For a nanoparticle of arbitrary shape, the box is first partitioned into
sub-boxes, and only sub-boxes inside the particle are integrated.

Key integrals computed:
  - Electromagnetic energy density:
      u_EM = (1/2) (ε |E|² + μ |H|²)
  - Absorbed power:
      P_abs = (1/2) ω ε₀ Im[ε] ∫ |E|² dV
  - Hot-carrier generation volume:
      V_hc = ∫ Θ(|E|² − E_th²) dV
"""

import numpy as np


def legendre_nodes_weights_1d(n):
    """
    Compute Gauss-Legendre nodes and weights on [−1, 1].

    Parameters
    ----------
    n : int
        Number of nodes (1 ≤ n ≤ 10 supported directly; falls back to numpy).

    Returns
    -------
    x : ndarray
    w : ndarray
    """
    if n < 1:
        raise ValueError("n must be positive.")
    if n <= 100:
        x, w = np.polynomial.legendre.leggauss(n)
        return x, w
    else:
        raise ValueError("n too large for direct Gauss-Legendre computation.")


def gauss_legendre_3d_set(a, b, nx, ny, nz):
    """
    Set up a 3D tensor-product Gauss-Legendre quadrature rule.

    Parameters
    ----------
    a : ndarray, shape (3,)
        Lower limits.
    b : ndarray, shape (3,)
        Upper limits.
    nx, ny, nz : int
        Orders in each direction.

    Returns
    -------
    x, y, z : ndarray
        Quadrature nodes.
    w : ndarray
        Quadrature weights.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.any(b <= a):
        raise ValueError("Upper limits must exceed lower limits.")

    xx, wx = legendre_nodes_weights_1d(nx)
    yy, wy = legendre_nodes_weights_1d(ny)
    zz, wz = legendre_nodes_weights_1d(nz)

    # Map to [a, b]
    xx = ((1.0 - xx) * a[0] + (1.0 + xx) * b[0]) / 2.0
    wx = wx * (b[0] - a[0]) / 2.0

    yy = ((1.0 - yy) * a[1] + (1.0 + yy) * b[1]) / 2.0
    wy = wy * (b[1] - a[1]) / 2.0

    zz = ((1.0 - zz) * a[2] + (1.0 + zz) * b[2]) / 2.0
    wz = wz * (b[2] - a[2]) / 2.0

    n_total = nx * ny * nz
    x = np.zeros(n_total)
    y = np.zeros(n_total)
    z = np.zeros(n_total)
    w = np.zeros(n_total)

    idx = 0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                x[idx] = xx[i]
                y[idx] = yy[j]
                z[idx] = zz[k]
                w[idx] = wx[i] * wy[j] * wz[k]
                idx += 1

    return x, y, z, w


def integrate_over_box(f, a, b, nx=6, ny=6, nz=6):
    """
    Integrate a scalar function f(x,y,z) over a box using 3D Gauss-Legendre.

    Parameters
    ----------
    f : callable
        Function f(x, y, z) returning float or ndarray.
    a, b : ndarray, shape (3,)
    nx, ny, nz : int

    Returns
    -------
    result : float
    """
    x, y, z, w = gauss_legendre_3d_set(a, b, nx, ny, nz)
    values = f(x, y, z)
    return float(np.sum(w * values))


def electromagnetic_energy_density_integral(epsilon, mu, E_field_func, H_field_func,
                                            a, b, nx=6, ny=6, nz=6):
    """
    Compute total electromagnetic energy in a volume:

        U = (1/2) ∫ [ ε |E|² + μ |H|² ] dV

    Parameters
    ----------
    epsilon : float or complex
        Permittivity.
    mu : float
        Permeability.
    E_field_func, H_field_func : callable
        Functions returning |E|² and |H|² at arrays of points.
    a, b : ndarray
    nx, ny, nz : int

    Returns
    -------
    U : float
        Total energy (J).
    """
    x, y, z, w = gauss_legendre_3d_set(a, b, nx, ny, nz)
    e2 = E_field_func(x, y, z)
    h2 = H_field_func(x, y, z)
    integrand = 0.5 * (np.real(epsilon) * e2 + mu * h2)
    return float(np.sum(w * integrand))


def absorbed_power_integral(omega, epsilon, E_field_func,
                            a, b, nx=6, ny=6, nz=6):
    """
    Compute absorbed power from Ohmic losses:

        P_abs = (1/2) ω ε₀ Im[ε] ∫ |E|² dV

    Parameters
    ----------
    omega : float
    epsilon : complex
    E_field_func : callable
    a, b : ndarray

    Returns
    -------
    P_abs : float
    """
    eps0 = 8.854187817e-12
    x, y, z, w = gauss_legendre_3d_set(a, b, nx, ny, nz)
    e2 = E_field_func(x, y, z)
    integrand = 0.5 * omega * eps0 * np.imag(epsilon) * e2
    # Ensure non-negative absorbed power (Im[ε] < 0 means absorption)
    integrand = np.maximum(integrand, 0.0)
    return float(np.sum(w * integrand))


def test_exactness_monomial(a, b, max_total_degree, nx=6, ny=6, nz=6):
    """
    Verify quadrature exactness by integrating monomials x^i y^j z^k
    up to total degree T = i+j+k.

    Exact integral:
        I = [(b₁^{i+1}−a₁^{i+1})/(i+1)] × [(b₂^{j+1}−a₂^{j+1})/(j+1)]
            × [(b₃^{k+1}−a₃^{k+1})/(k+1)]

    Parameters
    ----------
    a, b : ndarray
    max_total_degree : int
    nx, ny, nz : int

    Returns
    -------
    errors : list of float
        Relative errors for each monomial tested.
    """
    errors = []
    for t in range(max_total_degree + 1):
        for k in range(t + 1):
            for j in range(t - k + 1):
                i = t - j - k
                p = np.array([i, j, k])

                exact = (
                    (b[0] ** (p[0] + 1) - a[0] ** (p[0] + 1)) / (p[0] + 1) *
                    (b[1] ** (p[1] + 1) - a[1] ** (p[1] + 1)) / (p[1] + 1) *
                    (b[2] ** (p[2] + 1) - a[2] ** (p[2] + 1)) / (p[2] + 1)
                )

                def monomial(xx, yy, zz):
                    return (xx ** p[0]) * (yy ** p[1]) * (zz ** p[2])

                approx = integrate_over_box(monomial, a, b, nx, ny, nz)
                if abs(exact) < 1e-30:
                    err = abs(approx)
                else:
                    err = abs(approx - exact) / abs(exact)
                errors.append(err)
    return errors
