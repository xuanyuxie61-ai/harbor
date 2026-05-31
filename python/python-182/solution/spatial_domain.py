"""
spatial_domain.py

Spatial domain construction and finite-element basis evaluation for
Bayesian inference on annular geometries.

Reimplemented from seed projects:
  - annulus_grid: Cartesian and Fibonacci point generation in annuli
  - fem_basis: Lagrange basis on the reference triangle
  - circle_distance: geometric statistics for spatial correlation validation
"""
import math
import numpy as np


def annulus_grid(r1: float, r2: float, n_r: int, n_theta: int):
    """
    Generate a structured polar grid inside an annulus [r1, r2] x [0, 2pi).

    Returns:
        x, y: arrays of shape (n_points,)
    """
    if r1 < 0 or r2 <= r1 or n_r < 1 or n_theta < 1:
        raise ValueError("annulus_grid: invalid parameters")
    rs = np.linspace(r1, r2, n_r)
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    R, T = np.meshgrid(rs, thetas, indexing='ij')
    x = R.flatten() * np.cos(T.flatten())
    y = R.flatten() * np.sin(T.flatten())
    return x, y


def annulus_grid_fibonacci(r1: float, r2: float, n: int):
    """
    Generate near-uniform points in an annulus via Fibonacci (golden-angle) spiral.

    Points follow:
        r_i = sqrt( r1^2 + (r2^2 - r1^2) * (i-0.5) / n )
        theta_i = 2 * pi * i / phi^2   where phi = (1+sqrt(5))/2

    Returns:
        x, y: arrays of shape (n,)
    """
    if n < 1 or r1 < 0 or r2 <= r1:
        raise ValueError("annulus_grid_fibonacci: invalid parameters")
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    golden_angle = 2.0 * math.pi / (phi * phi)
    i = np.arange(1, n + 1, dtype=float)
    r = np.sqrt(r1 * r1 + (r2 * r2 - r1 * r1) * (i - 0.5) / n)
    theta = golden_angle * i
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y


def fem_basis_2d(i: int, j: int, k: int, x: float, y: float) -> float:
    """
    Evaluate the Lagrange basis function L_{i,j,k}(x,y) of degree D=i+j+k
    on the reference triangle with vertices (0,0), (1,0), (0,1).

    The basis satisfies the Kronecker-delta property at nodes
    (X,Y) = (i/D, j/D).

    Parameters:
        i, j, k: integer barycentric coordinates, i+j+k = D
        x, y: evaluation point (should satisfy x>=0, y>=0, x+y<=1)

    Returns:
        Basis value at (x,y)
    """
    if i < 0 or j < 0 or k < 0:
        raise ValueError("fem_basis_2d: indices must be non-negative")
    d = i + j + k
    if d == 0:
        return 1.0

    lijk = 1.0
    cijk = 1.0
    for p in range(i):
        lijk *= (d * x - p)
        cijk *= (i - p)
    for p in range(j):
        lijk *= (d * y - p)
        cijk *= (j - p)
    for p in range(k):
        lijk *= (d * (x + y) - (d - p))
        cijk *= ((i + j) - (d - p))

    if abs(cijk) < 1e-15:
        return 0.0
    return lijk / cijk


def fem_basis_eval_on_triangle(degree: int, x: float, y: float):
    """
    Evaluate all Lagrange basis functions of given degree on the reference triangle.

    Returns:
        vals: list of (i,j,k,value)
    """
    results = []
    for ii in range(degree + 1):
        for jj in range(degree + 1 - ii):
            kk = degree - ii - jj
            val = fem_basis_2d(ii, jj, kk, x, y)
            results.append((ii, jj, kk, val))
    return results


def circle_distance_pdf(d: float) -> float:
    """
    Theoretical PDF for the distance between two random points on the unit circle.

    pdf(d) = 1 / (pi * sqrt(1 - 0.25 * d^2)),   0 <= d <= 2
    """
    if d < 0.0 or d > 2.0:
        return 0.0
    denom = math.pi * math.sqrt(max(0.0, 1.0 - 0.25 * d * d))
    if denom < 1e-15:
        return 0.0
    return 1.0 / denom


def circle_distance_exact_mean() -> float:
    """Exact mean chord distance: 4/pi."""
    return 4.0 / math.pi


def circle_distance_exact_variance() -> float:
    """Exact variance: 2 - 16/pi^2."""
    return 2.0 - 16.0 / (math.pi * math.pi)
