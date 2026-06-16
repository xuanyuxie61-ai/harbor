"""
高阶有限差分模块
High-order finite difference stencils for transverse derivatives
(2nd, 4th, 6th, 8th order central differences).

Algorithms sourced from:
  - 410_fem2d_predator_prey_fast (fast FE stencils)
  - 991_r8pp (packed matrix storage for stencil coefficients)
  - 846_paraheat_functional (parabolic diffusion operators)
"""
import numpy as np


def stencil_2nd_order_1d():
    """
    2nd order central difference for 1st derivative:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)

    Returns (coefficients, offsets): ([-0.5, 0.5], [-1, 1])
    Such that f'(x) ≈ Σ c_k f(x + k*h) / h
    """
    return np.array([-0.5, 0.5]), np.array([-1, 1])


def stencil_4th_order_1st_derivative():
    """
    4th order central difference for 1st derivative:
        f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)

    Returns (coefficients, offsets)
    """
    coeffs = np.array([1.0/12.0, -8.0/12.0, 8.0/12.0, -1.0/12.0])
    offsets = np.array([-2, -1, 1, 2])
    return coeffs, offsets


def stencil_6th_order_1st_derivative():
    """
    6th order central difference for 1st derivative:
        f'(x) ≈ (-f(x-3h) + 9f(x-2h) - 45f(x-h) + 45f(x+h) - 9f(x+h) + f(x+3h)) / (60h)

    Returns (coefficients, offsets)
    """
    coeffs = np.array([-1.0/60.0, 9.0/60.0, -45.0/60.0, 45.0/60.0, -9.0/60.0, 1.0/60.0])
    offsets = np.array([-3, -2, -1, 1, 2, 3])
    return coeffs, offsets


def stencil_8th_order_1st_derivative():
    """
    8th order central difference for 1st derivative:
        f'(x) ≈ (f(x-4h) - 32f(x-3h) + 56f(x-2h) - 896f(x-h) + 896f(x+h)
                 - 56f(x+2h) + 32f(x+3h) - f(x+4h)) / (280h)

    Wait, that's not quite right. Let me use the standard formula:
        f'(x) ≈ (3f(x-4h) - 32f(x-3h) + 168f(x-2h) - 672f(x-h)
                 + 672f(x+h) - 168f(x+2h) + 32f(x+3h) - 3f(x+4h)) / (840h)

    Returns (coefficients, offsets)
    """
    coeffs = np.array([3.0/840.0, -32.0/840.0, 168.0/840.0, -672.0/840.0,
                       672.0/840.0, -168.0/840.0, 32.0/840.0, -3.0/840.0])
    offsets = np.array([-4, -3, -2, -1, 1, 2, 3, 4])
    return coeffs, offsets


def stencil_2nd_order_2nd_derivative():
    """
    2nd order central difference for 2nd derivative (Laplacian):
        f''(x) ≈ (f(x-h) - 2f(x) + f(x+h)) / h²

    Returns (coefficients, offsets)
    """
    coeffs = np.array([1.0, -2.0, 1.0])
    offsets = np.array([-1, 0, 1])
    return coeffs, offsets


def stencil_4th_order_2nd_derivative():
    """
    4th order central difference for 2nd derivative:
        f''(x) ≈ (-f(x-2h) + 16f(x-h) - 30f(x) + 16f(x+h) - f(x+2h)) / (12h²)

    Returns (coefficients, offsets)
    """
    coeffs = np.array([-1.0/12.0, 16.0/12.0, -30.0/12.0, 16.0/12.0, -1.0/12.0])
    offsets = np.array([-2, -1, 0, 1, 2])
    return coeffs, offsets


def stencil_6th_order_2nd_derivative():
    """
    6th order central difference for 2nd derivative:
        f''(x) ≈ (2f(x-3h) - 27f(x-2h) + 270f(x-h) - 490f(x)
                  + 270f(x+h) - 27f(x+2h) + 2f(x+3h)) / (180h²)

    Returns (coefficients, offsets)
    """
    coeffs = np.array([2.0/180.0, -27.0/180.0, 270.0/180.0, -490.0/180.0,
                       270.0/180.0, -27.0/180.0, 2.0/180.0])
    offsets = np.array([-3, -2, -1, 0, 1, 2, 3])
    return coeffs, offsets


def apply_1d_derivative(f, dx, order=4, axis=0):
    """
    Apply 1st derivative operator along specified axis.

    Parameters
    ----------
    f : ndarray
        Input field (2D or 3D)
    dx : float
        Grid spacing
    order : int
        Order of accuracy (2, 4, 6, or 8)
    axis : int
        Axis along which to differentiate

    Returns
    -------
    ndarray
        Derivative field
    """
    if order == 2:
        coeffs, offsets = stencil_2nd_order_1d()
    elif order == 4:
        coeffs, offsets = stencil_4th_order_1st_derivative()
    elif order == 6:
        coeffs, offsets = stencil_6th_order_1st_derivative()
    elif order == 8:
        coeffs, offsets = stencil_8th_order_1st_derivative()
    else:
        raise ValueError(f"Unsupported order {order}. Use 2, 4, 6, or 8.")

    half_width = max(abs(offsets))
    result = np.zeros_like(f)

    # Move target axis to position 0 for easier slicing
    f_moved = np.moveaxis(f, axis, 0)
    result_moved = np.zeros_like(f_moved)
    n = f_moved.shape[0]

    for c, off in zip(coeffs, offsets):
        # Shift array by offset
        if off > 0:
            shifted = np.zeros_like(f_moved)
            shifted[off:] = f_moved[:-off]
        elif off < 0:
            shifted = np.zeros_like(f_moved)
            shifted[:off] = f_moved[-off:]
        else:
            shifted = f_moved
        result_moved += c * shifted

    # Zero out boundary regions where stencil is incomplete
    result_moved[:half_width] = 0.0
    result_moved[-half_width:] = 0.0

    result = np.moveaxis(result_moved, 0, axis)
    return result / dx


def apply_2nd_derivative(f, dx, order=2, axis=0):
    """
    Apply 2nd derivative operator along specified axis.

    Parameters
    ----------
    f : ndarray
        Input field
    dx : float
        Grid spacing
    order : int
        Order of accuracy (2, 4, or 6)
    axis : int
        Axis along which to differentiate

    Returns
    -------
    ndarray
        Second derivative field
    """
    if order == 2:
        coeffs, offsets = stencil_2nd_order_2nd_derivative()
    elif order == 4:
        coeffs, offsets = stencil_4th_order_2nd_derivative()
    elif order == 6:
        coeffs, offsets = stencil_6th_order_2nd_derivative()
    else:
        raise ValueError(f"Unsupported order {order}. Use 2, 4, or 6.")

    half_width = max(abs(offsets))
    f_moved = np.moveaxis(f, axis, 0)
    result_moved = np.zeros_like(f_moved)
    n = f_moved.shape[0]

    for c, off in zip(coeffs, offsets):
        if off > 0:
            shifted = np.zeros_like(f_moved)
            shifted[off:] = f_moved[:-off]
        elif off < 0:
            shifted = np.zeros_like(f_moved)
            shifted[:off] = f_moved[-off:]
        else:
            shifted = f_moved
        result_moved += c * shifted

    # Boundary handling
    result_moved[:half_width] = 0.0
    result_moved[-half_width:] = 0.0

    result = np.moveaxis(result_moved, 0, axis)
    return result / (dx**2)


def laplacian_2d(f, dx, dy, order=2):
    """
    Compute 2D Laplacian ∇²f = ∂²f/∂x² + ∂²f/∂y²

    Parameters
    ----------
    f : ndarray (nx, ny)
        Input field
    dx, dy : float
        Grid spacing in x and y
    order : int
        Order of accuracy

    Returns
    -------
    ndarray
        Laplacian field
    """
    d2f_dx2 = apply_2nd_derivative(f, dx, order=order, axis=0)
    d2f_dy2 = apply_2nd_derivative(f, dy, order=order, axis=1)
    return d2f_dx2 + d2f_dy2


def gradient_2d(f, dx, dy, order=4):
    """
    Compute 2D gradient ∇f = (∂f/∂x, ∂f/∂y)

    Parameters
    ----------
    f : ndarray (nx, ny)
        Input field
    dx, dy : float
        Grid spacing
    order : int
        Order of accuracy

    Returns
    -------
    df_dx, df_dy : ndarray
        Gradient components
    """
    df_dx = apply_1d_derivative(f, dx, order=order, axis=0)
    df_dy = apply_1d_derivative(f, dy, order=order, axis=1)
    return df_dx, df_dy


def divergence_2d(fx, fy, dx, dy, order=4):
    """
    Compute 2D divergence ∇·F = ∂fx/∂x + ∂fy/∂y

    Parameters
    ----------
    fx, fy : ndarray (nx, ny)
        Vector field components
    dx, dy : float
        Grid spacing
    order : int
        Order of accuracy

    Returns
    -------
    ndarray
        Divergence field
    """
    dfx_dx = apply_1d_derivative(fx, dx, order=order, axis=0)
    dfy_dy = apply_1d_derivative(fy, dy, order=order, axis=1)
    return dfx_dx + dfy_dy


def curl_2d(fx, fy, dx, dy, order=4):
    """
    Compute 2D curl (scalar) ∇×F = ∂fy/∂x - ∂fx/∂y

    Parameters
    ----------
    fx, fy : ndarray (nx, ny)
        Vector field components
    dx, dy : float
        Grid spacing
    order : int
        Order of accuracy

    Returns
    -------
    ndarray
        Curl field (scalar in 2D)
    """
    dfy_dx = apply_1d_derivative(fy, dx, order=order, axis=0)
    dfx_dy = apply_1d_derivative(fx, dy, order=order, axis=1)
    return dfy_dx - dfx_dy


def biharmonic_2d(f, dx, dy, order=2):
    """
    Compute 2D biharmonic operator ∇⁴f = ∇²(∇²f)

    Parameters
    ----------
    f : ndarray
        Input field
    dx, dy : float
        Grid spacing
    order : int
        Order of accuracy

    Returns
    -------
    ndarray
        Biharmonic field
    """
    lap_f = laplacian_2d(f, dx, dy, order=order)
    return laplacian_2d(lap_f, dx, dy, order=order)


def compact_fd_1st_derivative(f, dx, beta=0.25):
    """
    Compact (Padé) finite difference for 1st derivative (4th order):
        β f'_{i-1} + f'_i + β f'_{i+1} = α (f_{i+1} - f_{i-1}) / (2h)
                                         + γ (f_{i+2} - f_{i-2}) / (4h)

    For β = 1/4, α = 3/2, γ = 0 (standard 4th order compact).

    This is an implicit scheme requiring tridiagonal solve.

    Parameters
    ----------
    f : ndarray (nx, ny)
        Input field
    dx : float
        Grid spacing
    beta : float
        Compact scheme parameter

    Returns
    -------
    ndarray
        Derivative along x-axis
    """
    nx, ny = f.shape
    result = np.zeros_like(f)

    # RHS: explicit part
    for i in range(2, nx - 2):
        for j in range(ny):
            rhs = (3.0 / 2.0) * (f[i+1, j] - f[i-1, j]) / (2.0 * dx)
            # Store in result temporarily
            result[i, j] = rhs

    # Solve tridiagonal system along x for each y
    alpha = 1.0 - 2.0 * beta
    for j in range(ny):
        # Thomas algorithm for tridiagonal system
        # beta * f'_{i-1} + f'_i + beta * f'_{i+1} = rhs_i
        a = np.full(nx, beta)
        b = np.ones(nx)
        c = np.full(nx, beta)
        d = result[:, j].copy()

        # Forward sweep
        for i in range(1, nx):
            m = a[i] / b[i-1]
            b[i] -= m * c[i-1]
            d[i] -= m * d[i-1]

        # Back substitution
        result[nx-1, j] = d[nx-1] / b[nx-1]
        for i in range(nx-2, -1, -1):
            result[i, j] = (d[i] - c[i] * result[i+1, j]) / b[i]

    # Zero boundaries
    result[:2, :] = 0.0
    result[-2:, :] = 0.0

    return result


def tridiagonal_solve(a, b, c, d):
    """
    Solve tridiagonal system using Thomas algorithm.
    From 991_r8pp (packed storage for banded systems).

    a[i] * x[i-1] + b[i] * x[i] + c[i] * x[i+1] = d[i]

    Parameters
    ----------
    a, b, c, d : ndarray (n,)
        Lower diagonal, main diagonal, upper diagonal, RHS

    Returns
    -------
    ndarray
        Solution vector x
    """
    n = len(d)
    a, b, c, d = a.copy(), b.copy(), c.copy(), d.copy()

    # Forward elimination
    for i in range(1, n):
        if abs(b[i-1]) < 1e-14:
            raise ValueError("Zero pivot in tridiagonal solve")
        m = a[i] / b[i-1]
        b[i] -= m * c[i-1]
        d[i] -= m * d[i-1]

    # Back substitution
    x = np.zeros(n)
    if abs(b[n-1]) < 1e-14:
        raise ValueError("Zero pivot in back substitution")
    x[n-1] = d[n-1] / b[n-1]
    for i in range(n-2, -1, -1):
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]

    return x
