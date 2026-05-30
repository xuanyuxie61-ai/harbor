
import numpy as np
from typing import Tuple, List


def minimal_surface_catenoid(
    X: np.ndarray,
    Y: np.ndarray,
    a: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if a <= 0:
        raise ValueError("Parameter a must be positive")

    R = np.sqrt(X ** 2 + Y ** 2)

    R = np.maximum(R, 1.01 / a)

    U = np.arccosh(a * R) / a

    denom = R * np.sqrt((a * R) ** 2 - 1.0)
    denom = np.maximum(denom, 1e-15)

    Ux = X / denom
    Uy = Y / denom


    a2R2 = (a * R) ** 2
    factor = (a2R2 - 1.0) ** 1.5
    factor = np.maximum(factor, 1e-15)

    Uxx = (Y ** 2 * (a2R2 - 1.0) + X ** 2) / (R ** 3 * factor)
    Uxy = -X * Y / (R ** 3 * factor)
    Uyy = (X ** 2 * (a2R2 - 1.0) + Y ** 2) / (R ** 3 * factor)

    return U, Ux, Uy, Uxx, Uxy, Uyy


def minimal_surface_scherk(
    X: np.ndarray,
    Y: np.ndarray,
    a: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if a <= 0:
        raise ValueError("Parameter a must be positive")


    X = np.clip(X, -0.99 * np.pi / (2.0 * a), 0.99 * np.pi / (2.0 * a))
    Y = np.clip(Y, -0.99 * np.pi / (2.0 * a), 0.99 * np.pi / (2.0 * a))

    cos_aX = np.cos(a * X)
    cos_aY = np.cos(a * Y)
    cos_aX = np.maximum(np.abs(cos_aX), 1e-15) * np.sign(cos_aX + 1e-15)

    U = np.log(cos_aY / cos_aX) / a
    Ux = np.tan(a * X)
    Uy = -np.tan(a * Y)
    Uxx = a * (np.tan(a * X) ** 2 + 1.0)
    Uxy = np.zeros_like(X)
    Uyy = -a * (np.tan(a * Y) ** 2 + 1.0)

    return U, Ux, Uy, Uxx, Uxy, Uyy


def minimal_surface_residual(
    Uxx: np.ndarray,
    Uxy: np.ndarray,
    Uyy: np.ndarray,
    Ux: np.ndarray,
    Uy: np.ndarray
) -> np.ndarray:
    R = (1.0 + Ux ** 2) * Uyy - 2.0 * Ux * Uy * Uxy + (1.0 + Uy ** 2) * Uxx
    return R


def point_in_polygon(
    x0: float,
    y0: float,
    poly_x: np.ndarray,
    poly_y: np.ndarray
) -> bool:
    n = len(poly_x)
    if n < 3:
        raise ValueError("Polygon must have at least 3 vertices")
    if len(poly_y) != n:
        raise ValueError("poly_x and poly_y must have same length")

    inside = False
    for i in range(n):
        ip1 = (i + 1) % n

        yi = poly_y[i]
        yip1 = poly_y[ip1]


        if ((yi > y0) != (yip1 > y0)) or (y0 == yi and yip1 > y0):

            xi = poly_x[i]
            xip1 = poly_x[ip1]

            if abs(yip1 - yi) < 1e-15:
                continue

            x_intersect = xi + (y0 - yi) * (xip1 - xi) / (yip1 - yi)

            if x0 < x_intersect:
                inside = not inside

    return inside


def quantum_state_bloch_region(
    state: np.ndarray,
    region_polygon_x: np.ndarray,
    region_polygon_y: np.ndarray
) -> bool:
    dim = len(state)
    if dim != 2:
        raise ValueError("This function only works for single-qubit states (dim=2)")

    alpha = state[0]
    beta = state[1]

    x_bloch = 2.0 * (alpha * np.conj(beta)).real
    y_bloch = 2.0 * (alpha * np.conj(beta)).imag

    return point_in_polygon(x_bloch, y_bloch, region_polygon_x, region_polygon_y)


def geometric_quantum_kernel(
    x: np.ndarray,
    x_prime: np.ndarray,
    surface_type: str = "catenoid",
    a: float = 1.0
) -> float:
    if len(x) < 2 or len(x_prime) < 2:
        raise ValueError("Input vectors must have at least 2 dimensions")


    X = np.array([[x[0], x_prime[0]]])
    Y = np.array([[x[1], x_prime[1]]])

    if surface_type == "catenoid":
        U, _, _, _, _, _ = minimal_surface_catenoid(X, Y, a)
    elif surface_type == "scherk":
        U, _, _, _, _, _ = minimal_surface_scherk(X, Y, a)
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")


    diff = U[0, 0] - U[0, 1]
    kernel = np.exp(-diff ** 2)
    return kernel


def quantum_feature_space_volume(
    n_qubits: int,
    n_samples: int = 1000
) -> float:
    dim = 2 ** n_qubits
    count = 0

    for _ in range(n_samples):

        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)


        probs = np.abs(psi) ** 2
        entropy = -np.sum(probs * np.log(probs + 1e-15))
        max_entropy = np.log(dim)


        if entropy > 0.5 * max_entropy:
            count += 1

    return count / n_samples
