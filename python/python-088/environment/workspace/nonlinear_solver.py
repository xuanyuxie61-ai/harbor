
import numpy as np
from typing import Optional, Tuple


def poly_eval(coeffs: np.ndarray, z: np.ndarray) -> np.ndarray:
    result = np.zeros_like(z, dtype=complex)
    for c in reversed(coeffs):
        result = result * z + c
    return result


def wdk_roots(
    coeffs: np.ndarray, tol: float = 1e-12, max_iter: int = 1000
) -> np.ndarray:
    d = len(coeffs) - 1
    if d < 1:
        return np.array([])


    leading = coeffs[0]
    if abs(leading) < 1e-15:
        raise ValueError("Leading coefficient is zero")
    coeffs = coeffs / leading


    R = 1.0 + np.max(np.abs(coeffs[1:]))


    theta = np.linspace(0, 2 * np.pi, d + 1)[:-1]
    roots = R * np.exp(1j * theta)

    for iteration in range(max_iter):
        roots_old = roots.copy()

        for i in range(d):
            zi = roots_old[i]

            denom = 1.0 + 0j
            for j in range(d):
                if i != j:
                    denom *= (zi - roots[j])
            if abs(denom) < 1e-30:
                denom = 1e-30
            roots[i] = zi - poly_eval(np.concatenate([[1.0], coeffs[1:]])[::-1], np.array([zi]))[0] / denom

        max_change = np.max(np.abs(roots - roots_old))
        if max_change < tol:
            break

    return roots


def newton_raphson_scalar(
    f, df, x0: float, tol: float = 1e-12, max_iter: int = 100
) -> float:
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        dfx = df(x)
        if abs(dfx) < 1e-15:
            break
        x_new = x - fx / dfx
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def newton_raphson_system(
    F, JF, x0: np.ndarray, tol: float = 1e-10, max_iter: int = 50
) -> np.ndarray:
    x = x0.copy().astype(float)
    for _ in range(max_iter):
        Fx = F(x)
        if np.linalg.norm(Fx) < tol:
            break
        J = JF(x)
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:

            dx = np.linalg.solve(J + np.eye(len(x)) * 1e-8, -Fx)
        x = x + dx
        if np.linalg.norm(dx) < tol:
            break
    return x


def fixed_point_iteration(
    g, x0: float, tol: float = 1e-10, max_iter: int = 1000
) -> float:
    x = x0
    for _ in range(max_iter):
        x_new = g(x)
        if abs(x_new - x) < tol:
            return x_new
        x = x_new
    return x


def companion_matrix_eigenvalues(coeffs: np.ndarray) -> np.ndarray:
    d = len(coeffs) - 1
    if d < 1:
        return np.array([])


    leading = coeffs[0]
    if abs(leading) < 1e-15:
        raise ValueError("Leading coefficient is zero")
    a = coeffs[1:] / leading

    C = np.zeros((d, d))
    C[:-1, 1:] = np.eye(d - 1)
    C[:, 0] = -a[::-1]

    return np.linalg.eigvals(C)


def durand_kerner_step(
    coeffs: np.ndarray, roots: np.ndarray
) -> np.ndarray:
    d = len(roots)
    new_roots = roots.copy()
    for i in range(d):
        denom = 1.0 + 0j
        for j in range(d):
            if i != j:
                denom *= (roots[i] - roots[j])
        if abs(denom) < 1e-30:
            denom = 1e-30
        pz = poly_eval(coeffs[::-1], np.array([roots[i]]))[0]
        new_roots[i] = roots[i] - pz / denom
    return new_roots


def polynomial_characteristic_values(
    A: np.ndarray, B: Optional[np.ndarray] = None
) -> np.ndarray:
    if B is None:
        return np.linalg.eigvals(A)
    else:
        return scipy_eigvals_generalized(A, B)


def scipy_eigvals_generalized(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    try:
        B_inv = np.linalg.inv(B)
        return np.linalg.eigvals(B_inv @ A)
    except np.linalg.LinAlgError:

        B_pinv = np.linalg.pinv(B)
        return np.linalg.eigvals(B_pinv @ A)


def complex_iterative_refine(
    f, z0: complex, tol: float = 1e-12, max_iter: int = 100
) -> complex:
    z = z0
    for _ in range(max_iter):
        z_new = f(z)
        if abs(z_new - z) < tol:
            return z_new
        z = z_new
    return z
