
import numpy as np
from typing import Tuple, List
import itertools


def hypercube_monomial_integral(exponents: Tuple[int, ...]) -> float:
    result = 1.0
    for alpha in exponents:
        if alpha < 0:
            raise ValueError("Exponents must be non-negative")
        result /= (alpha + 1.0)
    return result


def hermite_monomial_integral_1d(alpha: int, weight_type: str = "physicist") -> float:
    if alpha < 0:
        raise ValueError("Exponent must be non-negative")

    if alpha % 2 == 1:
        return 0.0

    import math
    double_fact = 1.0
    for k in range(alpha - 1, 0, -2):
        double_fact *= k

    result = double_fact * np.sqrt(np.pi) / (2.0 ** (alpha / 2.0))

    if weight_type == "probabilist":
        result *= 2.0 ** ((alpha + 1.0) / 2.0)

    return result


def gauss_legendre_points_weights_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be positive")
    x, w = np.polynomial.legendre.leggauss(n)

    x = 0.5 * (x + 1.0)
    w = 0.5 * w
    return x, w


def gauss_hermite_points_weights_1d(n: int,
                                     weight_type: str = "physicist") -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be positive")
    x, w = np.polynomial.hermite.hermgauss(n)

    if weight_type == "probabilist":
        x *= np.sqrt(2.0)
        w *= np.sqrt(2.0)

    return x, w


def tensor_product_quadrature_1d_to_nd(points_1d: List[np.ndarray],
                                        weights_1d: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    d = len(points_1d)
    grids = np.meshgrid(*points_1d, indexing='ij')
    points_nd = np.stack([g.ravel() for g in grids], axis=-1)

    w_grids = np.meshgrid(*weights_1d, indexing='ij')
    weights_nd = np.prod(np.stack([g.ravel() for g in w_grids], axis=-1), axis=-1)

    return points_nd, weights_nd


def validate_hypercube_quadrature(dim: int, n_points: int,
                                  degree_max: int = 5) -> dict:
    x_1d, w_1d = gauss_legendre_points_weights_1d(n_points)
    points_1d = [x_1d.copy() for _ in range(dim)]
    weights_1d = [w_1d.copy() for _ in range(dim)]
    pts, wts = tensor_product_quadrature_1d_to_nd(points_1d, weights_1d)

    errors = []
    max_degree_passed = -1

    for total_degree in range(degree_max + 1):

        for exponents in itertools.combinations_with_replacement(range(total_degree + 1), dim):
            if sum(exponents) != total_degree:
                continue

            for alpha in set(itertools.permutations(exponents)):
                exact = hypercube_monomial_integral(alpha)
                numerical = np.sum(wts * np.prod(pts ** np.array(alpha), axis=1))
                if abs(exact) > 1e-14:
                    rel_err = abs(numerical - exact) / abs(exact)
                else:
                    rel_err = abs(numerical)
                errors.append({
                    "exponents": alpha,
                    "degree": total_degree,
                    "exact": exact,
                    "numerical": numerical,
                    "relative_error": rel_err,
                })


        degree_errors = [e for e in errors if e["degree"] == total_degree]
        if all(e["relative_error"] < 1e-12 for e in degree_errors):
            max_degree_passed = total_degree
        else:
            break

    return {
        "dim": dim,
        "n_points_per_dim": n_points,
        "total_points": pts.shape[0],
        "max_degree_passed": max_degree_passed,
        "errors": errors,
    }


def validate_hermite_quadrature_1d(n_points: int,
                                   degree_max: int = 10,
                                   weight_type: str = "physicist") -> dict:
    x, w = gauss_hermite_points_weights_1d(n_points, weight_type)
    max_degree_passed = -1
    errors = []

    for alpha in range(degree_max + 1):
        exact = hermite_monomial_integral_1d(alpha, weight_type)
        numerical = np.sum(w * (x ** alpha))
        if abs(exact) > 1e-14:
            rel_err = abs(numerical - exact) / abs(exact)
        else:
            rel_err = abs(numerical)
        errors.append({
            "degree": alpha,
            "exact": exact,
            "numerical": numerical,
            "relative_error": rel_err,
        })
        if rel_err < 1e-12:
            max_degree_passed = alpha
        else:
            break

    return {
        "n_points": n_points,
        "weight_type": weight_type,
        "max_degree_passed": max_degree_passed,
        "errors": errors,
    }


def smolyak_sparse_grid_1d_to_nd(level: int, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    if level < 0:
        raise ValueError("level must be non-negative")
    if dim < 1:
        raise ValueError("dim must be positive")


    n = max(1, level + 1)
    x, w = gauss_legendre_points_weights_1d(n)
    pts_1d = [x for _ in range(dim)]
    wts_1d = [w for _ in range(dim)]
    return tensor_product_quadrature_1d_to_nd(pts_1d, wts_1d)


def ensemble_mean_integral(ensemble_values: np.ndarray,
                           quadrature_points: np.ndarray,
                           quadrature_weights: np.ndarray) -> Tuple[float, float]:
    mean_vals = np.mean(ensemble_values, axis=0)
    mu = np.sum(quadrature_weights * mean_vals)
    sigma_sq = np.sum(quadrature_weights * (mean_vals - mu) ** 2)
    return float(mu), float(sigma_sq)
