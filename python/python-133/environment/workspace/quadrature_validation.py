
import numpy as np
from typing import Tuple, List
from math import factorial


def hermite_exact_integral_1d(power: int) -> float:
    if power % 2 == 1:
        return 0.0
    m = power // 2

    double_fact = factorial(2 * m) // (2 ** m * factorial(m))
    return double_fact * np.sqrt(np.pi) / (2.0 ** m)


def hermite_exact_integral_nd(exponents: np.ndarray) -> float:
    exponents = np.asarray(exponents, dtype=int)
    result = 1.0
    for e in exponents:
        result *= hermite_exact_integral_1d(int(e))
    return result


def monomial_value(dim_num: int, point_num: int,
                   exponents: np.ndarray,
                   x: np.ndarray) -> np.ndarray:
    exponents = np.asarray(exponents, dtype=int)
    x = np.asarray(x)
    if x.shape[0] != dim_num:
        x = x.T

    values = np.ones(point_num)
    for d in range(dim_num):
        if exponents[d] != 0:
            values *= x[d, :] ** exponents[d]
    return values


def comp_next_composition(n: int, k: int) -> List[np.ndarray]:
    compositions = []
    a = np.zeros(k, dtype=int)
    a[0] = n
    more = (a[-1] != n)
    compositions.append(a.copy())
    h = 0
    t = n

    while more:
        if 1 < t:
            h = 0
        h += 1
        t = a[h - 1]
        a[h - 1] = 0
        a[0] = t - 1
        a[h] += 1
        more = (a[-1] != n)
        compositions.append(a.copy())

    return compositions


def vector_representative(dim: int, base: int, vec: np.ndarray) -> np.ndarray:
    return np.sort(vec)


def vector_equivalent_next(vec: np.ndarray) -> Tuple[np.ndarray, bool]:
    vec = np.asarray(vec)


    return vec, False


def symmetrize_monomial_coeffs(dim: int, coeffs: np.ndarray,
                               exponents_list: List[np.ndarray]) -> np.ndarray:
    n = len(exponents_list)
    coeffs_new = coeffs.copy()


    groups = {}
    for i, exp in enumerate(exponents_list):
        rep = tuple(np.sort(exp))
        if rep not in groups:
            groups[rep] = []
        groups[rep].append(i)


    for rep, indices in groups.items():
        avg_coeff = np.mean(coeffs[indices])
        for idx in indices:
            coeffs_new[idx] = avg_coeff

    return coeffs_new


def validate_quadrature_rule(grid_point: np.ndarray,
                             grid_weight: np.ndarray,
                             dim_num: int,
                             degree_max: int) -> dict:
    n_points = grid_point.shape[1]
    errors_by_degree = {}
    max_error = 0.0
    total_tests = 0

    for degree in range(degree_max + 1):
        compositions = comp_next_composition(degree, dim_num)
        degree_errors = []

        for comp in compositions:
            exponents = comp

            v = monomial_value(dim_num, n_points, exponents, grid_point)
            quad_val = np.dot(grid_weight, v)


            exact_val = hermite_exact_integral_nd(exponents)

            err = abs(quad_val - exact_val)
            degree_errors.append(err)
            max_error = max(max_error, err)
            total_tests += 1

        errors_by_degree[degree] = {
            'count': len(degree_errors),
            'max_error': max(degree_errors) if degree_errors else 0.0,
            'mean_error': np.mean(degree_errors) if degree_errors else 0.0,
        }

    return {
        'max_error': max_error,
        'total_tests': total_tests,
        'errors_by_degree': errors_by_degree,
    }


def convergence_order_estimate(errors: List[float], n_points: List[int]) -> float:
    log_n = np.log(n_points)
    log_err = np.log(errors)

    A = np.vstack([log_n, np.ones(len(log_n))]).T
    p, _ = np.linalg.lstsq(A, log_err, rcond=None)[0]
    return -float(p)
