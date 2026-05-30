
import numpy as np
from typing import Tuple, List


def safe_gamma_ratio(numerators: np.ndarray, denominator: float) -> float:
    numerators = np.asarray(numerators, dtype=float)
    if np.any(numerators <= 0):
        raise ValueError("Gamma 参数必须为正数")
    if denominator <= 0:
        raise ValueError("Gamma 分母参数必须为正数")
    log_num = np.sum([_log_gamma_approx(x) for x in numerators])
    log_den = _log_gamma_approx(denominator)
    return np.exp(log_num - log_den)


def _log_gamma_approx(z: float) -> float:
    if z <= 0:
        raise ValueError("z 必须为正数")
    if z < 1e-5:
        return -np.log(z) - 0.5772156649015329 * z
    g = 7.0
    coeffs = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ]
    z = z - 1.0
    x = coeffs[0]
    for i in range(1, len(coeffs)):
        x += coeffs[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * np.log(2.0 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)


def gamma_func(z: float) -> float:
    if z <= 0:
        raise ValueError("Gamma 参数必须为正数")
    return np.exp(_log_gamma_approx(z))


def arc_cosine_safe(c: float) -> float:
    c = float(c)
    if c < -1.0:
        c = -1.0
    elif c > 1.0:
        c = 1.0
    return np.arccos(c)


def vec_norm(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-15:
        return 0.0
    return n


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = vec_norm(v)
    if n < 1e-15:
        return v.copy()
    return v / n


def clamp_value(x: float, xmin: float, xmax: float) -> float:
    if xmin > xmax:
        raise ValueError("xmin 必须不大于 xmax")
    if x < xmin:
        return xmin
    if x > xmax:
        return xmax
    return x


def solve_quadratic(a: float, b: float, c: float) -> Tuple[float, float]:
    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            raise ValueError("a 和 b 同时接近零，不是二次方程")
        root = -c / b
        return (root, root)
    disc = b * b - 4.0 * a * c
    if disc < 0:
        if disc > -1e-12:
            disc = 0.0
        else:
            raise ValueError("判别式为负，无实根")
    sqrt_disc = np.sqrt(disc)
    if b >= 0:
        q = -0.5 * (b + sqrt_disc)
    else:
        q = -0.5 * (b - sqrt_disc)
    root1 = q / a
    root2 = c / q if abs(q) > 1e-15 else (-b - np.copysign(sqrt_disc, b)) / (2.0 * a)
    return (root1, root2)


def gcd_vector(v: np.ndarray) -> int:
    v = np.asarray(v, dtype=int)
    if v.size == 0:
        return 0
    g = abs(int(v[0]))
    for val in v[1:]:
        g = _gcd_two(g, abs(int(val)))
        if g == 1:
            break
    return g


def _gcd_two(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


def check_well_posed_diophantine(a: np.ndarray, b: int) -> bool:
    a = np.asarray(a, dtype=int)
    if np.any(a <= 0):
        return False
    if b < 0:
        return False
    g = gcd_vector(a)
    if g == 0:
        return False
    return (b % g) == 0


def cliff_rng_next(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return float('nan')
    val = -100.0 * np.log(x)
    return val - np.floor(val)


def cliff_rng_sequence(seed: float, n: int) -> np.ndarray:
    if seed <= 0.0 or seed >= 1.0:
        seed = 0.3
    seq = np.zeros(n, dtype=float)
    x = seed
    for i in range(n):
        x = cliff_rng_next(x)
        if np.isnan(x):
            x = 0.3
        seq[i] = x
    return seq


def compute_bandwidth(adj: List[List[int]], perm: List[int]) -> int:
    n = len(adj)
    if n == 0:
        return 0
    inv_perm = [0] * n
    for new_idx, old_idx in enumerate(perm):
        inv_perm[old_idx] = new_idx
    bw = 0
    for old_i in range(n):
        new_i = inv_perm[old_i]
        for old_j in adj[old_i]:
            new_j = inv_perm[old_j]
            bw = max(bw, abs(new_i - new_j))
    return bw + 1
