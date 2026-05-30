
import numpy as np
from scipy.special import erfc
from typing import Tuple, List


def normal_01_cdf_inv(p: float) -> float:
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0


    a1 = -3.969683028665376e+01
    a2 = 2.209460984245205e+02
    a3 = -2.759285104469687e+02
    a4 = 1.383577518672690e+02
    a5 = -3.066479806614716e+01
    a6 = 2.506628277459239e+00

    b1 = -5.447609879822406e+01
    b2 = 1.615858368580409e+02
    b3 = -1.556989798598866e+02
    b4 = 6.680131188771972e+01
    b5 = -1.328068155288572e+01

    c1 = -7.784894002430293e-03
    c2 = -3.223964580411365e-01
    c3 = -2.400758277161838e+00
    c4 = -2.549732539343734e+00
    c5 = 4.374664141464968e+00
    c6 = 2.938163982698783e+00

    d1 = 7.784695709041462e-03
    d2 = 3.224671290700398e-01
    d3 = 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0)
    else:
        q = np.sqrt(-2.0 * np.log(1.0 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
             ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0)


    e = 0.5 * erfc(-x / np.sqrt(2.0)) - p
    u = e * np.sqrt(2.0 * np.pi) * np.exp(x * x / 2.0)
    x = x - u / (1.0 + x * u / 2.0)
    return float(x)


def r8vec_normal_01_sorted(n: int) -> np.ndarray:
    if n <= 0:
        return np.array([])

    u = np.sort(np.random.uniform(0.0, 1.0, size=n))
    return np.array([normal_01_cdf_inv(float(ui)) for ui in u])


def rng_cliff_next(x: float) -> float:
    if x <= 0.0 or x >= 1.0:
        return np.nan
    return np.mod(-100.0 * np.log(x), 1.0)


def cliff_sequence(n: int, seed: float = 0.314159265) -> np.ndarray:
    if not (0.0 < seed < 1.0):
        seed = 0.314159265
    seq = np.zeros(n)
    x = seed
    for i in range(n):
        x = rng_cliff_next(x)
        if np.isnan(x):
            x = 0.5
        seq[i] = x
    return seq


def lights_out_matrix(mrow: int, ncol: int) -> np.ndarray:
    N = mrow * ncol
    A = np.zeros((N, N), dtype=int)

    def idx(i: int, j: int) -> int:
        if 1 <= i <= mrow and 1 <= j <= ncol:
            return (i - 1) * ncol + (j - 1)
        return -1

    for i in range(1, mrow + 1):
        for j in range(1, ncol + 1):
            c = idx(i, j)
            A[c, c] = 1
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nbr = idx(i + di, j + dj)
                if nbr >= 0:
                    A[nbr, c] = 1
    return A


def connectivity_mod2(occupied: np.ndarray, mrow: int, ncol: int) -> bool:
    if occupied.size != mrow * ncol:
        raise ValueError("occupied 长度必须等于 mrow*ncol")
    grid = occupied.reshape((mrow, ncol)).astype(bool)


    visited = np.zeros((mrow, ncol), dtype=bool)
    queue = []
    for i in range(mrow):
        if grid[i, 0]:
            queue.append((i, 0))
            visited[i, 0] = True

    while queue:
        i, j = queue.pop(0)
        if j == ncol - 1:
            return True
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < mrow and 0 <= nj < ncol:
                if grid[ni, nj] and not visited[ni, nj]:
                    visited[ni, nj] = True
                    queue.append((ni, nj))
    return False


def generate_fracture_network_params(num_fractures: int,
                                     a_min: float = 0.5,
                                     D_f: float = 2.2) -> dict:

    sorted_normals = r8vec_normal_01_sorted(num_fractures)

    lengths = a_min * np.exp(np.abs(sorted_normals))


    cliff = cliff_sequence(num_fractures, seed=0.271828182)
    strikes = 360.0 * cliff
    dips = 90.0 * np.random.uniform(0.0, 1.0, size=num_fractures)


    positions = np.random.randn(num_fractures, 3)
    positions[:, 0] *= 200.0
    positions[:, 1] *= 50.0
    positions[:, 2] *= 30.0

    return {
        "lengths": lengths,
        "strikes": strikes,
        "dips": dips,
        "positions": positions,
    }
