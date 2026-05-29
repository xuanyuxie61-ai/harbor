"""
utils.py

Utility modules reimplemented from seed projects:
  - latin_edge: Latin hypercube stratified sampling for MCMC initialization
  - calendar_nyt: Julian Ephemeris Date to NYT issue conversion for synthetic
    observation batch indexing
  - unicycle: single-cycle permutation enumeration for proposal diagnostics
  - circle_distance: geometric probability on the unit circle
"""
import math
import numpy as np


def latin_edge(dim_num: int, point_num: int, rng):
    """
    Latin hypercube edge sampling.

    In each dimension, the coordinates are a random permutation of
    {0, 1, ..., point_num-1} / (point_num - 1).

    Parameters:
        dim_num: spatial dimension
        point_num: number of points (must be >= 1)
        rng: object with a .uniform() method returning U[0,1]

    Returns:
        x: shape (dim_num, point_num) array in [0,1]^dim_num
    """
    if point_num < 1:
        raise ValueError("latin_edge: point_num must be >= 1")
    x = np.zeros((dim_num, point_num), dtype=float)
    if point_num == 1:
        x[:, 0] = 0.5
        return x
    for i in range(dim_num):
        perm = np.arange(point_num)
        # Fisher-Yates shuffle using rng
        for j in range(point_num - 1, 0, -1):
            idx = int(rng.uniform() * (j + 1))
            perm[j], perm[idx] = perm[idx], perm[j]
        x[i, :] = perm / (point_num - 1.0)
    return x


def _ymd_to_jed(y: int, m: int, d: int) -> float:
    """Gregorian calendar date to Julian Ephemeris Day (simplified)."""
    if m <= 2:
        y -= 1
        m += 12
    A = y // 100
    B = 2 - A + A // 4
    jd = math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1)) + d + B - 1524.5
    return jd


def _jed_to_ymd(jed: float):
    """Julian Ephemeris Day to Gregorian (y, m, d, fraction)."""
    jd = jed + 0.5
    Z = math.floor(jd)
    F = jd - Z
    if Z < 2299161:
        A = Z
    else:
        alpha = math.floor((Z - 1867216.25) / 36524.25)
        A = Z + 1 + alpha - math.floor(alpha / 4.0)
    B = A + 1524
    C = math.floor((B - 122.1) / 365.25)
    D = math.floor(365.25 * C)
    E = math.floor((B - D) / 30.6001)
    day = B - D - math.floor(30.6001 * E) + F
    if E < 14:
        month = E - 1
    else:
        month = E - 13
    if month > 2:
        year = C - 4716
    else:
        year = C - 4715
    return int(year), int(month), day


def jed_to_nyt(jed: float):
    """
    Convert Julian Ephemeris Date to New York Times volume/issue.
    Simplified reimplementation preserving the core epoch arithmetic
    and major historical corrections.

    Returns:
        volume, issue as integers
    """
    jed_epoch = _ymd_to_jed(1851, 9, 17)
    if jed <= jed_epoch:
        return -1, -1

    issue = math.floor(jed - jed_epoch)
    y, m, _ = _jed_to_ymd(jed)
    volume = y - 1851 + 1
    if (m == 9 and _ < 18) or m < 9:
        volume -= 1

    corrections = [
        (1852, 1, 2, -1),
        (1852, 7, 6, -1),
        (1853, 7, 2, -1),
        (1854, 7, 6, -1),
        (1855, 7, 5, -1),
        (1855, 9, 25, +1),
        (1855, 9, 29, -1),
        (1856, 1, 4, -1),
        (1856, 7, 7, -1),
        (1857, 1, 3, -1),
        (1858, 1, 2, -1),
        (1858, 7, 6, -1),
        (1859, 7, 5, -1),
        (1860, 1, 3, -1),
        (1860, 7, 5, -1),
        (1861, 1, 2, -1),
        (1861, 4, 21, +1),
        (1861, 4, 28, +1),
        (1861, 5, 5, -2),
        (1861, 7, 5, -1),
        (1862, 1, 2, -1),
        (1862, 7, 5, -1),
        (1863, 1, 2, -1),
        (1863, 9, 28, -1),
        (1863, 9, 30, +1),
        (1864, 1, 2, -1),
        (1864, 7, 5, -1),
        (1865, 1, 3, -1),
        (1865, 7, 5, -1),
        (1866, 1, 2, -1),
        (1867, 1, 2, -1),
    ]
    for cy, cm, cd, delta in corrections:
        if jed >= _ymd_to_jed(cy, cm, cd):
            issue += delta

    # Sunday issues before 1905
    jed_1905_04_22 = _ymd_to_jed(1905, 4, 22)
    days = math.floor(min(jed, jed_1905_04_22) - jed_epoch)
    sundays = math.floor((days + 3) / 7.0)
    issue -= sundays

    # 1898 inflation
    if jed >= _ymd_to_jed(1898, 2, 7):
        issue += 500

    # 1978 strike
    jed_1978_08_10 = _ymd_to_jed(1978, 8, 10)
    jed_1978_11_05 = _ymd_to_jed(1978, 11, 5)
    if jed >= jed_1978_08_10:
        issue -= math.floor(min(jed, jed_1978_11_05) - jed_1978_08_10) + 1

    # 2000 correction
    if jed >= _ymd_to_jed(2000, 1, 1):
        issue -= 500

    return int(volume), int(issue)


def nyt_to_jed(volume: int, issue: int):
    """
    Inverse conversion: NYT volume/issue -> approximate JED.
    Uses bisection around the epoch.
    """
    low = _ymd_to_jed(1851, 9, 18)
    high = _ymd_to_jed(2025, 1, 1)
    for _ in range(80):
        mid = (low + high) * 0.5
        v, i = jed_to_nyt(mid)
        if v < 0:
            low = mid
            continue
        if (v, i) == (volume, issue):
            return mid
        if i < issue or v < volume:
            low = mid
        else:
            high = mid
    return (low + high) * 0.5


def perm_lex_next(n: int, p: np.ndarray, rank: int):
    """
    Lexicographic permutation successor.

    Parameters:
        n: length of permutation
        p: current permutation (1-based values)
        rank: current rank (-1 to start)

    Returns:
        p_next, rank_next
    """
    p = np.array(p, dtype=int)
    if rank <= -1:
        p = np.arange(1, n + 1)
        return p, 0

    # Find rightmost ascent
    i = n - 2
    while i >= 0 and p[i] > p[i + 1]:
        i -= 1

    if i < 0:
        p = np.arange(1, n + 1)
        return p, -1

    j = n - 1
    while p[j] < p[i]:
        j -= 1

    p[i], p[j] = p[j], p[i]
    p[i + 1 :] = p[i + 1 :][::-1]
    return p, rank + 1


def unicycle_next(n: int, u: np.ndarray, rank: int):
    """
    Generate the next unicycle (single n-cycle permutation starting with 1)
    in lexicographic order.

    Parameters:
        n: order
        u: current unicycle
        rank: current rank (-1 to start)

    Returns:
        u_next, rank_next
    """
    if rank <= -1:
        u = np.arange(1, n + 1)
        return u, 0

    p = u[1:].copy() - 1
    p, rank = perm_lex_next(n - 1, p, rank)
    u[1:] = p + 1
    return u, rank


def circle_unit_sample(rng):
    """Sample a point uniformly on the unit circle."""
    theta = 2.0 * math.pi * rng.uniform()
    return np.array([math.cos(theta), math.sin(theta)])


def circle_distance_chord(theta1: float, theta2: float) -> float:
    """
    Euclidean chord distance between two points on the unit circle
    with given polar angles.
    """
    dx = math.cos(theta1) - math.cos(theta2)
    dy = math.sin(theta1) - math.sin(theta2)
    return math.hypot(dx, dy)


def circle_distance_stats(n: int, rng):
    """
    Monte Carlo estimation of mean and variance of chord distance
    between two random points on the unit circle.

    Parameters:
        n: number of sample pairs
        rng: random generator with .uniform()

    Returns:
        mu, var
    """
    t = np.empty(n, dtype=float)
    for i in range(n):
        p = circle_unit_sample(rng)
        q = circle_unit_sample(rng)
        t[i] = np.linalg.norm(p - q)
    mu = float(np.mean(t))
    if n > 1:
        var = float(np.sum((t - mu) ** 2) / (n - 1))
    else:
        var = 0.0
    return mu, var
