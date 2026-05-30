
import numpy as np
from typing import List, Tuple, Optional






HEX_DIRECTIONS = {
    '1': np.array([1, 0]),
    '2': np.array([0, 1]),
    '3': np.array([-1, 1]),
    '4': np.array([-1, 0]),
    '5': np.array([0, -1]),
    '6': np.array([1, -1]),
}


def boundary_range_hex(word: str, start: np.ndarray = None) -> Tuple[int, int, int, int]:
    if start is None:
        start = np.array([0, 0])
    i, j = int(start[0]), int(start[1])
    imin = imax = i
    jmin = jmax = j

    for ch in word:
        if ch in HEX_DIRECTIONS:
            di, dj = HEX_DIRECTIONS[ch]
            i += di
            j += dj
            imin = min(imin, i)
            imax = max(imax, i)
            jmin = min(jmin, j)
            jmax = max(jmax, j)

    return imin, imax, jmin, jmax


def boundary_trace_hex(word: str, start: np.ndarray = None) -> np.ndarray:
    if start is None:
        start = np.array([0, 0])
    pts = [start.copy()]
    p = start.copy()
    for ch in word:
        if ch in HEX_DIRECTIONS:
            p = p + HEX_DIRECTIONS[ch]
            pts.append(p.copy())
    return np.array(pts)


def pram_boundary_word() -> Tuple[str, np.ndarray]:
    w = (
        "AAAAAAAA"
        "CCCCCCCC"
        "EEEEEEEEEEEEEE"
        "CCCCCCCCCCCCCCCCCCCCCCCC"
        "FffFFffF"
        "GGGGGGGGGGGG"
        "JjjJJjjJ"
        "IIIIIIIIIIIIIIIIIIII"
        "KKKKKKKKKKKKKKKKKKKKKK"
    )
    p = np.array([0, 0])
    return w, p






def diophantine_nd_nonnegative(a: List[int], b: int) -> np.ndarray:
    a = np.array(a, dtype=int)
    a = a[a > 0]
    n = len(a)
    if n == 0:
        return np.array([]).reshape(0, 0)
    if b < 0:
        return np.array([]).reshape(0, n)

    solutions = []
    y = np.zeros(n, dtype=int)
    j = 0
    r = b

    while True:

        r = b
        for idx in range(j):
            r -= a[idx] * y[idx]

        if j < n:
            y[j] = r // a[j]
            j += 1
        else:
            if r == 0:
                solutions.append(y.copy())

            while j > 0:
                j -= 1
                if y[j] > 0:
                    y[j] -= 1
                    j += 1
                    break
            else:
                break

    if len(solutions) == 0:
        return np.array([]).reshape(0, n)
    sol = np.array(solutions)

    sol = sol[np.lexsort(sol.T[::-1])]
    return sol


def frobenius_number_2d(a: int, b: int) -> int:
    if np.gcd(a, b) != 1:
        return -1
    return a * b - a - b






def generate_hcp_lattice_2d(nx: int, ny: int,
                             lattice_constant: float = 1.0) -> np.ndarray:
    a = lattice_constant
    pts = []
    for j in range(ny):
        for i in range(nx):
            x = a * (i + 0.5 * j)
            y = a * (np.sqrt(3.0) / 2.0) * j
            pts.append([x, y])
    return np.array(pts)


def generate_square_lattice_2d(nx: int, ny: int,
                                lattice_constant: float = 1.0) -> np.ndarray:
    a = lattice_constant
    x = np.arange(nx) * a
    y = np.arange(ny) * a
    xv, yv = np.meshgrid(x, y)
    return np.column_stack([xv.ravel(), yv.ravel()])


def apply_periodic_boundary_hex(points: np.ndarray,
                                 lattice_constant: float = 1.0) -> np.ndarray:
    a = lattice_constant
    a1 = np.array([a, 0.0])
    a2 = np.array([0.5 * a, np.sqrt(3.0) / 2.0 * a])
    





    n2 = 2.0 * points[:, 1] / (np.sqrt(3.0) * a)
    n1 = points[:, 0] / a - 0.5 * n2
    

    n1 = n1 - np.floor(n1)
    n2 = n2 - np.floor(n2)
    

    new_x = n1 * a + 0.5 * n2 * a
    new_y = n2 * np.sqrt(3.0) / 2.0 * a
    return np.column_stack([new_x, new_y])


def lattice_miller_index_to_direction(h: int, k: int, l: int = 0,
                                       crystal_system: str = "hexagonal") -> np.ndarray:
    if crystal_system == "hexagonal":


        U = 2 * h + k
        V = 2 * k + h
        return np.array([U, V])
    else:
        return np.array([h, k, l])


def voronoi_cell_area_hex(lattice_constant: float = 1.0) -> float:
    return np.sqrt(3.0) / 2.0 * lattice_constant ** 2


def coordination_number(crystal_structure: str = "fcc") -> int:
    mapping = {
        "fcc": 12,
        "bcc": 8,
        "hcp": 12,
        "sc": 6,
        "diamond": 4,
        "hexagonal": 6,
        "square": 4,
    }
    return mapping.get(crystal_structure.lower(), 6)
