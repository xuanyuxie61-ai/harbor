
import numpy as np
from typing import Tuple, List


def sort_rc(n: int, indx: int, isgn: int,
            i_save: List[int], j_save: List[int],
            k_save: List[int], l_save: List[int],
            n_save: List[int]) -> Tuple[int, int, int]:
    if n < 1:
        return 0, 0, 0

    if indx == 0:
        k_save[0] = n // 2
        l_save[0] = k_save[0]
        n_save[0] = n

    elif indx < 0:
        if indx == -2:
            if isgn < 0:
                i_save[0] += 1
            j_save[0] = l_save[0]
            l_save[0] = i_save[0]
            return -1, i_save[0], j_save[0]

        if isgn > 0:
            return 2, i_save[0], j_save[0]

        if k_save[0] <= 1:
            if n_save[0] == 1:
                i_save[0] = 0
                j_save[0] = 0
                return 0, 0, 0
            else:
                i_save[0] = n_save[0]
                n_save[0] -= 1
                j_save[0] = 1
                return 1, i_save[0], j_save[0]

        k_save[0] -= 1
        l_save[0] = k_save[0]

    elif indx == 1:
        l_save[0] = k_save[0]

    while True:
        i_save[0] = 2 * l_save[0]

        if i_save[0] == n_save[0]:
            j_save[0] = l_save[0]
            l_save[0] = i_save[0]
            return -1, i_save[0], j_save[0]
        elif i_save[0] <= n_save[0]:
            j_save[0] = i_save[0] + 1
            return -2, i_save[0], j_save[0]

        if k_save[0] <= 1:
            break

        k_save[0] -= 1
        l_save[0] = k_save[0]

    if n_save[0] == 1:
        i_save[0] = 0
        j_save[0] = 0
        return 0, 0, 0
    else:
        i_save[0] = n_save[0]
        n_save[0] -= 1
        j_save[0] = 1
        return 1, i_save[0], j_save[0]


def external_sort_array(arr: np.ndarray) -> np.ndarray:
    arr = arr.copy()
    n = arr.shape[0]
    if n <= 1:
        return arr

    i_save = [-1]
    j_save = [-1]
    k_save = [-1]
    l_save = [-1]
    n_save = [-1]

    indx = 0
    isgn = 0

    while True:
        indx, i, j = sort_rc(n, indx, isgn, i_save, j_save, k_save, l_save, n_save)
        if indx < 0:
            isgn = 1 if arr[i - 1] > arr[j - 1] else -1
        elif indx > 0:
            arr[i - 1], arr[j - 1] = arr[j - 1], arr[i - 1]
        else:
            break

    return arr


def safe_divide(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    b_safe = np.where(np.abs(b) < eps, np.sign(b) * eps if np.sign(b) != 0 else eps, b)
    return a / b_safe


def sigmoid(x: np.ndarray, steepness: float = 1.0, midpoint: float = 0.0) -> np.ndarray:
    z = steepness * (x - midpoint)
    z = np.clip(z, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def validate_parameters(params: dict, bounds: dict) -> bool:
    for name, value in params.items():
        if name in bounds:
            low, high = bounds[name]
            if not (low <= value <= high):
                raise ValueError(
                    f"validate_parameters: 参数 {name}={value} 超出边界 [{low}, {high}]"
                )
    return True


def compute_gini_coefficient(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = np.where(values < 0, 0.0, values)
    n = values.shape[0]
    if n == 0:
        return 0.0

    total = np.sum(values)
    if total < 1e-15:
        return 0.0


    sorted_vals = np.sort(values)
    cumsum = np.cumsum(sorted_vals)
    B = np.sum(cumsum) / (n * total)
    return float(1.0 - 2.0 * B + 1.0 / n)


def morse_potential(r: np.ndarray, epsilon: float = 1.0,
                    r_eq: float = 1.0, alpha: float = 6.0) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    dr = r - r_eq
    exp_term = np.exp(-alpha * dr)
    return epsilon * ((1.0 - exp_term) ** 2 - 1.0)
