
import numpy as np
from typing import Tuple, List, Dict


def i4mat_rref2(A: np.ndarray, max_iter: int = 10000) -> Tuple[np.ndarray, int]:
    A_work = A.copy().astype(np.int64)
    m, n = A_work.shape
    rank = 0

    for col in range(n):
        if rank >= m:
            break


        pivot_row = -1
        for row in range(rank, m):
            if A_work[row, col] != 0:
                pivot_row = row
                break

        if pivot_row == -1:
            continue


        if pivot_row != rank:
            A_work[[pivot_row, rank]] = A_work[[rank, pivot_row]]


        if A_work[rank, col] < 0:
            A_work[rank] = -A_work[rank]


        row_vals = A_work[rank]
        nonzero_vals = row_vals[row_vals != 0]
        if len(nonzero_vals) > 0:
            g = np.gcd.reduce(np.abs(nonzero_vals))
            if g > 1:
                A_work[rank] = A_work[rank] // g


        for row in range(m):
            if row != rank and A_work[row, col] != 0:

                factor = A_work[row, col]
                pivot = A_work[rank, col]


                lcm_val = np.lcm(abs(factor), abs(pivot))
                if lcm_val == 0:
                    continue

                row_mult = lcm_val // abs(factor)
                pivot_mult = lcm_val // abs(pivot)

                if factor * pivot < 0:
                    pivot_mult = -pivot_mult

                A_work[row] = row_mult * A_work[row] - pivot_mult * A_work[rank]


                nonzero_vals = A_work[row][A_work[row] != 0]
                if len(nonzero_vals) > 0:
                    g = np.gcd.reduce(np.abs(nonzero_vals))
                    if g > 1:
                        A_work[row] = A_work[row] // g

        rank += 1

    return A_work, rank


def levenshtein_distance(s1: List, s2: List) -> int:
    m, n = len(s1), len(s2)

    if m == 0:
        return n
    if n == 0:
        return m


    prev = np.arange(n + 1, dtype=int)
    curr = np.zeros(n + 1, dtype=int)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost
            )
        prev, curr = curr, prev

    return int(prev[n])


def build_projection_matrix(n_rays: int, n_pixels_x: int, n_pixels_y: int,
                            domain_size: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    pixel_size = domain_size / max(n_pixels_x, n_pixels_y)
    n_pixels = n_pixels_x * n_pixels_y
    M = 2 * n_rays

    A = np.zeros((M, n_pixels), dtype=int)
    ray_angles = np.zeros(M)


    for r in range(n_rays):
        y_ray = domain_size * (r + 0.5) / n_rays

        for px in range(n_pixels_x):
            for py in range(n_pixels_y):
                pixel_y_min = py * domain_size / n_pixels_y
                pixel_y_max = (py + 1) * domain_size / n_pixels_y

                if pixel_y_min <= y_ray < pixel_y_max:
                    pixel_idx = py * n_pixels_x + px

                    A[r, pixel_idx] = int(domain_size / (n_pixels_x * pixel_size))

        ray_angles[r] = 0.0


    for r in range(n_rays):
        x_ray = domain_size * (r + 0.5) / n_rays

        for px in range(n_pixels_x):
            for py in range(n_pixels_y):
                pixel_x_min = px * domain_size / n_pixels_x
                pixel_x_max = (px + 1) * domain_size / n_pixels_x

                if pixel_x_min <= x_ray < pixel_x_max:
                    pixel_idx = py * n_pixels_x + px
                    A[n_rays + r, pixel_idx] = int(domain_size / (n_pixels_y * pixel_size))

        ray_angles[n_rays + r] = np.pi / 2.0

    return A, ray_angles


def solve_tomography_svd(A: np.ndarray, travel_times: np.ndarray,
                         regularization: float = 1e-4) -> np.ndarray:

    U, s, Vt = np.linalg.svd(A.astype(float), full_matrices=False)


    filter_factors = s / (s**2 + regularization**2)


    m = Vt.T @ (filter_factors * (U.T @ travel_times))

    return m


def analyze_system_identifiability(A: np.ndarray) -> Dict:

    A_rref, rank = i4mat_rref2(A)

    m, n = A.shape
    nullity = n - rank


    A_float = A.astype(float)
    U, s, Vt = np.linalg.svd(A_float, full_matrices=False)


    nonzero_singular = s[s > 1e-10]
    if len(nonzero_singular) > 0:
        condition_number = nonzero_singular[0] / nonzero_singular[-1]
    else:
        condition_number = np.inf

    return {
        'matrix_shape': (m, n),
        'rank': rank,
        'nullity': nullity,
        'is_overdetermined': m > n,
        'is_underdetermined': m < n,
        'condition_number': float(condition_number),
        'n_nonzero_singular': len(nonzero_singular),
        'identifiability_ratio': float(rank / n) if n > 0 else 0.0
    }


def classify_tissue_sequences(scan_sequences: List[List[float]],
                              reference_sequences: List[List[float]],
                              labels: List[str]) -> Dict:
    if len(reference_sequences) != len(labels):
        raise ValueError("参考序列数与标签数必须相同")

    results = []

    for scan in scan_sequences:
        distances = []
        for ref in reference_sequences:

            scan_quantized = [int(round(x * 10)) for x in scan]
            ref_quantized = [int(round(x * 10)) for x in ref]
            dist = levenshtein_distance(scan_quantized, ref_quantized)
            distances.append(dist)

        min_idx = int(np.argmin(distances))
        results.append({
            'predicted_label': labels[min_idx],
            'min_distance': distances[min_idx],
            'all_distances': distances
        })

    return {
        'n_scans': len(scan_sequences),
        'n_classes': len(labels),
        'classifications': results
    }
