
import numpy as np
from typing import Tuple, List





def test_partial_digest(k: int, dmax: int) -> Tuple[np.ndarray, np.ndarray]:
    if k < 2:
        raise ValueError("test_partial_digest: k must be >= 2.")
    if dmax < k - 1:
        raise ValueError("test_partial_digest: dmax must be >= k - 1.")


    if dmax - 1 >= k - 2:
        interior = np.random.choice(np.arange(1, dmax), size=k - 2, replace=False)
    else:
        interior = np.array([], dtype=int)

    locate = np.sort(np.concatenate(([0], interior, [dmax])))
    d = i4vec_distances(k, locate)
    return locate, d


def i4vec_distances(n: int, locate: np.ndarray) -> np.ndarray:
    if n < 2:
        raise ValueError("i4vec_distances: n must be >= 2.")
    if locate.shape[0] != n:
        raise ValueError("i4vec_distances: locate length must equal n.")

    nd = n * (n - 1) // 2
    d = np.zeros(nd, dtype=int)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d[idx] = abs(int(locate[j]) - int(locate[i]))
            idx += 1

    return d


def ksub_random(n: int, k: int) -> np.ndarray:
    if n < k:
        raise ValueError("ksub_random: n must be >= k.")
    if k < 0:
        raise ValueError("ksub_random: k must be >= 0.")
    if k == 0:
        return np.array([], dtype=int)
    return np.random.choice(np.arange(1, n + 1), size=k, replace=False)





def validate_distance_matrix(
    dist: np.ndarray,
    rtol: float = 1.0e-10,
    atol: float = 1.0e-10,
) -> dict:
    if dist.ndim != 2:
        raise ValueError("validate_distance_matrix: dist must be 2D.")
    n = dist.shape[0]
    if dist.shape[1] != n:
        raise ValueError("validate_distance_matrix: dist must be square.")

    results = {
        "is_square": True,
        "is_nonnegative": True,
        "zero_diagonal": True,
        "is_symmetric": True,
        "triangle_inequality": True,
        "violations": [],
    }


    if np.any(dist < -atol):
        results["is_nonnegative"] = False
        results["violations"].append("Negative distance found.")


    diag_norm = np.linalg.norm(np.diag(dist))
    if diag_norm > atol:
        results["zero_diagonal"] = False
        results["violations"].append(f"Non-zero diagonal: norm={diag_norm:.3e}")


    asym_norm = np.linalg.norm(dist - dist.T)
    if asym_norm > atol:
        results["is_symmetric"] = False
        results["violations"].append(f"Asymmetric matrix: norm={asym_norm:.3e}")


    max_violation = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j] + atol:
                    violation = dist[i, j] - (dist[i, k] + dist[k, j])
                    max_violation = max(max_violation, violation)
    if max_violation > atol:
        results["triangle_inequality"] = False
        results["violations"].append(f"Triangle inequality violated by {max_violation:.3e}")

    return results





def validate_backbone_distances(
    ca_coords: np.ndarray,
    expected_distances: np.ndarray,
    expected_pairs: List[Tuple[int, int]],
    tolerance: float = 1.5,
) -> dict:
    if ca_coords.ndim != 2 or ca_coords.shape[1] != 3:
        raise ValueError("validate_backbone_distances: ca_coords must be (n, 3).")
    n_res = ca_coords.shape[0]
    m = len(expected_pairs)
    if expected_distances.shape[0] != m:
        raise ValueError("validate_backbone_distances: expected_distances length must match pairs.")

    computed = np.zeros(m, dtype=float)
    passed = np.zeros(m, dtype=bool)

    for idx, (i, j) in enumerate(expected_pairs):
        if not (0 <= i < n_res and 0 <= j < n_res):
            raise ValueError(f"validate_backbone_distances: residue index out of range at pair {idx}.")
        d = np.linalg.norm(ca_coords[i] - ca_coords[j])
        computed[idx] = d
        passed[idx] = abs(d - expected_distances[idx]) <= tolerance

    results = {
        "n_constraints": m,
        "n_passed": int(np.count_nonzero(passed)),
        "n_failed": int(m - np.count_nonzero(passed)),
        "pass_rate": float(np.mean(passed)),
        "max_error_A": float(np.max(np.abs(computed - expected_distances))),
        "rmsd_A": float(np.sqrt(np.mean((computed - expected_distances) ** 2))),
        "passed_mask": passed,
        "computed_distances": computed,
    }
    return results
