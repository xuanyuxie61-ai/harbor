import numpy as np


def magic4_matrix(n):
    if n % 4 != 0:
        raise ValueError(f"magic4_matrix requires n to be a multiple of 4, got {n}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")


    M = np.arange(1, n * n + 1).reshape(n, n)


    n_sq = n * n
    for i in range(n):
        for j in range(n):

            m1 = np.abs(i - j) % 4
            m2 = (i + j + 1) % 4
            if m1 == 0 or m2 == 0:
                M[i, j] = n_sq + 1 - M[i, j]

    return M


def normalized_magic_weights(n):
    M = magic4_matrix(n)
    return M.astype(np.float64) / np.sum(M)


def ball_unit_sample(n_samples, dim=3, seed=None):
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if seed is not None:
        np.random.seed(seed)


    dirs = np.random.randn(n_samples, dim)
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    dirs = dirs / norms


    u = np.random.rand(n_samples)
    r = u ** (1.0 / dim)

    return dirs * r.reshape(-1, 1)


def distance_stats(points_a, points_b):
    if points_a.shape[1] != points_b.shape[1]:
        raise ValueError("Dimension mismatch")


    diffs = points_a[:, np.newaxis, :] - points_b[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)

    return {
        'mean': float(np.mean(dists)),
        'std': float(np.std(dists)),
        'min': float(np.min(dists)),
        'max': float(np.max(dists)),
        'distances': dists.flatten()
    }


def safe_divide(a, b, fill_value=0.0):
    b = np.asarray(b, dtype=np.float64)
    result = np.full_like(np.asarray(a, dtype=np.float64), fill_value)
    mask = np.abs(b) > 1e-15
    result[mask] = a[mask] / b[mask]
    return result


def clip_with_warning(arr, lo, hi, name="array"):
    arr = np.asarray(arr)
    clipped = np.clip(arr, lo, hi)
    n_violations = np.sum((arr < lo) | (arr > hi))
    if n_violations > 0:

        pass
    return clipped
