
import numpy as np


def haar_1d_transform(u):
    n = u.size
    if n < 1:
        raise ValueError("Input vector must be non-empty.")
    v = u.astype(float).copy()
    s = np.sqrt(2.0)
    w = np.zeros(n)

    k = 1
    while k * 2 <= n:
        k *= 2

    while k > 1:
        k = k // 2
        w[:k] = (v[:2 * k:2] + v[1:2 * k:2]) / s
        w[k:2 * k] = (v[:2 * k:2] - v[1:2 * k:2]) / s
        v[:2 * k] = w[:2 * k]
    return v


def haar_1d_inverse(v):
    n = v.size
    if n < 1:
        raise ValueError("Input vector must be non-empty.")
    u = v.astype(float).copy()
    s = np.sqrt(2.0)
    w = np.zeros(n)


    k = 1
    while k * 2 <= n:
        k *= 2


    scale = 1
    while scale < k:
        w[:2 * scale] = u[:2 * scale]
        u[:2 * scale:2] = (w[:scale] + w[scale:2 * scale]) / s
        u[1:2 * scale:2] = (w[:scale] - w[scale:2 * scale]) / s
        scale *= 2
    return u


def haar_2d_transform(u):
    m, n = u.shape
    if m < 1 or n < 1:
        raise ValueError("Input array must be non-empty.")
    v = u.astype(float).copy()


    for j in range(n):
        v[:, j] = haar_1d_transform(v[:, j])


    for i in range(m):
        v[i, :] = haar_1d_transform(v[i, :])

    return v


def haar_2d_inverse(v):
    m, n = v.shape
    if m < 1 or n < 1:
        raise ValueError("Input array must be non-empty.")
    u = v.astype(float).copy()


    for i in range(m):
        u[i, :] = haar_1d_inverse(u[i, :])


    for j in range(n):
        u[:, j] = haar_1d_inverse(u[:, j])

    return u


def extract_multiresolution_hotspots(field, threshold_factor=2.0):
    m, n = field.shape
    if m < 2 or n < 2:
        raise ValueError("Field must be at least 2×2.")

    v = haar_2d_transform(field)
    scales = []
    coefficients = {}


    num_levels = int(np.floor(np.log2(min(m, n))))

    for level in range(1, num_levels + 1):
        block = min(m, n) // (2 ** level)
        if block < 1:
            break





        detail_region = v[block:2 * block, block:2 * block]
        if detail_region.size == 0:
            continue
        std_val = np.std(detail_region)
        mean_val = np.mean(np.abs(detail_region))
        coefficients[level] = {
            'std': std_val,
            'mean_abs': mean_val,
            'max_abs': np.max(np.abs(detail_region))
        }
        if std_val > 0 and mean_val > threshold_factor * std_val:
            scales.append(block)

    return scales, coefficients
