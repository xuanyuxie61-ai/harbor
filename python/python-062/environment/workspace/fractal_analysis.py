
import numpy as np


def box_counting(field, threshold=None, max_level=6):
    if field.ndim != 2:
        raise ValueError("box_counting: 目前仅支持二维场")

    if threshold is None:
        threshold = np.median(field)


    binary = (field > threshold).astype(int)

    nx, ny = binary.shape
    min_dim = min(nx, ny)
    max_level = min(max_level, int(np.log2(min_dim)))

    scales = []
    counts = []

    for level in range(1, max_level + 1):
        box_size = 2 ** level
        n_boxes_x = nx // box_size
        n_boxes_y = ny // box_size

        if n_boxes_x < 1 or n_boxes_y < 1:
            break

        count = 0
        for i in range(n_boxes_x):
            for j in range(n_boxes_y):
                ix0 = i * box_size
                ix1 = (i + 1) * box_size
                jy0 = j * box_size
                jy1 = (j + 1) * box_size

                if np.any(binary[ix0:ix1, jy0:jy1] > 0):
                    count += 1

        epsilon = 1.0 / n_boxes_x
        scales.append(epsilon)
        counts.append(count)

    scales = np.array(scales, dtype=np.float64)
    counts = np.array(counts, dtype=np.float64)


    if len(scales) >= 2:
        log_eps = np.log(1.0 / scales)
        log_N = np.log(np.maximum(counts, 1))


        A = np.vstack([log_eps, np.ones_like(log_eps)]).T
        D_f, _ = np.linalg.lstsq(A, log_N, rcond=None)[0]
    else:
        D_f = 0.0

    return D_f, scales, counts


def compute_intermittency_factor(field, window_size=8):
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        def uniform_filter(arr, size, mode='nearest'):
            from scipy.ndimage import uniform_filter as uf
            return uf(arr, size=size, mode=mode)

    local_mean = uniform_filter(field, size=window_size, mode='nearest')
    local_mean_safe = np.where(np.abs(local_mean) < 1e-15, 1e-15, local_mean)


    local_var = uniform_filter((field - local_mean)**2, size=window_size, mode='nearest')
    cv = np.sqrt(np.clip(local_var, 0, None)) / np.abs(local_mean_safe)
    cv = np.clip(cv, 0.0, 100.0)


    mu = float(np.mean(cv))
    return mu


def richardson_cascade_spectrum(k, epsilon, C=1.5, mu=0.25):

    nu = 1.5e-5
    eta = max((nu**3 / max(epsilon, 1e-12)) ** 0.25, 1e-10)


    k_safe = np.where(k <= 0, 1e-10, k)
    E = C * (max(epsilon, 1e-12) ** (2.0 / 3.0)) * (k_safe ** (-5.0 / 3.0))


    corr = (k_safe * eta) ** (-mu / 9.0)
    corr = np.clip(corr, 1e-30, 1e30)
    E = E * corr


    arg = -5.0 * (k_safe * eta) ** (4.0 / 3.0)
    arg = np.clip(arg, -700, 700)
    E = E * np.exp(arg)

    E = np.where(k <= 0, 0.0, E)
    return E
