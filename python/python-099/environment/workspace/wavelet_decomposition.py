
import numpy as np
import math


def haar_1d_transform(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    N = u.size
    if N == 0:
        return u.copy()


    n_transform = 2 ** int(math.floor(math.log2(N)))
    if n_transform < 2:
        return u.copy()

    v = u.copy()
    n = n_transform
    while n > 1:
        n_half = n // 2
        temp = v.copy()
        for j in range(n_half):
            a = temp[2 * j]
            b = temp[2 * j + 1]
            v[j] = (a + b) / math.sqrt(2.0)
            v[n_half + j] = (a - b) / math.sqrt(2.0)
        n = n_half

    return v


def haar_1d_inverse(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    N = v.size
    if N == 0:
        return v.copy()

    n_transform = 2 ** int(math.floor(math.log2(N)))
    if n_transform < 2:
        return v.copy()

    u = v.copy()
    n = 2
    while n <= n_transform:
        n_half = n // 2
        temp = u.copy()
        for j in range(n_half):
            a = temp[j]
            b = temp[n_half + j]
            u[2 * j] = (a + b) / math.sqrt(2.0)
            u[2 * j + 1] = (a - b) / math.sqrt(2.0)
        n = n * 2

    return u


def haar_2d_transform(U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    M, N = U.shape
    if M == 0 or N == 0:
        return U.copy()

    V = U.copy()

    for i in range(M):
        V[i, :] = haar_1d_transform(V[i, :])

    for j in range(N):
        V[:, j] = haar_1d_transform(V[:, j])
    return V


def haar_2d_inverse(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    M, N = V.shape
    if M == 0 or N == 0:
        return V.copy()

    U = V.copy()

    for j in range(N):
        U[:, j] = haar_1d_inverse(U[:, j])

    for i in range(M):
        U[i, :] = haar_1d_inverse(U[i, :])
    return U


def detect_reflection_peaks_haar(
    signal: np.ndarray,
    threshold_factor: float = 2.0,
) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    if signal.size < 4:
        return np.array([], dtype=int)

    v = haar_1d_transform(signal)
    n_transform = 2 ** int(math.floor(math.log2(signal.size)))
    if n_transform < 4:
        return np.array([], dtype=int)


    n_half = n_transform // 2
    details = v[n_half:n_transform]
    std_d = float(np.std(details))
    if std_d < 1e-15:
        return np.array([], dtype=int)

    threshold = threshold_factor * std_d
    local_peaks = np.where(np.abs(details) > threshold)[0]

    peak_indices = np.unique(np.clip(2 * local_peaks, 0, signal.size - 1))
    return peak_indices


def multiscale_energy_distribution(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=float)
    N = signal.size
    if N < 2:
        return np.array([np.sum(signal ** 2)], dtype=float)

    n_transform = 2 ** int(math.floor(math.log2(N)))
    v = haar_1d_transform(signal)

    energies = []

    energies.append(float(v[0] ** 2))

    n = n_transform
    pos = 1
    while n > 1:
        n_half = n // 2
        detail_block = v[pos:pos + n_half]
        energies.append(float(np.sum(detail_block ** 2)))
        pos += n_half
        n = n_half

    return np.array(energies, dtype=float)
