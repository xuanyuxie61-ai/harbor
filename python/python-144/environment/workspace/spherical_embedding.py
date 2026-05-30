
import numpy as np
from scipy.optimize import least_squares


def sphere_distance1(lat1: float, lon1: float, lat2: float, lon2: float,
                      r: float = 1.0) -> float:

    cos_term = (np.sin(lat1) * np.sin(lat2) +
                np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1))
    cos_term = np.clip(cos_term, -1.0, 1.0)
    return r * np.arccos(cos_term)


def ll_to_xyz(r: float, ll: np.ndarray) -> np.ndarray:
    n = ll.shape[1]
    xyz = np.zeros((3, n))
    xyz[0, :] = r * np.cos(ll[1, :]) * np.cos(ll[0, :])
    xyz[1, :] = r * np.sin(ll[1, :]) * np.cos(ll[0, :])
    xyz[2, :] = -r * np.sin(ll[0, :])
    return xyz


def xyz_to_ll(xyz: np.ndarray, r: float = 1.0) -> np.ndarray:
    x, y, z = xyz[0, :], xyz[1, :], xyz[2, :]
    lat = np.arcsin(np.clip(-z / r, -1.0, 1.0))
    lon = np.arctan2(y, x)
    return np.vstack([lat, lon])


def map_spherical_residual(ll_vec: np.ndarray, r: float, city_num: int,
                           distance: np.ndarray) -> np.ndarray:
    ll = ll_vec.reshape(2, city_num)
    n1 = 3
    n2 = (city_num * (city_num - 1)) // 2
    f = np.zeros(n1 + n2)
    k = 0

    f[k] = ll[0, 0]
    k += 1
    f[k] = ll[1, 0]
    k += 1

    f[k] = ll[1, 1]
    k += 1

    for i in range(city_num):
        for j in range(i + 1, city_num):
            d_computed = sphere_distance1(ll[0, i], ll[1, i],
                                           ll[0, j], ll[1, j], r)
            f[k] = distance[i, j] - d_computed
            k += 1
    return f


def correlation_to_spherical_embedding(corr: np.ndarray, r: float = 1.0,
                                        random_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    n = corr.shape[0]
    if n < 3:
        raise ValueError("correlation_to_spherical_embedding: 资产数至少为3。")

    corr = 0.5 * (corr + corr.T)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    distance = np.arccos(np.clip(corr, -1.0, 1.0))

    ll0 = rng.random(2 * n)
    result = least_squares(
        lambda vec: map_spherical_residual(vec, r, n, distance),
        ll0,
        method="lm",
        max_nfev=2000 * n,
        ftol=1e-10,
        xtol=1e-10,
    )
    ll = result.x.reshape(2, n)
    xyz = ll_to_xyz(r, ll)
    return xyz


def circle01_sample_random(n: int, rng: np.random.Generator = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.random(n)
    x = np.zeros((2, n))
    x[0, :] = np.cos(2.0 * np.pi * theta)
    x[1, :] = np.sin(2.0 * np.pi * theta)
    return x


def spherical_diversity_index(xyz: np.ndarray) -> float:
    n = xyz.shape[1]
    if n == 0:
        return 0.0
    centroid = np.mean(xyz, axis=1)
    norm_c = np.linalg.norm(centroid)
    diversity = 2.0 - 2.0 * norm_c ** 2
    return float(max(diversity, 0.0))


def angular_distance_matrix(xyz: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(xyz, axis=0, keepdims=True)
    if np.any(norms < 1e-12):
        raise ValueError("angular_distance_matrix: 存在零范数向量。")
    unit = xyz / norms
    inner = np.clip(unit.T @ unit, -1.0, 1.0)
    return np.arccos(inner)
