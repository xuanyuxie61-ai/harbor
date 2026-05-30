
import numpy as np


def alnorm(z: float) -> float:
    z = float(z)
    if z < 0.0:
        return 1.0 - alnorm(-z)
    y = 0.5 * z * z
    if z <= 1.28:

        a1 = 0.4361836
        a2 = -0.1201676
        a3 = 0.9372980
        b1 = 0.33267
        p = np.exp(-y) * (a1 + a2 * z + a3 * (z ** 2)) / (1.0 + b1 * z)
        return 1.0 - p
    else:

        p = 0.0


        if z > 37.0:
            return 1.0
        phi_z = 0.3989422804014327 * np.exp(-y)
        p = phi_z / z * (1.0 - 1.0 / (z * z) + 3.0 / (z ** 4) - 15.0 / (z ** 6))
        return 1.0 - p


def normp(z: float) -> tuple:
    z = float(z)
    pdf = 0.3989422804014327 * np.exp(-0.5 * z * z)
    if z < 0.0:
        phi, q_neg, _ = normp(-z)
        return 1.0 - q_neg, phi, pdf
    if z == 0.0:
        return 0.5, 0.5, pdf

    from math import erf
    phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    q = 1.0 - phi
    return phi, q, pdf


def nprob(z: float) -> tuple:
    z = float(z)
    pdf = 0.3989422804014327 * np.exp(-0.5 * z * z)
    if z == 0.0:
        return 0.5, 0.5, pdf
    if z < 0.0:
        phi_neg, q_neg, _ = nprob(-z)
        return q_neg, phi_neg, pdf
    if z > 37.0:
        return 1.0, 0.0, pdf

    from math import erf
    phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    q = 1.0 - phi
    return phi, q, pdf


def standard_normal_cdf(z: np.ndarray, method: str = "alnorm") -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if method == "scipy":
        from scipy.special import ndtr
        return ndtr(z)
    func = {"alnorm": alnorm, "normp": lambda x: normp(x)[0],
            "nprob": lambda x: nprob(x)[0]}.get(method, alnorm)
    return np.vectorize(func)(z)


def box_muller_transform(n: int, seed: int = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_pairs = (n + 1) // 2
    u1 = rng.random(n_pairs)
    u2 = rng.random(n_pairs)

    u1 = np.where(u1 < 1e-15, 1e-15, u1)
    r = np.sqrt(-2.0 * np.log(u1))
    theta = 2.0 * np.pi * u2
    samples = np.concatenate([r * np.cos(theta), r * np.sin(theta)])
    return samples[:n]


def reparameterized_gaussian_sample(mean: np.ndarray, std: np.ndarray,
                                    seed: int = None) -> np.ndarray:
    shape = np.shape(mean)
    n = int(np.prod(shape))
    eps = box_muller_transform(n, seed).reshape(shape)
    std = np.asarray(std)

    std = np.where(std <= 0.0, 1e-8, std)
    return np.asarray(mean) + std * eps


def gaussian_kl_divergence(mu1: float, sigma1: float, mu2: float,
                           sigma2: float) -> float:
    s1 = max(sigma1, 1e-15)
    s2 = max(sigma2, 1e-15)
    kl = np.log(s2 / s1) + (s1 * s1 + (mu1 - mu2) ** 2) / (2.0 * s2 * s2) - 0.5
    return float(kl)
