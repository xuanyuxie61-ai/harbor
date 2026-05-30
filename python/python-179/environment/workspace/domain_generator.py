
import numpy as np
from typing import Tuple
from system_utils import robust_sqrt, clip_to_range






def chicken_egg_half_profile(B: float, L: float, w: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    B = float(B)
    L = float(L)
    w = float(w)

    B = clip_to_range(B, 1e-6, L)
    w = clip_to_range(w, -L / 4 + 1e-6, L / 4 - 1e-6)
    numerator = L * L - 4.0 * x * x
    denominator = L * L + 8.0 * w * x + 4.0 * w * w

    numerator = np.maximum(numerator, 0.0)
    denominator = np.maximum(denominator, 1e-12)
    y = 0.5 * B * robust_sqrt(numerator / denominator)
    return y


def pyriform_egg_half_profile(B: float, L: float, w: float, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    B = float(B)
    L = float(L)
    w = float(w)
    B = clip_to_range(B, 1e-6, L)
    w = clip_to_range(w, -L / 4 + 1e-6, L / 4 - 1e-6)
    term1 = L * L - 4.0 * x * x
    term2 = L * L + 2.0 * w * L + 4.0 * x * (L + 2.0 * w)
    denom = (L**4 + 4.0 * w * L**3 + 4.0 * w * w * L * L
             + 16.0 * w * w * x * x + 32.0 * w**3 * x + 16.0 * w**4)
    numerator = term1 * term2
    numerator = np.maximum(numerator, 0.0)
    denom = np.maximum(denom, 1e-12)
    y = 0.5 * B * robust_sqrt(numerator / denom)
    return y


def universal_egg_half_profile(B: float, L: float, w: float, D: float,
                                x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    B = clip_to_range(float(B), 1e-6, float(L))
    L = float(L)
    w = clip_to_range(float(w), -L / 4 + 1e-6, L / 4 - 1e-6)
    D = clip_to_range(float(D), 1e-6, B)
    y_chicken = chicken_egg_half_profile(B, L, w, x)
    y_pyriform = pyriform_egg_half_profile(B, L, w, x)

    lam = (B - D) / B
    lam = clip_to_range(lam, 0.0, 1.0)
    y = (1.0 - lam) * y_chicken + lam * y_pyriform
    return y






def chebyshev_nodes_1d(a: float, b: float, n: int) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be >= 2")
    i = np.arange(n, dtype=float)
    t = np.cos((2.0 * i + 1.0) * np.pi / (2.0 * n))
    x = 0.5 * (a + b) + 0.5 * (b - a) * t
    return x


def generate_parametric_radial_domain(n: int = 64,
                                       B: float = 1.0,
                                       L: float = 2.0,
                                       w: float = 0.1,
                                       D: float = 0.6) -> np.ndarray:
    x = chebyshev_nodes_1d(-L / 2.0, L / 2.0, n)
    return x


def compute_radial_cross_section(x: np.ndarray,
                                  B: float = 1.0,
                                  L: float = 2.0,
                                  w: float = 0.1,
                                  D: float = 0.6) -> np.ndarray:
    r = universal_egg_half_profile(B, L, w, D, x)
    return r






def hand_outline_polygon(n_points: int = 100) -> np.ndarray:
    K = 8
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)

    c = np.zeros(2 * K + 1, dtype=complex)
    c[K] = 1.0 + 0.0j
    c[K + 1] = 0.35 - 0.12j
    c[K - 1] = 0.35 + 0.12j
    c[K + 2] = -0.08 + 0.20j
    c[K - 2] = -0.08 - 0.20j
    c[K + 3] = 0.15 - 0.05j
    c[K - 3] = 0.15 + 0.05j
    c[K + 4] = -0.05 + 0.10j
    c[K - 4] = -0.05 - 0.10j
    c[K + 5] = 0.03 - 0.02j
    c[K - 5] = 0.03 + 0.02j

    z = np.zeros(n_points, dtype=complex)
    for k in range(-K, K + 1):
        z += c[K + k] * np.exp(1j * k * theta)

    xy = np.column_stack((z.real, z.imag))
    xy -= xy.mean(axis=0)

    max_norm = np.max(np.linalg.norm(xy, axis=1))
    if max_norm > 0:
        xy /= max_norm
    return xy


def hand_ellipse_fourier_approx(n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    xy = hand_outline_polygon(n_points)
    Xc = xy - xy.mean(axis=0)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    pcs = Vt.T[:, :3]
    return xy, pcs
