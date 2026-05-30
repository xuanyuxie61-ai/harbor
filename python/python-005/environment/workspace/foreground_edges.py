# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple





def generate_double_c_foreground(n1: int = 300, n2: int = 300,
                                  seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    r1 = rng.uniform(2.0, 5.0, n1)
    theta1 = rng.uniform(np.pi / 2.0, 3.0 * np.pi / 2.0, n1)
    x1 = r1 * np.cos(theta1)
    y1 = r1 * np.sin(theta1)

    r2 = rng.uniform(2.0, 5.0, n2)
    theta2 = rng.uniform(3.0 * np.pi / 2.0, 5.0 * np.pi / 2.0, n2)
    x2 = r2 * np.cos(theta2)
    y2 = r2 * np.sin(theta2) + 3.5
    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])
    labels = np.concatenate([np.zeros(n1), np.ones(n2)])

    perm = rng.permutation(len(x))
    return x[perm], y[perm], labels[perm]


def foreground_temperature_profile(theta: np.ndarray,
                                    amplitude: float = 100.0,
                                    width: float = 0.2) -> np.ndarray:
    signal = amplitude * np.exp(-(theta / width) ** 2)
    noise = np.random.randn(len(theta)) * 5.0
    return signal + noise





def detect_edges_1d(y: np.ndarray, x: np.ndarray = None,
                     window: int = 5, threshold: float = 0.5) -> List[int]:
    n = len(y)
    if x is None:
        x = np.arange(n, dtype=float)
    edges = []
    for i in range(window, n - window):

        xl = x[i - window:i]
        yl = y[i - window:i]

        xr = x[i + 1:i + window + 1]
        yr = y[i + 1:i + window + 1]

        Al = np.vstack([xl, np.ones(len(xl))]).T
        Ar = np.vstack([xr, np.ones(len(xr))]).T
        try:
            a_l, _ = np.linalg.lstsq(Al, yl, rcond=None)[0]
            a_r, _ = np.linalg.lstsq(Ar, yr, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        denom = max(abs(a_l), abs(a_r), 1e-12)
        if abs(a_l - a_r) / denom > threshold:
            edges.append(i)
    return edges


def shepp_logan_2d(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    ellipses = [
        (1.0, 0.69, 0.92, 0.0, 0.0, 0.0),
        (-0.98, 0.6624, 0.8740, 0.0, -0.0184, 0.0),
        (-0.02, 0.1100, 0.3100, 0.22, 0.0, -18.0),
        (-0.02, 0.1600, 0.4100, -0.22, 0.0, 18.0),
        (0.01, 0.2100, 0.2500, 0.0, 0.35, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, 0.1, 0.0),
        (0.01, 0.0460, 0.0460, 0.0, -0.1, 0.0),
        (0.01, 0.0460, 0.0230, -0.08, -0.605, 0.0),
        (0.01, 0.0230, 0.0230, 0.0, -0.606, 0.0),
        (0.01, 0.0230, 0.0460, 0.06, -0.605, 0.0),
    ]
    z = np.zeros_like(x)
    for A, a, b, xc, yc, alpha_deg in ellipses:
        alpha = np.radians(alpha_deg)
        xt = (x - xc) * np.cos(alpha) + (y - yc) * np.sin(alpha)
        yt = (x - xc) * np.sin(alpha) - (y - yc) * np.cos(alpha)
        mask = (xt ** 2 / (a ** 2) + yt ** 2 / (b ** 2)) <= 1.0
        z += A * mask

    return z * 50.0


def gradient_edge_detector_2d(image: np.ndarray, threshold: float = 10.0) -> np.ndarray:
    ny, nx = image.shape
    edges = np.zeros((ny, nx), dtype=bool)
    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            gx = 0.5 * (image[i + 1, j] - image[i - 1, j])
            gy = 0.5 * (image[i, j + 1] - image[i, j - 1])
            grad = np.sqrt(gx ** 2 + gy ** 2)
            if grad > threshold:
                edges[i, j] = True
    return edges





def compute_residual_rms(cmb_map: np.ndarray,
                          foreground_map: np.ndarray,
                          mask: np.ndarray) -> float:
    total = cmb_map + foreground_map
    vals = total[mask]
    if len(vals) == 0:
        return 0.0
    return float(np.sqrt(np.mean(vals ** 2)))
