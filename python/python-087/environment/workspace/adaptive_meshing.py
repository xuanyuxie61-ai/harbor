
import numpy as np
from typing import Tuple, List, Callable


def discrete_pdf_sample_1d(pdf_values: np.ndarray, x_grid: np.ndarray,
                           n_samples: int) -> np.ndarray:
    pdf = np.asarray(pdf_values, dtype=np.float64)
    x = np.asarray(x_grid, dtype=np.float64)
    if len(pdf) != len(x):
        raise ValueError("pdf_values 与 x_grid 长度必须一致")
    if np.any(pdf < 0):
        raise ValueError("PDF 值必须非负")
    total = np.trapezoid(pdf, x)
    if total <= 1e-18:
        raise ValueError("PDF 积分为零")
    pdf_norm = pdf / total

    cdf = np.zeros_like(x)
    for i in range(1, len(x)):
        cdf[i] = cdf[i - 1] + 0.5 * (pdf_norm[i - 1] + pdf_norm[i]) * (x[i] - x[i - 1])
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    u = np.random.rand(n_samples)

    samples = np.interp(u, cdf, x)
    return samples


def cvt_1d_lloyd(n_generators: int, pdf_func: Callable[[np.ndarray], np.ndarray],
                 x_range: Tuple[float, float], n_samples: int = 20000,
                 it_max: int = 50, tol: float = 1e-6) -> np.ndarray:
    a, b = x_range
    if a >= b:
        raise ValueError("区间无效")
    gens = np.linspace(a, b, n_generators)
    x_fine = np.linspace(a, b, 2000)
    pdf_fine = pdf_func(x_fine)

    pdf_fine = np.maximum(pdf_fine, 0.0)
    for it in range(it_max):
        samples = discrete_pdf_sample_1d(pdf_fine, x_fine, n_samples)

        dist = np.abs(samples[:, None] - gens[None, :])
        labels = np.argmin(dist, axis=1)
        new_gens = np.zeros_like(gens)
        moved = False
        for k in range(n_generators):
            mask = labels == k
            if np.sum(mask) == 0:

                new_gens[k] = np.random.uniform(a, b)
                moved = True
            else:
                new_gens[k] = np.mean(samples[mask])
        new_gens = np.sort(new_gens)
        shift = np.max(np.abs(new_gens - gens))
        gens = new_gens
        if shift < tol:
            break
    return gens


def curvature_density(x: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    w = np.asarray(displacement, dtype=np.float64)
    if len(x) != len(w):
        raise ValueError("x 与 displacement 长度不一致")
    n = len(x)
    if n < 3:
        return np.zeros_like(x)
    dx = np.gradient(x)
    dw = np.gradient(w, x)
    d2w = np.gradient(dw, x)
    curvature = np.abs(d2w) / (1.0 + dw ** 2) ** 1.5

    curvature = np.maximum(curvature, 1e-6 * np.max(curvature))
    return curvature


def simplex_coordinates_nd(n: int) -> np.ndarray:
    if n < 1:
        raise ValueError("维度 n 必须 ≥ 1")
    a = (1.0 - np.sqrt(n + 1.0)) / n
    b_sq = 1.0 - n * a * a
    if b_sq < 0:
        raise ValueError("数值不稳定，b² < 0")
    b = np.sqrt(b_sq)
    verts = np.zeros((n + 1, n), dtype=np.float64)
    for i in range(n + 1):
        if i < n:
            verts[i, :i] = a
            verts[i, i] = (1.0 if i == 0 else b)
        else:
            verts[i, :] = a

    centroid = verts.mean(axis=0)
    verts -= centroid
    return verts


def helix_parametrization(s: np.ndarray, R: float, pitch: float) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if R <= 0:
        raise ValueError("曲率半径 R 必须为正")
    theta = s / R
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z = pitch * theta / (2.0 * np.pi)
    return np.column_stack((x, y, z))


def circle_parametrization(theta: np.ndarray, R: float) -> np.ndarray:
    theta = np.asarray(theta, dtype=np.float64)
    return np.column_stack((R * np.cos(theta), R * np.sin(theta)))


def map_nodes_to_space_curve(s_nodes: np.ndarray, curve_func: Callable[[np.ndarray], np.ndarray],
                             tangent_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    coords = curve_func(s_nodes)

    tangents = np.zeros_like(coords)
    if len(s_nodes) >= 2:
        tangents[0] = coords[1] - coords[0]
        tangents[-1] = coords[-1] - coords[-2]
        tangents[1:-1] = coords[2:] - coords[:-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-14, 1.0, norms)
    tangents = tangents / norms
    return coords, tangents
