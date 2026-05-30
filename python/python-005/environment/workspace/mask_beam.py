# -*- coding: utf-8 -*-

import numpy as np
from typing import List, Tuple
from utils import gamma_lanczos, binomial, robust_divide





def polygon_moment(nv: int, x: np.ndarray, y: np.ndarray,
                   p: int, q: int) -> float:
    if nv < 3:
        return 0.0
    nu = 0.0
    for i in range(nv):
        j = (i + 1) % nv
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]
        s = 0.0
        for a in range(p + 1):
            for b in range(q + 1):
                s += (binomial(p, a) * binomial(q, b) *
                      (xi ** (p - a)) * (xj ** a) *
                      (yi ** (q - b)) * (yj ** b))
        cross = xj * yi - xi * yj
        denom = (p + q + 2) * (p + q + 1) * binomial(p + q, p)
        if denom == 0:
            continue
        nu += cross * s / denom
    return nu


def polygon_area(x: np.ndarray, y: np.ndarray) -> float:
    nv = len(x)
    if nv < 3:
        return 0.0
    area = 0.0
    for i in range(nv):
        j = (i + 1) % nv
        area += x[i] * y[j] - x[j] * y[i]
    return 0.5 * area


def polygon_centroid(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    A = polygon_area(x, y)
    if abs(A) < 1e-15:
        return 0.0, 0.0
    cx = polygon_moment(len(x), x, y, 1, 0) / A
    cy = polygon_moment(len(x), x, y, 0, 1) / A
    return cx, cy


def polygon_central_moment(x: np.ndarray, y: np.ndarray,
                           p: int, q: int) -> float:
    cx, cy = polygon_centroid(x, y)
    x_shifted = x - cx
    y_shifted = y - cy
    return polygon_moment(len(x), x_shifted, y_shifted, p, q)





def point_in_polygon(px: float, py: float,
                     x: np.ndarray, y: np.ndarray) -> bool:
    nv = len(x)
    inside = False
    j = nv - 1
    for i in range(nv):
        xi, yi = x[i], y[i]
        xj, yj = x[j], y[j]

        if abs((py - yi) * (xj - xi) - (px - xi) * (yj - yi)) < 1e-12:
            if min(xi, xj) <= px <= max(xi, xj) and min(yi, yj) <= py <= max(yi, yj):
                return True

        if ((yi > py) != (yj > py)):
            xinters = (xj - xi) * (py - yi) / (yj - yi + 1e-15) + xi
            if px < xinters:
                inside = not inside
        j = i
    return inside





def disk_monomial_integral(r: float, e1: int, e2: int) -> float:
    if e1 < 0 or e2 < 0:
        return 0.0
    if e1 % 2 == 1 or e2 % 2 == 1:
        return 0.0
    if r <= 0.0:
        return 0.0
    g1 = gamma_lanczos((e1 + 1.0) / 2.0)
    g2 = gamma_lanczos((e2 + 1.0) / 2.0)
    g3 = gamma_lanczos((e1 + e2) / 2.0 + 1.0)
    exponent = e1 + e2 + 2
    return 2.0 * g1 * g2 / (g3 * exponent) * (r ** exponent)


def disk_area(r: float) -> float:
    return np.pi * r * r


def disk_uniform_sample(r: float, n: int) -> np.ndarray:
    samples = np.random.randn(n, 2)
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    samples = samples / norms
    radii = r * np.sqrt(np.random.rand(n, 1))
    return samples * radii





def gaussian_beam_window(l: int, fwhm_arcmin: float) -> float:
    sigma = np.radians(fwhm_arcmin / 60.0) / np.sqrt(8.0 * np.log(2.0))
    return np.exp(-l * (l + 1.0) * sigma ** 2 / 2.0)


def beam_convolved_Cl(Cl: np.ndarray, lmax: int, fwhm_arcmin: float) -> np.ndarray:
    out = np.zeros(len(Cl))
    for idx, l in enumerate(range(2, lmax + 3)):
        if idx >= len(Cl):
            break
        Bl = gaussian_beam_window(l, fwhm_arcmin)
        out[idx] = Bl * Bl * Cl[idx]
    return out





class SurveyMask:

    def __init__(self, boundary_x: np.ndarray, boundary_y: np.ndarray):
        self.x = boundary_x
        self.y = boundary_y
        self.nv = len(boundary_x)
        self._area = polygon_area(self.x, self.y)
        self._cx, self._cy = polygon_centroid(self.x, self.y)

    def area(self) -> float:
        return abs(self._area)

    def centroid(self) -> Tuple[float, float]:
        return self._cx, self._cy

    def ellipticity(self) -> float:
        mu20 = polygon_central_moment(self.x, self.y, 2, 0)
        mu02 = polygon_central_moment(self.x, self.y, 0, 2)
        mu11 = polygon_central_moment(self.x, self.y, 1, 1)
        trace = mu20 + mu02
        det = mu20 * mu02 - mu11 * mu11
        disc = np.sqrt(max(((mu20 - mu02) / 2.0) ** 2 + mu11 ** 2, 0.0))
        lambda_plus = trace / 2.0 + disc
        lambda_minus = trace / 2.0 - disc
        if lambda_plus <= 1e-15:
            return 0.0
        a = np.sqrt(lambda_plus)
        b = np.sqrt(max(lambda_minus, 0.0))
        return 1.0 - b / a

    def contains(self, px: float, py: float) -> bool:
        return point_in_polygon(px, py, self.x, self.y)

    def fsky(self) -> float:
        return self.area() / (4.0 * np.pi)
