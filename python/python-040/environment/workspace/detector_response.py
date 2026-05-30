#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, Optional


def advection_diffusion_energy_deposit(
    nx: int = 101,
    nt: int = 1000,
    c: float = 1.0,
    diff_coeff: float = 0.001,
    x_min: float = 0.0,
    x_max: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 3:
        raise ValueError("nx 必须 >= 3")
    if nt < 1:
        raise ValueError("nt 必须 >= 1")
    if c < 0.0:
        raise ValueError("漂移速度 c 必须非负")

    dx = (x_max - x_min) / (nx - 1)
    dt = 1.0 / nt



    sigma = c * dt / dx
    d_factor = diff_coeff * dt / (dx ** 2)

    if sigma > 1.0:

        dt = dx / c * 0.9
        sigma = c * dt / dx
        nt = max(int(1.0 / dt) + 1, nt)
    if d_factor > 0.5:
        dt = 0.5 * dx ** 2 / diff_coeff * 0.9
        d_factor = diff_coeff * dt / (dx ** 2)
        nt = max(int(1.0 / dt) + 1, nt)

    x = np.linspace(x_min, x_max, nx)
    u = np.zeros(nx)


    mask = (0.4 <= x) & (x <= 0.6)
    u[mask] = (10.0 * x[mask] - 4.0) ** 2 * (6.0 - 10.0 * x[mask]) ** 2


    im1 = np.array([nx - 1] + list(range(nx - 1)))
    ip1 = np.array(list(range(1, nx)) + [0])

    for _ in range(nt):

        u_new = u.copy()

        u_new -= sigma * 0.5 * (u[ip1] - u[im1])
        u_new += (sigma ** 2) * 0.5 * (u[ip1] - 2.0 * u + u[im1])

        u_new += d_factor * (u[ip1] - 2.0 * u + u[im1])
        u = u_new

    return x, u


def news_edge_detector(
    data: np.ndarray,
    threshold: float = 0.1,
    normalize: bool = True
) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("输入必须是二维数组")
    m, n = data.shape
    if m < 3 or n < 3:

        return np.zeros_like(data)


    b = np.zeros((m + 2, n + 2), dtype=float)
    b[1:m+1, 1:n+1] = data

    b[0, 1:n+1] = b[1, 1:n+1]
    b[m+1, 1:n+1] = b[m, 1:n+1]
    b[1:m+1, 0] = b[1:m+1, 1]
    b[1:m+1, n+1] = b[1:m+1, n]

    b[0, 0] = (b[0, 1] + b[1, 0]) / 2.0
    b[m+1, 0] = (b[m+1, 1] + b[m, 0]) / 2.0
    b[0, n+1] = (b[0, n] + b[1, n+1]) / 2.0
    b[m+1, n+1] = (b[m+1, n] + b[m, n+1]) / 2.0


    e = np.zeros((m + 2, n + 2), dtype=float)
    e[1:m+1, 1:n+1] = np.abs(-b[0:m, 1:n+1] + b[2:m+2, 1:n+1]) \
                     + np.abs(-b[1:m+1, 0:n] + b[1:m+1, 2:n+2])


    e = e[1:m+1, 1:n+1]

    if normalize:
        e_min = np.min(e)
        e_max = np.max(e)
        if e_max > e_min:
            e = (e - e_min) / (e_max - e_min)
        else:
            e = np.zeros_like(e)


    e = np.where(e > threshold, e, 0.0)

    return e


def detector_hit_map(
    n_pixels: int = 64,
    noise_level: float = 0.01,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)

    hit_map = np.zeros((n_pixels, n_pixels))




    for track_idx in range(2):
        if track_idx == 0:
            x0, y0 = n_pixels * 0.15, n_pixels * 0.15
            angle = np.pi / 4.0
        else:
            x0, y0 = n_pixels * 0.85, n_pixels * 0.15
            angle = 3.0 * np.pi / 4.0


        length = n_pixels * 0.7
        n_steps = int(length * 2)

        for step in range(n_steps):
            t = step / n_steps
            x = x0 + t * length * np.cos(angle)
            y = y0 + t * length * np.sin(angle)


            curvature = 0.05 * np.sin(t * np.pi)
            x += curvature * n_pixels * 0.1 * np.sin(angle)
            y -= curvature * n_pixels * 0.1 * np.cos(angle)

            ix = int(round(x))
            iy = int(round(y))


            sigma = 1.5
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    jx, jy = ix + dx, iy + dy
                    if 0 <= jx < n_pixels and 0 <= jy < n_pixels:
                        dist2 = dx ** 2 + dy ** 2
                        energy = np.exp(-dist2 / (2.0 * sigma ** 2))

                        if t > 0.8:
                            energy *= (1.0 + 2.0 * (t - 0.8) / 0.2)
                        hit_map[jx, jy] += energy


    noise = np.random.normal(0.0, noise_level * np.max(hit_map), (n_pixels, n_pixels))
    hit_map += noise
    hit_map = np.maximum(hit_map, 0.0)


    edge_map = news_edge_detector(hit_map, threshold=0.15)

    return hit_map, edge_map


def aperiodic_detector_geometry(
    nmax: int = 3,
    scale: float = 1.0
) -> np.ndarray:
    if not (0 <= nmax <= 6):
        raise ValueError("nmax 必须在 [0, 6] 范围内")


    cos30 = np.cos(np.pi / 6.0)
    sin30 = np.sin(np.pi / 6.0)


    base_vertices = np.array([
        [0.0, 0.0],
        [cos30, sin30],
        [cos30 + cos30, sin30 + 0.5],
        [cos30, sin30 + 1.0],
        [0.0, 1.0],
        [-cos30, sin30 + 1.0],
        [-cos30, sin30],
    ]) * scale


    centers = [np.mean(base_vertices, axis=0)]



    angles = [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    for n in range(1, nmax + 1):
        new_centers = []
        factor = scale * (1.5 ** n)
        for center in centers:
            for angle_deg in angles:
                angle = np.radians(angle_deg + 15.0 * n)
                offset = np.array([factor * np.cos(angle), factor * np.sin(angle)])
                new_centers.append(center + offset)
        centers.extend(new_centers)

    coords = np.array(centers)

    coords_rounded = np.round(coords, decimals=8)
    unique_coords = np.unique(coords_rounded, axis=0)

    return unique_coords * scale


def detector_energy_resolution(
    energy: np.ndarray,
    a_stoch: float = 0.1,
    b_const: float = 0.01,
    c_noise: float = 0.5
) -> np.ndarray:
    energy = np.atleast_1d(energy)
    energy = np.maximum(energy, 1e-6)

    sigma_e = np.sqrt(a_stoch ** 2 * energy + b_const ** 2 * energy ** 2 + c_noise ** 2)


    sigma_e = np.minimum(sigma_e, 0.99 * energy)

    measured = energy + np.random.normal(0.0, sigma_e)
    measured = np.maximum(measured, 0.0)

    return measured
