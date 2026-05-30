# -*- coding: utf-8 -*-

import time
import numpy as np


def log_message(message: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{ts}] {message}")


def print_numeric_matrix(mat, title="", fmt="{:12.6f}"):
    if title:
        print(f"\n{title}")
    mat = np.atleast_2d(np.asarray(mat, dtype=float))
    for row in mat:
        line = " ".join(fmt.format(v) for v in row)
        print(line)


def uniform_on_sphere_phong(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xyz = rng.standard_normal(size=(n, 3))
    norms = np.linalg.norm(xyz, axis=1, keepdims=True)

    norms = np.where(norms < 1e-15, 1.0, norms)
    return xyz / norms


def brownian_displacement(n_steps, dim=3, dt=1.0, D=0.5, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    sigma = np.sqrt(2.0 * D * dt)
    return rng.normal(loc=0.0, scale=sigma, size=(n_steps, dim))


def direction_uniform_nd(dim, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    v = rng.standard_normal(size=dim)
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        v[0] = 1.0
        norm = 1.0
    return v / norm


def safe_divide(a, b, default=0.0):
    b = np.asarray(b, dtype=float)
    result = np.empty_like(b, dtype=float)
    mask = np.abs(b) > 1e-15
    result[mask] = a / b[mask]
    result[~mask] = default
    return result


def clip_gradient(grad, max_norm=1e3):
    gnorm = np.linalg.norm(grad)
    if gnorm > max_norm:
        return grad * (max_norm / gnorm)
    return grad
