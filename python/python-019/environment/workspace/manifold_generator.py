
import numpy as np


def circle_loop(center, radius, n_points=100):
    theta = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    m = center[0] + radius * np.cos(theta)
    gamma = center[1] + radius * np.sin(theta)
    return np.column_stack((m, gamma))


def helix_loop(center, radius, pitch, n_points=200, turns=2):
    theta = np.linspace(0.0, 2.0 * np.pi * turns, n_points)
    m = center[0] + radius * np.cos(theta)
    gamma = center[1] + radius * np.sin(theta)
    t = center[2] + pitch * theta / (2.0 * np.pi)
    return np.column_stack((m, gamma, t))


def nonlinear_curve(n_points=100, dim=5):
    z = np.linspace(0.0, 2.0 * np.pi, n_points)
    params = np.zeros((n_points, dim))
    for d in range(dim):
        freq = d + 1
        params[:, d] = np.cos(freq * z) / np.sqrt(freq)
    return params


def simplex_parameter_space(dim, n_points=75, std=0.2):
    vertices = _regular_simplex_vertices(dim)
    params = np.zeros((n_points, dim))
    labels = np.zeros(n_points, dtype=int)
    points_per_vertex = n_points // (dim + 1)
    idx = 0
    for v in range(dim + 1):
        n = points_per_vertex if v < dim else (n_points - idx)
        params[idx:idx + n] = vertices[v] + std * np.random.randn(n, dim)
        labels[idx:idx + n] = v
        idx += n
    return params, labels


def _regular_simplex_vertices(n):
    x = np.zeros((n, n + 1))
    for j in range(n):
        x[j, j] = 1.0
    a = (1.0 - np.sqrt(1.0 + n)) / n
    x[:, n] = a
    c = x.sum(axis=1) / (n + 1)
    x = x - c[:, None]
    s = np.linalg.norm(x[:, 0])
    x = x / s
    return x.T


def adiabatic_cycle_around_ep(ep_center, ep_radius, n_points=200):
    return circle_loop(ep_center, ep_radius, n_points)
