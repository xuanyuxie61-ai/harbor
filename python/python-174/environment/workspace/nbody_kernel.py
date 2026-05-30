
import numpy as np


def monomial_value(exponents, points):
    points = np.atleast_2d(points)
    N, D = points.shape
    value = np.ones(N)
    for j in range(D):
        e = int(exponents[j])
        if e != 0:
            value *= np.power(points[:, j], e)
    return value


def coulomb_potential_direct(points, charges, epsilon=1e-10):
    points = np.asarray(points, dtype=float)
    charges = np.asarray(charges, dtype=float)
    N = points.shape[0]
    if charges.shape[0] != N:
        raise ValueError("charges长度必须等于points行数")

    potential = np.zeros(N)
    for i in range(N):
        diff = points[i] - points
        dist = np.linalg.norm(diff, axis=1)

        dist[i] = np.inf

        dist = np.where(dist < epsilon, epsilon, dist)
        potential[i] = np.sum(charges / dist)
    return potential


def coulomb_force_direct(points, charges, epsilon=1e-10):
    points = np.asarray(points, dtype=float)
    charges = np.asarray(charges, dtype=float)
    N = points.shape[0]
    forces = np.zeros((N, 3))
    for i in range(N):
        diff = points[i] - points
        dist = np.linalg.norm(diff, axis=1)
        dist[i] = np.inf
        dist = np.where(dist < epsilon, epsilon, dist)
        inv_r3 = 1.0 / (dist ** 3)
        forces[i] = np.sum((charges * inv_r3)[:, None] * diff, axis=0)
    return forces


def pwc_kernel_approx(r_min, r_max, n_segments):
    if r_min <= 0 or r_max <= r_min or n_segments <= 0:
        raise ValueError("参数非法")
    breaks = np.linspace(r_min, r_max, n_segments + 1)
    centers = 0.5 * (breaks[:-1] + breaks[1:])
    values = 1.0 / centers
    return breaks, values


def evaluate_pwc_kernel(r, breaks, values):
    r = np.asarray(r)
    result = np.zeros_like(r, dtype=float)
    for k in range(len(values)):
        mask = (r >= breaks[k]) & (r < breaks[k + 1])
        result[mask] = values[k]

    result[r < breaks[0]] = values[0]
    result[r >= breaks[-1]] = values[-1]
    return result


def build_transition_matrix_from_neighbors(neighbor_counts, n_states):
    neighbor_counts = np.asarray(neighbor_counts, dtype=float)
    if neighbor_counts.shape != (n_states, n_states):
        raise ValueError("neighbor_counts形状不匹配")
    T = np.zeros((n_states, n_states))
    for i in range(n_states):
        row_sum = np.sum(neighbor_counts[i, :])
        if row_sum < 1e-15:
            T[i, :] = 1.0 / n_states
        else:
            T[i, :] = neighbor_counts[i, :] / row_sum
    return T


def kernel_gradient_laplacian(points, charges, target, epsilon=1e-10):
    diff = target - points
    dist = np.linalg.norm(diff, axis=1)
    dist = np.where(dist < epsilon, epsilon, dist)
    inv_r = 1.0 / dist
    inv_r3 = inv_r ** 3
    potential = np.sum(charges * inv_r)
    gradient = np.sum((charges * inv_r3)[:, None] * diff, axis=0)

    laplacian = 0.0
    return float(potential), gradient, float(laplacian)
