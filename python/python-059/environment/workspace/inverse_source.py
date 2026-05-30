
import numpy as np
from math import sqrt, exp, pi


class InverseSourceError(Exception):
    pass


def concentration_to_pseudo_distance(C, Q=1.0, D=1.0, L=100.0):
    C = np.asarray(C, dtype=np.float64)
    if np.any(C <= 0):
        raise InverseSourceError("concentration_to_pseudo_distance: 浓度必须为正")

    arg = Q / (4.0 * pi * D * L * C)


    w = np.zeros_like(arg)
    large = arg > 10.0
    small = arg <= 1.0
    medium = ~(large | small)

    w[small] = arg[small]
    w[medium] = np.log(arg[medium]) - np.log(np.log(arg[medium]) + 1e-15)

    w[large] = np.log(arg[large]) - np.log(np.log(arg[large]))
    w = np.maximum(w, 0.0)
    return L * w


def compute_position_from_distance(dist_matrix, dim=3):
    D = np.asarray(dist_matrix, dtype=np.float64)
    N = D.shape[0]
    if D.shape[0] != D.shape[1]:
        raise InverseSourceError("compute_position_from_distance: 距离矩阵必须为方阵")

    D2 = D ** 2
    J = np.eye(N) - np.ones((N, N)) / N
    B = -0.5 * J @ D2 @ J


    B = 0.5 * (B + B.T)
    eigvals, eigvecs = np.linalg.eigh(B)


    idx = np.argsort(eigvals)[::-1][:dim]
    Lambda = np.diag(np.maximum(eigvals[idx], 0.0))
    V = eigvecs[:, idx]

    positions = V @ np.sqrt(Lambda)
    return positions.T


def map_residuals(x_flat, dim, num_stations, dist_matrix):
    positions = x_flat.reshape((dim, num_stations))
    residuals = []
    n1 = (dim * (dim + 1)) // 2 if num_stations >= dim else (num_stations * (num_stations + 1)) // 2
    n2 = (num_stations * (num_stations - 1)) // 2


    k = 0
    for city in range(min(dim, num_stations)):
        for d_idx in range(city, min(dim, num_stations)):
            residuals.append(positions[d_idx, city])
            k += 1

    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            d_comp = np.linalg.norm(positions[:, i] - positions[:, j])
            d_obs = dist_matrix[i, j]
            residuals.append(d_obs - d_comp)

    return np.array(residuals, dtype=np.float64)


def inverse_source_location(
    station_positions,
    concentrations,
    Q=1.0,
    D_diff=1.0,
    L=100.0,
    dim=3,
):
    N = len(concentrations)
    if station_positions.shape[0] != N:
        raise InverseSourceError("inverse_source_location: 站点数与浓度数不匹配")


    pseudo_d = concentration_to_pseudo_distance(concentrations, Q, D_diff, L)


    dist_matrix = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            dist_matrix[i, j] = abs(pseudo_d[i] - pseudo_d[j])
            dist_matrix[j, i] = dist_matrix[i, j]


    pos_est = compute_position_from_distance(dist_matrix, dim)



    weights = concentrations / np.sum(concentrations)
    source_guess = np.average(station_positions, axis=0, weights=weights)


    grad = np.zeros(dim)
    for i in range(N):
        for j in range(i + 1, N):
            if concentrations[i] > concentrations[j]:
                vec = station_positions[i] - station_positions[j]
                d = np.linalg.norm(vec) + 1e-6
                grad += (concentrations[i] - concentrations[j]) * vec / d

    if np.linalg.norm(grad) > 1e-12:
        grad = grad / np.linalg.norm(grad)

        step = np.mean(pseudo_d) * 0.1
        source_pos = source_guess - step * grad
    else:
        source_pos = source_guess


    predicted = np.zeros(N)
    for i in range(N):
        d = np.linalg.norm(station_positions[i] - source_pos) + 1e-6
        predicted[i] = Q / (4.0 * pi * D_diff * d) * exp(-d / L)

    residual_norm = np.linalg.norm(concentrations - predicted)
    return source_pos, residual_norm


def position_to_distance(city_dim, city_num, positions):
    dist = np.zeros((city_num, city_num), dtype=np.float64)
    for i in range(city_num):
        for j in range(i + 1, city_num):
            d = np.linalg.norm(positions[:, i] - positions[:, j])
            dist[i, j] = d
            dist[j, i] = d
    return dist
