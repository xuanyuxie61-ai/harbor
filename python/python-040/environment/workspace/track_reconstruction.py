#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from typing import Tuple, List, Optional






def path_length(path: np.ndarray, dist_matrix: np.ndarray) -> float:
    n = path.size
    total = 0.0
    for i in range(n - 1):
        total += dist_matrix[path[i], path[i + 1]]
    total += dist_matrix[path[-1], path[0]]
    return total


def tsp_track_association(
    hits_per_layer: List[np.ndarray],
    max_iter: int = 5000
) -> Tuple[np.ndarray, float]:
    if len(hits_per_layer) < 2:
        raise ValueError("至少需要两层击中点")


    all_hits = []
    layer_ids = []
    for lid, hits in enumerate(hits_per_layer):
        hits = np.atleast_2d(hits)
        for h in hits:
            all_hits.append(h)
            layer_ids.append(lid)

    all_hits = np.array(all_hits)
    n = all_hits.shape[0]
    layer_ids = np.array(layer_ids)


    dist = np.zeros((n, n))
    lambda_penalty = 10.0 * np.max(np.std(all_hits, axis=0)) if n > 1 else 1.0

    for i in range(n):
        for j in range(i + 1, n):
            d_spatial = np.linalg.norm(all_hits[i] - all_hits[j])

            d_layer = abs(layer_ids[i] - layer_ids[j])
            if d_layer == 0:
                d_spatial += 1e6
            d_total = d_spatial + lambda_penalty * max(0, 2 - d_layer)
            dist[i, j] = d_total
            dist[j, i] = d_total


    p = np.random.permutation(n)
    best_len = path_length(p, dist)
    best_path = p.copy()

    for _ in range(max_iter):

        pt1 = np.random.randint(n)
        pt2 = np.random.randint(n)
        lo, hi = min(pt1, pt2), max(pt1, pt2)
        q = np.arange(n)
        q[lo:hi+1] = q[lo:hi+1][::-1]
        p_new = p[q]
        new_len = path_length(p_new, dist)
        if new_len < best_len:
            p = p_new
            best_len = new_len
            best_path = p.copy()


        pt1 = np.random.randint(n)
        pt2 = np.random.randint(n - 1) if n > 1 else 0
        q = np.delete(np.arange(n), pt1)
        q = np.insert(q, pt2, pt1)
        p_new = p[q]
        new_len = path_length(p_new, dist)
        if new_len < best_len:
            p = p_new
            best_len = new_len
            best_path = p.copy()

    return best_path, best_len






def hermite_cubic_value(
    x1: float, f1: float, d1: float,
    x2: float, f2: float, d2: float,
    n: int, x: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.atleast_1d(x)
    h = x2 - x1
    if abs(h) < 1e-14:
        return np.full_like(x, f1), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)

    t = (x - x1) / h

    t = np.clip(t, -0.5, 1.5)


    h00 = 2.0 * t ** 3 - 3.0 * t ** 2 + 1.0
    h10 = t ** 3 - 2.0 * t ** 2 + t
    h01 = -2.0 * t ** 3 + 3.0 * t ** 2
    h11 = t ** 3 - t ** 2


    dh00 = 6.0 * t ** 2 - 6.0 * t
    dh10 = 3.0 * t ** 2 - 4.0 * t + 1.0
    dh01 = -6.0 * t ** 2 + 6.0 * t
    dh11 = 3.0 * t ** 2 - 2.0 * t


    ddh00 = 12.0 * t - 6.0
    ddh10 = 6.0 * t - 4.0
    ddh01 = -12.0 * t + 6.0
    ddh11 = 6.0 * t - 2.0


    dddh00 = 12.0 * np.ones_like(t)
    dddh10 = 6.0 * np.ones_like(t)
    dddh01 = -12.0 * np.ones_like(t)
    dddh11 = 6.0 * np.ones_like(t)

    f = f1 * h00 + h * d1 * h10 + f2 * h01 + h * d2 * h11
    d = (f1 * dh00 + h * d1 * dh10 + f2 * dh01 + h * d2 * dh11) / h
    s = (f1 * ddh00 + h * d1 * ddh10 + f2 * ddh01 + h * d2 * ddh11) / (h ** 2)
    t3 = (f1 * dddh00 + h * d1 * dddh10 + f2 * dddh01 + h * d2 * dddh11) / (h ** 3)

    return f, d, s, t3


def hermite_cubic_spline(
    xn: np.ndarray,
    fn: np.ndarray,
    dn: np.ndarray,
    x_eval: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xn = np.asarray(xn).ravel()
    fn = np.asarray(fn).ravel()
    dn = np.asarray(dn).ravel()
    x_eval = np.asarray(x_eval).ravel()

    nn = xn.size
    if nn < 2:
        raise ValueError("至少需要两个节点")
    if not np.all(np.diff(xn) > 0):
        raise ValueError("xn 必须严格递增")

    n_eval = x_eval.size
    f_out = np.zeros(n_eval)
    d_out = np.zeros(n_eval)
    s_out = np.zeros(n_eval)
    t_out = np.zeros(n_eval)


    for i in range(n_eval):
        xv = x_eval[i]

        if xv <= xn[0]:
            idx = 0
        elif xv >= xn[-1]:
            idx = nn - 2
        else:
            idx = np.searchsorted(xn, xv, side='right') - 1
            idx = max(0, min(idx, nn - 2))

        ff, dd, ss, tt = hermite_cubic_value(
            xn[idx], fn[idx], dn[idx],
            xn[idx + 1], fn[idx + 1], dn[idx + 1],
            1, np.array([xv])
        )
        f_out[i] = ff[0]
        d_out[i] = dd[0]
        s_out[i] = ss[0]
        t_out[i] = tt[0]

    return f_out, d_out, s_out, t_out


def estimate_momentum_from_curvature(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coords: Optional[np.ndarray] = None,
    magnetic_field: float = 3.8
) -> float:
    n = x_coords.size
    if n < 3:
        return 0.0

    x = np.asarray(x_coords)
    y = np.asarray(y_coords)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    u = x - x_mean
    v = y - y_mean

    Suu = np.sum(u ** 2)
    Svv = np.sum(v ** 2)
    Suv = np.sum(u * v)
    Suuu = np.sum(u ** 3)
    Svvv = np.sum(v ** 3)
    Suvv = np.sum(u * v ** 2)
    Svuu = np.sum(v * u ** 2)

    denom = 2.0 * (Suu * Svv - Suv ** 2)
    if abs(denom) < 1e-14:
        return 0.0

    uc = (Svv * (Suuu + Suvv) - Suv * (Svvv + Svuu)) / denom
    vc = (Suu * (Svvv + Svuu) - Suv * (Suuu + Suvv)) / denom

    R = np.sqrt(uc ** 2 + vc ** 2 + (Suu + Svv) / n)


    R = max(R, 1e-4)

    p_T = 0.3 * magnetic_field * R


    if z_coords is not None and z_coords.size >= 2:
        dz = z_coords[-1] - z_coords[0]
        dr = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
        if abs(dr) > 1e-10:
            tan_lambda = dz / dr
            p_total = p_T * np.sqrt(1.0 + tan_lambda ** 2)
            return p_total

    return p_T






def fem1d_track_fit(
    track_length: np.ndarray,
    energy_deposit: np.ndarray,
    n_nodes: int = 20,
    weight_a: float = 1.0,
    weight_d: float = 0.1,
    weight_b: float = 10.0
) -> Tuple[np.ndarray, np.ndarray]:
    track_length = np.asarray(track_length).ravel()
    energy_deposit = np.asarray(energy_deposit).ravel()

    if track_length.size != energy_deposit.size:
        raise ValueError("track_length 与 energy_deposit 长度不匹配")

    n_data = track_length.size
    x_min = np.min(track_length)
    x_max = np.max(track_length)


    node_x = np.linspace(x_min, x_max, n_nodes)


    data_l = np.searchsorted(node_x, track_length, side='right') - 1
    data_l = np.clip(data_l, 0, n_nodes - 2)
    data_r = data_l + 1


    is_legal = (track_length >= x_min) & (track_length <= x_max)
    eq_num = int(np.sum(is_legal)) + (n_nodes - 2) + 2

    A = np.zeros((eq_num, n_nodes))
    b_vec = np.zeros(eq_num)

    eq_i = 0

    for i in range(n_data):
        if is_legal[i]:
            l = data_l[i]
            r = data_r[i]
            h = node_x[r] - node_x[l]
            if abs(h) < 1e-14:
                A[eq_i, l] = weight_a
            else:
                A[eq_i, l] = weight_a * (node_x[r] - track_length[i]) / h
                A[eq_i, r] = weight_a * (track_length[i] - node_x[l]) / h
            b_vec[eq_i] = weight_a * energy_deposit[i]
            eq_i += 1


    for i in range(1, n_nodes - 1):
        h_left = node_x[i] - node_x[i - 1]
        h_right = node_x[i + 1] - node_x[i]
        if h_left > 1e-14 and h_right > 1e-14:
            A[eq_i, i - 1] = weight_d / h_left
            A[eq_i, i] = -weight_d / h_left - weight_d / h_right
            A[eq_i, i + 1] = weight_d / h_right
            eq_i += 1


    A[eq_i, 0] = weight_b
    b_vec[eq_i] = 0.0
    eq_i += 1

    A[eq_i, n_nodes - 1] = weight_b
    b_vec[eq_i] = 0.0
    eq_i += 1


    A = A[:eq_i, :]
    b_vec = b_vec[:eq_i]


    node_c, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)

    return node_x, node_c


def particle_id_from_dedx(
    dedx_samples: np.ndarray,
    momentum: float
) -> str:
    mean_dedx = np.median(dedx_samples)
    if mean_dedx <= 0.0 or momentum <= 0.0:
        return 'unknown'






    log_mom = np.log10(momentum)
    log_dedx = np.log10(mean_dedx)


    if log_dedx > 0.8 - 0.3 * log_mom:
        return 'electron'
    elif abs(log_dedx - 0.2) < 0.3:
        return 'muon'
    elif log_dedx > 0.4 and momentum < 2.0:
        return 'pion'
    else:
        return 'muon'
