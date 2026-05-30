
import numpy as np


def vandermonde_solve_bjorck_pereyra(nodes, rhs):
    nodes = np.asarray(nodes).ravel()
    rhs = np.asarray(rhs)
    N = nodes.size

    if rhs.ndim == 1:
        rhs = rhs.reshape(-1, 1)
    if rhs.shape[0] != N:
        raise ValueError("rhs must have length equal to number of nodes.")


    for j in range(N - 1):
        for i in range(j + 1, N):
            if abs(nodes[i] - nodes[j]) < 1e-14:
                return np.zeros_like(rhs), 1

    x = rhs.astype(complex).copy()


    for k in range(N - 1):
        for i in range(N - 1, k, -1):
            x[i, :] = x[i, :] - nodes[k] * x[i - 1, :]


    for k in range(N - 1, -1, -1):
        if k < N - 1:
            for i in range(k + 1, N):
                x[i, :] = x[i, :] / (nodes[i] - nodes[i - k - 1])
        for i in range(k, N - 1):
            x[i, :] = x[i, :] - x[i + 1, :]

    if rhs.shape[1] == 1:
        x = x.ravel()
    return x, 0


def vandermonde_determinant(nodes):
    N = nodes.size
    det_val = 1.0 + 0.0j
    for j in range(N):
        for i in range(j + 1, N):
            det_val *= (nodes[i] - nodes[j])
    return det_val


def barycentric_lagrange_interpolate(nodes, values, z):
    nodes = np.asarray(nodes)
    values = np.asarray(values)
    z = np.asarray(z)
    N = nodes.size


    w = np.ones(N, dtype=complex)
    for j in range(N):
        for k in range(N):
            if k != j:
                w[j] /= (nodes[j] - nodes[k])

    z_flat = z.ravel()
    pz = np.zeros_like(z_flat, dtype=complex)

    for idx, zz in enumerate(z_flat):

        exact = np.isclose(zz, nodes)
        if np.any(exact):
            pz[idx] = values[np.argmax(exact)]
            continue

        num = np.sum(w * values / (zz - nodes))
        den = np.sum(w / (zz - nodes))
        if abs(den) < 1e-30:
            pz[idx] = np.nan
        else:
            pz[idx] = num / den

    return pz.reshape(z.shape)


def interpolate_energy_band(param_points, energy_points, eval_points):
    return barycentric_lagrange_interpolate(param_points, energy_points, eval_points)


def characteristic_polynomial_from_roots(roots):
    roots = np.asarray(roots)
    coeffs = [1.0]
    for r in roots:
        coeffs = np.convolve(coeffs, [1.0, -r])
    return coeffs
