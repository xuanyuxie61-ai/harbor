
import numpy as np
from numpy.linalg import solve, lstsq


def phi_mq(r, r0):
    return np.sqrt(r**2 + r0**2)


def phi_tps(r, r0):
    r = np.where(r < 1e-15, 1e-15, r)
    return r**2 * np.log(r / (r0 + 1e-15))


def phi_gaussian(r, r0):
    return np.exp(-r**2 / (r0**2 + 1e-30))


def phi_imq(r, r0):
    return 1.0 / np.sqrt(r**2 + r0**2 + 1e-30)


def rbf_weights(m, nd, xd, r0, phi_func, pd):
    A = np.zeros((nd, nd), dtype=float)
    for i in range(nd):
        for j in range(nd):
            r = np.linalg.norm(xd[:, i] - xd[:, j])
            A[i, j] = phi_func(r, r0)


    A += np.eye(nd) * 1e-10 * np.trace(A) / nd

    try:
        w = solve(A, pd)
    except np.linalg.LinAlgError:
        w = lstsq(A, pd, rcond=None)[0]
    return w


def rbf_interpolate(m, nd, xd, r0, phi_func, w, ni, xi):
    pi = np.zeros(ni, dtype=float)
    for i in range(ni):
        d = xd - xi[:, i][:, None]
        r = np.sqrt(np.sum(d**2, axis=0))
        v = phi_func(r, r0)
        pi[i] = np.dot(v, w)
    return pi


def rbf_gradient(m, nd, xd, r0, phi_func, w, xi):
    grad = np.zeros(m, dtype=float)
    for j in range(nd):
        diff = xi[:, 0] - xd[:, j]
        r = np.linalg.norm(diff)
        r = max(r, 1e-15)

        if phi_func == phi_mq:
            dphi = r / np.sqrt(r**2 + r0**2)
        elif phi_func == phi_tps:
            dphi = 2.0 * r * np.log(r / (r0 + 1e-15)) + r
        elif phi_func == phi_gaussian:
            dphi = -2.0 * r / (r0**2 + 1e-30) * np.exp(-r**2 / (r0**2 + 1e-30))
        elif phi_func == phi_imq:
            dphi = -r / ((r**2 + r0**2) ** 1.5 + 1e-30)
        else:
            dphi = 0.0

        grad += w[j] * dphi * diff / r
    return grad


def pressure_laplacian_rbf(m, nd, xd, r0, phi_func, w, xi):
    laplacian = 0.0
    for j in range(nd):
        diff = xi[:, 0] - xd[:, j]
        r = np.linalg.norm(diff)
        r = max(r, 1e-15)

        if phi_func == phi_mq:

            d2phi = r0**2 / ((r**2 + r0**2) ** 1.5 + 1e-30)
        elif phi_func == phi_gaussian:
            d2phi = (4.0 * r**2 / (r0**4 + 1e-30) - 2.0 / (r0**2 + 1e-30)) * np.exp(-r**2 / (r0**2 + 1e-30))
        else:
            d2phi = 0.0


        if phi_func == phi_mq:
            dphi = r / np.sqrt(r**2 + r0**2)
            lap_phi = d2phi + (m - 1.0) / r * dphi
        elif phi_func == phi_gaussian:
            dphi = -2.0 * r / (r0**2 + 1e-30) * np.exp(-r**2 / (r0**2 + 1e-30))
            lap_phi = d2phi + (m - 1.0) / r * dphi
        else:
            lap_phi = 0.0

        laplacian += w[j] * lap_phi
    return laplacian


def adaptive_rbf_scale(xd):
    nd = xd.shape[1]
    min_dists = []
    for i in range(nd):
        dists = np.sqrt(np.sum((xd - xd[:, i][:, None])**2, axis=0))
        dists[i] = np.inf
        min_dists.append(np.min(dists))
    return 0.5 * np.mean(min_dists)


def reconstruct_3d_pressure_field(bubble_center, bubble_radius, p_wall, p_far,
                                  n_data=100, n_eval=50, rbf_type='mq'):
    phi_map = {
        'mq': phi_mq,
        'tps': phi_tps,
        'gaussian': phi_gaussian,
        'imq': phi_imq,
    }
    phi_func = phi_map.get(rbf_type, phi_mq)


    m = 3
    xd = np.random.randn(m, n_data)
    norms = np.sqrt(np.sum(xd**2, axis=0))
    norms = np.maximum(norms, 1e-15)
    xd = xd / norms


    radii = bubble_radius * (1.0 + 4.0 * np.random.uniform(0.0, 1.0, size=n_data))
    xd = xd * radii + bubble_center[:, None]


    pd = p_far + (p_wall - p_far) * (bubble_radius / radii)

    r0 = adaptive_rbf_scale(xd)
    w = rbf_weights(m, n_data, xd, r0, phi_func, pd)


    x_eval = np.linspace(bubble_center[0] - 3*bubble_radius,
                         bubble_center[0] + 3*bubble_radius, n_eval)
    y_eval = np.linspace(bubble_center[1] - 3*bubble_radius,
                         bubble_center[1] + 3*bubble_radius, n_eval)
    z_eval = np.linspace(bubble_center[2] - 3*bubble_radius,
                         bubble_center[2] + 3*bubble_radius, n_eval)

    X, Y, Z = np.meshgrid(x_eval, y_eval, z_eval, indexing='ij')
    xi_grid = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    ni = xi_grid.shape[1]
    p_eval = rbf_interpolate(m, n_data, xd, r0, phi_func, w, ni, xi_grid)

    return xi_grid, p_eval
