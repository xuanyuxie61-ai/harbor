import numpy as np


def phi_multiquadric(r, r0):
    return np.sqrt(r * r + r0 * r0)


def phi_inverse_multiquadric(r, r0):
    return 1.0 / np.sqrt(r * r + r0 * r0)


def phi_thin_plate_spline(r, r0):
    result = np.zeros_like(r)
    mask = r > 1e-15
    result[mask] = r[mask] ** 2 * np.log(r[mask] / r0)
    return result


def phi_gaussian(r, r0):
    return np.exp(-0.5 * r * r / (r0 * r0))


def rbf_weights(data_points, data_values, r0=1.0, basis='multiquadric'):
    data_points = np.asarray(data_points, dtype=np.float64)
    data_values = np.asarray(data_values, dtype=np.float64)
    nd = data_points.shape[0]

    if len(data_values) != nd:
        raise ValueError("data_values length must match data_points rows")


    basis_funcs = {
        'multiquadric': phi_multiquadric,
        'inverse_mq': phi_inverse_multiquadric,
        'tps': phi_thin_plate_spline,
        'gaussian': phi_gaussian
    }
    if basis not in basis_funcs:
        raise ValueError(f"Unknown basis: {basis}")
    phi_func = basis_funcs[basis]


    A = np.zeros((nd, nd), dtype=np.float64)
    for i in range(nd):
        diff = data_points[i] - data_points
        r = np.linalg.norm(diff, axis=1)
        A[i, :] = phi_func(r, r0)


    reg = 1e-10 * np.eye(nd)
    A += reg


    weights = np.linalg.solve(A, data_values)
    return weights


def rbf_interpolate(query_points, data_points, weights, r0=1.0, basis='multiquadric'):
    query_points = np.asarray(query_points, dtype=np.float64)
    data_points = np.asarray(data_points, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    nq = query_points.shape[0]
    nd = data_points.shape[0]

    basis_funcs = {
        'multiquadric': phi_multiquadric,
        'inverse_mq': phi_inverse_multiquadric,
        'tps': phi_thin_plate_spline,
        'gaussian': phi_gaussian
    }
    phi_func = basis_funcs[basis]

    values = np.zeros(nq, dtype=np.float64)
    for i in range(nq):
        diff = query_points[i] - data_points
        r = np.linalg.norm(diff, axis=1)
        phi_vals = phi_func(r, r0)
        values[i] = np.dot(weights, phi_vals)

    return values


def rbf_gradient_2d(query_point, data_points, weights, r0=1.0, basis='multiquadric', h=1e-6):
    qp = np.asarray(query_point, dtype=np.float64).reshape(1, -1)


    qp_xp = qp.copy()
    qp_xp[0, 0] += h
    qp_xm = qp.copy()
    qp_xm[0, 0] -= h


    qp_yp = qp.copy()
    qp_yp[0, 1] += h
    qp_ym = qp.copy()
    qp_ym[0, 1] -= h

    fx = (rbf_interpolate(qp_xp, data_points, weights, r0, basis)[0] -
          rbf_interpolate(qp_xm, data_points, weights, r0, basis)[0]) / (2 * h)
    fy = (rbf_interpolate(qp_yp, data_points, weights, r0, basis)[0] -
          rbf_interpolate(qp_ym, data_points, weights, r0, basis)[0]) / (2 * h)

    return np.array([fx, fy])
