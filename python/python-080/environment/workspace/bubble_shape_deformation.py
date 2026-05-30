
import numpy as np
from numpy.linalg import eigvals, cond, inv
from scipy.linalg import sqrtm
from scipy.special import legendre


def ellipse_condition_number(A):
    eigenvals = eigvals(A)
    eigenvals = np.abs(eigenvals)
    eigenvals = np.maximum(eigenvals, 1e-15)
    return np.max(eigenvals) / np.min(eigenvals)


def bubble_ellipse_shape(V, A, R, num_points=100):

    A = 0.5 * (A + A.T)
    eigs = eigvals(A)
    if np.any(eigs <= 0):
        A = A + np.eye(2) * (1.1 * abs(np.min(eigs)) + 0.1)

    theta = np.linspace(0, 2 * np.pi, num_points)
    unit_circle = np.vstack([np.cos(theta), np.sin(theta)])


    try:
        A_inv_sqrt = inv(sqrtm(A))
    except (np.linalg.LinAlgError, ValueError):
        A_inv_sqrt = np.eye(2)

    points = V[:, None] + R * (A_inv_sqrt @ unit_circle)
    return points


def deformation_velocity_potential(R, dRdt, a_n, da_ndt, theta, N_modes=4):
    cos_theta = np.cos(theta)
    phi = np.zeros_like(theta)


    phi += (R**2 / np.maximum(np.abs(theta), 1e-15)) * dRdt * 1.0

    phi = R * dRdt

    for n in range(2, N_modes + 1):
        if n - 1 < len(a_n):
            Pn = legendre(n)(cos_theta)
            phi += (R / (n + 1)) * da_ndt[n - 1] * Pn

    return phi


def mode_amplitude_odes(t, y, R_eq, sigma, rho, mu, N_modes=4):
    N = N_modes - 1
    a = np.zeros(N_modes + 1)
    dadt = np.zeros(N_modes + 1)

    for n in range(2, N_modes + 1):
        idx = 2 * (n - 2)
        a[n] = y[idx]
        dadt[n] = y[idx + 1]

    R = y[-2]
    dRdt = y[-1]

    dydt = np.zeros(len(y))


    d2Rdt2 = -safe_divide(2.0 * sigma, rho * R**2) - 1.5 * dRdt**2 / (R + 1e-15)
    dydt[-2] = dRdt
    dydt[-1] = d2Rdt2

    for n in range(2, N_modes + 1):
        idx = 2 * (n - 2)
        an = a[n]
        dan_dt = dadt[n]


        restoring = -(n - 1) * sigma / (rho * (R**3 + 1e-30)) * (n + 2) * (n - 1) * an

        damping = -2.0 * mu * (n - 1) * (n + 2) / (rho * (R**2 + 1e-30)) * dan_dt

        coupling = (n - 1) * d2Rdt2 / (R + 1e-15) * an

        expansion = -3.0 * dRdt / (R + 1e-15) * dan_dt

        d2an_dt2 = restoring + damping + coupling + expansion

        dydt[idx] = dan_dt
        dydt[idx + 1] = d2an_dt2

    return dydt


def safe_divide(a, b, default=0.0):
    b = np.asarray(b)
    a = np.asarray(a)
    result = np.empty_like(a, dtype=float)
    mask = np.abs(b) > 1e-30
    result[mask] = a[mask] / b[mask]
    result[~mask] = default
    return result


def chaotic_microfragmentation(num_points=5000, iterations=3000):
    A = [
        np.array([[0.80, 0.00], [0.00, 0.80]]),
        np.array([[0.50, 0.00], [0.00, 0.50]]),
        np.array([[0.355, -0.355], [0.355, 0.355]]),
        np.array([[0.355, 0.355], [-0.355, 0.355]]),
    ]
    b = [
        np.array([0.10, 0.04]),
        np.array([0.25, 0.40]),
        np.array([0.266, 0.078]),
        np.array([0.378, 0.434]),
    ]

    x = np.random.rand(2, num_points)


    lyap_sum = 0.0
    lyap_count = 0
    delta = 1e-10

    for i in range(iterations):
        j = np.random.randint(0, 4, size=num_points)
        for k in range(num_points):
            Aj = A[j[k]]
            bj = b[j[k]]
            x[:, k] = Aj @ x[:, k] + bj


        if i < 100 and num_points > 1:
            x_pert = x[:, 0].copy()
            x_pert[0] += delta
            j0 = np.random.randint(0, 4)
            x_pert = A[j0] @ x_pert + b[j0]
            dist = np.linalg.norm(x_pert - x[:, 0])
            if dist > 1e-30:
                lyap_sum += np.log(dist / delta)
                lyap_count += 1

    lyapunov = lyap_sum / max(lyap_count, 1)
    return x, lyapunov


def compute_deformation_tensor(points):
    x = points[0, :]
    y = points[1, :]

    M = np.vstack([x**2, x * y, y**2]).T
    rhs = np.ones(len(x))
    coeffs, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)
    A_mat = np.array([[coeffs[0], coeffs[1] / 2.0], [coeffs[1] / 2.0, coeffs[2]]])
    return A_mat


def fragmentation_dimension(x):
    mins = np.min(x, axis=1)
    maxs = np.max(x, axis=1)
    L = np.max(maxs - mins)
    if L < 1e-15:
        return 0.0

    epsilons = L / (2.0 ** np.arange(1, 8))
    N_boxes = []
    for eps in epsilons:
        nx = int(np.ceil((maxs[0] - mins[0]) / eps)) + 1
        ny = int(np.ceil((maxs[1] - mins[1]) / eps)) + 1
        boxes = set()
        for k in range(x.shape[1]):
            ix = int((x[0, k] - mins[0]) / eps)
            iy = int((x[1, k] - mins[1]) / eps)
            boxes.add((ix, iy))
        N_boxes.append(len(boxes))

    log_eps = np.log(1.0 / epsilons)
    log_N = np.log(np.maximum(N_boxes, 1))

    D_box = np.polyfit(log_eps, log_N, 1)[0]
    return max(0.0, min(D_box, 2.0))
