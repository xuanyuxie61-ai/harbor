
import numpy as np


def se2_exp(v):
    vx, vy, vtheta = v
    theta = vtheta

    if abs(theta) < 1e-8:
        Vx = vx
        Vy = vy
    else:
        s = np.sin(theta)
        c = np.cos(theta)
        Vx = (vx * s + vy * (c - 1.0)) / theta
        Vy = (vy * s + vx * (1.0 - c)) / theta

    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s, Vx],
                     [s,  c, Vy],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def se2_log(T):
    R = T[0:2, 0:2]
    t = T[0:2, 2]
    theta = np.arctan2(R[1, 0], R[0, 0])

    if abs(theta) < 1e-8:
        vx = t[0]
        vy = t[1]
    else:
        A = np.sin(theta) / theta
        B = (1.0 - np.cos(theta)) / theta
        det = A * A + B * B
        if abs(det) < 1e-15:
            vx = t[0]
            vy = t[1]
        else:
            vx = (A * t[0] + B * t[1]) / det
            vy = (-B * t[0] + A * t[1]) / det

    return np.array([vx, vy, theta], dtype=np.float64)


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2.0 * np.pi
    while angle < -np.pi:
        angle += 2.0 * np.pi
    return angle


def is_positive_semidefinite(M, tol=1e-10):
    M = np.asarray(M, dtype=np.float64)
    if not np.allclose(M, M.T, atol=tol):
        return False
    eigvals = np.linalg.eigvalsh(M)
    return np.min(eigvals) >= -tol


def nearest_positive_semidefinite(M):
    M = np.asarray(M, dtype=np.float64)
    M_sym = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def mahalanobis_distance(x, mu, Sigma):
    diff = np.asarray(x) - np.asarray(mu)
    try:
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        Sigma_inv = np.linalg.pinv(Sigma)
    return np.sqrt(max(diff @ Sigma_inv @ diff, 0.0))


def chi2_confidence_interval(dim, confidence=0.95):


    from math import sqrt

    z = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    if dim <= 0:
        return 0.0
    approx = dim * (1.0 - 2.0 / (9.0 * dim) + z * sqrt(2.0 / (9.0 * dim))) ** 3
    return max(approx, 0.0)


def compute_trajectory_ate(estimated, ground_truth):
    est = np.asarray(estimated, dtype=np.float64)
    gt = np.asarray(ground_truth, dtype=np.float64)
    if est.shape != gt.shape:
        raise ValueError("shape mismatch")
    if est.ndim == 1:
        est = est.reshape(1, -1)
        gt = gt.reshape(1, -1)
    diffs = est[:, 0:2] - gt[:, 0:2]
    rmse = np.sqrt(np.mean(np.sum(diffs ** 2, axis=1)))
    return rmse


def robust_loss(residual, huber_delta=1.0):
    r = abs(float(residual))
    d = float(huber_delta)
    if r <= d:
        return 0.5 * r * r
    else:
        return d * (r - 0.5 * d)


def format_matrix_latex(M, name="M", precision=4):
    rows = []
    for row in M:
        rows.append(" & ".join(f"{v:.{precision}f}" for v in row))
    body = " \\\\\n".join(rows)
    return f"\\[{name} = \\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}\\]"
