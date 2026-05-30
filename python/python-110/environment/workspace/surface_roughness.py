
import numpy as np
from typing import Tuple
from utils import validate_array_1d, validate_array_2d


def coastline_perturb(
    boundary_points: np.ndarray,
    mu: float,
    n_iter: int = 1,
) -> np.ndarray:
    boundary_points = validate_array_2d(boundary_points, "boundary_points")
    if boundary_points.shape[0] != 2:
        raise ValueError("boundary_points must be 2 x N")
    if not (0.0 <= mu <= 0.5):
        mu = np.clip(mu, 0.0, 0.5)
    p = boundary_points.T.copy()
    for _ in range(n_iter):
        n = p.shape[0]
        sig = mu ** 2
        w = mu + sig * np.random.randn(n)

        p_next = np.roll(p, -1, axis=0)
        p_prev = np.roll(p, 1, axis=0)
        p_next2 = np.roll(p, -2, axis=0)
        perturb = (
            0.5 * (p + p_next)
            + w[:, None] * (p + p_next)
            - w[:, None] * (p_prev + p_next2)
        )
        perturb = np.roll(perturb, -1, axis=0)
        q = np.zeros((2 * n, 2), dtype=float)
        q[0:2 * n:2, :] = p
        q[1:2 * n:2, :] = perturb
        p = q
    return p.T


def fractal_dimension_box_counting(
    curve: np.ndarray,
    n_scales: int = 10,
) -> float:
    curve = validate_array_2d(curve, "curve")
    x = curve[0, :]
    y = curve[1, :]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
    L_max = max(x_max - x_min, y_max - y_min)
    if L_max < 1e-15:
        return 0.0
    epsilons = L_max / (2.0 ** np.arange(1, n_scales + 1))
    counts = []
    for eps in epsilons:
        nx = int(np.ceil((x_max - x_min) / eps))
        ny = int(np.ceil((y_max - y_min) / eps))
        if nx < 1:
            nx = 1
        if ny < 1:
            ny = 1
        occupied = np.zeros((nx, ny), dtype=bool)
        for i in range(curve.shape[1]):
            ix = int(np.floor((x[i] - x_min) / eps))
            iy = int(np.floor((y[i] - y_min) / eps))
            ix = np.clip(ix, 0, nx - 1)
            iy = np.clip(iy, 0, ny - 1)
            occupied[ix, iy] = True
        counts.append(float(np.sum(occupied)))
    counts = np.array(counts, dtype=float)
    epsilons = np.array(epsilons, dtype=float)

    valid = counts > 0
    if np.sum(valid) < 2:
        return 1.0
    log_eps = np.log(1.0 / epsilons[valid])
    log_N = np.log(counts[valid])

    A = np.vstack([log_eps, np.ones_like(log_eps)]).T
    D_f, _ = np.linalg.lstsq(A, log_N, rcond=None)[0]
    return float(D_f)


def roughness_induced_broadening(
    rms_roughness_nm: float,
    dot_radius_nm: float,
    m_star_ratio: float = 0.023,
) -> float:
    if rms_roughness_nm <= 0 or dot_radius_nm <= 0:
        raise ValueError("Roughness and radius must be positive")
    H_BAR = 1.054571817e-34
    M_E = 9.10938356e-31
    m_star = m_star_ratio * M_E
    R_dot = dot_radius_nm * 1e-9
    delta_r = rms_roughness_nm * 1e-9

    E_conf = H_BAR ** 2 / (2.0 * m_star * R_dot ** 2)

    Delta_E = E_conf * (delta_r / R_dot)
    Gamma_ang = Delta_E / H_BAR
    return float(Gamma_ang)


def generate_rough_quantum_dot_boundary(
    R_nominal: float,
    n_vertices: int = 64,
    mu_perturb: float = 0.02,
    n_iter: int = 3,
) -> Tuple[np.ndarray, float]:
    if R_nominal <= 0:
        raise ValueError("Radius must be positive")
    theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
    x = R_nominal * np.cos(theta)
    y = R_nominal * np.sin(theta)
    boundary = np.vstack([x, y])
    rough_boundary = coastline_perturb(boundary, mu_perturb, n_iter)
    D_f = fractal_dimension_box_counting(rough_boundary)
    return rough_boundary, D_f


def effective_potential_perturbation(
    r_grid: np.ndarray,
    R_nominal: float,
    rms_roughness: float,
    barrier_height_eV: float = 0.5,
) -> np.ndarray:
    r_grid = validate_array_1d(r_grid, "r_grid")
    if R_nominal <= 0 or rms_roughness < 0:
        raise ValueError("Invalid geometric parameters")
    EV_TO_J = 1.602176634e-19
    V0 = barrier_height_eV * EV_TO_J
    delta = rms_roughness
    if delta < 1e-12:
        return np.zeros_like(r_grid)

    amp = 0.1 * V0 * (2.0 * np.random.rand() - 1.0)
    V_pert = amp * np.exp(-0.5 * ((r_grid - R_nominal) / delta) ** 2)
    return V_pert
