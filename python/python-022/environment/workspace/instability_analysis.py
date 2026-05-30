
import numpy as np
from typing import Tuple, List, Dict
from icf_parameters import NP, TP
from utils import clamp






def generate_surface_perturbation_ifs(n_points: int = 1000,
                                      amplitude: float = NP.PERTURBATION_AMPLITUDE,
                                      mode: int = NP.PERTURBATION_MODE) -> np.ndarray:

    transforms = [
        (np.array([[0.8, 0.0], [0.0, 0.8]]), np.array([0.1, 0.04])),
        (np.array([[0.5, 0.0], [0.0, 0.5]]), np.array([0.25, 0.4])),
        (np.array([[0.355, -0.355], [0.355, 0.355]]), np.array([0.266, 0.078])),
        (np.array([[0.355, 0.355], [-0.355, 0.355]]), np.array([0.378, 0.434])),
    ]

    x = np.array([0.5, 0.5])
    points = []


    for _ in range(100):
        idx = np.random.randint(0, len(transforms))
        A, b = transforms[idx]
        x = A @ x + b

    for _ in range(n_points):
        idx = np.random.randint(0, len(transforms))
        A, b = transforms[idx]
        x = A @ x + b
        points.append(x.copy())

    points = np.array(points)


    theta = np.linspace(0.0, 2.0 * np.pi, mode + 1)[:-1]
    perturbation = np.zeros(mode)
    for i in range(mode):

        sector_mask = (points[:, 0] >= i / mode) & (points[:, 0] < (i + 1) / mode)
        if np.any(sector_mask):
            perturbation[i] = amplitude * np.std(points[sector_mask, 1])
        else:
            perturbation[i] = amplitude * 0.1

    return perturbation






def atwood_number(rho_high: float, rho_low: float) -> float:
    denom = rho_high + rho_low
    if denom < 1.0e-30:
        return 0.0
    return (rho_high - rho_low) / denom


def rayleigh_taylor_growth_rate(rho_ablation: float, rho_corona: float,
                                 acceleration: float, mode_l: int,
                                 radius: float, v_ablation: float,
                                 beta_stabilization: float = 3.0) -> float:
    A_t = atwood_number(rho_ablation, rho_corona)
    if A_t <= 0.0 or radius <= 1.0e-15 or acceleration <= 0.0:
        return 0.0

    k = mode_l / radius
    g = acceleration

    term_classical = np.sqrt(max(A_t * k * g, 0.0))
    term_ablative = beta_stabilization * k * v_ablation

    gamma = term_classical - term_ablative
    return max(gamma, 0.0)


def richtmyer_meshkov_amplitude(eta_0: float, delta_v: float,
                                 A_t: float, mode_l: int, radius: float,
                                 t: float) -> float:
    if radius <= 1.0e-15 or t < 0.0:
        return eta_0
    k = mode_l / radius
    return eta_0 * (1.0 + k * A_t * delta_v * t)


def compute_mode_growth_spectrum(rho_profile: np.ndarray,
                                 r_cells: np.ndarray,
                                 u_nodes: np.ndarray,
                                 mode_range: range = range(1, 25)) -> Dict[int, float]:
    growth_rates = {}
    n_cells = len(r_cells)
    if n_cells < 2:
        return growth_rates


    interface_idx = 0
    max_grad = 0.0
    for i in range(1, n_cells - 1):
        dr = r_cells[i + 1] - r_cells[i - 1]
        if dr < 1.0e-15:
            continue
        grad = abs(rho_profile[i + 1] - rho_profile[i - 1]) / dr
        if grad > max_grad:
            max_grad = grad
            interface_idx = i

    if interface_idx == 0:
        return growth_rates

    rho_high = rho_profile[interface_idx]
    rho_low = rho_profile[min(interface_idx + 1, n_cells - 1)]
    R_int = r_cells[interface_idx]


    u_int = 0.5 * (u_nodes[interface_idx] + u_nodes[interface_idx + 1])
    accel = u_int**2 / max(R_int, 1.0e-15)


    v_abl = abs(u_int) * 0.1

    for l in mode_range:
        gamma = rayleigh_taylor_growth_rate(
            rho_high, rho_low, accel, l, R_int, v_abl
        )
        growth_rates[l] = gamma

    return growth_rates






def build_energy_flow_digraph() -> Tuple[np.ndarray, List[str]]:
    node_names = [
        "Laser", "E_thermal", "I_thermal", "Radiation",
        "Kinetic", "Fusion", "Neutron_loss", "Xray_loss"
    ]


    arcs = [
        (0, 1),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 5),
        (3, 7),
        (5, 2),
        (5, 6),
        (4, 2),
    ]

    n_nodes = len(node_names)
    adjacency_matrix = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in arcs:
        adjacency_matrix[i, j] = 1

    return adjacency_matrix, node_names


def energy_flow_pagerank(adjacency: np.ndarray, damping: float = 0.85,
                         tol: float = 1.0e-8, max_iter: int = 100) -> np.ndarray:
    n = adjacency.shape[0]
    out_degrees = np.sum(adjacency, axis=1)

    transition = np.zeros((n, n))
    for j in range(n):
        if out_degrees[j] > 0:
            transition[:, j] = adjacency[j, :] / out_degrees[j]
        else:
            transition[:, j] = 1.0 / n

    pr = np.ones(n) / n
    for _ in range(max_iter):
        pr_new = (1.0 - damping) / n + damping * transition @ pr
        if np.linalg.norm(pr_new - pr, ord=1) < tol:
            break
        pr = pr_new

    return pr


def analyze_instability_feedthrough(mode_growth: Dict[int, float],
                                    perturbation_spectrum: np.ndarray) -> float:
    total = 0.0
    for l, gamma in mode_growth.items():
        if l - 1 < len(perturbation_spectrum):
            eta_l = perturbation_spectrum[l - 1]
        else:
            eta_l = NP.PERTURBATION_AMPLITUDE / l
        total += (eta_l * gamma)**2
    return np.sqrt(total)
