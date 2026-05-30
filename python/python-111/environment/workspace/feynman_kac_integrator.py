
import numpy as np
from typing import Callable, Tuple


def brownian_walk_1d(x0: float, D: float, dt: float, n_steps: int,
                     boundary_left: float, boundary_right: float) -> Tuple[np.ndarray, float]:
    if boundary_left >= boundary_right:
        raise ValueError("boundary_left must be less than boundary_right")
    if x0 < boundary_left or x0 > boundary_right:
        raise ValueError("x0 must be within boundaries")
    
    trajectory = np.zeros(n_steps + 1)
    trajectory[0] = x0
    sigma = np.sqrt(2.0 * D * dt)
    
    for k in range(n_steps):
        trajectory[k + 1] = trajectory[k] + sigma * np.random.randn()
        if trajectory[k + 1] <= boundary_left or trajectory[k + 1] >= boundary_right:
            return trajectory[:k + 2], (k + 1) * dt

        if trajectory[k + 1] < boundary_left:
            trajectory[k + 1] = boundary_left + (boundary_left - trajectory[k + 1])
        if trajectory[k + 1] > boundary_right:
            trajectory[k + 1] = boundary_right - (trajectory[k + 1] - boundary_right)
    
    return trajectory, n_steps * dt


def feynman_kac_escape_probability(x0: float, potential: Callable[[float], float],
                                   D: float, dt: float, n_steps: int,
                                   boundary_left: float, boundary_right: float,
                                   n_trajectories: int = 10000) -> Tuple[float, float]:
    if n_trajectories < 1:
        raise ValueError("n_trajectories must be positive")
    
    results = np.zeros(n_trajectories)
    for i in range(n_trajectories):
        traj, _ = brownian_walk_1d(x0, D, dt, n_steps, boundary_left, boundary_right)

        final_pos = traj[-1]

        if final_pos >= boundary_right:
            weight = 1.0
        elif final_pos <= boundary_left:
            weight = 0.0
        else:

            weight = (final_pos - boundary_left) / (boundary_right - boundary_left)
        

        path_integral = 0.0
        for k in range(len(traj) - 1):
            vk = potential(traj[k])
            vk1 = potential(traj[k + 1])
            path_integral += 0.5 * dt * (vk + vk1)
        
        results[i] = weight * np.exp(-path_integral)
    
    prob = float(np.mean(results))
    std_err = float(np.std(results) / np.sqrt(n_trajectories))
    return prob, std_err


def mean_first_passage_time_1d(x0: float, potential: Callable[[float], float],
                               D: float, dt: float, n_steps: int,
                               boundary_left: float, boundary_right: float,
                               n_trajectories: int = 5000) -> Tuple[float, float]:
    fpt_list = []
    for _ in range(n_trajectories):
        _, tau = brownian_walk_1d(x0, D, dt, n_steps, boundary_left, boundary_right)
        fpt_list.append(tau)
    
    fpt_array = np.array(fpt_list)
    mfpt = float(np.mean(fpt_array))
    std_err = float(np.std(fpt_array) / np.sqrt(n_trajectories))
    return mfpt, std_err


def kramers_rate_approximation(barrier_height: float, kT: float,
                                D: float, curvature_top: float,
                                curvature_bottom: float) -> float:
    if barrier_height <= 0:
        raise ValueError("barrier_height must be positive")
    if curvature_bottom <= 0:
        raise ValueError("curvature_bottom must be positive")
    if curvature_top >= 0:
        raise ValueError("curvature_top must be negative")
    
    omega_m = np.sqrt(curvature_bottom)
    omega_b = np.sqrt(abs(curvature_top))
    prefactor = (omega_m * omega_b) / (2.0 * np.pi)
    rate = prefactor * np.exp(-barrier_height / kT)
    return float(rate)


def path_integral_free_energy(x_samples: np.ndarray, potential: Callable[[float], float],
                              D: float, dt: float, n_steps: int,
                              boundary_left: float, boundary_right: float,
                              n_paths_per_x: int = 2000) -> np.ndarray:
    N = len(x_samples)
    histogram = np.zeros(N)
    bin_edges = np.linspace(boundary_left, boundary_right, N + 1)
    
    total_samples = 0
    for x0 in x_samples:
        for _ in range(n_paths_per_x):
            traj, _ = brownian_walk_1d(x0, D, dt, n_steps, boundary_left, boundary_right)

            counts, _ = np.histogram(traj, bins=bin_edges)
            histogram += counts
            total_samples += len(traj)
    

    bin_widths = np.diff(bin_edges)
    prob = histogram / (total_samples * bin_widths + 1e-12)
    prob = np.maximum(prob, 1e-12)
    

    free_energy = -np.log(prob)
    free_energy -= free_energy.min()
    return free_energy
