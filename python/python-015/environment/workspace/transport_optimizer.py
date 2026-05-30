
import numpy as np
from typing import Tuple


def knapsack_rational(n: int, budget: float, gains: np.ndarray,
                      costs: np.ndarray) -> Tuple[np.ndarray, float, float]:
    x = np.zeros(n)
    total_cost = 0.0
    total_gain = 0.0
    
    for i in range(n):
        if costs[i] < 1e-15:

            x[i] = 1.0
            total_gain += gains[i]
            continue
        
        if budget <= total_cost + 1e-15:
            x[i] = 0.0
        elif total_cost + costs[i] <= budget:
            x[i] = 1.0
            total_cost += costs[i]
            total_gain += gains[i]
        else:

            remaining = budget - total_cost
            x[i] = remaining / costs[i]
            total_cost = budget
            total_gain += gains[i] * x[i]
    
    return x, total_cost, total_gain


def sort_by_profit_density(gains: np.ndarray, costs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    safe_costs = np.where(costs < 1e-15, 1e-15, costs)
    density = gains / safe_costs
    

    idx = np.argsort(-density)
    return gains[idx], costs[idx]


def transport_channel_selection(n_channels: int, e_fermi: float,
                                 energy_window: float,
                                 ham, bz_sampler) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    k_points = bz_sampler(n_channels)
    
    gains = np.zeros(n_channels)
    costs = np.zeros(n_channels)
    
    for i in range(n_channels):
        k = k_points[i]
        energies, _ = ham.eigenproblem(k)
        

        gap = abs(energies[1] - energies[0])
        gains[i] = 1.0 / (gap + 0.01)
        

        e_avg = 0.5 * (energies[0] + energies[1])
        costs[i] = abs(e_avg - e_fermi) + 0.01
    

    gains_sorted, costs_sorted = sort_by_profit_density(gains, costs)
    

    budget = energy_window * n_channels
    x, total_cost, total_gain = knapsack_rational(n_channels, budget, gains_sorted, costs_sorted)
    
    return x, total_cost, total_gain


def chiral_anomaly_conductance(ham, k_points: np.ndarray, e_field: np.ndarray,
                                b_field: np.ndarray, band_index: int = 0) -> np.ndarray:
    from berry_curvature import berry_curvature_numeric
    
    N = k_points.shape[0]
    conductance = np.zeros(N)
    
    e_dot_b = np.dot(e_field, b_field)
    e_norm = np.linalg.norm(e_field)
    b_norm = np.linalg.norm(b_field)
    
    for i in range(N):
        k = k_points[i]
        omega = berry_curvature_numeric(ham, k, band_index)
        







        raise NotImplementedError("Hole_3: 手征反常电导计算待实现")
    
    return conductance
