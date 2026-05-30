
import numpy as np
from typing import Tuple, Callable


def legendre_gauss_nodes_weights(n: int) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return nodes, weights


def jacobi_gauss_nodes_weights(n: int, alpha: float = 0.0, beta: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    if n < 1:
        raise ValueError("n must be positive")
    if alpha <= -1.0 or beta <= -1.0:
        raise ValueError("alpha and beta must be > -1")
    nodes, weights = np.polynomial.legendre.leggauss(n)




    if abs(alpha) < 1e-10 and abs(beta) < 1e-10:
        return nodes, weights
    

    shift = (beta - alpha) / (2.0 * n + alpha + beta + 2.0)
    nodes = nodes + shift * (1.0 - nodes ** 2)
    nodes = np.clip(nodes, -1.0, 1.0)
    weights = weights * (1.0 - nodes) ** alpha * (1.0 + nodes) ** beta
    weights = weights / np.sum(weights) * 2.0
    return nodes, weights


def map_to_physical_interval(
    nodes: np.ndarray,
    weights: np.ndarray,
    a: float,
    b: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if b <= a:
        raise ValueError("b must be greater than a")
    scale = 0.5 * (b - a)
    shift = 0.5 * (a + b)
    phys_nodes = shift + scale * nodes
    phys_weights = scale * weights
    return phys_nodes, phys_weights


def five_dimensional_gauss_quadrature(
    f: Callable[[np.ndarray], np.ndarray],
    n_per_dim: int = 8,
    mass_number: int = 235,
) -> float:
    from collective_coordinates import collective_coordinate_bounds
    
    bounds = collective_coordinate_bounds(mass_number)
    keys = ['beta2', 'beta3', 'beta4', 'beta5', 'delta']
    

    nodes_list = []
    weights_list = []
    
    for key in keys:
        a, b = bounds[key]
        if key == 'delta':

            nodes, weights = jacobi_gauss_nodes_weights(n_per_dim, alpha=0.0, beta=0.0)
        else:
            nodes, weights = legendre_gauss_nodes_weights(n_per_dim)
        phys_nodes, phys_weights = map_to_physical_interval(nodes, weights, a, b)
        nodes_list.append(phys_nodes)
        weights_list.append(phys_weights)
    

    total = 0.0
    n_total = n_per_dim ** 5
    

    for i2 in range(n_per_dim):
        b2 = nodes_list[0][i2]
        w2 = weights_list[0][i2]
        for i3 in range(n_per_dim):
            b3 = nodes_list[1][i3]
            w3 = weights_list[1][i3]
            for i4 in range(n_per_dim):
                b4 = nodes_list[2][i4]
                w4 = weights_list[2][i4]
                for i5 in range(n_per_dim):
                    b5 = nodes_list[3][i5]
                    w5 = weights_list[3][i5]
                    for idelta in range(n_per_dim):
                        delta_val = nodes_list[4][idelta]
                        wdelta = weights_list[4][idelta]
                        
                        q = np.array([[b2, b3, b4, b5, delta_val]])
                        val = f(q)[0]
                        total += val * w2 * w3 * w4 * w5 * wdelta
    
    return float(total)


def partition_function_integral(
    mass_number: int,
    charge_number: int,
    excitation_energy: float,
    n_per_dim: int = 6,
) -> float:
    from potential_energy_surface import potential_energy
    from diffusion_coefficient import nuclear_temperature
    
    T = nuclear_temperature(excitation_energy, mass_number)
    if T < 0.1:
        T = 0.1
    
    def integrand(q_array):
        vals = np.zeros(len(q_array))
        for i in range(len(q_array)):
            V = potential_energy(q_array[i], mass_number, charge_number)
            vals[i] = np.exp(-V / T)
        return vals
    
    Z = five_dimensional_gauss_quadrature(integrand, n_per_dim, mass_number)
    return float(Z)
