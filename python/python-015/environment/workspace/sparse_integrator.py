
import numpy as np
from typing import Tuple, Callable
from itertools import product


def clenshaw_curtis_weights(N: int) -> np.ndarray:
    if N == 1:
        return np.array([2.0])
    
    n = N - 1

    c = np.zeros(N)
    c[0::2] = 2.0 / np.arange(1, n + 2, 2)
    c[0] = 1.0
    if n % 2 == 0:
        c[-1] = 1.0 / (n + 1)
    else:
        c[-1] = 0.0
    

    c_extended = np.concatenate([c, c[-2:0:-1]])
    f = np.fft.ifft(c_extended).real
    w = 2 * f[:N]
    w[0] *= 0.5
    w[-1] *= 0.5
    
    return w


def clenshaw_curtis_nodes(N: int) -> np.ndarray:
    if N == 1:
        return np.array([0.0])
    
    j = np.arange(N)
    x = np.cos(np.pi * j / (N - 1))
    return x


def difference_weights(level: int) -> np.ndarray:
    if level == 0:
        return np.array([2.0])
    elif level == 1:

        w1 = clenshaw_curtis_weights(3)
        dw = np.zeros(3)
        dw[0] = w1[0]
        dw[1] = w1[1] - 2.0
        dw[2] = w1[2]
        return dw
    else:
        N_prev = 2 ** (level - 1) + 1
        N_curr = 2 ** level + 1
        
        w_prev = clenshaw_curtis_weights(N_prev)
        w_curr = clenshaw_curtis_weights(N_curr)
        

        dw = w_curr.copy()
        for i in range(N_prev):
            dw[2 * i] -= w_prev[i]
        
        return dw


def generate_index_set(dim: int, max_level: int) -> np.ndarray:
    indices = []
    
    def recursive_gen(remaining_dim: int, remaining_level: int, current: list):
        if remaining_dim == 1:
            for l in range(remaining_level + 1):
                indices.append(current + [l])
        else:
            for l in range(remaining_level + 1):
                recursive_gen(remaining_dim - 1, remaining_level - l, current + [l])
    

    recursive_gen(dim, max_level, [])
    

    result = []
    for idx in indices:
        if sum(idx) >= max_level - dim + 1:
            result.append(idx)
    
    return np.array(result, dtype=int)


def sparse_grid_quadrature(dim: int, max_level: int,
                            bounds: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        bounds = np.array([[-1.0, 1.0]] * dim)
    
    if bounds.shape != (dim, 2):
        raise ValueError("bounds必须是(dim, 2)数组")
    

    lengths = bounds[:, 1] - bounds[:, 0]
    midpoints = 0.5 * (bounds[:, 0] + bounds[:, 1])
    
    index_set = generate_index_set(dim, max_level)
    

    all_points = []
    all_weights = []
    
    for idx in index_set:

        level_sum = np.sum(idx)
        coeff = ((-1) ** (max_level - level_sum)) * comb(max_level - level_sum, dim - 1)
        

        sub_points_list = []
        sub_weights_list = []
        
        for d in range(dim):
            level = idx[d]
            if level == 0:
                nodes = np.array([0.0])
                dw = np.array([2.0])
            else:
                N = 2 ** level + 1
                nodes = clenshaw_curtis_nodes(N)
                dw = difference_weights(level)
            

            nodes = midpoints[d] + 0.5 * lengths[d] * nodes
            dw = dw * (lengths[d] / 2.0)
            
            sub_points_list.append(nodes)
            sub_weights_list.append(dw)
        

        for point_tuple in product(*sub_points_list):
            all_points.append(point_tuple)
        for weight_tuple in product(*sub_weights_list):
            all_weights.append(coeff * np.prod(weight_tuple))
    
    if len(all_points) == 0:
        return np.zeros((0, dim)), np.zeros(0)
    
    points = np.array(all_points)
    weights = np.array(all_weights)
    

    points_rounded = np.round(points, decimals=14)
    unique_points = []
    unique_weights = []
    
    visited = set()
    for i in range(len(points_rounded)):
        key = tuple(points_rounded[i])
        if key not in visited:
            visited.add(key)
            unique_points.append(points_rounded[i])

            w_sum = 0.0
            for j in range(len(points_rounded)):
                if tuple(points_rounded[j]) == key:
                    w_sum += weights[j]
            unique_weights.append(w_sum)
    
    points = np.array(unique_points)
    weights = np.array(unique_weights)
    
    return points, weights


def comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def integrate_sparse_grid(dim: int, max_level: int,
                           integrand: Callable[[np.ndarray], np.ndarray],
                           bounds: np.ndarray = None) -> float:
    points, weights = sparse_grid_quadrature(dim, max_level, bounds)
    
    if len(points) == 0:
        return 0.0
    
    values = integrand(points)
    result = np.dot(weights, values)
    return result
