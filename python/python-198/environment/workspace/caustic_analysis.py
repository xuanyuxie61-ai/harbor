
import numpy as np


def caustic_mapping(n, m):
    j = np.arange(n + 1)
    k = np.mod(m * j, n)
    
    theta_j = 2.0 * np.pi * j / n
    theta_k = 2.0 * np.pi * k / n
    
    points_j = np.column_stack((np.cos(theta_j), np.sin(theta_j)))
    points_k = np.column_stack((np.cos(theta_k), np.sin(theta_k)))
    
    edges = np.column_stack((j, k))
    return edges, points_j, points_k


def characteristic_burgers_1d(x0, u0, t):
    return x0 + u0 * t


def shock_formation_time(u0_func, x_grid):
    u0 = u0_func(x_grid)
    du = np.gradient(u0, x_grid)
    min_du = np.min(du)
    idx = np.argmin(du)
    
    if min_du >= 0:
        return np.inf, x_grid[idx]
    
    t_b = -1.0 / min_du
    return t_b, x_grid[idx]


def caustic_inspired_topology_field(nodes, elements, m=5, n=20):

    p1 = nodes[elements[:, 0]]
    p2 = nodes[elements[:, 1]]
    p3 = nodes[elements[:, 2]]
    centroid = (p1 + p2 + p3) / 3.0
    
    r = np.sqrt(centroid[:, 0] ** 2 + centroid[:, 1] ** 2)
    theta = np.arctan2(centroid[:, 1], centroid[:, 0])
    


    theta_mod = np.mod(theta * m, 2.0 * np.pi)
    

    velocity = np.sin(n * theta_mod) * np.exp(-2.0 * r ** 2)
    
    return velocity


def detect_gradient_catastrophe(x_history, u_history, t_array):
    grad_max_history = []
    for i, t in enumerate(t_array):
        u = u_history[i]
        if len(u) < 2:
            grad_max_history.append(0.0)
            continue

        x_sorted = np.sort(x_history)
        dx_min = np.min(np.diff(x_sorted))
        if dx_min < 1e-12:
            dx_min = 1e-12
        grad = np.gradient(u, dx_min)
        grad_max_history.append(np.max(np.abs(grad)))
    
    grad_max_history = np.array(grad_max_history)
    threshold = 10.0 * grad_max_history[0] if grad_max_history[0] > 0 else 100.0
    
    catastrophic = np.where(grad_max_history > threshold)[0]
    if len(catastrophic) > 0:
        return t_array[catastrophic[0]], grad_max_history
    return None, grad_max_history
