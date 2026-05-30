
import numpy as np
from typing import List, Tuple, Dict


def triangle_area(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    return 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))


def q_measure(triangles: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    n_tri = triangles.shape[0]
    q_values = np.zeros(n_tri)
    
    for i in range(n_tri):
        idx = triangles[i]
        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
        
        a = np.linalg.norm(p2 - p1)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p1 - p3)
        
        area = triangle_area(p1, p2, p3)
        
        if area < 1e-14:
            q_values[i] = 0.0
            continue
        
        denom = a**2 + b**2 + c**2
        if denom < 1e-14:
            q_values[i] = 0.0
            continue
        
        q_values[i] = 4.0 * np.sqrt(3.0) * area / denom
    
    return q_values


def alpha_measure(triangles: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    n_tri = triangles.shape[0]
    alpha_vals = np.zeros(n_tri)
    
    for i in range(n_tri):
        idx = triangles[i]
        p1, p2, p3 = nodes[idx[0]], nodes[idx[1]], nodes[idx[2]]
        
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        

        if a < 1e-14 or b < 1e-14 or c < 1e-14:
            alpha_vals[i] = 0.0
            continue
        

        cos1 = (b**2 + c**2 - a**2) / (2.0 * b * c)
        cos2 = (a**2 + c**2 - b**2) / (2.0 * a * c)
        cos3 = (a**2 + b**2 - c**2) / (2.0 * a * b)
        

        cos1 = np.clip(cos1, -1.0, 1.0)
        cos2 = np.clip(cos2, -1.0, 1.0)
        cos3 = np.clip(cos3, -1.0, 1.0)
        
        theta1 = np.arccos(cos1)
        theta2 = np.arccos(cos2)
        theta3 = np.arccos(cos3)
        
        min_angle = min(theta1, theta2, theta3)
        alpha_vals[i] = min_angle / (np.pi / 3.0)
    
    return alpha_vals


def gamma_measure(nodes: np.ndarray) -> float:
    n_nodes = nodes.shape[0]
    if n_nodes <= 1:
        return 1.0
    
    min_distances = np.zeros(n_nodes)
    
    for i in range(n_nodes):
        dists = np.linalg.norm(nodes - nodes[i], axis=1)
        dists[i] = np.inf
        min_distances[i] = np.min(dists)
    
    d_min = np.min(min_distances)
    d_max = np.max(min_distances)
    
    if d_max < 1e-14:
        return 1.0
    
    return d_min / d_max


def bandwidth_mesh(triangles: np.ndarray) -> int:
    bandwidth = 0
    for tri in triangles:
        i, j, k = tri
        local_bw = max(abs(i - j), abs(j - k), abs(i - k))
        bandwidth = max(bandwidth, local_bw)
    return bandwidth


def mesh_quality_report(triangles: np.ndarray, nodes: np.ndarray) -> Dict[str, float]:
    q_vals = q_measure(triangles, nodes)
    alpha_vals = alpha_measure(triangles, nodes)
    gamma = gamma_measure(nodes)
    bw = bandwidth_mesh(triangles)
    
    report = {
        'q_min': float(np.min(q_vals)),
        'q_mean': float(np.mean(q_vals)),
        'alpha_min': float(np.min(alpha_vals)),
        'alpha_mean': float(np.mean(alpha_vals)),
        'gamma': float(gamma),
        'bandwidth': int(bw),
        'num_triangles': triangles.shape[0],
        'num_nodes': nodes.shape[0]
    }
    return report


def reject_poor_triangles(triangles: np.ndarray, nodes: np.ndarray,
                          q_threshold: float = 0.1,
                          alpha_threshold: float = 0.1) -> np.ndarray:
    q_vals = q_measure(triangles, nodes)
    alpha_vals = alpha_measure(triangles, nodes)
    
    mask = (q_vals >= q_threshold) & (alpha_vals >= alpha_threshold)
    return mask
