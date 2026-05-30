
import numpy as np
from typing import Tuple, Callable


def taylor_green_vortex(x: np.ndarray, y: np.ndarray, t: float = 0.0,
                        nu: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    decay_v = np.exp(-2.0 * nu * t)
    decay_p = np.exp(-4.0 * nu * t)
    
    u = np.sin(x) * np.cos(y) * decay_v
    v = -np.cos(x) * np.sin(y) * decay_v
    p = 0.25 * (np.cos(2.0 * x) + np.cos(2.0 * y)) * decay_p
    
    return u, v, p


def cavity_flow_exact(x: np.ndarray, y: np.ndarray, Re: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:

    x = np.clip(x, 0.0, 1.0)
    y = np.clip(y, 0.0, 1.0)
    

    u = 16.0 * x**2 * (1.0 - x)**2 * y * (1.0 - y) * (2.0 * y - 1.0) * Re / 100.0
    v = -16.0 * x * (1.0 - x) * (2.0 * x - 1.0) * y**2 * (1.0 - y)**2 * Re / 100.0
    
    return u, v


def interpolate_to_grid(tri_nodes: np.ndarray, tri_elements: np.ndarray,
                        node_values: np.ndarray,
                        grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    nx = len(grid_x)
    ny = len(grid_y)
    grid_values = np.zeros((ny, nx))
    

    n_tri = tri_elements.shape[0]
    tri_min_x = np.zeros(n_tri)
    tri_max_x = np.zeros(n_tri)
    tri_min_y = np.zeros(n_tri)
    tri_max_y = np.zeros(n_tri)
    
    for t in range(n_tri):
        idx = tri_elements[t]
        pts = tri_nodes[idx]
        tri_min_x[t] = np.min(pts[:, 0])
        tri_max_x[t] = np.max(pts[:, 0])
        tri_min_y[t] = np.min(pts[:, 1])
        tri_max_y[t] = np.max(pts[:, 1])
    

    for j in range(ny):
        for i in range(nx):
            px = grid_x[i]
            py = grid_y[j]
            
            found = False
            for t in range(n_tri):

                if px < tri_min_x[t] or px > tri_max_x[t]:
                    continue
                if py < tri_min_y[t] or py > tri_max_y[t]:
                    continue
                

                idx = tri_elements[t]
                p1, p2, p3 = tri_nodes[idx[0]], tri_nodes[idx[1]], tri_nodes[idx[2]]
                
                denom = (p2[1] - p3[1]) * (p1[0] - p3[0]) + (p3[0] - p2[0]) * (p1[1] - p3[1])
                if abs(denom) < 1e-14:
                    continue
                
                lam1 = ((p2[1] - p3[1]) * (px - p3[0]) + (p3[0] - p2[0]) * (py - p3[1])) / denom
                lam2 = ((p3[1] - p1[1]) * (px - p3[0]) + (p1[0] - p3[0]) * (py - p3[1])) / denom
                lam3 = 1.0 - lam1 - lam2
                

                if lam1 >= -1e-10 and lam2 >= -1e-10 and lam3 >= -1e-10:
                    vals = node_values[idx]
                    grid_values[j, i] = lam1 * vals[0] + lam2 * vals[1] + lam3 * vals[2]
                    found = True
                    break
            
            if not found:

                dists = np.linalg.norm(tri_nodes - np.array([px, py]), axis=1)
                nearest = np.argmin(dists)
                grid_values[j, i] = node_values[nearest]
    
    return grid_values


def compute_flow_acoustic_source(flow_u: np.ndarray, flow_v: np.ndarray,
                                 acoustic_p: np.ndarray,
                                 dx: float, dy: float,
                                 k: float, c0: float = 1540.0) -> np.ndarray:
    ny, nx = acoustic_p.shape
    source = np.zeros_like(acoustic_p, dtype=complex)
    

    for j in range(ny):
        for i in range(nx):

            if i == 0:
                dpx = (acoustic_p[j, i + 1] - acoustic_p[j, i]) / dx
            elif i == nx - 1:
                dpx = (acoustic_p[j, i] - acoustic_p[j, i - 1]) / dx
            else:
                dpx = (acoustic_p[j, i + 1] - acoustic_p[j, i - 1]) / (2.0 * dx)
            

            if j == 0:
                dpy = (acoustic_p[j + 1, i] - acoustic_p[j, i]) / dy
            elif j == ny - 1:
                dpy = (acoustic_p[j, i] - acoustic_p[j - 1, i]) / dy
            else:
                dpy = (acoustic_p[j + 1, i] - acoustic_p[j - 1, i]) / (2.0 * dy)
            

            v_dot_grad_p = flow_u[j, i] * dpx + flow_v[j, i] * dpy
            source[j, i] = -2.0j * k / c0 * v_dot_grad_p
    
    return source


def mach_number_field(flow_u: np.ndarray, flow_v: np.ndarray,
                      c0: float = 1540.0) -> np.ndarray:
    velocity_magnitude = np.sqrt(flow_u**2 + flow_v**2)
    return velocity_magnitude / c0
