
import numpy as np
from scipy.spatial import Delaunay
from typing import Callable, Tuple, Optional


def signed_distance_circle(p: np.ndarray, xc: float, yc: float, r: float) -> np.ndarray:
    return np.sqrt((p[:, 0] - xc) ** 2 + (p[:, 1] - yc) ** 2) - r


def signed_distance_rectangle(p: np.ndarray, xlim: Tuple[float, float],
                              ylim: Tuple[float, float]) -> np.ndarray:
    dx = np.maximum(xlim[0] - p[:, 0], p[:, 0] - xlim[1])
    dy = np.maximum(ylim[0] - p[:, 1], p[:, 1] - ylim[1])
    d_inside = np.maximum(dx, dy)

    d_outside = np.sqrt(np.maximum(dx, 0) ** 2 + np.maximum(dy, 0) ** 2)
    d = np.where(np.logical_and(dx < 0, dy < 0), d_inside, d_outside)
    return d


def distmesh_2d(fd: Callable[[np.ndarray], np.ndarray],
                fh: Callable[[np.ndarray], np.ndarray],
                h0: float,
                bbox: Tuple[Tuple[float, float], Tuple[float, float]],
                pfix: Optional[np.ndarray] = None,
                iteration_max: int = 100,
                tol: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
    (xmin, xmax), (ymin, ymax) = bbox
    

    x_range = np.arange(xmin, xmax, h0)
    y_range = np.arange(ymin, ymax, h0 * np.sqrt(3) / 2)
    xx, yy = np.meshgrid(x_range, y_range)

    yy[1::2, :] += h0 / 2
    p_init = np.column_stack((xx.ravel(), yy.ravel()))
    

    if fh is not None:
        h_vals = fh(p_init)
        h_vals = np.maximum(h_vals, 1e-6)
        prob = np.minimum(1.0, (h0 / h_vals) ** 2)
        rng = np.random.default_rng(42)
        accept = rng.random(len(p_init)) < prob
        p = p_init[accept]
    else:
        p = p_init.copy()
    

    if pfix is not None and len(pfix) > 0:
        p = np.vstack([pfix, p])
    

    d = fd(p)
    p = p[d < 0]
    
    F_scale = 1.2
    
    for it in range(iteration_max):
        if len(p) < 3:
            raise ValueError("Too few points for triangulation")
        
        tri = Delaunay(p)
        t = tri.simplices
        

        edges = []
        for tri_elem in t:
            edges.extend([
                tuple(sorted([tri_elem[0], tri_elem[1]])),
                tuple(sorted([tri_elem[1], tri_elem[2]])),
                tuple(sorted([tri_elem[2], tri_elem[0]])),
            ])
        edges = list(set(edges))
        

        forces = np.zeros_like(p)
        bar_lengths = []
        for i, j in edges:
            dp = p[j] - p[i]
            L = np.linalg.norm(dp)
            if L < 1e-12:
                continue
            bar_lengths.append(L)
            

            hi = fh(np.array([p[i]]))[0] if fh is not None else h0
            hj = fh(np.array([p[j]]))[0] if fh is not None else h0
            h_bar = 0.5 * (hi + hj)
            L0 = h_bar * F_scale
            

            F = max(L0 - L, 0.0)
            fvec = (F / L) * dp
            forces[i] -= fvec
            forces[j] += fvec
        

        p_new = p + 0.2 * forces
        

        d_new = fd(p_new)

        eps = 1e-6
        grad = np.zeros_like(p_new)
        for dim in range(2):
            p_perturb = p_new.copy()
            p_perturb[:, dim] += eps
            grad[:, dim] = (fd(p_perturb) - d_new) / eps
        
        grad_norm = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norm = np.maximum(grad_norm, 1e-12)
        

        outside = d_new > 0
        p_new[outside] -= d_new[outside][:, None] * (grad[outside] / grad_norm[outside])
        

        if pfix is not None and len(pfix) > 0:
            p_new[:len(pfix)] = pfix
        

        max_disp = np.max(np.linalg.norm(p_new - p, axis=1))
        p = p_new
        if max_disp < tol * h0:
            break
    

    tri = Delaunay(p)
    t = tri.simplices
    return p, t


def simpqual(p: np.ndarray, t: np.ndarray) -> np.ndarray:
    quality = np.zeros(len(t))
    for i, tri in enumerate(t):
        v1, v2, v3 = p[tri[0]], p[tri[1]], p[tri[2]]
        a = np.linalg.norm(v2 - v1)
        b = np.linalg.norm(v3 - v2)
        c = np.linalg.norm(v1 - v3)
        

        s = 0.5 * (a + b + c)
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        

        R = (a * b * c) / (4.0 * max(area, 1e-12))

        r = area / max(s, 1e-12)
        
        quality[i] = r / max(R, 1e-12)
    return quality


def generate_reaction_coordinate_mesh(q_range: Tuple[float, float],
                                      rmsd_range: Tuple[float, float],
                                      h0: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    (qmin, qmax), (rmsdmin, rmsdmax) = q_range, rmsd_range
    bbox = ((qmin, qmax), (rmsdmin, rmsdmax))
    

    def fh(pp):
        q = pp[:, 0]
        rmsd = pp[:, 1]
        center_q = 0.5 * (qmin + qmax)
        dist = np.sqrt((q - center_q) ** 2 + (rmsd - 0.5 * (rmsdmin + rmsdmax)) ** 2)
        return h0 * (0.5 + 0.5 * dist / max(qmax - qmin, rmsdmax - rmsdmin))
    
    def fd(pp):
        return signed_distance_rectangle(pp, (qmin, qmax), (rmsdmin, rmsdmax))
    
    p, t = distmesh_2d(fd, fh, h0, bbox, iteration_max=80, tol=1e-3)
    return p, t
