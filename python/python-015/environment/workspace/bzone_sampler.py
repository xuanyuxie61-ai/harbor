
import numpy as np
from typing import Tuple, Callable, Optional


def cholesky_factor(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    n = a.shape[0]
    
    if a.shape[0] != a.shape[1]:
        raise ValueError("矩阵必须是方阵")
    

    if np.max(np.abs(a - a.T)) > 1e-12:
        raise ValueError("矩阵必须是对称的")
    

    try:
        L = np.linalg.cholesky(a)
    except np.linalg.LinAlgError:
        raise ValueError("矩阵必须是正定的")
    

    u = L.T
    return u


def uniform_in_sphere01_map(m: int, n: int) -> np.ndarray:
    x = np.random.randn(m, n)
    norms = np.linalg.norm(x, axis=0)
    

    norms = np.where(norms < 1e-15, 1.0, norms)
    

    u = np.random.rand(n)
    radius = u ** (1.0 / m)
    
    x = x / norms * radius
    return x


def ellipse_sample(n: int, a: np.ndarray, r: float) -> np.ndarray:
    m = a.shape[0]
    u = cholesky_factor(a)
    
    y = uniform_in_sphere01_map(m, n)
    y = r * y
    

    x = np.linalg.solve(u, y)
    return x


def ellipse_area(a: np.ndarray, r: float) -> float:
    m = a.shape[0]
    det_a = np.linalg.det(a)
    
    if det_a <= 0:
        raise ValueError("矩阵A的行列式必须为正")
    

    from scipy.special import gamma as scipy_gamma
    unit_sphere_vol = np.pi ** (m / 2.0) / scipy_gamma(m / 2.0 + 1.0)
    
    volume = unit_sphere_vol * (r ** m) / np.sqrt(det_a)
    return volume


def cvt_sampler_nonuniform(n_generators: int, sample_num: int, it_num: int,
                            density_func: Callable[[np.ndarray], np.ndarray],
                            bounds: np.ndarray = None) -> np.ndarray:
    if bounds is None:
        bounds = np.array([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]])
    
    m = bounds.shape[0]
    

    generators = np.zeros((n_generators, m))
    for dim in range(m):
        generators[:, dim] = np.random.uniform(bounds[dim, 0], bounds[dim, 1], n_generators)
    
    for it in range(it_num):

        samples = np.zeros((sample_num, m))
        for dim in range(m):
            samples[:, dim] = np.random.uniform(bounds[dim, 0], bounds[dim, 1], sample_num)
        

        densities = density_func(samples)
        

        densities = np.maximum(densities, 1e-15)
        


        k = np.zeros(sample_num, dtype=int)
        for i in range(sample_num):
            dists = np.linalg.norm(generators - samples[i], axis=1)
            k[i] = np.argmin(dists)
        

        new_generators = np.zeros_like(generators)
        for i in range(n_generators):
            mask = (k == i)
            if np.sum(mask) == 0:
                new_generators[i] = generators[i]
            else:
                weights = densities[mask]
                weighted_sum = np.sum(weights[:, None] * samples[mask], axis=0)
                total_weight = np.sum(weights)
                new_generators[i] = weighted_sum / total_weight
        
        generators = new_generators
    

    for dim in range(m):
        generators[:, dim] = np.clip(generators[:, dim], bounds[dim, 0], bounds[dim, 1])
    
    return generators


def uniform_kpoint_grid(bounds: np.ndarray, grid_size: int) -> np.ndarray:
    m = bounds.shape[0]
    
    axes = []
    for dim in range(m):
        axes.append(np.linspace(bounds[dim, 0], bounds[dim, 1], grid_size))
    
    mesh = np.meshgrid(*axes, indexing='ij')
    kpoints = np.stack([m.ravel() for m in mesh], axis=-1)
    return kpoints


def adaptive_weyl_node_sampler(n_points: int, weyl_nodes: np.ndarray,
                                node_radius: float = 0.5,
                                bz_bounds: np.ndarray = None) -> np.ndarray:
    if bz_bounds is None:
        bz_bounds = np.array([[-np.pi, np.pi], [-np.pi, np.pi], [-np.pi, np.pi]])
    
    n_nodes = weyl_nodes.shape[0] if weyl_nodes.ndim > 1 else 1
    

    n_near = int(0.7 * n_points)
    n_global = n_points - n_near
    
    samples_near = []
    if n_nodes > 0:
        points_per_node = n_near // n_nodes
        for i in range(n_nodes):
            node = weyl_nodes[i] if weyl_nodes.ndim > 1 else weyl_nodes

            a = np.eye(3)
            pts = ellipse_sample(points_per_node, a, node_radius).T
            pts += node

            for dim in range(3):
                pts[:, dim] = np.clip(pts[:, dim], bz_bounds[dim, 0], bz_bounds[dim, 1])
            samples_near.append(pts)
    

    global_samples = np.zeros((n_global, 3))
    for dim in range(3):
        global_samples[:, dim] = np.random.uniform(
            bz_bounds[dim, 0], bz_bounds[dim, 1], n_global
        )
    
    if len(samples_near) > 0:
        all_samples = np.vstack(samples_near + [global_samples])
    else:
        all_samples = global_samples
    

    if all_samples.shape[0] > n_points:
        idx = np.random.choice(all_samples.shape[0], n_points, replace=False)
        all_samples = all_samples[idx]
    
    return all_samples
