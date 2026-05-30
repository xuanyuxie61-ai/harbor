
import numpy as np
import math
from typing import Tuple, List






def generate_hexagonal_photoreceptor_array(radius: float, n_rings: int) -> np.ndarray:
    d = 2.0 * radius
    points = []
    

    points.append([0.0, 0.0])
    

    for n in range(1, n_rings + 1):

        for k in range(6 * n):


            edge = k // n
            pos_in_edge = k % n
            

            theta0 = edge * np.pi / 3.0
            theta1 = ((edge + 1) % 6) * np.pi / 3.0
            

            t = pos_in_edge / n if n > 0 else 0.0
            x = n * d * ((1 - t) * np.cos(theta0) + t * np.cos(theta1))
            y = n * d * ((1 - t) * np.sin(theta0) + t * np.sin(theta1))
            points.append([x, y])
    
    return np.array(points, dtype=np.float64)


def hexagon_moment_integral(p: int, q: int, vertices: np.ndarray) -> float:
    n = vertices.shape[0]
    value = 0.0
    
    for i in range(n):
        x0, y0 = vertices[i]
        x1, y1 = vertices[(i + 1) % n]
        dx = x1 - x0
        dy = y1 - y0
        

        for k in range(p + 1):
            for m in range(q + 2):
                binom_pk = math.comb(p, k)
                binom_q1m = math.comb(q + 1, m)
                coeff = binom_pk * binom_q1m
                x_term = (x0 ** (p - k)) * (dx ** k) if p - k >= 0 and k >= 0 else 0.0
                y_term = (y0 ** (q + 1 - m)) * (dy ** m) if q + 1 - m >= 0 and m >= 0 else 0.0
                denom = (q + 1) * (k + m + 1)
                if denom != 0:
                    value += coeff * x_term * y_term / denom
    
    return float(value)






def _orient2d(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _in_circumcircle(a: np.ndarray, b: np.ndarray, c: np.ndarray, p: np.ndarray) -> bool:
    ax, ay = a[0], a[1]
    bx, by = b[0], b[1]
    cx, cy = c[0], c[1]
    px, py = p[0], p[1]
    
    det = (
        (ax - px) * ((by - py) * (cx**2 + cy**2 - px**2 - py**2) 
                     - (cy - py) * (bx**2 + by**2 - px**2 - py**2))
        - (ay - py) * ((bx - px) * (cx**2 + cy**2 - px**2 - py**2)
                       - (cx - px) * (bx**2 + by**2 - px**2 - py**2))
        + (ax**2 + ay**2 - px**2 - py**2) * ((bx - px) * (cy - py) - (by - py) * (cx - px))
    )
    
    orient = _orient2d(a, b, c)
    return det * orient > 1e-12


def delaunay_triangulation_2d(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    triangles = []
    
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a, b, c = points[i], points[j], points[k]
                

                orient = _orient2d(a, b, c)
                if abs(orient) < 1e-12:
                    continue
                

                if orient < 0:
                    b, c = c, b
                    j_tmp, k_tmp = k, j
                else:
                    j_tmp, k_tmp = j, k
                

                is_delaunay = True
                for p_idx in range(n):
                    if p_idx in (i, j_tmp, k_tmp):
                        continue
                    if _in_circumcircle(a, c, b, points[p_idx]):
                        is_delaunay = False
                        break
                
                if is_delaunay:
                    triangles.append([i, j_tmp, k_tmp])
    
    return np.array(triangles, dtype=np.int64)






def extract_mesh_boundary(triangles: np.ndarray) -> np.ndarray:
    edges = []
    for tri in triangles:

        edges.append((tri[0], tri[1]))
        edges.append((tri[1], tri[2]))
        edges.append((tri[2], tri[0]))
    

    edge_set = set(edges)
    boundary_edges = []
    
    for e in edges:
        reverse = (e[1], e[0])
        if reverse not in edge_set:
            boundary_edges.append(e)
    
    if not boundary_edges:
        return np.array([]).reshape(0, 2)
    

    boundary_edges = list(dict.fromkeys(boundary_edges))
    

    ordered = [boundary_edges[0]]
    remaining = boundary_edges[1:]
    
    while remaining:
        current_end = ordered[-1][1]
        found = False
        for idx, e in enumerate(remaining):
            if e[0] == current_end:
                ordered.append(e)
                remaining.pop(idx)
                found = True
                break
        if not found:
            break
    
    return np.array(ordered, dtype=np.int64)






def triangle_quality_alpha(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:

    a2 = np.sum((q - r) ** 2)
    b2 = np.sum((p - r) ** 2)
    c2 = np.sum((p - q) ** 2)
    

    eps = 1e-14
    a = np.sqrt(a2)
    b = np.sqrt(b2)
    c = np.sqrt(c2)
    

    cos_p = np.clip((b2 + c2 - a2) / (2.0 * b * c + eps), -1.0, 1.0)
    cos_q = np.clip((a2 + c2 - b2) / (2.0 * a * c + eps), -1.0, 1.0)
    cos_r = np.clip((a2 + b2 - c2) / (2.0 * a * b + eps), -1.0, 1.0)
    
    angle_p = np.arccos(cos_p)
    angle_q = np.arccos(cos_q)
    angle_r = np.arccos(cos_r)
    
    min_angle = min(angle_p, angle_q, angle_r)
    alpha = min_angle / (np.pi / 3.0)
    
    return float(alpha)


def triangle_quality_q(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    a = np.linalg.norm(q - r)
    b = np.linalg.norm(p - r)
    c = np.linalg.norm(p - q)
    

    area = 0.5 * abs(_orient2d(p, q, r))
    
    s = 0.5 * (a + b + c)
    eps = 1e-14
    
    if s < eps or a < eps or b < eps or c < eps or area < eps:
        return 0.0
    
    r_in = area / s
    R_out = a * b * c / (4.0 * area + eps)
    q_measure = 2.0 * r_in / (R_out + eps)
    
    return float(q_measure)


def evaluate_mesh_quality(points: np.ndarray, triangles: np.ndarray) -> dict:
    n_tri = triangles.shape[0]
    alphas = []
    qs = []
    areas = []
    
    for tri in triangles:
        p, q, r = points[tri[0]], points[tri[1]], points[tri[2]]
        alphas.append(triangle_quality_alpha(p, q, r))
        qs.append(triangle_quality_q(p, q, r))
        areas.append(0.5 * abs(_orient2d(p, q, r)))
    
    alphas = np.array(alphas)
    qs = np.array(qs)
    areas = np.array(areas)
    
    boundary = extract_mesh_boundary(triangles)
    
    quality = {
        'alpha_min': float(np.min(alphas)),
        'alpha_ave': float(np.mean(alphas)),
        'q_min': float(np.min(qs)),
        'q_ave': float(np.mean(qs)),
        'area_min': float(np.min(areas)),
        'area_max': float(np.max(areas)),
        'area_ave': float(np.mean(areas)),
        'area_std': float(np.std(areas)),
        'boundary_segments': boundary,
        'num_triangles': n_tri,
        'num_boundary_edges': boundary.shape[0] if boundary.size > 0 else 0,
    }
    
    return quality






def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    n = points.shape[0]
    if n <= 3:
        return np.arange(n)
    

    start = 0
    for i in range(1, n):
        if points[i, 0] < points[start, 0] or \
           (abs(points[i, 0] - points[start, 0]) < 1e-12 and points[i, 1] < points[start, 1]):
            start = i
    
    hull = []
    current = start
    
    while True:
        hull.append(current)
        next_point = (current + 1) % n
        
        for i in range(n):
            if i == current:
                continue

            cross = _orient2d(points[current], points[next_point], points[i])
            if cross > 1e-12:
                next_point = i
            elif abs(cross) < 1e-12:

                d_next = np.sum((points[next_point] - points[current]) ** 2)
                d_i = np.sum((points[i] - points[current]) ** 2)
                if d_i > d_next:
                    next_point = i
        
        current = next_point
        if current == start:
            break
    
    return np.array(hull, dtype=np.int64)
