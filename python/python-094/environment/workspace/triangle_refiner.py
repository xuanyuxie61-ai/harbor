
import numpy as np


def triangle_refine_centroids(c, t):
    t = np.asarray(t, dtype=float)
    if c == 0:
        centroid = np.mean(t, axis=0)
        return centroid.reshape(1, 2)


    m01 = 0.5 * (t[0] + t[1])
    m12 = 0.5 * (t[1] + t[2])
    m20 = 0.5 * (t[2] + t[0])


    sub_triangles = [
        np.vstack([t[0], m01, m20]),
        np.vstack([m01, t[1], m12]),
        np.vstack([m20, m12, t[2]]),
        np.vstack([m01, m12, m20])
    ]

    centroids = []
    for sub in sub_triangles:
        centroids.append(triangle_refine_centroids(c - 1, sub))
    return np.vstack(centroids)


def triangle_refine_num(c):
    return 4 ** int(c)


def triangle_refine_quad(c, t, f):
    from geometry_utils import triangle_area
    centroids = triangle_refine_centroids(c, t)
    n = triangle_refine_num(c)
    area = triangle_area(t)
    if n == 0:
        return 0.0
    f_vals = f(centroids)
    return np.sum(f_vals) * area / n


def triangulation_q2l_to_linear(triangle_node1):
    triangle_node1 = np.asarray(triangle_node1, dtype=int)
    if triangle_node1.ndim != 2:
        raise ValueError("triangle_node1 must be 2D.")
    if triangle_node1.shape[0] != 6:
        raise ValueError("triangle_node1 must have 6 rows (quadratic triangles).")

    n_tri = triangle_node1.shape[1]
    triangle_node2 = np.zeros((3, 4 * n_tri), dtype=int)

    for tri1 in range(n_tri):
        n = triangle_node1[:, tri1]
        tri2 = tri1 * 4
        triangle_node2[:, tri2] = [n[0], n[3], n[5]]
        triangle_node2[:, tri2 + 1] = [n[1], n[4], n[3]]
        triangle_node2[:, tri2 + 2] = [n[2], n[5], n[4]]
        triangle_node2[:, tri2 + 3] = [n[3], n[4], n[5]]

    return triangle_node2


def adaptive_triangle_refine(triangles, nodes, indicator_func, threshold,
                              max_level=3, current_level=0):
    refined_triangles = []
    refined_levels = []

    for tri in triangles:
        tri = np.asarray(tri)
        if tri.ndim == 1 and nodes is not None:
            tri_coords = nodes[tri, :]
        else:
            tri_coords = tri

        centroid = np.mean(tri_coords, axis=0)
        indicator = indicator_func(centroid)

        if indicator > threshold and current_level < max_level:

            m01 = 0.5 * (tri_coords[0] + tri_coords[1])
            m12 = 0.5 * (tri_coords[1] + tri_coords[2])
            m20 = 0.5 * (tri_coords[2] + tri_coords[0])
            sub_tris = [
                np.vstack([tri_coords[0], m01, m20]),
                np.vstack([m01, tri_coords[1], m12]),
                np.vstack([m20, m12, tri_coords[2]]),
                np.vstack([m01, m12, m20])
            ]
            sub_results, sub_levels = adaptive_triangle_refine(
                sub_tris, None, indicator_func, threshold,
                max_level, current_level + 1)
            refined_triangles.extend(sub_results)
            refined_levels.extend(sub_levels)
        else:
            refined_triangles.append(tri_coords)
            refined_levels.append(current_level)

    return refined_triangles, refined_levels


class AdaptiveMeshRefinement:

    def __init__(self, base_nodes, base_triangles, max_level=3):
        self.nodes = np.asarray(base_nodes, dtype=float)
        self.triangles = np.asarray(base_triangles, dtype=int)
        self.max_level = int(max_level)

    def refine_by_gradient(self, field_values, gradient_threshold):
        field_values = np.asarray(field_values, dtype=float)
        triangles_list = [self.nodes[tri, :] for tri in self.triangles]

        def indicator(centroid):



            dists = np.sum((self.nodes - centroid) ** 2, axis=1)
            nearest = np.argpartition(dists, min(2, len(dists) - 1))[:3]
            vals = field_values[nearest]
            return np.std(vals) / (np.mean(dists[nearest]) ** 0.5 + 1e-10)

        refined, levels = adaptive_triangle_refine(
            triangles_list, None, indicator, gradient_threshold, self.max_level)
        return refined, levels

    def compute_refined_integral(self, f, base_indicator=None, threshold=1.0):
        triangles_list = [self.nodes[tri, :] for tri in self.triangles]

        if base_indicator is None:
            base_indicator = lambda c: 0.0

        refined, levels = adaptive_triangle_refine(
            triangles_list, None, base_indicator, threshold, self.max_level)

        total = 0.0
        for tri in refined:

            total += triangle_refine_quad(2, tri, f)

        return total, len(refined)
