
import numpy as np
from scipy.spatial import Delaunay

class AdaptiveOceanMesh:

    def __init__(self, bbox=(0.0, 1.0, 0.0, 1.0)):
        self.bbox = bbox

    @staticmethod
    def drectangle(p, x1, x2, y1, y2):
        dx = np.maximum(x1 - p[:, 0], p[:, 0] - x2)
        dy = np.maximum(y1 - p[:, 1], p[:, 1] - y2)
        d = np.maximum(dx, dy)

        inside = (dx < 0) & (dy < 0)
        d[inside] = np.maximum(dx[inside], dy[inside])
        return d

    @staticmethod
    def dcircle(p, xc, yc, r):
        return np.sqrt((p[:, 0] - xc)**2 + (p[:, 1] - yc)**2) - r

    @staticmethod
    def dunion(d1, d2):
        return np.minimum(d1, d2)

    @staticmethod
    def ddiff(d1, d2):
        return np.maximum(d1, -d2)

    def generate_mesh(self, fd, fh, h0=0.05, max_iter=50, dt_mesh=0.2, tol=1e-3):
        xmin, xmax, ymin, ymax = self.bbox
        xspan, yspan = xmax - xmin, ymax - ymin
        area0 = h0**2


        n_estimate = int(1.2 * xspan * yspan / area0)
        p = np.random.rand(n_estimate, 2)
        p[:, 0] = xmin + p[:, 0] * xspan
        p[:, 1] = ymin + p[:, 1] * yspan


        d = fd(p)
        p = p[d < -0.01 * h0]


        if len(p) > 0:
            h_vals = fh(p)
            acceptance = np.random.rand(len(p)) < (h0 / h_vals)**2
            p = p[acceptance]


        n_bnd = max(int(np.ceil(2.0 * (xspan + yspan) / h0)), 4)
        theta_bnd = np.linspace(0, 2*np.pi, n_bnd, endpoint=False)

        edge_pts = []
        n_edge = n_bnd // 4
        for edge in [(xmin, ymin, xmax, ymin), (xmax, ymin, xmax, ymax),
                     (xmax, ymax, xmin, ymax), (xmin, ymax, xmin, ymin)]:
            xs = np.linspace(edge[0], edge[2], n_edge, endpoint=False)
            ys = np.linspace(edge[1], edge[3], n_edge, endpoint=False)
            edge_pts.extend(list(zip(xs, ys)))
        if edge_pts:
            p_bnd = np.array(edge_pts)
            p = np.vstack([p, p_bnd]) if len(p) > 0 else p_bnd


        p = np.unique(np.round(p / (h0 * 0.1)) * (h0 * 0.1), axis=0)

        for it in range(max_iter):
            if len(p) < 3:
                break

            tri = Delaunay(p)
            t = tri.simplices


            e = np.sort(np.vstack([t[:, [0,1]], t[:, [1,2]], t[:, [2,0]]]), axis=1)
            e = np.unique(e, axis=0)


            bar_vec = p[e[:, 1]] - p[e[:, 0]]
            bar_len = np.sqrt(np.sum(bar_vec**2, axis=1))


            pmid = 0.5 * (p[e[:, 0]] + p[e[:, 1]])
            hmid = fh(pmid)
            L0 = hmid


            F_mag = np.maximum(L0 - bar_len, 0.0)
            F_vec = np.zeros_like(p)
            for k in range(len(e)):
                if bar_len[k] > 1e-12:
                    f = F_mag[k] * bar_vec[k] / bar_len[k]
                    F_vec[e[k, 0]] -= f
                    F_vec[e[k, 1]] += f


            p_new = p + dt_mesh * F_vec


            d_new = fd(p_new)
            boundary = np.abs(d_new) < 0.5 * h0
            if np.any(boundary):

                eps_proj = 1e-6
                grad = np.zeros_like(p_new[boundary])
                grad[:, 0] = (fd(p_new[boundary] + [eps_proj, 0]) - d_new[boundary]) / eps_proj
                grad[:, 1] = (fd(p_new[boundary] + [0, eps_proj]) - d_new[boundary]) / eps_proj
                gnorm = np.sqrt(np.sum(grad**2, axis=1))
                gnorm[gnorm < 1e-12] = 1.0
                p_new[boundary] -= d_new[boundary][:, None] * grad / gnorm[:, None]


            d_out = fd(p_new)
            keep = d_out < 0.01 * h0
            p_new = p_new[keep]

            disp = np.max(np.sqrt(np.sum((p_new - p[keep])**2, axis=1))) if len(p_new) > 0 else 0.0
            p = p_new
            if disp < tol * h0:
                break

        if len(p) < 3:

            nx = max(int(np.ceil((xmax - xmin) / h0)), 2)
            ny = max(int(np.ceil((ymax - ymin) / h0)), 2)
            xg = np.linspace(xmin, xmax, nx)
            yg = np.linspace(ymin, ymax, ny)
            X, Y = np.meshgrid(xg, yg, indexing='ij')
            p = np.column_stack([X.ravel(), Y.ravel()])
            tri = Delaunay(p)
            t = tri.simplices
            return p, t

        tri = Delaunay(p)
        t = tri.simplices
        return p, t

    def compute_mesh_quality(self, p, t):
        p1 = p[t[:, 0]]
        p2 = p[t[:, 1]]
        p3 = p[t[:, 2]]

        a2 = np.sum((p2 - p3)**2, axis=1)
        b2 = np.sum((p3 - p1)**2, axis=1)
        c2 = np.sum((p1 - p2)**2, axis=1)


        area = 0.5 * np.abs((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) -
                            (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))

        quality = 4.0 * np.sqrt(3.0) * area / (a2 + b2 + c2 + 1e-14)
        return quality


def cvt_lloyd_2d(points, density_func, n_iter=20):
    from scipy.spatial import Voronoi
    p = points.copy()
    for _ in range(n_iter):
        vor = Voronoi(p)
        new_p = np.zeros_like(p)
        for i, point in enumerate(p):
            region_idx = vor.point_region[i]
            vertices = vor.regions[region_idx]
            if -1 in vertices or len(vertices) < 3:
                new_p[i] = point
                continue
            poly = np.array([vor.vertices[v] for v in vertices])


            n_samples = 200
            min_xy = np.min(poly, axis=0)
            max_xy = np.max(poly, axis=0)
            samples = np.random.rand(n_samples, 2) * (max_xy - min_xy) + min_xy

            inside = np.zeros(n_samples, dtype=bool)
            for s in range(n_samples):
                x, y = samples[s]
                inside[s] = point_in_polygon(x, y, poly)
            if np.sum(inside) > 0:
                weights = density_func(samples[inside])
                weights = np.maximum(weights, 1e-10)
                new_p[i] = np.average(samples[inside], axis=0, weights=weights)
            else:
                new_p[i] = point
        p = new_p
    return p


def point_in_polygon(x, y, poly):
    n = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside
