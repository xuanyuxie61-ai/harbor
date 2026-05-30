# -*- coding: utf-8 -*-

import numpy as np


class AdaptiveMesh:

    def __init__(self, levelset, h_min=0.01, h_max=0.1, h_band=0.2):
        self.ls = levelset
        self.h_min = h_min
        self.h_max = h_max
        self.h_band = h_band

    def compute_size_function(self):
        phi = np.abs(self.ls.phi)
        h = self.h_min + (self.h_max - self.h_min) * np.tanh(phi / self.h_band)
        return h

    def refine_grid_uniform(self, factor=2):
        nx, ny = self.ls.nx, self.ls.ny
        xlim, ylim = self.ls.xlim, self.ls.ylim
        new_nx = nx * factor
        new_ny = ny * factor

        x_new = np.linspace(xlim[0], xlim[1], new_nx)
        y_new = np.linspace(ylim[0], ylim[1], new_ny)
        X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')


        x_old = self.ls.x
        y_old = self.ls.y
        phi_new = np.zeros((new_nx, new_ny), dtype=np.float64)

        for i in range(new_nx):
            for j in range(new_ny):
                xi = X_new[i, j]
                yi = Y_new[i, j]

                ii = np.searchsorted(x_old, xi) - 1
                jj = np.searchsorted(y_old, yi) - 1
                ii = np.clip(ii, 0, nx - 2)
                jj = np.clip(jj, 0, ny - 2)

                tx = (xi - x_old[ii]) / (x_old[ii + 1] - x_old[ii])
                ty = (yi - y_old[jj]) / (y_old[jj + 1] - y_old[jj])

                phi00 = self.ls.phi[ii, jj]
                phi10 = self.ls.phi[ii + 1, jj]
                phi01 = self.ls.phi[ii, jj + 1]
                phi11 = self.ls.phi[ii + 1, jj + 1]

                phi_new[i, j] = (1 - tx) * (1 - ty) * phi00 \
                                + tx * (1 - ty) * phi10 \
                                + (1 - tx) * ty * phi01 \
                                + tx * ty * phi11

        self.ls.nx = new_nx
        self.ls.ny = new_ny
        self.ls.x = x_new
        self.ls.y = y_new
        self.ls.dx = (xlim[1] - xlim[0]) / (new_nx - 1)
        self.ls.dy = (ylim[1] - ylim[0]) / (new_ny - 1)
        self.ls.phi = phi_new
        return self

    def cvt_optimize_nodes_2d(self, num_points=100, max_iter=50, tol=1e-4):
        xlim, ylim = self.ls.xlim, self.ls.ylim

        nx_pts = int(np.sqrt(num_points))
        ny_pts = int(np.ceil(num_points / nx_pts))
        x_pts = np.linspace(xlim[0] + 0.05, xlim[1] - 0.05, nx_pts)
        y_pts = np.linspace(ylim[0] + 0.05, ylim[1] - 0.05, ny_pts)
        X, Y = np.meshgrid(x_pts, y_pts, indexing='ij')
        points = np.column_stack([X.ravel(), Y.ravel()])[:num_points, :]


        points += 0.01 * (np.random.rand(*points.shape) - 0.5)


        alpha = 50.0


        phi = self.ls.phi
        x_grid = self.ls.x
        y_grid = self.ls.y
        rho_grid = 1.0 / (1.0 + alpha * phi ** 2)


        for it in range(max_iter):
            points_new = np.zeros_like(points)


            dxg = x_grid[1] - x_grid[0]
            dyg = y_grid[1] - y_grid[0]

            for pidx in range(len(points)):
                px, py = points[pidx]

                ix = np.argmin(np.abs(x_grid - px))
                iy = np.argmin(np.abs(y_grid - py))

                wx = max(1, int(0.2 / dxg))
                wy = max(1, int(0.2 / dyg))
                i0 = max(0, ix - wx)
                i1 = min(len(x_grid), ix + wx + 1)
                j0 = max(0, iy - wy)
                j1 = min(len(y_grid), iy + wy + 1)

                Xg, Yg = np.meshgrid(x_grid[i0:i1], y_grid[j0:j1], indexing='ij')
                Rg = rho_grid[i0:i1, j0:j1]


                dists = np.sqrt((Xg - px) ** 2 + (Yg - py) ** 2)

                min_dists = dists.copy()
                for qidx in range(len(points)):
                    if qidx == pidx:
                        continue
                    qx, qy = points[qidx]
                    d2 = np.sqrt((Xg - qx) ** 2 + (Yg - qy) ** 2)
                    min_dists = np.minimum(min_dists, d2)

                mask = dists <= min_dists + 1e-8
                if np.sum(mask) == 0:
                    points_new[pidx] = points[pidx]
                else:
                    wsum = np.sum(Rg[mask])
                    if wsum < 1e-14:
                        points_new[pidx] = points[pidx]
                    else:
                        cx = np.sum(Xg[mask] * Rg[mask]) / wsum
                        cy = np.sum(Yg[mask] * Rg[mask]) / wsum
                        points_new[pidx] = [cx, cy]

            diff = np.max(np.linalg.norm(points_new - points, axis=1))
            points = points_new.copy()
            if diff < tol:
                break

        return points

    @staticmethod
    def triangle_area(p1, p2, p3):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        p3 = np.asarray(p3)
        if p1.ndim == 1:
            return 0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1])
                          - (p2[1] - p1[1]) * (p3[0] - p1[0]))
        else:
            return 0.5 * ((p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1])
                          - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0]))

    @staticmethod
    def triangle_quality(p1, p2, p3):
        A = abs(AdaptiveMesh.triangle_area(p1, p2, p3))
        L1 = np.linalg.norm(p2 - p1)
        L2 = np.linalg.norm(p3 - p2)
        L3 = np.linalg.norm(p1 - p3)
        s2 = L1 ** 2 + L2 ** 2 + L3 ** 2
        if s2 < 1e-14:
            return 0.0
        Q = 4.0 * np.sqrt(3.0) * A / s2
        return Q

    def estimate_interface_mesh_quality(self):
        points = self.ls.get_zero_levelset_points()
        if len(points) < 3:
            return 0.0


        cx = np.mean(points[:, 0])
        cy = np.mean(points[:, 1])
        angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
        idx = np.argsort(angles)
        points_sorted = points[idx]

        qualities = []
        n = len(points_sorted)
        for i in range(n):
            p1 = points_sorted[i]
            p2 = points_sorted[(i + 1) % n]
            p3 = points_sorted[(i + 2) % n]
            q = self.triangle_quality(p1, p2, p3)
            qualities.append(q)

        return np.mean(qualities) if qualities else 0.0
