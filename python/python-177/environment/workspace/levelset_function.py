# -*- coding: utf-8 -*-

import numpy as np
from numerical_utils import central_diff_2nd, laplacian_2d


class LevelSetFunction:

    def __init__(self, nx, ny, xlim=(-1.0, 1.0), ylim=(-1.0, 1.0)):
        if nx < 5 or ny < 5:
            raise ValueError("LevelSetFunction: nx and ny must be at least 5")
        self.nx = nx
        self.ny = ny
        self.xlim = xlim
        self.ylim = ylim
        self.x = np.linspace(xlim[0], xlim[1], nx)
        self.y = np.linspace(ylim[0], ylim[1], ny)
        self.dx = (xlim[1] - xlim[0]) / (nx - 1)
        self.dy = (ylim[1] - ylim[0]) / (ny - 1)
        self.phi = np.zeros((nx, ny), dtype=np.float64)

    def init_circle(self, cx=0.0, cy=0.0, r=0.3):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        self.phi = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r
        return self

    def init_ellipse(self, cx=0.0, cy=0.0, a=0.4, b=0.2, theta=0.0):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        xr = (X - cx) * np.cos(theta) + (Y - cy) * np.sin(theta)
        yr = -(X - cx) * np.sin(theta) + (Y - cy) * np.cos(theta)

        d_alg = np.sqrt((xr / a) ** 2 + (yr / b) ** 2) - 1.0

        self.phi = d_alg * np.minimum(a, b)
        return self

    def init_star_shape(self, cx=0.0, cy=0.0, r0=0.3, amp=0.05, n_peaks=5):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        theta = np.arctan2(Y - cy, X - cx)
        r_target = r0 + amp * np.sin(n_peaks * theta)
        r_actual = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

        self.phi = r_actual - r_target
        return self

    def init_two_circles(self, c1=(-0.3, 0.0), c2=(0.3, 0.0), r=0.25):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        phi1 = np.sqrt((X - c1[0]) ** 2 + (Y - c1[1]) ** 2) - r
        phi2 = np.sqrt((X - c2[0]) ** 2 + (Y - c2[1]) ** 2) - r
        self.phi = np.minimum(phi1, phi2)
        return self

    def init_rectangle(self, cx=0.0, cy=0.0, w=0.5, h=0.3):
        X, Y = np.meshgrid(self.x, self.y, indexing='ij')
        dx = np.maximum(np.abs(X - cx) - w / 2.0, 0.0)
        dy = np.maximum(np.abs(Y - cy) - h / 2.0, 0.0)

        dist_out = np.sqrt(dx ** 2 + dy ** 2)

        dist_in = np.minimum(np.maximum(np.abs(X - cx) - w / 2.0, np.abs(Y - cy) - h / 2.0), 0.0)
        self.phi = dist_out + dist_in
        return self

    def compute_gradient_norm(self):
        phi = self.phi
        dx, dy = self.dx, self.dy
        phi_x = central_diff_2nd(phi, dx, axis=0)
        phi_y = central_diff_2nd(phi, dy, axis=1)
        grad_norm = np.sqrt(phi_x ** 2 + phi_y ** 2)

        grad_norm = np.maximum(grad_norm, 1e-12)
        return phi_x, phi_y, grad_norm

    def compute_curvature(self):



        raise NotImplementedError("HOLE_3: Curvature formula implementation missing")

    def compute_normal(self):
        phi_x, phi_y, grad_norm = self.compute_gradient_norm()
        nx = phi_x / grad_norm
        ny = phi_y / grad_norm
        return nx, ny

    def signed_distance_redistancing_brute(self, max_iter=50, dtau=0.5):
        phi0 = self.phi.copy()
        phi = phi0.copy()
        dx = self.dx
        ny_local = self.ny

        s0 = np.sign(phi0)
        s0 = np.where(s0 == 0, 1.0, s0)

        for _iter in range(max_iter):
            phi_new = phi.copy()
            for i in range(1, self.nx - 1):
                for j in range(1, ny_local - 1):

                    if s0[i, j] > 0:
                        dxp = phi[i + 1, j] - phi[i, j]
                        dxm = phi[i, j] - phi[i - 1, j]
                        dyp = phi[i, j + 1] - phi[i, j]
                        dym = phi[i, j] - phi[i, j - 1]
                    else:
                        dxp = phi[i, j] - phi[i - 1, j]
                        dxm = phi[i + 1, j] - phi[i, j]
                        dyp = phi[i, j] - phi[i, j - 1]
                        dym = phi[i, j + 1] - phi[i, j]

                    gx = np.maximum(dxp / dx, 0.0) ** 2 + np.minimum(dxm / dx, 0.0) ** 2
                    gy = np.maximum(dyp / dx, 0.0) ** 2 + np.minimum(dym / dx, 0.0) ** 2
                    grad_mag = np.sqrt(gx + gy)
                    phi_new[i, j] = phi[i, j] - dtau * s0[i, j] * (grad_mag - 1.0)
            phi = phi_new.copy()

            phi[0, :] = phi[1, :]
            phi[-1, :] = phi[-2, :]
            phi[:, 0] = phi[:, 1]
            phi[:, -1] = phi[:, -2]

        self.phi = phi
        return self

    def compute_volume(self):
        return np.sum(self.phi < 0) * self.dx * self.dy

    def compute_interface_length(self):
        phi = self.phi
        dx, dy = self.dx, self.dy
        eps = 1.5 * max(dx, dy)
        delta = np.zeros_like(phi)
        mask = np.abs(phi) < eps
        delta[mask] = (1.0 / (2.0 * eps)) * (1.0 + np.cos(np.pi * phi[mask] / eps))
        _, _, grad_norm = self.compute_gradient_norm()
        L = np.sum(delta * grad_norm) * dx * dy
        return L

    def get_zero_levelset_points(self):
        points = []
        phi = self.phi
        nx, ny = self.nx, self.ny
        x = self.x
        y = self.y


        for i in range(nx):
            for j in range(ny - 1):
                if phi[i, j] * phi[i, j + 1] < 0:
                    t = phi[i, j] / (phi[i, j] - phi[i, j + 1])
                    px = x[i]
                    py = y[j] + t * (y[j + 1] - y[j])
                    points.append((px, py))


        for i in range(nx - 1):
            for j in range(ny):
                if phi[i, j] * phi[i + 1, j] < 0:
                    t = phi[i, j] / (phi[i, j] - phi[i + 1, j])
                    px = x[i] + t * (x[i + 1] - x[i])
                    py = y[j]
                    points.append((px, py))

        return np.array(points) if points else np.zeros((0, 2))
