# -*- coding: utf-8 -*-

import numpy as np


class Reinitializer:

    def __init__(self, levelset, max_iter=100, tol=1e-6):
        self.ls = levelset
        self.max_iter = max_iter
        self.tol = tol
        self.dx = levelset.dx
        self.dy = levelset.dy

    def _smooth_sign(self, phi0):
        nx, ny = self.ls.nx, self.ls.ny
        h = min(self.dx, self.dy)


        grad = np.zeros_like(phi0)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dxp = (phi0[i + 1, j] - phi0[i, j]) / self.dx
                dxm = (phi0[i, j] - phi0[i - 1, j]) / self.dx
                dyp = (phi0[i, j + 1] - phi0[i, j]) / self.dy
                dym = (phi0[i, j] - phi0[i, j - 1]) / self.dy
                grad[i, j] = np.sqrt(0.5 * (dxp ** 2 + dxm ** 2 + dyp ** 2 + dym ** 2))


        grad[0, :] = grad[1, :]
        grad[-1, :] = grad[-2, :]
        grad[:, 0] = grad[:, 1]
        grad[:, -1] = grad[:, -2]

        S = phi0 / np.sqrt(phi0 ** 2 + grad ** 2 * h ** 2 + 1e-12)
        return S

    def _godunov_gradient_magnitude(self, phi):
        nx, ny = phi.shape
        grad_mag = np.zeros_like(phi)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                a = (phi[i, j] - phi[i - 1, j]) / self.dx
                b = (phi[i + 1, j] - phi[i, j]) / self.dx
                c = (phi[i, j] - phi[i, j - 1]) / self.dy
                d_val = (phi[i, j + 1] - phi[i, j]) / self.dy

                term_x = np.maximum(a, 0.0) ** 2 + np.minimum(b, 0.0) ** 2
                term_y = np.maximum(c, 0.0) ** 2 + np.minimum(d_val, 0.0) ** 2
                grad_mag[i, j] = np.sqrt(term_x + term_y)


        grad_mag[0, :] = grad_mag[1, :]
        grad_mag[-1, :] = grad_mag[-2, :]
        grad_mag[:, 0] = grad_mag[:, 1]
        grad_mag[:, -1] = grad_mag[:, -2]
        return grad_mag

    def reinitialize(self, phi0=None):
        if phi0 is None:
            phi0 = self.ls.phi.copy()
        else:
            phi0 = np.asarray(phi0, dtype=np.float64)

        phi = phi0.copy()
        S = self._smooth_sign(phi0)
        h = min(self.dx, self.dy)
        dtau = 0.1 * h

        for it in range(self.max_iter):
            grad_mag = self._godunov_gradient_magnitude(phi)

            phi_new = phi - dtau * S * np.clip(grad_mag - 1.0, -5.0, 5.0)


            phi_new[0, :] = phi_new[1, :]
            phi_new[-1, :] = phi_new[-2, :]
            phi_new[:, 0] = phi_new[:, 1]
            phi_new[:, -1] = phi_new[:, -2]


            phi_max = np.max(np.abs(phi0)) * 2.0 + 1.0
            phi_new = np.clip(phi_new, -phi_max, phi_max)

            diff = np.max(np.abs(phi_new - phi))
            phi = phi_new.copy()

            if diff < self.tol:
                break

        self.ls.phi = phi
        return it + 1, diff

    def reinitialize_jacobi_style(self, phi0=None, omega=1.0):
        if phi0 is None:
            phi0 = self.ls.phi.copy()
        phi = phi0.copy()
        S = self._smooth_sign(phi0)
        h = min(self.dx, self.dy)
        dtau = 0.1 * h

        for it in range(self.max_iter):
            grad_mag = self._godunov_gradient_magnitude(phi)
            phi_new = phi - omega * dtau * S * np.clip(grad_mag - 1.0, -5.0, 5.0)

            phi_new[0, :] = phi_new[1, :]
            phi_new[-1, :] = phi_new[-2, :]
            phi_new[:, 0] = phi_new[:, 1]
            phi_new[:, -1] = phi_new[:, -2]

            phi_max = np.max(np.abs(phi0)) * 2.0 + 1.0
            phi_new = np.clip(phi_new, -phi_max, phi_max)

            diff = np.max(np.abs(phi_new - phi))
            phi = phi_new.copy()
            if diff < self.tol:
                break

        self.ls.phi = phi
        return it + 1, diff

    def fast_marching_brute(self, phi0=None):
        if phi0 is None:
            phi0 = self.ls.phi.copy()

        nx, ny = phi0.shape
        x = self.ls.x
        y = self.ls.y


        points = []
        for i in range(nx):
            for j in range(ny - 1):
                if phi0[i, j] * phi0[i, j + 1] < 0:
                    t = phi0[i, j] / (phi0[i, j] - phi0[i, j + 1])
                    points.append((x[i], y[j] + t * (y[j + 1] - y[j])))
        for i in range(nx - 1):
            for j in range(ny):
                if phi0[i, j] * phi0[i + 1, j] < 0:
                    t = phi0[i, j] / (phi0[i, j] - phi0[i + 1, j])
                    points.append((x[i] + t * (x[i + 1] - x[i]), y[j]))

        if len(points) == 0:
            return phi0.copy()

        points = np.array(points)
        phi_new = np.zeros_like(phi0)

        for i in range(nx):
            for j in range(ny):
                dists = np.sqrt((points[:, 0] - x[i]) ** 2 + (points[:, 1] - y[j]) ** 2)
                dmin = np.min(dists)
                phi_new[i, j] = dmin if phi0[i, j] >= 0 else -dmin

        self.ls.phi = phi_new
        return phi_new

    def check_sdf_property(self):
        phi = self.ls.phi
        nx, ny = phi.shape
        grad_norm = np.zeros_like(phi)

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                dxp = (phi[i + 1, j] - phi[i, j]) / self.dx
                dxm = (phi[i, j] - phi[i - 1, j]) / self.dx
                dyp = (phi[i, j + 1] - phi[i, j]) / self.dy
                dym = (phi[i, j] - phi[i, j - 1]) / self.dy
                grad_norm[i, j] = np.sqrt(0.5 * (dxp ** 2 + dxm ** 2 + dyp ** 2 + dym ** 2))

        error = np.max(np.abs(grad_norm - 1.0))
        return error
