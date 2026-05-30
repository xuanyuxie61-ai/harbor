# -*- coding: utf-8 -*-

import numpy as np




_LEBEDEV_6_X = np.array([ 0.0,  0.0,  0.0,  0.0,  1.0, -1.0], dtype=np.float64)
_LEBEDEV_6_Y = np.array([ 0.0,  0.0,  1.0, -1.0,  0.0,  0.0], dtype=np.float64)
_LEBEDEV_6_Z = np.array([ 1.0, -1.0,  0.0,  0.0,  0.0,  0.0], dtype=np.float64)
_LEBEDEV_6_W = np.array([0.1666666666666667, 0.1666666666666667,
                         0.1666666666666667, 0.1666666666666667,
                         0.1666666666666667, 0.1666666666666667], dtype=np.float64)

_LEBEDEV_14_X = np.array([
    0.0, 0.0, 0.0, 0.0, 1.0, -1.0,
    0.5773502691896258, -0.5773502691896258,  0.5773502691896258,
   -0.5773502691896258,  0.5773502691896258, -0.5773502691896258,
    0.5773502691896258, -0.5773502691896258
], dtype=np.float64)
_LEBEDEV_14_Y = np.array([
    0.0, 0.0, 1.0, -1.0, 0.0, 0.0,
    0.5773502691896258, -0.5773502691896258,  0.5773502691896258,
   -0.5773502691896258, -0.5773502691896258,  0.5773502691896258,
   -0.5773502691896258,  0.5773502691896258
], dtype=np.float64)
_LEBEDEV_14_Z = np.array([
    1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
    0.5773502691896258, -0.5773502691896258, -0.5773502691896258,
    0.5773502691896258,  0.5773502691896258, -0.5773502691896258,
   -0.5773502691896258,  0.5773502691896258
], dtype=np.float64)
_LEBEDEV_14_W = np.array([
    0.0666666666666667, 0.0666666666666667, 0.0666666666666667, 0.0666666666666667,
    0.0666666666666667, 0.0666666666666667,
    0.0750000000000000, 0.0750000000000000, 0.0750000000000000, 0.0750000000000000,
    0.0750000000000000, 0.0750000000000000, 0.0750000000000000, 0.0750000000000000
], dtype=np.float64)


def lebedev_by_order(order):
    if order == 6:
        return _LEBEDEV_6_X.copy(), _LEBEDEV_6_Y.copy(), _LEBEDEV_6_Z.copy(), _LEBEDEV_6_W.copy()
    elif order == 14:
        return _LEBEDEV_14_X.copy(), _LEBEDEV_14_Y.copy(), _LEBEDEV_14_Z.copy(), _LEBEDEV_14_W.copy()
    else:
        raise ValueError(f"lebedev_by_order: unsupported order {order}, use 6 or 14")


class CurvatureFlow:

    def __init__(self, levelset):
        self.ls = levelset

    def compute_mean_curvature_flow_velocity(self):
        return self.ls.compute_curvature()

    def compute_willmore_flow_rhs(self):
        kappa = self.ls.compute_curvature()
        phi = self.ls.phi
        dx, dy = self.ls.dx, self.ls.dy
        nx, ny = self.ls.nx, self.ls.ny


        lap_kappa = np.zeros_like(kappa)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                lap_kappa[i, j] = (kappa[i + 1, j] - 2.0 * kappa[i, j] + kappa[i - 1, j]) / (dx * dx) \
                                  + (kappa[i, j + 1] - 2.0 * kappa[i, j] + kappa[i, j - 1]) / (dy * dy)

        lap_kappa[0, :] = lap_kappa[1, :]
        lap_kappa[-1, :] = lap_kappa[-2, :]
        lap_kappa[:, 0] = lap_kappa[:, 1]
        lap_kappa[:, -1] = lap_kappa[:, -2]

        _, _, grad_norm = self.ls.compute_gradient_norm()
        rhs = (lap_kappa + 0.5 * kappa ** 3) * grad_norm

        rhs = np.clip(rhs, -1e4, 1e4)
        return rhs

    @staticmethod
    def integrate_on_sphere_surface(f_vals, order=14):
        x, y, z, w = lebedev_by_order(order)
        if len(f_vals) != len(w):
            raise ValueError("integrate_on_sphere_surface: f_vals length must match quadrature nodes")
        integral = 4.0 * np.pi * np.sum(w * f_vals)
        return integral

    @staticmethod
    def sphere_distance(lat1, lon1, lat2, lon2, R=1.0):
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
        return R * c

    def compute_willmore_energy(self):
        phi = self.ls.phi
        kappa = self.ls.compute_curvature()
        _, _, grad_norm = self.ls.compute_gradient_norm()
        dx, dy = self.ls.dx, self.ls.dy
        eps = 1.5 * max(dx, dy)

        delta = np.zeros_like(phi)
        mask = np.abs(phi) < eps
        delta[mask] = (1.0 / (2.0 * eps)) * (1.0 + np.cos(np.pi * phi[mask] / eps))

        W = np.sum(kappa ** 2 * delta * grad_norm) * dx * dy
        return W

    def compute_surface_area(self):
        return self.ls.compute_interface_length()

    def compute_gauss_map_variance(self):
        nx_vec, ny_vec = self.ls.compute_normal()

        theta = np.arctan2(ny_vec, nx_vec)

        phi = self.ls.phi
        eps = 1.5 * max(self.ls.dx, self.ls.dy)
        mask = np.abs(phi) < eps
        if np.sum(mask) == 0:
            return 0.0
        theta_masked = theta[mask]
        var = np.var(theta_masked)
        return var
