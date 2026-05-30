# -*- coding: utf-8 -*-

import numpy as np
from numerical_utils import weno5_derivative, tvd_rk3_step, central_diff_2nd


class HJSolver:

    def __init__(self, levelset, epsilon=0.01, gamma=0.0):
        self.ls = levelset
        self.epsilon = epsilon
        self.gamma = gamma
        self.dx = levelset.dx
        self.dy = levelset.dy

    def _rhs_curvature_flow(self, phi):
        ls_tmp = type(self.ls)(self.ls.nx, self.ls.ny,
                               self.ls.xlim, self.ls.ylim)
        ls_tmp.phi = phi.copy()
        kappa = ls_tmp.compute_curvature()
        _, _, grad_norm = ls_tmp.compute_gradient_norm()
        rhs = self.epsilon * kappa * grad_norm

        rhs[0, :] = rhs[1, :]
        rhs[-1, :] = rhs[-2, :]
        rhs[:, 0] = rhs[:, 1]
        rhs[:, -1] = rhs[:, -2]
        return rhs

    def _rhs_advection(self, phi, u_field, v_field):
        dx, dy = self.dx, self.dy
        phi_x = central_diff_2nd(phi, dx, axis=0)
        phi_y = central_diff_2nd(phi, dy, axis=1)
        rhs = -(u_field * phi_x + v_field * phi_y)
        return rhs

    def _rhs_combined(self, phi, u_field, v_field, forcing):


        raise NotImplementedError("HOLE_2: Combined RHS implementation missing")

    def step_rk3(self, dt, u_field=None, v_field=None, forcing=None):
        phi0 = self.ls.phi.copy()

        if u_field is None:
            u_field = np.zeros_like(phi0)
        if v_field is None:
            v_field = np.zeros_like(phi0)
        if forcing is None:
            forcing = np.zeros_like(phi0)

        def rhs_func(phi):
            return self._rhs_combined(phi, u_field, v_field, forcing)

        phi_new = tvd_rk3_step(phi0, dt, rhs_func)
        self.ls.phi = phi_new
        return self

    def compute_cfl_dt(self, u_field=None, v_field=None, forcing=None, cfl=0.5):
        phi0 = self.ls.phi.copy()
        if u_field is None:
            u_field = np.zeros_like(phi0)
        if v_field is None:
            v_field = np.zeros_like(phi0)
        if forcing is None:
            forcing = np.zeros_like(phi0)

        ls_tmp = type(self.ls)(self.ls.nx, self.ls.ny,
                               self.ls.xlim, self.ls.ylim)
        ls_tmp.phi = phi0.copy()
        kappa = ls_tmp.compute_curvature()
        Vn = np.abs(self.epsilon * kappa) + np.abs(forcing)
        vel_max = np.max(Vn + np.abs(u_field) + np.abs(v_field))
        if vel_max < 1e-14:
            vel_max = 1.0
        dt_max = cfl * min(self.dx, self.dy) / vel_max
        return dt_max

    @staticmethod
    def exact_tanh_1d(x, t, x0=0.0, V=0.5, delta=0.05):
        return np.tanh((x - x0 - V * t) / delta)

    def compute_error_vs_exact(self, t, exact_func):
        X, Y = np.meshgrid(self.ls.x, self.ls.y, indexing='ij')
        phi_exact = exact_func(X, Y, t)
        diff = self.ls.phi - phi_exact
        error = np.sqrt(np.sum(diff ** 2) * self.dx * self.dy)
        return error


class ShearFlow:

    @staticmethod
    def simple_shear(X, Y, shear_rate=1.0):
        u = shear_rate * Y
        v = np.zeros_like(Y)
        return u, v

    @staticmethod
    def vortex_pair(X, Y, strength=1.0, center1=(-0.3, 0.0), center2=(0.3, 0.0)):
        def point_vortex(cx, cy, Gamma):
            r2 = (X - cx) ** 2 + (Y - cy) ** 2
            r2 = np.maximum(r2, 1e-8)
            u = -Gamma * (Y - cy) / (2.0 * np.pi * r2)
            v = Gamma * (X - cx) / (2.0 * np.pi * r2)
            return u, v

        u1, v1 = point_vortex(center1[0], center1[1], strength)
        u2, v2 = point_vortex(center2[0], center2[1], -strength)
        return u1 + u2, v1 + v2

    @staticmethod
    def oscillatory_shear(X, Y, t, freq=2.0, amp=1.0):
        u = amp * np.sin(2.0 * np.pi * freq * t) * Y
        v = np.zeros_like(Y)
        return u, v
