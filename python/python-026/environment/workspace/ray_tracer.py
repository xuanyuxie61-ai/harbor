# -*- coding: utf-8 -*-

import numpy as np
from physics_constants import plasma_frequency, C_LIGHT


class RayTracer:

    def __init__(self, omega0, eta_min=1e-4, max_steps=50000):
        self.omega0 = float(omega0)
        if self.omega0 <= 0:
            raise ValueError("激光角频率必须为正。")
        self.eta_min = float(eta_min)
        self.max_steps = int(max_steps)

    def _eta_and_gradient(self, x, y, density_interp_func):
        ne = density_interp_func(x, y)
        omega_p = plasma_frequency(ne)
        ratio = (omega_p / self.omega0) ** 2
        ratio = np.clip(ratio, 0.0, 1.0)
        eta = np.sqrt(1.0 - ratio)


        delta = max(1e-8 * max(abs(x), abs(y), 1.0), 1e-12)

        ne_px = density_interp_func(x + delta, y)
        ne_mx = density_interp_func(x - delta, y)
        ne_py = density_interp_func(x, y + delta)
        ne_my = density_interp_func(x, y - delta)

        omega_p_px = plasma_frequency(ne_px)
        omega_p_mx = plasma_frequency(ne_mx)
        omega_p_py = plasma_frequency(ne_py)
        omega_p_my = plasma_frequency(ne_my)

        ratio_px = np.clip((omega_p_px / self.omega0) ** 2, 0.0, 1.0)
        ratio_mx = np.clip((omega_p_mx / self.omega0) ** 2, 0.0, 1.0)
        ratio_py = np.clip((omega_p_py / self.omega0) ** 2, 0.0, 1.0)
        ratio_my = np.clip((omega_p_my / self.omega0) ** 2, 0.0, 1.0)

        eta_px = np.sqrt(1.0 - ratio_px)
        eta_mx = np.sqrt(1.0 - ratio_mx)
        eta_py = np.sqrt(1.0 - ratio_py)
        eta_my = np.sqrt(1.0 - ratio_my)

        grad_eta = np.array([
            (eta_px - eta_mx) / (2.0 * delta),
            (eta_py - eta_my) / (2.0 * delta)
        ], dtype=float)

        return eta, grad_eta

    def trace_ray(self, r0, k0, density_interp_func, domain_bounds,
                  ds_init=1e-7, ds_min=1e-12, ds_max=1e-5, courant_factor=0.5):
        r0 = np.asarray(r0, dtype=float)
        k0 = np.asarray(k0, dtype=float)
        if len(r0) != 2 or len(k0) != 2:
            raise ValueError("r0 和 k0 必须是二维向量。")

        (xmin, xmax), (ymin, ymax) = domain_bounds
        dx_dom = xmax - xmin
        dy_dom = ymax - ymin


        ne0 = density_interp_func(r0[0], r0[1])
        omega_p0 = plasma_frequency(ne0)
        ratio0 = np.clip((omega_p0 / self.omega0) ** 2, 0.0, 1.0)
        eta0 = np.sqrt(1.0 - ratio0)
        if eta0 < self.eta_min:
            return r0.reshape(1, -1), k0.reshape(1, -1), np.array([0.0]), 'cutoff'


        r = r0.copy()
        k = k0.copy()
        s = 0.0
        ds = ds_init

        traj = [r.copy()]
        k_traj = [k.copy()]
        s_list = [0.0]

        status = 'ok'

        for step in range(self.max_steps):

            eta_r, grad_eta = self._eta_and_gradient(r[0], r[1], density_interp_func)

            if eta_r < self.eta_min:
                status = 'cutoff'
                break

            k_norm = np.linalg.norm(k)
            if k_norm < 1e-20:
                status = 'stagnation'
                break



            def rhs(r_in, k_in):
                eta_tmp, grad_eta_tmp = self._eta_and_gradient(r_in[0], r_in[1], density_interp_func)
                if eta_tmp < self.eta_min:
                    return np.zeros(2), np.zeros(2)
                drds = k_in / np.linalg.norm(k_in)
                dkds = (self.omega0 / C_LIGHT) * grad_eta_tmp
                return drds, dkds


            dr1, dk1 = rhs(r, k)
            dr2, dk2 = rhs(r + 0.5 * ds * dr1, k + 0.5 * ds * dk1)
            dr3, dk3 = rhs(r + 0.5 * ds * dr2, k + 0.5 * ds * dk2)
            dr4, dk4 = rhs(r + ds * dr3, k + ds * dk3)

            r_new = r + (ds / 6.0) * (dr1 + 2 * dr2 + 2 * dr3 + dr4)
            k_new = k + (ds / 6.0) * (dk1 + 2 * dk2 + 2 * dk3 + dk4)


            if not (xmin <= r_new[0] <= xmax and ymin <= r_new[1] <= ymax):

                alpha = 1.0
                for dim, (lo, hi) in enumerate(zip([xmin, ymin], [xmax, ymax])):
                    if r_new[dim] < lo:
                        a = (lo - r[dim]) / (r_new[dim] - r[dim]) if r_new[dim] != r[dim] else 0.0
                        alpha = min(alpha, a)
                    elif r_new[dim] > hi:
                        a = (hi - r[dim]) / (r_new[dim] - r[dim]) if r_new[dim] != r[dim] else 0.0
                        alpha = min(alpha, a)
                r_new = r + alpha * (r_new - r)
                k_new = k + alpha * (k_new - k)
                s += alpha * ds
                traj.append(r_new.copy())
                k_traj.append(k_new.copy())
                s_list.append(s)
                status = 'domain_exit'
                break

            r = r_new
            k = k_new
            s += ds

            traj.append(r.copy())
            k_traj.append(k.copy())
            s_list.append(s)


            eta_next = eta_r
            ds_courant = courant_factor * min(dx_dom, dy_dom) * eta_next
            ds = np.clip(ds_courant, ds_min, ds_max)


            grad_norm = np.linalg.norm(grad_eta)
            if grad_norm > 1e3:
                ds = max(ds * 0.5, ds_min)

        else:
            status = 'max_steps'

        trajectory = np.array(traj)
        k_trajectory = np.array(k_traj)
        s_vals = np.array(s_list)

        return trajectory, k_trajectory, s_vals, status

    def trace_beam(self, positions, directions, density_interp_func, domain_bounds):
        N = positions.shape[0]
        results = []
        k0_base = self.omega0 / C_LIGHT
        for i in range(N):
            r0 = positions[i]
            d = directions[i]
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-20:
                d = np.array([1.0, 0.0])
                d_norm = 1.0
            k0 = k0_base * (d / d_norm)
            traj, k_traj, s_vals, status = self.trace_ray(
                r0, k0, density_interp_func, domain_bounds
            )
            results.append({
                'trajectory': traj,
                'k_trajectory': k_traj,
                's_vals': s_vals,
                'status': status,
                'final_position': traj[-1],
                'path_length': s_vals[-1]
            })
        return results
