# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix


class NewtonRaphsonSolver:

    def __init__(self, max_iter: int = 50, tol_force: float = 1e-6,
                 tol_disp: float = 1e-8, line_search: bool = True):
        self.max_iter = max_iter
        self.tol_force = tol_force
        self.tol_disp = tol_disp
        self.line_search = line_search
        self.history = []

    def solve(self, fem_model, external_force: np.ndarray, lambda_load: float,
              u0: np.ndarray = None) -> dict:
        n_dof = fem_model.n_dof
        f_ext = external_force
        if u0 is None:
            u = np.zeros(n_dof)
        else:
            u = np.array(u0, dtype=float)

        f_target = lambda_load * f_ext
        energy_history = []

        for it in range(self.max_iter):





















            raise NotImplementedError("Hole 2: 请实现 Newton-Raphson 核心迭代")



        return {
            'disp': u,
            'converged': False,
            'iterations': self.max_iter,
            'residual_norm': norm_r,
            'energy': energy_history[-1] if energy_history else 0.0,
            'history': self.history
        }

    def _armijo_line_search(self, fem_model, u, du, f_target, f_int_old,
                            c1: float = 1e-4, alpha_max: float = 1.0,
                            max_ls_iter: int = 10) -> float:
        alpha = alpha_max
        norm_r0 = np.linalg.norm(f_target - f_int_old)
        for _ in range(max_ls_iter):
            u_trial = u + alpha * du
            f_int_trial = fem_model.internal_force(u_trial)
            norm_r = np.linalg.norm(f_target - f_int_trial)
            if norm_r <= (1.0 - 2.0 * alpha * c1) * norm_r0:
                return alpha
            alpha *= 0.5
        return alpha


class PseudoTimeSolver:

    def __init__(self, damping_ratio: float = 0.9, dt: float = 0.01,
                 max_steps: int = 2000):
        self.damping_ratio = damping_ratio
        self.dt = dt
        self.max_steps = max_steps

    def solve(self, fem_model, external_force: np.ndarray, lambda_load: float,
              mass_lumping: np.ndarray = None) -> dict:
        n_dof = fem_model.n_dof
        f_target = lambda_load * external_force
        u = np.zeros(n_dof)
        v = np.zeros(n_dof)


        if mass_lumping is None:
            rho = fem_model.mat.rho
            t = fem_model.mesh.geom.t
            area = fem_model.mesh.geom.surface_area()
            m_val = rho * t * area / n_dof
            mass = np.full(n_dof, m_val)
        else:
            mass = np.array(mass_lumping)


        bottom, top = fem_model.mesh.get_boundary_nodes()
        fixed_dofs = []
        for nid in bottom:
            fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
        for nid in top:
            fixed_dofs.extend([nid * 3 + 0, nid * 3 + 1])
        if len(bottom) > 0:
            fixed_dofs.append(bottom[0] * 3 + 2)
        fixed_dofs = np.unique(fixed_dofs)
        free_dofs = np.setdiff1d(np.arange(n_dof), fixed_dofs)


        energy_history = []

        for step in range(self.max_steps):
            f_int = fem_model.internal_force(u)
            R = f_target - f_int

            a = np.zeros(n_dof)
            m_f = mass[free_dofs]
            m_f = np.where(m_f < 1e-14, 1e-14, m_f)
            a[free_dofs] = R[free_dofs] / m_f

            a[free_dofs] -= 2.0 * self.damping_ratio * np.sqrt(np.max(mass)) * v[free_dofs] / m_f

            if not np.all(np.isfinite(a[free_dofs])):
                break
            v += a * self.dt
            u += v * self.dt


            u[fixed_dofs] = 0.0
            v[fixed_dofs] = 0.0


            kinetic = 0.5 * np.sum(mass[free_dofs] * v[free_dofs] ** 2)
            potential = 0.5 * np.dot(u, f_int) - np.dot(u, f_target)
            if np.isfinite(kinetic) and np.isfinite(potential):
                energy_history.append(kinetic + potential)
            else:
                break


            if step > 50 and len(energy_history) >= 30:
                recent = np.abs(energy_history[-30:])
                avg_energy = np.mean(recent)
                max_energy = np.max(recent)
                if avg_energy > 1e-14 and max_energy / avg_energy < 1.05:
                    rel_res = np.linalg.norm(R[free_dofs]) / (np.linalg.norm(f_target[free_dofs]) + 1e-14)
                    if rel_res < 1e-3:
                        return {
                            'disp': u,
                            'converged': True,
                            'steps': step + 1,
                            'energy_history': energy_history,
                            'residual_norm': rel_res
                        }

        rel_res = np.linalg.norm(R[free_dofs]) / (np.linalg.norm(f_target[free_dofs]) + 1e-14)
        if not np.isfinite(rel_res):
            rel_res = 999.0
        return {
            'disp': u,
            'converged': False,
            'steps': step + 1,
            'energy_history': energy_history,
            'residual_norm': rel_res
        }
