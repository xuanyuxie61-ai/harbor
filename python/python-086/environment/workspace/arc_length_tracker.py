# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import spsolve, eigsh
from scipy.sparse import csr_matrix


class ArcLengthTracker:

    def __init__(self, initial_arc_length: float = 0.01, min_arc_length: float = 1e-5,
                 max_arc_length: float = 0.5, adaptivity: float = 0.5,
                 psi_scale: float = 1.0, max_recursion_depth: int = 5):
        self.ds = initial_arc_length
        self.ds_min = min_arc_length
        self.ds_max = max_arc_length
        self.adaptivity = adaptivity
        self.psi_scale = psi_scale
        self.max_recursion = max_recursion_depth
        self.path_history = []

    def _compute_psi(self, u: np.ndarray, lambda_val: float,
                     f_ext_norm: float) -> float:
        u_norm = np.linalg.norm(u) + 1e-14
        f_norm = f_ext_norm + 1e-14
        if abs(lambda_val) < 1e-10:
            psi = self.psi_scale * u_norm / f_norm
        else:
            psi = self.psi_scale * u_norm / (abs(lambda_val) * f_norm)

        psi_min = 1.0 / f_norm
        return max(psi, psi_min)

    def _solve_correction(self, fem_model, u_pred: np.ndarray, lambda_pred: float,
                          f_ext: np.ndarray, u0: np.ndarray, lambda0: float,
                          free_dofs: np.ndarray) -> tuple:
















        raise NotImplementedError("Hole 3: 请实现 _solve_correction")


    def track_path(self, fem_model, external_force: np.ndarray,
                   n_steps: int = 20, lambda_max: float = 2.0) -> dict:
        n_dof = fem_model.n_dof
        f_ext = external_force
        f_norm = np.linalg.norm(f_ext) + 1e-14


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

        if len(free_dofs) == 0:
            return {'path': [], 'bifurcation_points': [], 'n_steps': 0}


        K0 = fem_model.assemble_linear_stiffness()
        K0_ff = K0[free_dofs][:, free_dofs]
        u0 = np.zeros(n_dof)
        try:
            u0_f = spsolve(K0_ff, f_ext[free_dofs])
        except Exception:
            u0_f = spsolve(K0_ff + 1e-8 * csr_matrix(np.eye(len(free_dofs))), f_ext[free_dofs])
        u0[free_dofs] = u0_f
        lambda0 = 0.0

        path = [{
            'lambda': lambda0,
            'disp': u0.copy(),
            'max_disp': 0.0,
            'det_sign': 1.0,
            'min_eig': 1.0
        }]

        u = u0.copy()
        lam = lambda0

        prev_det_sign = 1.0
        prev_min_eig = 1.0
        bifurcation_points = []

        for step in range(n_steps):
            if abs(lam) >= lambda_max:
                break


            K_lin = fem_model.assemble_linear_stiffness()
            K_geo = fem_model.assemble_geometric_stiffness(u)
            K_T = K_lin + K_geo
            K_ff = K_T[free_dofs][:, free_dofs]
            try:
                delta_u_F = spsolve(K_ff, f_ext[free_dofs])
            except Exception:
                delta_u_F = spsolve(K_ff + 1e-8 * csr_matrix(np.eye(len(free_dofs))), f_ext[free_dofs])

            psi = self._compute_psi(u[free_dofs], lam, np.linalg.norm(f_ext[free_dofs]) + 1e-14)



            target_dlambda = 0.05
            tangent_norm_sq = np.dot(delta_u_F, delta_u_F) + psi ** 2
            if tangent_norm_sq < 1e-20:
                tangent_norm_sq = 1e-20
            tangent_norm = np.sqrt(tangent_norm_sq)
            self.ds = target_dlambda * tangent_norm
            self.ds = max(self.ds_min, min(self.ds, self.ds_max))


            lambda_dot = 1.0 / tangent_norm
            if step > 0 and len(path) >= 2:
                if (path[-1]['lambda'] - path[-2]['lambda']) < 0:
                    lambda_dot = -abs(lambda_dot)
                else:
                    lambda_dot = abs(lambda_dot)

            u_dot = np.zeros(n_dof)
            u_dot[free_dofs] = lambda_dot * delta_u_F


            du_pred = self.ds * u_dot
            dlambda_pred = self.ds * lambda_dot
            u_pred = u + du_pred
            lam_pred = lam + dlambda_pred


            converged = False
            for corr in range(15):
                du_corr, dlambda_corr = self._solve_correction(
                    fem_model, u_pred, lam_pred, f_ext, u, lam, free_dofs)
                u_pred += du_corr
                lam_pred += dlambda_corr


                f_int = fem_model.internal_force(u_pred)
                R = lam_pred * f_ext - f_int
                norm_r = np.linalg.norm(R[free_dofs])
                norm_f = np.linalg.norm((lam_pred * f_ext)[free_dofs]) + 1e-14


                du_c = u_pred[free_dofs] - u[free_dofs]
                dl_c = lam_pred - lam
                g_res = abs(np.dot(du_c, du_c) + psi ** 2 * dl_c ** 2 - self.ds ** 2)

                if norm_r / norm_f < 1e-5 and g_res < self.ds ** 2 * 1e-3:
                    converged = True
                    break


            if not converged or abs(lam_pred) > 10.0 * lambda_max or np.isnan(lam_pred):
                self.ds = max(self.ds * 0.25, self.ds_min)
                if self.ds <= self.ds_min:
                    break
                continue

            u = u_pred
            lam = lam_pred


            n_iter = corr + 1
            if n_iter <= 3:
                self.ds = min(self.ds * 1.5, self.ds_max)
            elif n_iter >= 10:
                self.ds = max(self.ds * 0.5, self.ds_min)


            K_T_corr = fem_model.assemble_linear_stiffness() + fem_model.assemble_geometric_stiffness(u)
            K_T_ff = K_T_corr[free_dofs][:, free_dofs]
            try:
                eigvals = eigsh(K_T_ff, k=1, which='SM', return_eigenvectors=False, tol=1e-2)
                min_eig = float(eigvals[0])
            except Exception:
                min_eig = float(np.linalg.cond(K_T_ff.toarray()))
                min_eig = 1.0 / min_eig if min_eig > 0 else 0.0

            det_sign = np.sign(min_eig) if abs(min_eig) > 1e-10 else 0.0
            if prev_det_sign * det_sign < 0 and step > 0:
                bifurcation_points.append({
                    'step': step,
                    'lambda': float(lam),
                    'min_eig': float(min_eig),
                    'prev_min_eig': float(prev_min_eig)
                })
            prev_det_sign = det_sign
            prev_min_eig = min_eig

            max_w = np.max(np.abs(u[2::3])) if n_dof >= 3 else 0.0
            path.append({
                'lambda': float(lam),
                'disp': u.copy(),
                'max_disp': float(max_w),
                'det_sign': float(det_sign),
                'min_eig': float(min_eig),
                'arc_length': float(self.ds)
            })

        return {
            'path': path,
            'bifurcation_points': bifurcation_points,
            'n_steps': len(path) - 1
        }

    def chirikov_stability_indicator(self, path: list) -> np.ndarray:
        n = len(path)
        if n < 4:
            return np.array([])
        indicators = np.zeros(n - 2)
        winding = []
        for i in range(1, n):
            dw = path[i]['max_disp'] - path[i - 1]['max_disp']
            dl = path[i]['lambda'] - path[i - 1]['lambda']
            winding.append(np.arctan2(dl, dw + 1e-14))

        for i in range(1, len(winding) - 1):
            nu = winding[i] - winding[i - 1]
            nu_next = winding[i + 1] - winding[i]
            jump = abs(nu_next - nu)
            if jump > np.pi / 4.0:
                indicators[i - 1] = 1.0
        return indicators
