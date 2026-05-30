# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


class StabilityAnalyzer:

    def __init__(self, fem_model):
        self.fem = fem_model

    def tangent_stiffness_eigenvalues(self, u: np.ndarray, k: int = 5) -> dict:
        n_dof = self.fem.n_dof
        K_lin = self.fem.assemble_linear_stiffness()
        K_geo = self.fem.assemble_geometric_stiffness(u)
        K_T = K_lin + K_geo

        bottom, top = self.fem.mesh.get_boundary_nodes()
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
            return {'eigenvalues': np.array([]), 'stable': True}

        K_ff = K_T[free_dofs][:, free_dofs]
        k_eff = min(k, len(free_dofs) - 1)
        if k_eff < 1:
            return {'eigenvalues': np.array([1.0]), 'stable': True}

        try:
            eigvals, eigvecs = eigsh(K_ff, k=k_eff, which='SM', tol=1e-4)
            stable = np.all(eigvals > -1e-8)
            return {
                'eigenvalues': eigvals,
                'eigenvectors': eigvecs,
                'stable': stable,
                'min_eig': float(np.min(eigvals))
            }
        except Exception:

            K_dense = K_ff.toarray()
            eigvals = np.linalg.eigvalsh(K_dense)
            stable = np.all(eigvals > -1e-8)
            return {
                'eigenvalues': eigvals[:k_eff],
                'eigenvectors': None,
                'stable': stable,
                'min_eig': float(np.min(eigvals))
            }

    def lyapunov_exponent_discrete(self, path: list, perturbation_scale: float = 1e-6,
                                   n_iter: int = 50) -> float:
        if len(path) < 3:
            return 0.0

        exponents = []
        for i in range(1, min(n_iter, len(path) - 1)):
            du = path[i + 1]['disp'] - path[i]['disp']
            du_prev = path[i]['disp'] - path[i - 1]['disp']
            dl = path[i + 1]['lambda'] - path[i]['lambda']
            dl_prev = path[i]['lambda'] - path[i - 1]['lambda']

            norm_curr = np.sqrt(np.dot(du, du) + dl ** 2) + 1e-14
            norm_prev = np.sqrt(np.dot(du_prev, du_prev) + dl_prev ** 2) + 1e-14
            ratio = norm_curr / norm_prev
            exponents.append(np.log(max(ratio, 1e-14)))

        if not exponents:
            return 0.0
        return float(np.mean(exponents))

    def chirikov_overlap_criterion(self, path: list, mode_spacing: int = 2) -> bool:
        if len(path) < 2:
            return False
        t = self.fem.mesh.geom.t
        max_w = max([p['max_disp'] for p in path])
        epsilon = max_w / t

        n_avg = 5.0
        delta_n = float(mode_spacing)
        epsilon_crit = (delta_n / (2.0 * n_avg)) ** 2
        return epsilon > epsilon_crit

    def koiter_bifurcation_class(self, path: list) -> str:
        if len(path) < 3:
            return "undetermined"
        slopes = []
        for i in range(1, len(path)):
            dxi = path[i]['max_disp'] - path[i - 1]['max_disp']
            dl = path[i]['lambda'] - path[i - 1]['lambda']
            if abs(dxi) > 1e-12:
                slopes.append(dl / dxi)

        if not slopes:
            return "undetermined"

        has_negative = any(s < 0 for s in slopes)
        has_extreme = any(slopes[i] * slopes[i - 1] < 0 for i in range(1, len(slopes)))

        if has_extreme:
            return "snap-through"
        elif has_negative:
            return "symmetric-unstable"
        else:
            return "symmetric-stable"

    def energy_barrier(self, path: list) -> float:
        if len(path) < 2:
            return 0.0

        lambda_values = [p['lambda'] for p in path]
        max_lambda = max(lambda_values)
        min_lambda = min(lambda_values)
        return float(max_lambda - min_lambda)
