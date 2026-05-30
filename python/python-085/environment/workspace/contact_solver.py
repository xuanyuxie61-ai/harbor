import numpy as np
from typing import Tuple, List, Optional
from mesh_generator import TriMesh2D
from fem_assembler import ElasticFEM2D, assemble_contact_gaps, assemble_contact_normals
from banded_solver import BandedSolver
from utils import macaulay_bracket, solve_2x2_symmetric, safe_divide


class SignoriniCoulombContact:

    def __init__(self, fem: ElasticFEM2D, contact_nodes: np.ndarray,
                 friction_coeff: float = 0.3, aug_lag_penalty: float = 1e9,
                 max_iter: int = 100, tol: float = 1e-8):
        self.fem = fem
        self.mesh = fem.mesh
        self.contact_nodes = np.array(contact_nodes, dtype=int)
        self.n_contact = len(contact_nodes)
        self.mu_friction = friction_coeff
        self.c_n = aug_lag_penalty
        self.c_t = aug_lag_penalty * 0.5
        self.max_iter = max_iter
        self.tol = tol
        self.normals = assemble_contact_normals(self.mesh, self.contact_nodes)

        self.lambda_n = np.zeros(self.n_contact)
        self.lambda_t = np.zeros(self.n_contact)

    def _compute_local_gap_and_slip(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g_n = assemble_contact_gaps(self.mesh, u, self.contact_nodes, rigid_surface_y=0.0)
        g_t = np.zeros(self.n_contact)
        for idx, node in enumerate(self.contact_nodes):
            g_t[idx] = u[2 * node]
        return g_n, g_t

    def _augmented_lagrange_update(self, g_n: np.ndarray, g_t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:



        pass

    def _assemble_contact_force(self, u: np.ndarray) -> np.ndarray:
        f_contact = np.zeros(2 * self.mesh.n_nodes)
        g_n, g_t = self._compute_local_gap_and_slip(u)
        active = g_n < 1e-6
        for idx, node in enumerate(self.contact_nodes):
            if active[idx]:
                p_n = max(self.lambda_n[idx] - self.c_n * g_n[idx], 0.0)
                trial_t = self.lambda_t[idx] + self.c_t * g_t[idx]
                max_t = self.mu_friction * p_n
                p_t = np.clip(trial_t, -max_t, max_t)

                f_contact[2 * node] += p_t
                f_contact[2 * node + 1] += p_n
        return f_contact

    def _contact_stiffness_penalty(self) -> np.ndarray:
        n_dof = 2 * self.mesh.n_nodes
        K_c = np.zeros((n_dof, n_dof))

        for idx, node in enumerate(self.contact_nodes):
            K_c[2 * node + 1, 2 * node + 1] += self.c_n
            K_c[2 * node, 2 * node] += self.c_t
        return K_c

    def solve_static(self, f_ext: np.ndarray,
                     fixed_nodes: Optional[np.ndarray] = None,
                     fixed_values: Optional[np.ndarray] = None,
                     dof_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
        n_dof = 2 * self.mesh.n_nodes
        K = self.fem.assemble_global_stiffness()

        if fixed_nodes is not None and fixed_values is not None:
            K_mod, F_mod = self.fem.apply_dirichlet_bc(K, f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
        else:
            K_mod, F_mod = K.copy(), f_ext.copy()
        try:
            u = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            u = np.zeros(n_dof)

        self.lambda_n = np.zeros(self.n_contact)
        self.lambda_t = np.zeros(self.n_contact)

        history = {"residuals": [], "active_set_sizes": []}

        for it in range(self.max_iter):
            g_n, g_t = self._compute_local_gap_and_slip(u)
            active = g_n < 1e-4
            n_active = int(np.sum(active))
            history["active_set_sizes"].append(n_active)


            lambda_n_new, lambda_t_new = self._augmented_lagrange_update(g_n, g_t)


            f_contact = self._assemble_contact_force(u)


            F_total = f_ext + f_contact
            if fixed_nodes is not None and fixed_values is not None:
                K_mod, F_mod = self.fem.apply_dirichlet_bc(K, F_total, fixed_nodes, fixed_values, dof_mask=dof_mask)
            else:
                K_mod, F_mod = K.copy(), F_total.copy()
            try:
                u_new = np.linalg.solve(K_mod, F_mod)
            except np.linalg.LinAlgError:
                break


            du = np.linalg.norm(u_new - u) / max(np.linalg.norm(u_new), 1e-12)
            dlambda = np.linalg.norm(lambda_n_new - self.lambda_n) / max(np.linalg.norm(lambda_n_new), 1e-12)
            res = max(du, dlambda)
            history["residuals"].append(res)

            u = u_new
            self.lambda_n = lambda_n_new
            self.lambda_t = lambda_t_new

            if res < self.tol:
                break

        history["iterations"] = it + 1
        history["final_residual"] = res if it > 0 else 0.0
        return u, history

    def compute_contact_pressure(self, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        g_n, g_t = self._compute_local_gap_and_slip(u)
        p_n = np.maximum(self.lambda_n + self.c_n * g_n, 0.0)
        trial_t = self.lambda_t + self.c_t * g_t
        max_t = self.mu_friction * p_n
        p_t = np.clip(trial_t, -max_t, max_t)
        return p_n, p_t

    def compute_friction_dissipation(self, u: np.ndarray, v: np.ndarray) -> float:
        _, p_t = self.compute_contact_pressure(u)
        diss = 0.0
        for idx, node in enumerate(self.contact_nodes):
            v_t = v[2 * node]
            diss += p_t[idx] * v_t
        return diss


def active_set_newton_contact(fem: ElasticFEM2D, contact_nodes: np.ndarray,
                               f_ext: np.ndarray, friction_coeff: float = 0.3,
                               max_iter: int = 50, tol: float = 1e-10,
                               fixed_nodes: Optional[np.ndarray] = None,
                               fixed_values: Optional[np.ndarray] = None,
                               dof_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, dict]:
    mesh = fem.mesh
    n_dof = 2 * mesh.n_nodes
    K = fem.assemble_global_stiffness()
    u = np.zeros(n_dof)

    history = {"residuals": [], "active_sets": []}

    for it in range(max_iter):
        g_n = assemble_contact_gaps(mesh, u, contact_nodes, rigid_surface_y=0.0)
        active = g_n <= 1e-6
        history["active_sets"].append(int(np.sum(active)))


        n_con = np.sum(active)
        if n_con > 0:
            active_nodes = contact_nodes[active]

            bc_nodes = active_nodes.copy()
            bc_values = np.zeros((n_con, 2))
            for idx, node in enumerate(active_nodes):
                bc_values[idx, 1] = -mesh.nodes[node, 1]

            if fixed_nodes is not None and fixed_values is not None:
                all_nodes = np.concatenate([fixed_nodes, bc_nodes])
                all_values = np.vstack([fixed_values, bc_values])
                if dof_mask is not None:
                    bc_mask = np.ones((len(bc_nodes), 2), dtype=bool)
                    all_mask = np.vstack([dof_mask, bc_mask])
                else:
                    all_mask = None
                K_mod, F_mod = fem.apply_dirichlet_bc(K, f_ext, all_nodes, all_values, dof_mask=all_mask)
            else:
                K_mod, F_mod = fem.apply_dirichlet_bc(K, f_ext, bc_nodes, bc_values)
        else:
            if fixed_nodes is not None and fixed_values is not None:
                K_mod, F_mod = fem.apply_dirichlet_bc(K, f_ext, fixed_nodes, fixed_values, dof_mask=dof_mask)
            else:
                K_mod, F_mod = K.copy(), f_ext.copy()

        try:
            u_new = np.linalg.solve(K_mod, F_mod)
        except np.linalg.LinAlgError:
            break

        res = np.linalg.norm(u_new - u) / max(np.linalg.norm(u_new), 1e-12)
        history["residuals"].append(res)
        u = u_new
        if res < tol:
            break

    history["iterations"] = it + 1
    return u, history
