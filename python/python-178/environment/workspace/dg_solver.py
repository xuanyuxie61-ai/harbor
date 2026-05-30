
import numpy as np
from typing import Callable, Optional, Tuple
from mesh_io import TetrahedralMesh
from dg_basis import ModalBasisTet, NodalBasisTet
from tetrahedron_geometry import (reference_to_physical_tet4, jacobian_tet4,
                                   shape_function_linear_tet4)
from quadrature_rules import (TETRAHEDRON_QUADRATURE_RULES,
                               integrate_tetrahedron_monte_carlo)
from euler_equations import (primitive_to_conservative, conservative_to_primitive,
                              rusanov_flux, roe_flux, manufactured_solution_3d,
                              manufactured_source_3d, speed_of_sound, flux_dot_n)
from limiter import optimize_limiting_parameter, dg_slope_limiter
from rbf_reconstruction import rbf_troubled_cell_indicator


class DGSolver3D:
    def __init__(self, mesh: TetrahedralMesh, poly_order: int = 2,
                 use_modal: bool = True, flux_type: str = 'rusanov'):
        self.mesh = mesh
        self.p = poly_order
        self.use_modal = use_modal
        if use_modal:
            self.basis = ModalBasisTet(poly_order)
        else:
            self.basis = NodalBasisTet(poly_order)
        self.dof_per_elem = self.basis.dof
        self.n_vars = 5
        self.n_elem = mesh.n_elem

        self.U = np.zeros((self.n_elem, self.dof_per_elem, self.n_vars), dtype=np.float64)
        self.flux_type = flux_type

        self.quad_order = min(4, 2 * poly_order + 1)
        if self.quad_order not in TETRAHEDRON_QUADRATURE_RULES:
            self.quad_order = max(TETRAHEDRON_QUADRATURE_RULES.keys())
        self._precompute_quadrature()
        self._precompute_face_quadrature()

    def _precompute_quadrature(self):
        rule = TETRAHEDRON_QUADRATURE_RULES[self.quad_order]
        self.quad_pts_ref = rule['points']
        self.quad_wts_ref = rule['weights']
        nq = len(self.quad_wts_ref)
        self.elem_quad_pts = np.zeros((self.n_elem, nq, 3), dtype=np.float64)
        self.elem_quad_wts = np.zeros((self.n_elem, nq), dtype=np.float64)
        self.elem_quad_jac = np.zeros((self.n_elem, nq, 3, 3), dtype=np.float64)
        self.elem_detJ = np.zeros((self.n_elem, nq), dtype=np.float64)
        self.elem_basis_vals = np.zeros((self.n_elem, nq, self.dof_per_elem), dtype=np.float64)
        self.elem_basis_grads = np.zeros((self.n_elem, nq, self.dof_per_elem, 3), dtype=np.float64)
        for e in range(self.n_elem):
            verts = self.mesh.nodes[self.mesh.elements[e]]
            for q in range(nq):
                xi, eta, zeta = self.quad_pts_ref[q]
                self.elem_quad_pts[e, q] = reference_to_physical_tet4(verts, xi, eta, zeta)
                J = jacobian_tet4(verts)
                self.elem_quad_jac[e, q] = J
                detJ = abs(np.linalg.det(J))
                if detJ < 1e-30:
                    detJ = 1e-30
                self.elem_detJ[e, q] = detJ
                self.elem_quad_wts[e, q] = self.quad_wts_ref[q] * detJ

            xi = self.quad_pts_ref[:, 0]
            eta = self.quad_pts_ref[:, 1]
            zeta = self.quad_pts_ref[:, 2]
            self.elem_basis_vals[e] = self.basis.evaluate(xi, eta, zeta)

            for q in range(nq):
                grad_ref = self.basis.gradient(
                    self.quad_pts_ref[q, 0],
                    self.quad_pts_ref[q, 1],
                    self.quad_pts_ref[q, 2]
                )
                J = jacobian_tet4(verts)
                try:
                    JinvT = np.linalg.inv(J).T
                except np.linalg.LinAlgError:
                    JinvT = np.eye(3)
                self.elem_basis_grads[e, q] = grad_ref @ JinvT

    def _precompute_face_quadrature(self):

        self.face_quad_ref = np.array([
            [0.5, 0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
        ], dtype=np.float64)
        self.face_quad_wts = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0], dtype=np.float64)
        nq_face = 3
        self.face_quad_pts = {}
        self.face_quad_wts_phys = {}
        self.face_normals = {}
        self.face_areas = {}
        self.face_basis_vals = {}
        for e in range(self.n_elem):
            for f in range(4):
                normal, area = self.mesh.face_normal_and_area(e, f)
                self.face_normals[(e, f)] = normal
                self.face_areas[(e, f)] = area
                pts = []
                wts = []

                verts = self.mesh.nodes[self.mesh.elements[e]]
                face_verts_global = self.mesh.elements[e][
                    np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])[f]
                ]
                face_verts = self.mesh.nodes[face_verts_global]
                for q in range(nq_face):

                    lam = self.face_quad_ref[q]
                    pt = lam[0] * face_verts[0] + lam[1] * face_verts[1] + lam[2] * face_verts[2]
                    pts.append(pt)
                    wts.append(self.face_quad_wts[q] * 2.0 * area)
                self.face_quad_pts[(e, f)] = np.array(pts, dtype=np.float64)
                self.face_quad_wts_phys[(e, f)] = np.array(wts, dtype=np.float64)

                ref_pts = []
                for pt in pts:
                    from tetrahedron_geometry import physical_to_reference_tet4
                    ref_pt = physical_to_reference_tet4(verts, pt)
                    ref_pts.append(ref_pt)
                ref_pts = np.array(ref_pts)
                self.face_basis_vals[(e, f)] = self.basis.evaluate(
                    ref_pts[:, 0], ref_pts[:, 1], ref_pts[:, 2]
                )

    def set_initial_condition(self, ic_func: Callable):
        for e in range(self.n_elem):
            verts = self.mesh.nodes[self.mesh.elements[e]]
            nq = len(self.quad_wts_ref)

            M = np.zeros((self.dof_per_elem, self.dof_per_elem), dtype=np.float64)
            rhs = np.zeros((self.dof_per_elem, self.n_vars), dtype=np.float64)
            for q in range(nq):
                w = self.elem_quad_wts[e, q]
                phi = self.elem_basis_vals[e, q]
                x, y, z = self.elem_quad_pts[e, q]
                u_val = ic_func(x, y, z)
                M += w * np.outer(phi, phi)
                rhs += w * np.outer(phi, u_val)
            try:
                coeffs = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                coeffs = np.linalg.lstsq(M, rhs, rcond=None)[0]
            self.U[e] = coeffs

    def _eval_solution_at_quad(self, e: int, q: int) -> np.ndarray:
        phi = self.elem_basis_vals[e, q]
        return phi @ self.U[e]

    def _eval_solution_at_face_quad(self, e: int, f: int, q: int) -> np.ndarray:
        phi = self.face_basis_vals[(e, f)][q]
        return phi @ self.U[e]

    def compute_rhs(self, t: float,
                    source_func: Optional[Callable] = None,
                    boundary_func: Optional[Callable] = None) -> np.ndarray:
        rhs = np.zeros_like(self.U)
        nq = len(self.quad_wts_ref)
        nq_face = 3

        for e in range(self.n_elem):
            verts = self.mesh.nodes[self.mesh.elements[e]]
            for q in range(nq):
                u_q = self._eval_solution_at_quad(e, q)
                w = self.elem_quad_wts[e, q]
                phi = self.elem_basis_vals[e, q]
                grad_phi = self.elem_basis_grads[e, q]

                fx = flux_dot_n(u_q, 1.0, 0.0, 0.0)
                fy = flux_dot_n(u_q, 0.0, 1.0, 0.0)
                fz = flux_dot_n(u_q, 0.0, 0.0, 1.0)

                for j in range(self.dof_per_elem):
                    rhs[e, j] += w * (fx * grad_phi[j, 0] +
                                       fy * grad_phi[j, 1] +
                                       fz * grad_phi[j, 2])

                if source_func is not None:
                    x, y, z = self.elem_quad_pts[e, q]
                    s_q = source_func(x, y, z, t)
                    rhs[e, :] += w * np.outer(phi, s_q)










        raise NotImplementedError("Hole 2: surface integral / numerical flux is not implemented.")

        for e in range(self.n_elem):
            M = np.zeros((self.dof_per_elem, self.dof_per_elem), dtype=np.float64)
            for q in range(nq):
                w = self.elem_quad_wts[e, q]
                phi = self.elem_basis_vals[e, q]
                M += w * np.outer(phi, phi)
            try:
                Minv = np.linalg.inv(M)
            except np.linalg.LinAlgError:
                Minv = np.linalg.pinv(M)
            rhs[e] = Minv @ rhs[e]
        return rhs

    def apply_limiter(self):
        if self.p == 0:
            return

        averages = np.zeros((self.n_elem, self.n_vars), dtype=np.float64)
        for e in range(self.n_elem):
            vol = 0.0
            for q in range(len(self.quad_wts_ref)):
                w = self.elem_quad_wts[e, q]
                vol += w
                u_q = self._eval_solution_at_quad(e, q)
                averages[e] += w * u_q
            averages[e] /= (vol + 1e-30)

        centroids = np.zeros((self.n_elem, 3), dtype=np.float64)
        for e in range(self.n_elem):
            centroids[e] = self.mesh.nodes[self.mesh.elements[e]].mean(axis=0)
        for e in range(self.n_elem):

            neighbors = []
            neighbor_centroids = []
            for f in range(4):
                en = self.mesh.face_elements[e, f]
                if en >= 0:
                    neighbors.append(en)
                    neighbor_centroids.append(centroids[en])
            if len(neighbors) == 0:
                continue
            neighbor_avgs = averages[neighbors]
            neighbor_centroids = np.array(neighbor_centroids, dtype=np.float64)
            for var in range(self.n_vars):
                uh_var = self.U[e, :, var]
                avg = averages[e, var]
                nbr = neighbor_avgs[:, var]
                limited = dg_slope_limiter(uh_var, avg, nbr,
                                            centroids[e], neighbor_centroids)
                self.U[e, :, var] = limited

    def compute_element_average(self, e: int) -> np.ndarray:
        vol = 0.0
        avg = np.zeros(self.n_vars, dtype=np.float64)
        for q in range(len(self.quad_wts_ref)):
            w = self.elem_quad_wts[e, q]
            vol += w
            avg += w * self._eval_solution_at_quad(e, q)
        return avg / (vol + 1e-30)

    def compute_global_average(self) -> np.ndarray:
        total_vol = 0.0
        avg = np.zeros(self.n_vars, dtype=np.float64)
        for e in range(self.n_elem):
            elem_avg = self.compute_element_average(e)
            vol = sum(self.elem_quad_wts[e, :])
            total_vol += vol
            avg += vol * elem_avg
        return avg / (total_vol + 1e-30)

    def compute_total_mass(self) -> float:
        mass = 0.0
        for e in range(self.n_elem):
            for q in range(len(self.quad_wts_ref)):
                w = self.elem_quad_wts[e, q]
                u_q = self._eval_solution_at_quad(e, q)
                mass += w * u_q[0]
        return mass

    def compute_total_energy(self) -> float:
        energy = 0.0
        for e in range(self.n_elem):
            for q in range(len(self.quad_wts_ref)):
                w = self.elem_quad_wts[e, q]
                u_q = self._eval_solution_at_quad(e, q)
                energy += w * u_q[4]
        return energy

    def compute_l2_error(self, exact_func: Callable, t: float) -> float:
        err = 0.0
        for e in range(self.n_elem):
            for q in range(len(self.quad_wts_ref)):
                w = self.elem_quad_wts[e, q]
                u_num = self._eval_solution_at_quad(e, q)
                x, y, z = self.elem_quad_pts[e, q]
                u_ex = exact_func(x, y, z, t)
                diff = u_num - u_ex
                err += w * np.dot(diff, diff)
        return np.sqrt(err)
