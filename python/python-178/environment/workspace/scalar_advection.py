
import numpy as np
from typing import Callable, Optional
from mesh_io import TetrahedralMesh
from dg_basis import ModalBasisTet
from tetrahedron_geometry import reference_to_physical_tet4, jacobian_tet4
from quadrature_rules import TETRAHEDRON_QUADRATURE_RULES


class ScalarDGSolver3D:
    def __init__(self, mesh: TetrahedralMesh, poly_order: int = 2,
                 ax: float = 1.0, ay: float = 0.5, az: float = 0.25):
        self.mesh = mesh
        self.p = poly_order
        self.basis = ModalBasisTet(poly_order)
        self.dof_per_elem = self.basis.dof
        self.n_elem = mesh.n_elem
        self.a = np.array([ax, ay, az], dtype=np.float64)
        self.U = np.zeros((self.n_elem, self.dof_per_elem), dtype=np.float64)
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
        self.elem_basis_vals = np.zeros((self.n_elem, nq, self.dof_per_elem), dtype=np.float64)
        self.elem_basis_grads = np.zeros((self.n_elem, nq, self.dof_per_elem, 3), dtype=np.float64)
        for e in range(self.n_elem):
            verts = self.mesh.nodes[self.mesh.elements[e]]
            for q in range(nq):
                xi, eta, zeta = self.quad_pts_ref[q]
                self.elem_quad_pts[e, q] = reference_to_physical_tet4(verts, xi, eta, zeta)
                J = jacobian_tet4(verts)
                detJ = abs(np.linalg.det(J))
                if detJ < 1e-30:
                    detJ = 1e-30
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
        faces = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]], dtype=np.int64)
        for e in range(self.n_elem):
            for f in range(4):
                normal, area = self.mesh.face_normal_and_area(e, f)
                self.face_normals[(e, f)] = normal
                self.face_areas[(e, f)] = area
                face_verts_global = self.mesh.elements[e][faces[f]]
                face_verts = self.mesh.nodes[face_verts_global]
                pts = []
                wts = []
                for q in range(nq_face):
                    lam = self.face_quad_ref[q]
                    pt = lam[0] * face_verts[0] + lam[1] * face_verts[1] + lam[2] * face_verts[2]
                    pts.append(pt)
                    wts.append(self.face_quad_wts[q] * 2.0 * area)
                self.face_quad_pts[(e, f)] = np.array(pts, dtype=np.float64)
                self.face_quad_wts_phys[(e, f)] = np.array(wts, dtype=np.float64)

                verts = self.mesh.nodes[self.mesh.elements[e]]
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
            rhs = np.zeros(self.dof_per_elem, dtype=np.float64)
            for q in range(nq):
                w = self.elem_quad_wts[e, q]
                phi = self.elem_basis_vals[e, q]
                x, y, z = self.elem_quad_pts[e, q]
                u_val = ic_func(x, y, z)
                M += w * np.outer(phi, phi)
                rhs += w * phi * u_val
            try:
                self.U[e] = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                self.U[e] = np.linalg.lstsq(M, rhs, rcond=None)[0]

    def _eval_at_quad(self, e: int, q: int) -> float:
        return float(self.elem_basis_vals[e, q] @ self.U[e])

    def _eval_at_face_quad(self, e: int, f: int, q: int) -> float:
        return float(self.face_basis_vals[(e, f)][q] @ self.U[e])

    def compute_rhs(self, boundary_func: Optional[Callable] = None) -> np.ndarray:
        rhs = np.zeros_like(self.U)
        nq = len(self.quad_wts_ref)
        nq_face = 3
        ax, ay, az = self.a

        for e in range(self.n_elem):
            for q in range(nq):
                u_q = self._eval_at_quad(e, q)
                w = self.elem_quad_wts[e, q]
                grad_phi = self.elem_basis_grads[e, q]
                for j in range(self.dof_per_elem):
                    rhs[e, j] += w * u_q * (ax * grad_phi[j, 0] +
                                             ay * grad_phi[j, 1] +
                                             az * grad_phi[j, 2])








        raise NotImplementedError("Hole 3: upwind flux for scalar advection is not implemented.")

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

    def compute_total_integral(self) -> float:
        total = 0.0
        for e in range(self.n_elem):
            for q in range(len(self.quad_wts_ref)):
                w = self.elem_quad_wts[e, q]
                total += w * self._eval_at_quad(e, q)
        return total

    def compute_l2_error(self, exact_func: Callable, t: float) -> float:
        err = 0.0
        for e in range(self.n_elem):
            for q in range(len(self.quad_wts_ref)):
                w = self.elem_quad_wts[e, q]
                x, y, z = self.elem_quad_pts[e, q]
                val = self._eval_at_quad(e, q)
                ex = exact_func(x, y, z, t)
                if not (np.isfinite(val) and np.isfinite(ex)):
                    continue
                diff = val - ex
                err += w * diff * diff
        if not np.isfinite(err) or err < 0.0:
            return 0.0
        return np.sqrt(err)

    def compute_linf_error(self, exact_func: Callable, t: float) -> float:
        max_err = 0.0
        for e in range(self.n_elem):
            for q in range(len(self.quad_wts_ref)):
                x, y, z = self.elem_quad_pts[e, q]
                diff = abs(self._eval_at_quad(e, q) - exact_func(x, y, z, t))
                max_err = max(max_err, diff)
        return max_err
