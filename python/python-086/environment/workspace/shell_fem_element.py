# -*- coding: utf-8 -*-

import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple


class ShellMaterial:

    def __init__(self, E: float, nu: float, rho: float = 7850.0):
        if E <= 0.0:
            raise ValueError("杨氏模量必须为正")
        if not (0.0 <= nu < 0.5):
            raise ValueError("泊松比必须在 [0, 0.5) 范围内")
        self.E = float(E)
        self.nu = float(nu)
        self.rho = float(rho)

    def extensional_rigidity(self, t: float) -> float:
        return self.E * t / (1.0 - self.nu ** 2)

    def bending_rigidity(self, t: float) -> float:
        return self.E * t ** 3 / (12.0 * (1.0 - self.nu ** 2))

    def membrane_matrix(self, t: float) -> np.ndarray:
        C = self.extensional_rigidity(t)
        nu = self.nu
        return C * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)]
        ])

    def bending_matrix(self, t: float) -> np.ndarray:
        D = self.bending_rigidity(t)
        nu = self.nu
        return D * np.array([
            [1.0, nu, 0.0],
            [nu, 1.0, 0.0],
            [0.0, 0.0, 0.5 * (1.0 - nu)]
        ])


class ShellFEModel:

    def __init__(self, mesh, material: ShellMaterial):
        self.mesh = mesh
        self.mat = material
        self.n_dof_per_node = 3
        self.n_nodes = mesh.n_nodes
        self.n_dof = self.n_nodes * self.n_dof_per_node

        self._compute_element_geometry()

    def _compute_element_geometry(self):
        self.elem_areas = np.zeros(self.mesh.n_elem)
        self.elem_dNdx = np.zeros((self.mesh.n_elem, 3, 3))
        self.elem_dNdy = np.zeros((self.mesh.n_elem, 3, 3))
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            coords = self.mesh.nodes[nodes]




            R = self.mesh.geom.R
            theta = np.arctan2(coords[:, 1], coords[:, 0])
            x_axial = coords[:, 2]

            x1, y1 = theta[0], x_axial[0]
            x2, y2 = theta[1], x_axial[1]
            x3, y3 = theta[2], x_axial[2]
            J = np.array([
                [x2 - x1, x3 - x1],
                [y2 - y1, y3 - y1]
            ])
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-14:
                detJ = 1e-14
            self.elem_areas[eid] = 0.5 * abs(detJ) * R
            Jinv = np.linalg.inv(J)

            dN_dxi = np.array([-1.0, 1.0, 0.0])
            dN_deta = np.array([-1.0, 0.0, 1.0])
            dNdx = Jinv[0, 0] * dN_dxi + Jinv[0, 1] * dN_deta
            dNdy = Jinv[1, 0] * dN_dxi + Jinv[1, 1] * dN_deta
            self.elem_dNdx[eid, :, 0] = dNdx
            self.elem_dNdx[eid, :, 1] = dNdy
            self.elem_dNdx[eid, :, 2] = np.zeros(3)

    def _b_matrix_membrane(self, eid: int) -> np.ndarray:





        raise NotImplementedError("Hole 1: 请实现 _b_matrix_membrane")


    def _b_matrix_bending(self, eid: int) -> np.ndarray:
        R = self.mesh.geom.R
        A = self.elem_areas[eid]
        if A < 1e-20:
            A = 1e-20



        Bb = np.zeros((3, 9))
        nodes = self.mesh.elements[eid]
        coords = self.mesh.nodes[nodes]

        e1 = coords[1] - coords[0]
        e2 = coords[2] - coords[1]
        e3 = coords[0] - coords[2]

        for i in range(3):
            col = i * 3 + 2

            le = np.linalg.norm([e1, e2, e3][i])
            if le > 0:
                Bb[0, col] = -1.0 / (le * R)
                Bb[1, col] = -1.0 / (le * R ** 2)
        return Bb

    def assemble_linear_stiffness(self) -> csr_matrix:
        t = self.mesh.geom.t
        Cm = self.mat.membrane_matrix(t)
        Cb = self.mat.bending_matrix(t)
        row_idx = []
        col_idx = []
        data = []
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            Bm = self._b_matrix_membrane(eid)
            Bb = self._b_matrix_bending(eid)
            Ae = self.elem_areas[eid]
            ke = (Bm.T @ Cm @ Bm + Bb.T @ Cb @ Bb) * Ae

            dofs = []
            for nid in nodes:
                dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
            dofs = np.array(dofs, dtype=int)
            for i in range(9):
                for j in range(9):
                    if abs(ke[i, j]) > 1e-18:
                        row_idx.append(dofs[i])
                        col_idx.append(dofs[j])
                        data.append(ke[i, j])
        K = csr_matrix((data, (row_idx, col_idx)), shape=(self.n_dof, self.n_dof))
        return K

    def assemble_geometric_stiffness(self, disp: np.ndarray) -> csr_matrix:
        t = self.mesh.geom.t
        Cm = self.mat.membrane_matrix(t)
        row_idx = []
        col_idx = []
        data = []
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            Bm = self._b_matrix_membrane(eid)
            Ae = self.elem_areas[eid]
            dofs = []
            for nid in nodes:
                dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
            dofs = np.array(dofs, dtype=int)
            ue = disp[dofs]

            eps = Bm @ ue

            sigma = Cm @ eps
            Nx, Ntheta, Nxtheta = sigma[0], sigma[1], sigma[2]


            R = self.mesh.geom.R
            for i in range(3):
                for j in range(3):
                    ii = nodes[i] * 3 + 2
                    jj = nodes[j] * 3 + 2
                    val = (Nx * 1.0 + Ntheta / (R ** 2) + Nxtheta * 0.5) * Ae / 9.0
                    if abs(val) > 1e-18:
                        row_idx.append(ii)
                        col_idx.append(jj)
                        data.append(val)
        Kg = csr_matrix((data, (row_idx, col_idx)), shape=(self.n_dof, self.n_dof))
        return Kg

    def internal_force(self, disp: np.ndarray) -> np.ndarray:
        t = self.mesh.geom.t
        Cm = self.mat.membrane_matrix(t)
        Cb = self.mat.bending_matrix(t)
        fint = np.zeros(self.n_dof)
        for eid in range(self.mesh.n_elem):
            nodes = self.mesh.elements[eid]
            Bm = self._b_matrix_membrane(eid)
            Bb = self._b_matrix_bending(eid)
            Ae = self.elem_areas[eid]
            dofs = []
            for nid in nodes:
                dofs.extend([nid * 3 + 0, nid * 3 + 1, nid * 3 + 2])
            dofs = np.array(dofs, dtype=int)
            ue = disp[dofs]
            eps = Bm @ ue
            kap = Bb @ ue
            N = Cm @ eps
            M = Cb @ kap
            fe = (Bm.T @ N + Bb.T @ M) * Ae
            fint[dofs] += fe
        return fint
